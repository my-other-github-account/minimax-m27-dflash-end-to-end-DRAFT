# Section 1 — Hidden-State Trace Generation (llama.cpp + GGUF)

> **Active path.** Single-machine (or a small handful of disjoint-shard machines) producing `[ntok, 6, 3072]` bf16 hidden-state traces from a GGUF-quantized verifier via `llama-dump-hiddens`. No FP8 weights required, no Ray, no 4-node fabric.
>
> The previous vLLM TP=4 / MiniMax-M2.7-FP8 / 4-node recipe is preserved at `repro/legacy/01-generation-vllm-fp8-tp4.md` for anyone who needs to reproduce the original FP8 trace pool exactly. **It is no longer the recommended path** — see the legacy doc's banner for why.

---

## TL;DR

- **One** machine with the IQ4_XS GGUF (4 shards, ~108 GB on disk) is enough to start.
- Add more machines as **disjoint-shard workers** when you want parallelism — each worker gets a half-open `[start, end)` index range and writes to a shared (or rsync'd) trace directory. No coordination protocol; resumability + atomic writes do the work.
- Output: `hs_<idx>.safetensors`, shape `[ntok, 6, 3072]` (5 user-chosen layer taps + final residual), bf16 storage.
- Confirmed end-to-end: this recipe produced the trace pool that trained an IQ4 GGUF-only drafter past the FP8 SMOKE baseline (`p_1=13.0%` at epoch 4 on 3967 traces, vs FP8 SMOKE `9.7%` at epoch 3 on 1176 traces).

## Placeholder key

The doc and harvested scripts use shell-style placeholders so you can drop in your own values. Set them once at the top of your shell session:

| Placeholder | What it is | Example |
|---|---|---|
| `${WORKSPACE}` | Top-level work dir on each machine | `/opt/dflash` |
| `${WORK}` | Per-machine trace-gen workspace (state, prompts, traces, logs) | `${WORKSPACE}/iq4_tracegen` |
| `${MODELS}` | Where verifier GGUF shards sit | `${WORKSPACE}/models/MiniMax-M2.7-GGUF` |
| `${VENV}` | Python venv with `safetensors`, `datasets`, `numpy`, `torch` | `${WORKSPACE}/venvs/dflash` |
| `${BUUN}` | Path to the `buun-llama.cpp` build (provides `llama-dump-hiddens`) | `${WORK}/buun-llama-cpp/build/bin` |
| `node1` … `nodeN` | Hostnames of generation worker boxes | whatever your DNS / `/etc/hosts` resolves |
| `<HEAD_IP>` / `<HEAD_HOST>` | Machine that hosts the canonical trace pool (where files end up) | e.g. `node1` |

The single-machine path needs only `${WORKSPACE}`, `${MODELS}`, `${VENV}`, `${BUUN}`. Multi-machine adds the others.

---

## 1.1 Hardware assumed

| Item | Single-machine baseline | Cluster (recommended) |
|---|---|---|
| Machines | 1 GB10-class (or any host with ≥ 110 GB DRAM/unified memory + NVMe ≥ 200 GB free) | 2–4 of the same |
| GPU | Optional but recommended (CUDA accelerates `llama-dump-hiddens`) | one per worker |
| Disk per worker | ~120 GB (GGUF) + 50–80 GB (per-worker traces) | same |
| Interconnect | n/a | any TCP/SSH; we use QSFP `192.168.200.x/24`, but plain 1 GbE works for `rsync` of finished traces |
| Persistent storage for the pool | one directory on `<HEAD_HOST>`, NFS or rsync target | same |

We've run this with 1 machine (slow but works), 3 machines (`spark-2`/`spark-3`/`spark-4` with disjoint shards `[0,2500)`/`[2500,4500)`/`[4500,6515)`), and 4 machines. The scaling is embarrassingly parallel — each worker is independent, no Ray, no NCCL.

## 1.2 Software pinned versions

| Component | Version / commit | Notes |
|---|---|---|
| `buun-llama.cpp` (provides `llama-dump-hiddens`) | tip of `dump-hiddens` branch as of 2026-04-29 | `cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)` |
| Python | 3.12 in `${VENV}` | needs `safetensors`, `datasets`, `numpy`, `torch` (CPU-only is fine — only used to write tensors) |
| OS | Ubuntu 22.04+, kernel ≥ 6.x | |

## 1.3 Models required

| Model | Path | Size | Notes |
|---|---|---|---|
| MiniMax-M2.7 IQ4_XS GGUF | `${MODELS}/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf` (+ shards 2/3/4) | ~108 GB total | The only verifier artifact you need. **Download once on `<HEAD_HOST>` and either replicate to each worker or NFS-mount.** |

The trainer (Section 2) needs three additional **bridge tensors** extracted from this same GGUF — that step happens on `<HEAD_HOST>` after generation, not during. See `repro/scripts/iq4_tracegen/extract_gguf_bridge.py`.

---

## 1.4 Pre-flight checklist (per-machine)

Run this on each machine that will produce traces:

```bash
# 1. GGUF shard 1 readable
[ -f ${MODELS}/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf ] \
  && echo "  gguf OK" || echo "  GGUF MISSING"

# 2. buun-llama.cpp built
[ -x ${BUUN}/llama-dump-hiddens ] && echo "  binary OK" || echo "  BINARY MISSING"

# 3. Python venv has the imports
${VENV}/bin/python -c "import safetensors, datasets, numpy, torch; print('  venv OK')"

# 4. Disk space (need ~50-80 GB per worker for traces)
df -BG ${WORK} | tail -1
```

A failed `gguf OK` or `binary OK` is fatal — **do not proceed**, fix the missing artifact first. The script will error loudly otherwise.

## 1.5 Build the prompt dataset (once, on `<HEAD_HOST>`)

The trace generator reads from a saved Arrow dataset (`datasets.load_from_disk`). Each row provides a tokenized chat prompt; the worker writes one trace file per row.

If you're reproducing exactly, use the same prompt source the FP8 pool used: ShareGPT-cleaned, deduped, ≤ 2048 tokens — our copy lives at `${WORKSPACE}/iq4_tracegen/prompts_fp8/`. If you're building fresh, the schema is just: an `input_ids` (`int32[seq_len]`) column. See `repro/scripts/iq4_tracegen/filter_dataset_v2.py` for the cleanup we apply.

The prompt dataset must be **identically resolved** on every worker (same row at the same index). The simplest way: build it once on `<HEAD_HOST>`, then `rsync -a ${WORK}/prompts_fp8/  node-N:${WORK}/prompts_fp8/` to each worker.

## 1.6 Single-machine launch

This is the simplest possible invocation — one worker, no parallelism, one log to watch:

```bash
WORKER_ID=A START=0 END=6515 \
  bash repro/scripts/iq4_tracegen/iq4_worker.sh
```

`iq4_worker.sh` is a thin tmux wrapper around `gen_traces_v2.py`. It:

- Self-aborts if the host is `spark-5` (operational hard rule in this project — see `repro/plan/00-NEVER-touch-spark-5.md`). Remove the `case` block at the top if your hosts don't include a forbidden machine.
- Creates `tmux` session `iq4_worker_$WORKER_ID`. Re-running with the same `WORKER_ID` kills+respawns idempotently.
- Logs to `${WORK}/logs/worker_${WORKER_ID}_<timestamp>.log`.
- Tracks durable progress in `${WORK}/state_worker_${WORKER_ID}.json` (FM27 hash chain — see `repro/plan/00-resumability-doctrine.md`).

Watch progress:

```bash
tail -f ${WORK}/logs/worker_A_*.log
ls ${WORK}/traces/hidden_states/ | wc -l
```

Stop cleanly:

```bash
tmux send-keys -t iq4_worker_A C-c
# (the SIGINT handler finishes the in-flight trace, then exits)
```

Resume: just rerun the same `WORKER_ID=A START=0 END=6515 bash …`. It picks up exactly where it left off.

## 1.7 Multi-machine launch — disjoint shards

Pick disjoint half-open ranges. Three-machine layout used in production:

| Worker | Host | `START` | `END` |
|---|---|---|---|
| A | `node1` | 0 | 2500 |
| B | `node2` | 2500 | 4500 |
| C | `node3` | 4500 | 6515 |

```bash
# On node1:
WORKER_ID=A START=0    END=2500 bash repro/scripts/iq4_tracegen/iq4_worker.sh
# On node2:
WORKER_ID=B START=2500 END=4500 bash repro/scripts/iq4_tracegen/iq4_worker.sh
# On node3:
WORKER_ID=C START=4500 END=6515 bash repro/scripts/iq4_tracegen/iq4_worker.sh
```

**Disjointness invariant**: every pair of worker ranges should have empty intersection. This is enforced by your inputs — there is no runtime check. If you accidentally overlap, the skip-existing logic on the destination directory makes the second writer a no-op (atomic writes prevent corruption), but you waste compute. Verify before launching:

```bash
python3 -c "ranges = [(0,2500), (2500,4500), (4500,6515)]; \
            assert all(set(range(*a)).isdisjoint(range(*b)) \
            for i,a in enumerate(ranges) for b in ranges[i+1:]), 'OVERLAP'; \
            print('disjoint OK')"
```

Pool the traces on `<HEAD_HOST>` periodically (or at the end). One direction over QSFP:

```bash
# On <HEAD_HOST>:
for n in node1 node2 node3; do
  rsync -a --info=progress2 \
    ${n}:${WORK}/traces/hidden_states/ \
    ${WORK}/traces/hidden_states/
done
```

`gen_traces_v2.py` writes atomically (`.tmp_*` prefix → `fsync` → `rename` + dir-`fsync`), so an in-flight `rsync` against an actively-writing worker dir is safe — partial files are not visible by their final names.

## 1.8 What `gen_traces_v2.py` actually does

Per row of the prompt dataset, in index order:

1. Skip if `hs_<idx>.safetensors` already exists in the output dir.
2. Spawn `${BUUN}/llama-dump-hiddens` as a subprocess: pipes the tokenized prompt in via stdin, captures hidden-state tensors via the binary's binary-framed stdout protocol.
3. Buffer the per-layer hidden states for the 6 chosen layers (default `[2, 16, 30, 45, 59, 61]` — the last is the final residual).
4. Stack into a single `[ntok, 6, 3072]` tensor in bf16.
5. Write atomically (`.tmp_*.part` → `fsync` → `rename` → dir-`fsync`) to `hs_<idx>.safetensors`.
6. Update `state_worker_<ID>.json` with the new completed index, recompute the FM27 hash chain, atomic-write that too.
7. SIGTERM/SIGINT handler: finish the current trace (atomic write means a kill mid-`subprocess.communicate` either fully writes or fully drops; never half-written), exit 0.

The `[ntok, 6, 3072]` shape and bf16 dtype match the FP8 pipeline's trace schema **bit-for-bit** in layout. Layer indices `[2, 16, 30, 45, 59, 61]` are the same ones the FP8 trainer was wired to.

## 1.9 Sanity-check the produced pool

After your first ~50 traces, inspect:

```bash
${VENV}/bin/python -c "
from safetensors.torch import load_file
import torch, glob
files = sorted(glob.glob('${WORK}/traces/hidden_states/hs_*.safetensors'))[:5]
for f in files:
    t = load_file(f)['hidden_states']
    print(f'{f}: shape={tuple(t.shape)} dtype={t.dtype} '
          f'nan={torch.isnan(t).any().item()} inf={torch.isinf(t).any().item()} '
          f'mean={t.float().mean().item():.4f} std={t.float().std().item():.4f}')
"
```

Expect:

- `shape=(N, 6, 3072)` with `N` = prompt length (varies per row).
- `dtype=torch.bfloat16`.
- `nan=False`, `inf=False` — the IQ4 path does **not** produce the deep-layer overflow that plagued the FP8/vLLM path. There is no validator-gate quarantine step; if you see NaN/Inf, file an issue.
- `std` per layer roughly in the same range as the FP8 reference cohort (deep layers wider than shallow). For empirical comparison: run `repro/scripts/iq4_tracegen/iq4_fp8_compare.py` against a directory of FP8 reference traces if you have them.

If a worker produced a corrupt trace, just delete the file — re-running the worker fills it in atomically on the next pass.

## 1.10 Reuse checklist

Before declaring the pool ready for training (Section 2):

- [ ] Every index in `[0, N)` has a `hs_<idx>.safetensors` (no holes): `ls hs_*.safetensors | wc -l` should equal `N`.
- [ ] No NaN/Inf anywhere — sample-check 1% of files with the snippet in §1.9.
- [ ] All workers' `state_worker_<ID>.json` files validate (FM27 hash chain intact). The wrapper class in `gen_traces_v2.py` (`State._load_validated`) raises if not.
- [ ] Bridge tensors extracted: run `repro/scripts/iq4_tracegen/extract_gguf_bridge.py` on `<HEAD_HOST>` to produce `${WORK}/verifier_meta/{config.json,model.safetensors}` from GGUF shard 2 (Q8_0 / Q6_K / F32 → bf16 dequantization for `embed_tokens` / `lm_head` / `model.norm`). Section 2 needs this.
- [ ] Filtered + shuffled dataset built: `python3 repro/scripts/iq4_tracegen/filter_dataset_v2.py` (drops rows with no trace, shuffles before `split_ratio` to avoid the worker-shard-leaks-into-val bug from v1 — see `repro/plan/2026-04-30-iq4-gguf-only-end-to-end.md`).

## 1.11 Files in this section

| Path | Purpose |
|---|---|
| `repro/scripts/iq4_tracegen/iq4_worker.sh` | Per-machine launcher — disjoint-shard tmux wrapper around `gen_traces_v2.py` |
| `repro/scripts/iq4_tracegen/gen_traces_v2.py` | The actual trace generator. Atomic, resumable, kill-safe |
| `repro/scripts/iq4_tracegen/extract_gguf_bridge.py` | Dequantizes 3 tensors from GGUF shard 2 → bf16 safetensors. Run on `<HEAD_HOST>` before training |
| `repro/scripts/iq4_tracegen/filter_dataset_v2.py` | Builds the shuffled, filtered prompt dataset for §2's trainer |
| `repro/scripts/iq4_tracegen/iq4_fp8_compare.py` | Optional empirical comparison of new pool vs an FP8 reference cohort |
| `repro/scripts/iq4_tracegen/rip_spark1.sh` | (Spark-cluster-specific) rip non-genomics data off `<HEAD_HOST>` to free disk for the pool. Adapt or delete |
| `repro/plan/2026-04-30-iq4-gguf-only-end-to-end.md` | Tick doc with end-to-end results, the v1 split bug + v2 fix, results table vs FP8 SMOKE |
| `repro/plan/00-resumability-doctrine.md` | Atomic-writes / state.json / skip-existing rules every script in this section follows |

## 1.12 Why we left vLLM behind

In one sentence: **the FP8/vLLM path needed 4 nodes, replicated 215 GB weights to each, and still quarantined ~45% of files for deep-layer overflow.** The GGUF/llama.cpp path needs 1 node, has no overflow problem, and the resulting traces train a drafter that already exceeds the FP8 SMOKE baseline at epoch 2.

If you have a 4-node fabric idle and need to reproduce the original FP8 6515-file pool exactly, the legacy recipe is at `repro/legacy/01-generation-vllm-fp8-tp4.md`. The legacy supporting scripts (validator daemon with R62/R64 patches, vLLM TP=4 launch script, datagen client with retry logic, EngineCore-wedge recovery procedure) are in `repro/legacy/`.

---

**Next:** Section 2 — Training the DFlash drafter on these traces. See `repro/02-training.md`.
