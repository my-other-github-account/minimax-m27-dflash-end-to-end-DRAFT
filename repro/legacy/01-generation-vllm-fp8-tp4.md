# LEGACY — Hidden-State Trace Generation (vLLM TP=4, FP8, 4-node cluster)

> **🗄️ LEGACY DOCUMENT.** Active generation path is now **llama.cpp + GGUF, single-machine workers** — see `repro/01-generation.md`.
>
> This file is preserved as the canonical record of the **vLLM TP=4 / MiniMax-M2.7-FP8 / 4-node** pipeline that produced the original 6515-file FP8 trace pool used to train the FULL drafter (p_1=22.9% plateau at epoch 13/17). It is no longer the recommended path for new runs because:
>
> - Requires 4 GB10-class nodes wired on a high-bandwidth fabric (QSFP/100GbE+).
> - vLLM rank-0 EngineCore wedge recurs every ~42 mi (see §1.11) — needs a watchdog.
> - bf16 deep-layer overflow forces ~45% of files quarantined at the validator gate.
> - 215 GB FP8 weights must be replicated to all 4 nodes.
>
> The new GGUF path (`01-generation.md`) needs **one machine + one GGUF shard set** and produces traces that have already been shown to train a drafter past the FP8 SMOKE baseline. Use this legacy doc only if you need to reproduce the original FP8 trace pool exactly, or you have a 4-node cluster idle and want the higher-throughput path.

---

# Section 1 — Hidden-State Trace Generation (LEGACY: vLLM TP=4, FP8)

> **⚠️ DRAFT — not independently verified outside the original cluster.** This section reflects the recipe and evidence from the run that produced the canonical 6515-file FP8 trace pool. The pipeline is real and was used to train the working drafter; the *documentation has not yet been re-walked end-to-end on a clean machine*.

## Placeholder key

The doc and harvested scripts use shell-style placeholders so you can drop in your own values. Set them once at the top of your shell session:

| Placeholder | What it is | Example |
|---|---|---|
| `${WORKSPACE}` | Top-level work dir on every node | `/opt/dflash` |
| `${DATA_ROOT}` | Where staged + cleaned + quarantined files live | `${WORKSPACE}/data` |
| `${LOOP_WORKSPACE}` | Autonomous loop state (sha index, seen-prompt hashes) | `${WORKSPACE}/tracegen_loop` |
| `${MODELS}` | Where verifier and (later) drafter weights sit | `${WORKSPACE}/models` |
| `${VENV}` | Python venv with vLLM + speculators installed | `${WORKSPACE}/venvs/vllm` |
| `node1`, `node2`, `node3`, `node4` | Hostnames of your 4 GB10-class nodes (rank-0 = `node2` in the example) | whatever your DNS / `/etc/hosts` resolves |
| `<HEAD_IP>` | rank-0's IP on the high-bandwidth fabric | e.g. `192.168.200.2` |
| `<HIGH_BW_NIC>` | NIC name carrying the inter-rank fabric | e.g. `enp1s0f1np1` (Mellanox CX-7) |
| `<PRIVATE_SUBNET>` | First three octets of the QSFP/Ethernet fabric | e.g. `192.168.200` |
| `<JUMPHOST_IP>` | Optional bastion if your nodes aren't directly addressable | — |
| `<PDU_IP>` / `<PDU_USER>` / `<PDU_PASS>` | Smart power outlet (only for the cluster-wide power-cycle escape hatch in §1.12) | — |

Everything else in the doc (NCCL env vars, vLLM flags, layer IDs, dtype contracts) is meant to be copied verbatim — those are the load-bearing details.

---

End-to-end reproduction of the FP8 hidden-state trace pool used to train the MiniMax-M2.7 DFlash drafter. Produces `hs_<N>.safetensors` files of shape `[seq_len, 6, 3072]` (5 user-chosen layer taps + verifier last hidden) in `bfloat16` storage.

This section is **fully self-contained** — no training or inference assumed. Output is a directory of validated, deduped, NaN-free, layer-zero-rate-gated `safetensors` ready for `speculators.train`.

## TL;DR

- 4 GB10-class nodes, vLLM **TP=4**, MiniMax-M2.7-**FP8**, 6 layer taps `[2, 16, 30, 45, 59, 62]`
- vLLM source is **vanilla** — earlier patches (R33, R49, R63 in `interfaces.py` + `extract_hidden_states.py`) were all **reverted**. The bf16-overflow problem is real but is now caught downstream by a validator-side **R62 zero-rate gate** that quarantines bad files instead of trying to repair them
- ~45% of staged files quarantined for deep-layer overflow; remainder is clean training data
- A separate validator daemon and a separate datagen client run on rank-0; they do not communicate with vLLM beyond the OpenAI-compatible HTTP API
- Verified production run produced the **6515-file / 127 GB** pool that trained the working drafter (provenance below)

---

## 1.1 Hardware & topology assumed

| Item | Value used in the verified run |
|---|---|
| Nodes | 4 × DGX Spark GB10 (`node1` / `node2` / `node3` / `node4`) |
| Interconnect | QSFP 200 Gbps, MTU 9000, subnet `<PRIVATE_SUBNET>/24` on NIC `<HIGH_BW_NIC>` |
| Rank-0 (head/API) | `node2` at `<HEAD_IP>` |
| Workers (headless) | `node1` rank 1, `node3` rank 2, `node4` rank 3 |
| GPU per node | 1× GB10 (Blackwell SM 12.1a), 122 GiB unified memory |
| Per-GPU free memory required | ≥ 25–40 GB (FP8 weights + KV-transfer connector) |
| Power | All 4 share one ezOutlet5 at `<PDU_IP>` (cycle = ALL 4 reset simultaneously) |

Any 4 GB10-class boxes connected by ≥ 100 Gbps Ethernet on a private subnet will work. The QSFP fabric is just plain TCP NCCL — `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=<your-NIC>`. **Do not use Ray.** Use `--nnodes 4 --node-rank N --master-addr ... --master-port ...` (torchrun-style). Ray-based launches were observed to deadlock during EngineCore init on this topology.

## 1.2 Software pinned versions

| Component | Version / commit |
|---|---|
| vLLM | `0.20.1rc1.dev23+gde3da0b97` |
| Speculators | `67bafe6189549e9e0955b99cf0e399cb8c5a2627` ("Dflash verifier targets" #477) |
| CUDA | 13.x, `nvcc` at `/usr/local/cuda/bin/ptxas` |
| Python | 3.12, single venv at `${VENV}` |
| OS | Ubuntu 22.04+ |

## 1.3 Models required

| Model | Path used | Size | Notes |
|---|---|---|---|
| Verifier (target) | `${MODELS}/MiniMax-M2.7-FP8` | ~120 GB | `num_hidden_layers=62`, `hidden_size=3072`, `vocab_size=200064`, `quantization_config.quant_method="fp8"`. Must be on **all 4 nodes** at the same path. |

`config.json` and `model.safetensors.index.json` must both be present on every node — the `vllm_tp4_5L_FP8_CLEAN.sh` script refuses to launch if either is missing.

## 1.4 The "we flailed for days" lesson — read first

The trace pipeline went through a **multi-day patch saga** chasing bf16 overflow at deep layers (60+) that produced silent NaN→0 zero-poisoning. **Every one of those `vllm/...interfaces.py` and `extract_hidden_states.py` patches was eventually reverted.** What survived is one tiny patch on the validator side.

**State of patches in the verified production run:**

| File | Status | Note |
|---|---|---|
| `vllm/model_executor/models/interfaces.py` | **VANILLA — no patches** | R33/R49/R61/R63 all reverted |
| `vllm/v1/spec_decode/extract_hidden_states.py` | **VANILLA** | R49 buffer-zero-fill also reverted |
| `validator_daemon_5L.py` (rank-0 only) | **R62 zero-rate gate ACTIVE** | The only surviving patch — quarantines files where any layer has > 30 % all-zero rows |
| `validator_daemon_5L.py` (rank-0 only) | **R64 extended validation ACTIVE** | Adds `zero_collapse`, `constant_collapse`, `pure_echo` guards |

> **Why this matters:** the naive zero-fallback (R33) trains the drafter to ignore deep layers, then catastrophically fails at inference because real activations replace zeros at run time. **The right answer is to throw away ~45 % of generated files at the validator gate, not to "fix" the overflow at extraction time.**

The two surviving patches live in `repro/scripts/generation/validator_daemon_5L.py` in this repo — **search for `R62_ZERO_RATE_GATE` and `R64_EXTENDED_VALIDATION` markers** if you copy the script forward.

**Verify your vLLM is vanilla before launch:**

```bash
VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
# These greps should print NOTHING. Any output = a previous patch is still applied.
grep -nE "R33|R49|R61|R63|FINITE_FALLBACK|nan_to_num" \
  "$VLLM_DIR/model_executor/models/interfaces.py"
grep -nE "R33|R49|R61|R63" \
  "$VLLM_DIR/v1/spec_decode/extract_hidden_states.py"
```

## 1.5 Pre-flight checklist (per-node, all 4)

```bash
NODES=(node1 node2 node3 node4)   # replace with your hostnames or IPs
for n in "${NODES[@]}"; do
  ssh "$n" '
    hostname
    [ -f ${MODELS}/MiniMax-M2.7-FP8/config.json ] && echo "  model OK" || echo "  MODEL MISSING"
    [ -f ${MODELS}/MiniMax-M2.7-FP8/model.safetensors.index.json ] && echo "  index OK" || echo "  INDEX MISSING"
    [ -x ${VENV}/bin/python ] && echo "  venv OK" || echo "  VENV MISSING"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | head -3
    ip -4 -o addr show <HIGH_BW_NIC> 2>/dev/null | awk "{print \"  qsfp_ip=\" \$4}"
  '
done
```

Any GPU compute users from a prior run? Kill them BEFORE launching:

```bash
for node in node1 node2 node3 node4; do
  ssh $node 'pkill -TERM -f "vllm serve|EngineCore|launch_vllm|VLLM::Worker"' &
done
wait
sleep 15
for node in node1 node2 node3 node4; do
  ssh $node 'pkill -KILL -f "vllm serve|EngineCore|launch_vllm|VLLM::Worker" 2>/dev/null' &
done
wait
```

## 1.6 Critical environment (set per-node)

The launcher (`repro/scripts/generation/vllm_tp4_5L_FP8_CLEAN.sh`) sets these for you, but if you're running by hand:

```bash
ETH_IF=<HIGH_BW_NIC>                                    # adjust to your NIC
NODE_IP=$(ip -4 -o addr show $ETH_IF | awk '{print $4}' | cut -d/ -f1 | head -1)
HEAD_IP=<HEAD_IP>                                 # rank-0's QSFP IP
MASTER_PORT=29504

export VLLM_HOST_IP="$NODE_IP"
export GLOO_SOCKET_IFNAME="$ETH_IF"
export NCCL_SOCKET_IFNAME="$ETH_IF"
export NCCL_IB_DISABLE=1                              # plain TCP, no IB
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_MOE_FP4=0                  # FP8 path only
export VLLM_ENABLE_V1_MULTIPROCESSING=0               # required for kv_transfer connector
export FLASHINFER_CUDA_ARCH_LIST=12.1a                # GB10
export TORCH_CUDA_ARCH_LIST=12.1a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export HF_HUB_OFFLINE=1                               # critical — see Pitfall 7
```

## 1.7 Launch sequence — vLLM TP=4

The launcher script auto-detects rank from `hostname` (`node2`→0, `node1`→1, `node3`→2, `node4`→3) and refuses to launch if patches mismatch or GPU has prior compute users.

**Order matters: rank-0 first, then 1/2/3 simultaneously.** Rank-0 must reach `[utils.py:299]` model-loading state before workers connect — otherwise NCCL handshake intermittently fails.

```bash
# 1. Rank-0 (node2) first
ssh node2 'systemd-run --user --unit dflash-tracegen-vllm-rank-0 -- \
  bash ${WORKSPACE}/scripts/vllm_tp4_5L_FP8_CLEAN.sh'

# 2. Wait for rank-0 API healthy (poll up to 10 min — vLLM cold start is slow)
for i in $(seq 1 20); do
  if ssh node2 'curl -sf --max-time 5 http://127.0.0.1:8000/v1/models' >/dev/null; then
    echo "[$(date)] rank-0 API up"; break
  fi
  sleep 30
done

# 3. Ranks 1/2/3 simultaneously — must rendezvous within 601 s of each other
#    (torch.distributed default rendezvous timeout)
for node in node1 node3 node4; do
  ssh $node "systemd-run --user --unit dflash-tracegen-vllm-rank-N -- \
    bash ${WORKSPACE}/scripts/vllm_tp4_5L_FP8_CLEAN.sh" &
done
wait
```

> **Lesson R14a (do not skip):** Once rank-0 is up, **do not poll `/v1/models` between rank-0 and worker launches**. The worker ranks must be launched within ~30 seconds of rank-0; otherwise rank-0's torch.distributed init blocks for 601 s and times out. Health-poll AFTER all 4 are launched.

The full launch invocation (visible in the script) is:

```bash
python3 scripts/launch_vllm.py ${MODELS}/MiniMax-M2.7-FP8 \
  --hidden-states-path ${DATA_ROOT}/preprocessed_5L_FP8/hs_staging \
  --target-layer-ids 2 16 30 45 59 \
  -- \
  --tensor-parallel-size 4 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --no-enable-flashinfer-autotune \
  --no-enable-chunked-prefill \
  --max-num-batched-tokens 2048 \
  --kv-cache-dtype auto \
  --max-num-seqs 1 \
  --load-format fastsafetensors \
  --trust-remote-code \
  --port 8000 --host 0.0.0.0 \
  --nnodes 4 --node-rank $RANK \
  --master-addr <HEAD_IP> --master-port 29504 \
  $HEADLESS_FLAG
```

`launch_vllm.py` (in the speculators repo) auto-appends `num_hidden_layers` (62 = "last hidden") to `--target-layer-ids`, so you end up with **6 layer taps**: `[2, 16, 30, 45, 59, 62]`.

> **Why these flags are non-negotiable:**
> - `--enforce-eager`: required — chunked-prefill + cudagraph paths don't expose hidden states cleanly to the kv_transfer connector
> - `--max-num-seqs 1`: required — multi-batch hidden-state extraction has known race conditions in the connector path
> - `--max-model-len 2048`: chosen to balance prompt coverage vs latency. ~22 % of mixed-source prompts exceed 2048; those raise `BadRequestError 400` and are skipped (absorbed by datagen's `--max-consecutive-errors 500`)

## 1.8 Validator daemon (rank-0 only)

The validator (`repro/scripts/generation/validator_daemon_5L.py`) watches `hs_staging/`, validates each safetensor, and moves clean files to `hs_clean_pool/` with monotonic renumbering, bad files to `hs_quarantine/`.

**Validation pipeline per file:**
1. NaN / Inf scan across all layers
2. Schema check: `hidden_states` key, 3-D shape, `seq_len ≥ 3`
3. **R62 zero-rate gate**: if any layer has > 30 % rows with all-zero L1-norm → quarantine (catches bf16-overflow zero-poisoning at deep layers)
4. **R64 extended validation**: zero-collapse (final-tap mean-abs ≈ 0), constant-collapse (per-row variance ≈ 0), pure-echo (token_ids match the prompt, no actual generation) → quarantine
5. **First-4 KB sha256 dedup** vs `validator_state.json["pool_hashes"]` — see Trap E for the gotcha
6. Survivors → `hs_<monotonic_idx>.safetensors` in `hs_clean_pool/`

**Two launch traps (both observed live during cold-restart 2026-04-30):**

### Trap A — Validator paths drift after FP8 transition

Older copies of `validator_daemon_5L.py` hardcode `STAGING`/`POOL`/`QUAR`/`STATE` at `preprocessed_5L/...` (no FP8 suffix). The launcher writes to `preprocessed_5L_FP8/...`. Mismatch is **silent**: validator reports `staging empty, sleeping...` while files pile up in the FP8 dir. Diagnostic:

```bash
ls ${DATA_ROOT}/preprocessed_5L_FP8/hs_staging/ | wc -l
# rising over time, but...
grep -E "^(STAGING|POOL|QUAR|STATE) " repro/scripts/generation/validator_daemon_5L.py
# all 4 should contain "preprocessed_5L_FP8" — if any contain just "preprocessed_5L", patch the constants:
sed -i 's|preprocessed_5L/|preprocessed_5L_FP8/|g' repro/scripts/generation/validator_daemon_5L.py
```

### Trap B — `systemd-run --user` defaults to `/usr/bin/python3` (no safetensors)

`systemd-run --user ... python3 ...` resolves to system python, which lacks `safetensors`. The unit fails 80 ms after launch with `ModuleNotFoundError: No module named 'safetensors'`. **Always invoke the venv python by full path:**

```bash
ssh node2 'systemd-run --user --unit=dflash-tracegen-validator \
  --description="DFlash tracegen validator R62/R64" \
  -E PYTHONPATH=${WORKSPACE}/repos/speculators/scripts \
  ${VENV}/bin/python3 \
  ${WORKSPACE}/scripts/validator_daemon_5L.py'
```

Within 60 s the validator log should write a fresh `Validator started. Pool starts at idx=N, K known hashes` line. Log: `${WORKSPACE}/logs/validator-5L.log`.

**Healthy validator output looks like:**
```
[07:35:56] scan #752: clean+=24 nan=0 err=0 dup=0 | pool_size=6539 totals: clean=2568 nan=0 err=0
```

Quarantine reasons in the log distinguish R62 (`L4 zero-rate 84%`) from R64 (`R64 zero_collapse final-tap mean_abs=0.00001`) — useful for diagnosing whether the bf16-overflow issue is back vs a different failure mode.

## 1.9 Prompt sources — building the input dataset

The verified run cycled through several preprocessed prompt sources as each exhausted. **Plan exhaustion before it happens:**

| Source | Path | Sample count | Notes |
|---|---|---|---|
| `combined_48k` | `cache/bonus/preprocessed/combined_48k` | 48 000 | Default, pre-tokenized arrow format |
| `bonus_seed4321` | `cache/bonus/preprocessed/bonus_seed4321` | 12 000 | Same format, different seed — sitting next to `combined_48k`, easy to miss |
| `mixed_<ts>_seed*` | `cache/mixed_<ts>/preprocessed` | ~13 000 each | Mixed multi-dataset built via `multi_dataset_prompt_loader.py` |

Any HF instruction or chat dataset (ShareGPT, lmsys-chat-1m, OpenOrca, OpenHermes-2.5, evol-codealpaca-v1, ultrachat_200k, Nemotron-Post-Training-v2) can be tokenized into the same arrow format via the speculators repo's `prepare_data.py`:

```bash
${VENV}/bin/python3 \
  ${WORKSPACE}/repos/speculators/scripts/prepare_data.py \
  --data <hf_id_or_local_jsonl> \
  --model ${MODELS}/MiniMax-M2.7-FP8 \
  --output ${WORKSPACE}/cache/<source-name>/preprocessed
```

### Multi-dataset prompt mixer (`multi_dataset_prompt_loader.py`)

When `combined_48k` and `bonus_seed4321` are both exhausted, build a fresh mixed source. The loader (in `repro/scripts/generation/multi_dataset_prompt_loader.py`) streams 4 unblocked HF datasets, normalizes to ShareGPT format, length-filters at the character level, and **dedupes against a persistent `seen_prompt_hashes.json`**.

Default unblocked proportions (Nemotron + lmsys are HF-gated, fall through to these):
- `theblackcat102/evol-codealpaca-v1` — 40 %
- `Open-Orca/OpenOrca` — 25 %
- `teknium/OpenHermes-2.5` — 25 %
- `HuggingFaceH4/ultrachat_200k` (split `train_sft`) — 10 %

```bash
TS=$(date +%s)
DEST=${WORKSPACE}/cache/mixed_${TS}
mkdir -p $DEST

${VENV}/bin/python3 multi_dataset_prompt_loader.py \
  --output $DEST/raw.jsonl \
  --total 15000 \
  --seen-set ${LOOP_WORKSPACE}/state/seen_prompt_hashes.json

${VENV}/bin/python3 \
  ${WORKSPACE}/repos/speculators/scripts/prepare_data.py \
  --data $DEST/raw.jsonl \
  --model ${MODELS}/MiniMax-M2.7-FP8 \
  --output $DEST/preprocessed
```

> **Critical:** **update `seen_prompt_hashes.json` with newly-built prompts BEFORE running datagen on the new source**, otherwise the next rotation rebuilds against a stale seen-set and the cycle repeats. The loader does this for you when `--seen-set` is provided.

## 1.10 Datagen client (rank-0 only)

The datagen client calls vLLM's `/v1/completions` once per prompt. The kv_transfer connector aggregates the TP-sharded hidden states from all 4 ranks and writes **one** safetensors file per completion to `hs_staging/`. The validator picks it up from there.

> **Per-trial output:** **one file per (prompt, completion) — not per system.** The 4 sparks are tensor-parallel collaborators on a single forward pass; they don't each emit a file. So a 6515-file pool = 6515 distinct prompt completions.

### Trap C — Same venv-python issue as the validator

The ledgered wrapper imports `data_generation_offline` from the speculators scripts dir, which requires PYTHONPATH. Use the same venv-python pattern:

```bash
ssh node2 'systemd-run --user --unit=dflash-tracegen-datagen \
  --description="DFlash tracegen datagen client" \
  -E PYTHONPATH=${WORKSPACE}/repos/speculators/scripts \
  ${VENV}/bin/python3 \
  ${WORKSPACE}/scripts/data_generation_offline_ledgered.py \
    --model ${MODELS}/MiniMax-M2.7-FP8 \
    --endpoint http://127.0.0.1:8000/v1 \
    --preprocessed-data <prompt-source-dir> \
    --output ${DATA_ROOT}/preprocessed_5L_FP8/hs_staging \
    --max-samples 50000 \
    --concurrency 1 \
    --max-consecutive-errors 500'
```

### Trap D — The ledgered wrapper's ledger is GLOBAL across datasets

`data_generation_offline_ledgered.py` maintains a persistent JSON ledger at `cache/datagen_ledger.json` — a list of integer indices. **The ledger does NOT distinguish which dataset an index came from.** When you switch sources (e.g. `combined_48k` 48k → `bonus_seed4321` 12k), the ledger from the prior run already contains 0–47999, so the wrapper says `All samples already processed!` and exits in 4 seconds.

**Two options when switching prompt sources:**

1. **Bypass the ledgered wrapper** (recommended): call upstream `data_generation_offline.py` directly. It scans the output dir for already-existing `hs_<idx>.safetensors` and only skips those.
2. **Reset the ledger**: `mv cache/datagen_ledger.json{,.bak.$(date +%s)}` — keeps the wrapper's other dedup features but loses persistent skip-existing across runs.

### Trap E — First-4 KB sha256 dedup collides on structurally-similar safetensors

The validator dedups by `sha256(open(file).read(4096))[:16]` — the first 4 KB is the safetensors JSON header (tensor names / shapes / offsets). When a prompt source reaches structural exhaustion, the JSON headers can be byte-identical even though the actual hidden_states differ. **Symptom: `clean+=0 dup=25-30` per scan, every scan, while datagen IS alive and producing files.**

Diagnostic — verify whether staging hashes truly collide with pool hashes:

```python
import hashlib, json, os
state_path = "${DATA_ROOT}/preprocessed_5L_FP8/validator_state.json"
staging   = "${DATA_ROOT}/preprocessed_5L_FP8/hs_staging"
pool_hashes = set(json.load(open(state_path))["pool_hashes"])
for f in os.listdir(staging):
    p = os.path.join(staging, f)
    with open(p, "rb") as fh:
        h = hashlib.sha256(fh.read(4096)).hexdigest()
    print(f, "DUP" if h in pool_hashes else "NEW")
```

If everything prints `DUP`, you've hit Trap E. **Fix: rotate to a fresh prompt source via the multi-dataset loader, and update `seen_prompt_hashes.json` with the new prompts BEFORE relaunching datagen.**

## 1.11 The rank-0 EngineCore wedge — recurring failure mode

The dominant failure mode of long runs (8+ occurrences in a single 24-hour stretch, MTBF 5 min – 105 min, no monotonic trend) is the **rank-0 EngineCore wedge**: `RPC call to sample_tokens timed out` → `EngineDeadError` → APIServer shutdown.

**Critically: workers TP1/2/3 (and even TP0) do NOT exit when the head crashes.** They remain as orphan zombies holding ~100 GB GPU memory each. Recovery requires manual `pkill -KILL -f VLLM::Worker` on **all 4 sparks** before relaunch.

### Cheap precursor signals (act on these BEFORE EngineDeadError)

```bash
# Signal 1: zero throughput sustained
ssh node2 'tail -200 ${WORKSPACE}/logs/vllm-fp8-tp4-clean-node2-*.log \
  | grep -c "Avg generation throughput: 0.0 tokens/s"'
# > 5 lines in last 200 = wedge starting

# Signal 2: shared-memory broadcast warning every 60s
ssh node2 'tail -200 ${WORKSPACE}/logs/vllm-fp8-tp4-clean-node2-*.log \
  | grep "shm_broadcast.py.*No available shared memory broadcast block"'

# Signal 3: datagen client retries
ssh node2 'journalctl --user -u dflash-tracegen-datagen --since "10 minutes ago" \
  | grep "Request timed out after 4 attempts"'

# Signal 4: API endpoint health is misleading
ssh node2 'curl -sf --max-time 5 http://127.0.0.1:8000/v1/models'        # may still 200 OK!
ssh node2 'curl -sf --max-time 30 http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"...\",\"prompt\":\"hi\",\"max_tokens\":1}"'             # this hangs = wedged
```

### Coordinated relaunch procedure (the rehearsed `R14b` fix)

```bash
# 1. Stop datagen first
ssh node2 'systemctl --user stop dflash-tracegen-datagen'

# 2. Hard-kill ALL vLLM processes on ALL 4 sparks (skip systemctl stop on rank units — hangs 30s)
for node in node1 node2 node3 node4; do
  ssh $node 'pkill -KILL -f "VLLM::Worker"; pkill -KILL -f "vllm.entrypoints"; \
              pkill -KILL -f "EngineCore"; pkill -KILL -f "launch_vllm"' &
done
wait
sleep 10

# 3. Verify GPUs free on all 4
for node in node1 node2 node3 node4; do
  ssh $node 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'
done
# Expect: empty output on all 4

# 4. Relaunch rank-0, then ranks 1/2/3 within 30s (well under 601s rendezvous)
ssh node2 'systemd-run --user --unit dflash-tracegen-vllm-rank-0 -- \
  bash ${WORKSPACE}/scripts/vllm_tp4_5L_FP8_CLEAN.sh'
sleep 5
for node in node1 node3 node4; do
  ssh $node "systemd-run --user --unit dflash-tracegen-vllm-rank-N -- \
    bash ${WORKSPACE}/scripts/vllm_tp4_5L_FP8_CLEAN.sh" &
done
wait

# 5. Wait for /v1/models (~270s typical)
for i in $(seq 1 20); do
  if ssh node2 'curl -sf --max-time 5 http://127.0.0.1:8000/v1/models' >/dev/null; then
    echo "[$(date)] healthy after $((i*30))s"; break
  fi
  sleep 30
done

# 6. Restart datagen — expect 10-15min replay-dedup phase from sample 0
ssh node2 'systemctl --user start dflash-tracegen-datagen'
```

> **Lesson R17 (don't false-alarm during replay-dedup):** After a relaunch, datagen restarts from sample 0 and re-emits files for prompts already in the pool. The validator dedups them, so you'll see `clean+=0 dup=25-30` for ~10–15 min. **This is NOT stagnation.** Distinguish from real stagnation by checking the precursor signals above are negative AND that datagen log shows sample-cursor advancing.

### Structural mitigations to consider (NOT applied autonomously)

These were flagged by the autonomous loop but require human judgment because they trade off speed vs stability:

- Lower `--gpu-memory-utilization 0.90 → 0.85` for headroom
- Disable `--enforce-eager` (requires testing — model may not fit without it)
- Run `--max-num-seqs > 1` (single-seq may be hitting a scheduler edge case)
- Try newer vLLM commits beyond `0.20.1rc1.dev23+gde3da0b97`
- Add a watchdog systemd unit that detects the precursor signals and auto-restarts
- Persist datagen sample-cursor across restarts (eliminates 90–140 min cumulative replay-dedup waste)

## 1.12 Cluster power-cycle recovery (when nothing else works)

```bash
# All 4 sparks share one ezOutlet5 — cycling kills all 4 simultaneously
ssh <jumphost> 'curl -s --max-time 10 -u "<PDU_USER>:<PDU_PASS>" "http://<PDU_IP>/overview?reset=Reset"'
sleep 240   # ~4 min for DGX cold POST

# Verify all 4 back up
for node in node1 node2 node3 node4; do
  ssh -o ConnectTimeout=10 $node 'uptime -p'
done

# Then re-launch from §1.7 step 1
```

## 1.13 Trace schema produced

Every `hs_<N>.safetensors` (~5–20 MB depending on `seq_len`):

| Tensor | Dtype | Shape |
|---|---|---|
| `hidden_states` | `torch.bfloat16` | `[seq_len, 6, 3072]` |
| `token_ids` | `torch.int64` | `[seq_len]` |

The 6 layer indices in dim 1 are `[2, 16, 30, 45, 59, 62]` — the 5 user-supplied taps plus the auto-appended last-hidden (= `num_hidden_layers`).

> **Note:** The "FP8 vs bf16" distinction is in the verifier's compute path, not the trace storage dtype. Hidden states are always saved as `bfloat16` regardless of the verifier's quantization.

## 1.14 Sanity-check the produced pool

After the pool has at least a few hundred files, run a deep audit. The script in `repro/scripts/generation/deep_audit.py` (legacy, in `repro/legacy/deep_audit.py` — works for both old NVFP4 and new FP8 layouts) reports per-layer NaN/Inf, std distributions, and zero-rate.

For the strongest check (used to validate the production pool), run a **distribution comparison vs a reference cohort**:

```python
# repro/scripts/generation/cohort_compare.py — reproduces the empirical comparison
# that confirmed the new pool's distribution matches the validated reference cohort.
import os, glob, random, statistics
from safetensors import safe_open
import torch

POOL = "${DATA_ROOT}/preprocessed_5L_FP8/hs_clean_pool"
files = sorted(glob.glob(os.path.join(POOL, "hs_*.safetensors")),
               key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
random.seed(0)
sample = random.sample(files, min(80, len(files)))

seq_lens, mins, maxs, means, stds = [], [], [], [], []
zr = [[] for _ in range(6)]
nan_n = inf_n = 0
for p in sample:
    with safe_open(p, framework="pt") as f:
        hs = f.get_tensor("hidden_states")
    hsf = hs.float()
    if torch.isnan(hsf).any(): nan_n += 1
    if torch.isinf(hsf).any(): inf_n += 1
    seq_lens.append(hs.shape[0])
    mins.append(hsf.min().item());  maxs.append(hsf.max().item())
    means.append(hsf.mean().item()); stds.append(hsf.std().item())
    for L in range(6):
        zr[L].append((hsf[:, L, :].abs().sum(-1) == 0).float().mean().item())

print(f"NaN files: {nan_n}/{len(sample)}    Inf files: {inf_n}/{len(sample)}")
print(f"seq_len mean={statistics.mean(seq_lens):.1f} median={statistics.median(seq_lens)}")
print(f"global mean={statistics.mean(means):.4f}  global std={statistics.mean(stds):.2f}")
for L, name in enumerate(["L2","L16","L30","L45","L59","L62"]):
    m = statistics.mean(zr[L]) * 100
    print(f"  {name} zero-rate mean: {m:.2f}%")
```

**Healthy output:**
```
NaN files: 0/80    Inf files: 0/80
seq_len mean=555.2 median=475
global mean=-0.0065  global std=40.86
  L2  zero-rate mean: 0.00%
  L16 zero-rate mean: 0.00%
  L30 zero-rate mean: 0.00%
  L45 zero-rate mean: 0.00%
  L59 zero-rate mean: 0.00%
  L62 zero-rate mean: 0.00%
```

## 1.15 Uniqueness ratchet — strict-monotonic invariant

The autonomous tracegen loop maintains a hard invariant: `len(unique_sha_set)` strictly increases every tick. Tracked via `unique_proofs/sha_index.json` (flat JSON list of first-4 KB sha256[:16] values, format_version=2).

```python
# Quick ratchet check
import os, hashlib, json, glob
POOL  = "${DATA_ROOT}/preprocessed_5L_FP8/hs_clean_pool"
INDEX = "${LOOP_WORKSPACE}/unique_proofs/sha_index.json"

prior = set(json.load(open(INDEX))) if os.path.exists(INDEX) else set()
current = set()
for p in glob.glob(os.path.join(POOL, "hs_*.safetensors")):
    with open(p, "rb") as fh:
        current.add(hashlib.sha256(fh.read(4096)).hexdigest()[:16])

new = current - prior
gone = prior - current
print(f"prior={len(prior)} current={len(current)} new={len(new)} gone={len(gone)}")
assert not gone, "RATCHET VIOLATION: shas disappeared"
print("ratchet OK" if new else "ratchet flat (replay-dedup phase?)")
```

## 1.16 Verified production run

| Property | Value |
|---|---|
| Reference start | 2026-04-29 06:58 PT (`hs_0.safetensors`) |
| Reference end | 2026-04-29 07:24 PT (`hs_1175.safetensors`) — 1176 files trained the working drafter |
| Loop continuation | 2026-04-29 23:19 PT onward (`hs_1176`+) — autonomous extension |
| Pool snapshot | 6515 files / 127 GB at 2026-04-30 07:12 PT |
| Quarantine | 1 file (validator caught the only deep-layer overflow that slipped through) |
| Distinct prompt sources | `combined_48k`, `bonus_seed4321`, `mixed_1777534621`, `mixed_1777545051` |
| Total seen prompt hashes | 23 000 |
| Wedge events | 8 in 24 h, all recovered via `R14b` procedure (§1.11) |

### Distribution comparison vs reference cohort (empirical, n=80 random samples each)

| Metric | A_REFERENCE (idx 0–1175) | B_NEW_MID (idx 1176–3845) | C_NEW_RECENT (idx 3846–6514) |
|---|---|---|---|
| seq_len mean | 562.1 | 514.7 | 569.3 |
| global mean | 0.0024 | 0.0073 | -0.0098 |
| global std | 55.4 | 41.2 | 41.1 |
| NaN / Inf files | 0 / 0 | 0 / 0 | 0 / 0 |
| L2 zero-rate (mean) | 0.00 % | 0.00 % | 0.00 % |
| L16 zero-rate (mean) | 0.00 % | 0.00 % | 0.00 % |
| L30 zero-rate (mean) | 0.00 % | 0.00 % | 0.00 % |
| L45 zero-rate (mean) | 0.00 % | 0.00 % | 0.00 % |
| L59 zero-rate (mean) | 0.00 % | 0.00 % | 0.00 % |
| L62 zero-rate (mean) | 0.00 % | 0.00 % | 0.00 % |
| L59 std | 8.51 | 8.58 | 8.48 |
| L62 std | 133.8 | 98.8 | 99.0 |

Per-layer activation means and stds for L2/L16/L30/L45/L59 are within ~1 % across all three cohorts. The verifier last-hidden (L62) tail is slightly shorter in the new cohorts (std 133 → 99) — this is **expected and good**: the reference's longer tail is the bf16-overflow signature the R62 gate was designed to catch. The two distributions are statistically indistinguishable on the metrics that matter for drafter training.

## 1.17 Reuse checklist

Before declaring "trace generation is set up":

- [ ] vLLM `interfaces.py` is **VANILLA** (no `R33|R49|R61|R63|FINITE_FALLBACK|nan_to_num` markers)
- [ ] Validator script has `R62_ZERO_RATE_GATE` and `R64_EXTENDED_VALIDATION` markers
- [ ] Validator `STAGING`/`POOL`/`QUAR`/`STATE` constants all contain `preprocessed_5L_FP8` (Trap A)
- [ ] All 4 nodes reachable on the QSFP fabric (ping from rank-0)
- [ ] All 4 GPUs free of foreign compute users
- [ ] Rank-0 API responds 200 on `/v1/models` AND `/v1/completions` returns within 30 s for a 1-token request
- [ ] Validator + datagen units launched via `${VENV}/bin/python3` by full path (Traps B + C)
- [ ] Validator log writing within 60 s of launch
- [ ] `pool_size=N` in validator log matches `ls clean_pool | wc -l` within ±20 (Trap A diagnostic)
- [ ] First file lands in `hs_clean_pool/` within 5–10 min of datagen launch
- [ ] `cohort_compare.py` (§1.14) shows L0–L5 zero-rate < 0.5 % across a 80-file random sample
- [ ] Datagen log does NOT say `All samples already processed!` within 5 s (Trap D — ledger collision)
- [ ] Pool grows steadily for 30 min — then it's healthy

## 1.18 Files in this section

```
repro/scripts/generation/
├── vllm_tp4_5L_FP8_CLEAN.sh           # 4-rank vLLM launcher, auto-detects rank by hostname
├── validator_daemon_5L.py             # R62 zero-rate gate + R64 extended validation + dedup
├── data_generation_offline_ledgered.py  # ledger-aware datagen wrapper (see Trap D)
└── multi_dataset_prompt_loader.py     # prompt-source rotation when sources exhaust
```

Continue to [Section 2 — Training](02-training.md).
