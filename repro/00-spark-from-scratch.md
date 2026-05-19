# §0 — Spark From Scratch

End-to-end bringup on a fresh DGX Spark (or any single Linux node with an NVIDIA GPU + ~150 GB free disk + Python 3.12). At the end of this doc, you have a working trace-generation worker producing self-describing fp8 traces, a paired training dataset, a smoke-tested DFlash trainer, and a saved drafter checkpoint — all driven through the `dflash_llama` Python API.

> **Tested on:** NVIDIA GB10 (DGX Spark), Ubuntu 22.04, Python 3.12, torch 2.10. Should work on any compute capability ≥ 8.0 GPU with bf16 + fp8 storage.

**All paths in this doc are relative to the repo root.** Run every command from the directory you cloned the repo into.

---

## Step 1: clone, install, smoke-test

```bash
git clone https://github.com/my-other-github-account/minimax-m27-dflash-end-to-end-DRAFT.git dflash-llama
cd dflash-llama

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install pytest

pytest tests/ -q                      # 45 passed in ~0.5s
```

If your venv lacks `pip` (some locked-down systems), the library also runs from `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python3 -c "import dflash_llama; print(dflash_llama.__version__)"
```

---

## Step 2: build `llama-dump-hiddens` (reproducibly)

```bash
bash scripts/build_llama_dump_hiddens.sh
```

What this does:

1. Clones [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) at a **pinned tag** (default `master-fff0e0e`) into `build/llama.cpp-dflash/`.
2. Drops our vendored [`vendor/dump-hiddens/`](../vendor/dump-hiddens/) source files into `build/llama.cpp-dflash/examples/dump-hiddens/`.
3. Wires it into the cmake graph (idempotent — appends one `add_subdirectory(...)` line if not already there).
4. Builds with CUDA + Release. The binary that lands in `build/llama.cpp-dflash/build/bin/` is `llama-dump-hiddens-worker` — the persistent JSONL stdin/stdout worker used by the batched trace-server (see [§8 — Persistent batched-decode trace server](08-tracegen-server.md)). It is the only execution path the library ships, and it produces ~60+ traces/min single-host on a GB10 — ~2.5× faster than the legacy spawn-per-prompt path that used to exist.

Output: `build/llama.cpp-dflash/build/bin/llama-dump-hiddens-worker`. The script prints this path on stdout for piping:

```bash
LLAMA_DUMP_BIN=$(bash scripts/build_llama_dump_hiddens.sh | tail -1)
```

Knobs (all optional, env vars):

| var | default | meaning |
|---|---|---|
| `LLAMACPP_PIN` | `master-fff0e0e` | upstream tag/commit |
| `BUILD_CUDA` | `1` | `0` for CPU-only build |
| `JOBS` | `nproc` | parallel build jobs |

Re-running the script is safe — it skips the clone if the directory exists and just rebuilds.

---

## Step 3: pick a verifier (no on-disk fiddling)

The library accepts model **slugs** — both for the HF config and for GGUF weights. It downloads to a per-user cache (`~/.cache/dflash-llama/`, configurable via `DFLASH_LLAMA_HOME`) on first use and reuses on subsequent runs.

```python
from dflash_llama import load_verifier

verifier = load_verifier(
    "minimax-m2.7-iq4-xs",
    hf_repo="MiniMaxAI/MiniMax-M2",
    gguf_repo="unsloth/MiniMax-M2-GGUF",
    gguf_quant="UD-IQ4_XS",
)
```

That's the entire model-loading recipe. No `/path/to/...`, no manual `cp` of `config.json` and tokenizer files, no shard-counting. The library:

- Downloads the small files (`config.json`, `tokenizer.json`, `chat_template.jinja`, `model.safetensors.index.json`, plus the `configuration_*.py` / `modeling_*.py` for trust-remote-code models) from `MiniMaxAI/MiniMax-M2` — but **not** the multi-hundred-GB weights, which the trainer doesn't need.
- Downloads only the `UD-IQ4_XS/*` GGUF shards from `unsloth/MiniMax-M2-GGUF`.
- Returns a `BaseVerifier` with `hf_path` and `gguf_path` already set to the cached locations.

Mix-and-match:

```python
# Hub for the small files, local GGUF (e.g. you pre-staged it on a cluster)
load_verifier("minimax-m2.7-iq4-xs", hf_repo="MiniMaxAI/MiniMax-M2",
              gguf_path="./models/iq4/shard-00001.gguf")

# Local for both (no network access)
load_verifier("minimax-m2.7-iq4-xs",
              hf_path="./hf-cache/MiniMax-M2",
              gguf_path="./models/iq4/shard-00001.gguf")
```

A `verifier_meta` stub is **not required** if you give the trainer either `hf_repo` or `hf_path` — speculators reads only `lm_head.weight`, `model.embed_tokens.weight`, and `model.norm.weight`, all of which are in the HF safetensors index for the public weights. (See [§stub recipe](#optional-stub-verifier_meta-when-you-cant-download-weights) at the bottom for the air-gapped fallback.)

---

## Step 4: build a prompts dataset

You need an HF Dataset on disk with at least an `input_ids` column (Sequence of int). Optionally include `loss_mask` (Sequence of bool).

```bash
python3 - <<'PY'
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-M2", trust_remote_code=True)
src = load_dataset("allenai/tulu-3-sft-mixture", split="train")

rows = []
for example in src.select(range(800_000)):
    msgs = example["messages"]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    if 16 < len(ids) < 2048:
        rows.append({"input_ids": ids})

Dataset.from_list(rows).save_to_disk("data/prompts_tulu3")
print(f"wrote {len(rows)} prompts to data/prompts_tulu3")
PY
```

Output: `data/prompts_tulu3/`. Workers index into this by row number.

---

## Step 5: install speculators

The DFlash training step shells out to the [speculators](https://github.com/neuralmagic/speculators) trainer.

```bash
pip install speculators                              # if a release covers your needs
# OR pinned to a known-good commit:
pip install git+https://github.com/neuralmagic/speculators.git@67bafe6

# Optionally apply the four NaN-handling patches (only matters for legacy
# fp8 traces — the v3 self-describing format never has NaN):
SPECULATORS_DIR=$(python3 -c "import speculators, pathlib; print(pathlib.Path(speculators.__file__).parent.parent)")
for p in patches/speculators/*.patch; do
    git -C "$SPECULATORS_DIR" apply "$p" || echo "  (skipped, already applied)"
done
```

After this, set the env var pointing at speculators' `train.py` (the DFlashTrainer reads this; there's no hard-coded path):

```bash
export SPECULATORS_TRAIN_SCRIPT=$(python3 -c "
import speculators, pathlib
p = pathlib.Path(speculators.__file__).parent.parent / 'scripts' / 'train.py'
print(p)
")
```

> If you cloned speculators manually, set `SPECULATORS_TRAIN_SCRIPT` to whatever the path is on your system. The library uses this env var as the only way to find the trainer entry point — no more brittle defaults.

---

## Step 6: generate traces (single host)

```bash
mkdir -p data/traces data/state data/logs
LLAMA_DUMP_BIN=$(bash scripts/build_llama_dump_hiddens.sh | tail -1)

python3 -m dflash_llama.cli generate \
    --verifier minimax-m2.7-iq4-xs \
    --hf-repo MiniMaxAI/MiniMax-M2 \
    --gguf-repo unsloth/MiniMax-M2-GGUF \
    --gguf-quant UD-IQ4_XS \
    --binary "$LLAMA_DUMP_BIN" \
    --prompts data/prompts_tulu3 \
    --rows 0:6500 \
    --out data/traces \
    --state data/state/gen_state.json \
    --max-seq-len 2048 \
    2>&1 | tee data/logs/gen.log
```

You should see `[gen] completed=10 skipped=0 failed=0` lines streaming. ~2-3 seconds per trace on a single GB10. Resumable — re-running skips existing files.

The Python API equivalent (used internally by the CLI):

```python
from dflash_llama import TraceGenerator, load_verifier

verifier = load_verifier(
    "minimax-m2.7-iq4-xs",
    hf_repo="MiniMaxAI/MiniMax-M2",
    gguf_repo="unsloth/MiniMax-M2-GGUF",
    gguf_quant="UD-IQ4_XS",
)
gen = TraceGenerator(
    verifier=verifier,
    storage="fp8_per_tensor_scale",
    backend="tracegen_client",
    backend_kwargs={
        "binary": "build/llama.cpp-dflash/build/bin/llama-dump-hiddens-worker",
        "auto_start": True,
        "ctx": 16384,
        "ngl": 99,
        "override_tensor": "exps=CPU",
        "request_timeout": 600,
    },
)
gen.generate(
    prompts="data/prompts_tulu3",
    output_dir="data/traces",
    rows=range(0, 6500),
    state_path="data/state/gen_state.json",
    batch_width=8,
)
```

---

## Step 7: generate traces (multi-host, disjoint shards)

For N hosts, partition the row range into disjoint shards. Each host runs a worker on its own slice; output goes to a per-host directory and gets rsynced together at the end.

Example for 4 hosts, 200K total prompts:

| host    | shard | rows                | tmux session         |
|---------|-------|---------------------|----------------------|
| spark-1 | D     | `0:50000`           | `dflash_v3_api_D`    |
| spark-2 | B     | `50000:100000`      | `dflash_v3_api_B`    |
| spark-3 | A     | `100000:150000`     | `dflash_v3_api_A`    |
| spark-4 | C     | `150000:200000`     | `dflash_v3_api_C`    |

The library API worker `scripts/worker_api_v3.py` (in this repo) wraps `TraceGenerator.generate()` with arg parsing. Launch in tmux:

```bash
HOST=spark-1; SHARD=D; LO=0; HI=50000

ssh "$HOST" bash -lc "tmux new-session -d -s dflash_v3_api_${SHARD} '
    cd ~/dflash-llama && source .venv/bin/activate && \
    python3 scripts/worker_api_v3.py \
        --shard-id ${SHARD} \
        --rows ${LO}:${HI} \
        --out data/traces \
        --state data/state/state_worker_${SHARD}.json \
        --prompts data/prompts_tulu3 \
        --verifier-name minimax-m2.7-iq4-xs \
        --hf-repo MiniMaxAI/MiniMax-M2 \
        --gguf-repo unsloth/MiniMax-M2-GGUF \
        --gguf-quant UD-IQ4_XS \
        --binary build/llama.cpp-dflash/build/bin/llama-dump-hiddens \
        2>&1 | tee data/logs/api_worker_${SHARD}_\$(date +%Y%m%d_%H%M%S).log
'"
```

Every trace file is named `hs_<global_row_idx>.safetensors`. Because shards are disjoint, all hosts' `data/traces/` directories rsync into one place without collision:

```bash
mkdir -p data/traces_consolidated
for h in spark-1 spark-2 spark-3 spark-4; do
    rsync -a "$h:dflash-llama/data/traces/" data/traces_consolidated/
done
```

> **Use QSFP IPs for inter-host transfers.** WiFi hostnames are ~10 MB/s; QSFP fabric is ~322 MB/s. Map your host alias to the QSFP IP in `~/.ssh/config`.

---

## Step 8: train

```bash
export SPECULATORS_TRAIN_SCRIPT=$(python3 -c "
import speculators, pathlib
print(pathlib.Path(speculators.__file__).parent.parent / 'scripts' / 'train.py')
")

python3 - <<'PY'
from dflash_llama import DFlashTrainer, load_verifier

verifier = load_verifier(
    "minimax-m2.7-iq4-xs",
    hf_repo="MiniMaxAI/MiniMax-M2",          # <-- no path needed
)

trainer = DFlashTrainer(
    traces_dir="data/traces_consolidated",
    verifier=verifier,
    paired_dir="data/paired",
    num_layers=5,
    draft_vocab_size=32768,
)

prep_report = trainer.prepare(force=True)
print("prepare:", prep_report)

smoke = trainer.smoke(
    timeout_sec=90,
    save_path="data/smoke_ckpt",
    log_path="data/smoke.log",
    port=29503,
)
print(f"smoke ok={smoke.ok}  first_loss={smoke.first_loss}  last_loss={smoke.last_loss}")
assert smoke.ok, "smoke failed — see data/smoke.log"

result = trainer.train(
    save_to="data/ckpt",
    epochs=17,
    lr=3e-5,
    max_anchors=512,
    total_seq_len=2048,
    log_freq=5,
    scheduler_warmup_steps=100,
    save_best=True,
    port=29504,
)
assert result["rc"] == 0
print(f"train rc=0  log={result['log_path']}  save={result['save_path']}")

trainer.offline_eval(checkpoint="data/ckpt/0", max_batches=60)
PY
```

Expected runtime on a single GB10 with ~5K traces: smoke=90s, train ~3 hours for 17 epochs at `--num-workers 1`. Output:

```
data/ckpt/
├── 0/                       # best checkpoint
│   ├── model.safetensors    # ~2.1 GB DFlash drafter
│   ├── config.json
│   └── val_metrics.json
└── train_<timestamp>.log
```

---

## Step 9: verify

Round-trip a freshly produced trace through the library to confirm zero NaN:

```python
from dflash_llama import load_trace
import torch
t = load_trace("data/traces/hs_0.safetensors")
print("shape  :", t["hidden_states"].shape)            # (seq_len, n_layers, hidden_size)
print("dtype  :", t["hidden_states"].dtype)            # torch.bfloat16 (decoded from fp8)
print("any NaN:", torch.isnan(t["hidden_states"]).any().item())   # False
print("abs_max:", t["hidden_states"].abs().max().item())          # may be > 448 — that's fine
```

---

## Common gotchas

| symptom | cause | fix |
|---|---|---|
| `FileNotFoundError: 'torchrun'` | venv not on PATH | Library auto-resolves torchrun next to `sys.executable`; if you still hit this, ensure `python3 -c "import torch.distributed.run"` works in your venv. |
| `Column 'seq_len' doesn't exist` | old paired arrow | Re-run `trainer.prepare(force=True)`. |
| `torch.equal(): must be Tensor, not list` | paired arrow saved without torch format | Re-run `trainer.prepare(force=True)`. |
| `Expected local file missing: model-00000-of-00130.safetensors` | speculators trying to read full sharded weights | The library ships HF-only configs — make sure you're passing `hf_repo` or a path to a directory containing `model.safetensors.index.json` whose weight_map references only the bridge tensors (or the full safetensors shards if disk allows). |
| smoke `rc != 124` and no loss lines | training crashed before first log | Check `data/smoke.log` for the actual rank0 traceback. |
| `[gen] row N failed: timed out after 600s` | one prompt too long or model fell back to CPU | Normal at the < 1% level; raise `--per-trace-timeout` if persistent. |

---

## Optional: stub `verifier_meta` (when you can't download weights)

If you're on an air-gapped cluster and can't pull the safetensors shards, you can extract the 3 tensors speculators actually reads (`lm_head.weight`, `model.embed_tokens.weight`, `model.norm.weight`) from a GGUF you already have:

```bash
python3 scripts/build_verifier_meta_stub.py \
    --gguf data/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00002-of-00004.gguf \
    --out  data/verifier_meta/model.safetensors \
    --also-write-index
```

Then point the verifier at the local stub:

```python
v = load_verifier("minimax-m2.7-iq4-xs", hf_path="data/verifier_meta", gguf_path=".../shard-00001.gguf")
```

The stub is ~2.4 GB for MiniMax-M2.7. Cluster shortcut: build once on one host, `rsync -a` over QSFP (~8s at 322 MB/s).

---

## What you have at the end

- `data/traces/` — `hs_<i>.safetensors` files, self-describing fp8 (zero NaN by construction)
- `data/paired/prompts/` — HF Dataset (input_ids, loss_mask, seq_len, source_name, source_row_idx) + `t2d.npy`, `d2t.npy`, `token_freq.pt`
- `data/paired/hidden_states/` — symlink farm aligned with arrow row order
- `data/ckpt/0/model.safetensors` — trained DFlash drafter, ready for vLLM speculative decoding (with the `dflash` speculator type)
