# §0 — Spark From Scratch

End-to-end bringup on a fresh DGX Spark (or any single Linux node with an NVIDIA GPU + ~150 GB free disk + Python 3.12). At the end of this doc, you have a working trace-generation worker producing self-describing fp8 traces, a paired training dataset, a smoke-tested DFlash trainer, and a saved drafter checkpoint — all driven through the `dflash_llama` Python API.

> **Tested on:** NVIDIA GB10 (DGX Spark), Ubuntu 22.04, Python 3.12, torch 2.10. Should work on any compute capability ≥ 8.0 GPU with bf16 + fp8 storage.

---

## Prerequisites

You need:

1. **A llama.cpp build with the `llama-dump-hiddens` patch** (the trace-extraction binary). We use [our buun fork](https://github.com/my-other-github-account/buun-llama-cpp).
2. **A GGUF of the verifier model.** This walkthrough uses MiniMax-M2.7 quantized to UD-IQ4_XS (~109 GB). Substitute your own.
3. **The HF model directory of the same verifier** — only `config.json`, `tokenizer.json`, and `model.safetensors.index.json` are strictly required, but the trainer is happiest if you give it a stub `model.safetensors` containing just `lm_head.weight`, `model.embed_tokens.weight`, `model.norm.weight`. See [the verifier_meta stub recipe](#step-3-build-the-verifier_meta-stub).
4. **A prompts dataset** as an HF Dataset on disk with at least an `input_ids` column. We use a tokenized slice of `allenai/tulu-3-sft-mixture`.
5. **The [speculators](https://github.com/neuralmagic/speculators) repo cloned locally** (commit `67bafe6` or later, with the 4 NaN-handling patches in `patches/speculators/` applied if you need them).

---

## Step 1: clone + install

```bash
ssh you@spark-1
cd ~

# clone this library
git clone https://github.com/my-other-github-account/minimax-m27-dflash-end-to-end-DRAFT.git dflash-llama
cd dflash-llama

# create a venv with python ≥ 3.10 (3.12 recommended — speculators uses py3.12 features)
python3.12 -m venv ~/venvs/dflash
source ~/venvs/dflash/bin/activate

pip install --upgrade pip
pip install -e .
pip install pytest                       # for the test suite
pytest tests/ -q                         # 36 passed in ~10s
```

If you cannot install in the venv (no pip, locked-down system), the library also runs from `PYTHONPATH`:

```bash
export PYTHONPATH=$HOME/dflash-llama/src:$PYTHONPATH
python3 -c "import dflash_llama; print(dflash_llama.__version__)"
```

---

## Step 2: build `llama-dump-hiddens`

```bash
cd ~
git clone https://github.com/my-other-github-account/buun-llama-cpp.git
cd buun-llama-cpp
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-dump-hiddens -j$(nproc)

# verify
./bin/llama-dump-hiddens --help 2>&1 | head -5
```

Resulting binary path: `~/buun-llama-cpp/build/bin/llama-dump-hiddens`. We will refer to it as `LLAMA_DUMP_BIN` below.

---

## Step 3: build the `verifier_meta` stub

The speculators trainer reads `model.safetensors.index.json` and tries to load every shard listed in `weight_map`. For a 130-shard model this would download / read 200 GB of weights it doesn't actually use — only `lm_head.weight`, `model.embed_tokens.weight`, and `model.norm.weight` are actually consumed by the DFlash trainer.

We build a one-shard stub:

```bash
mkdir -p ~/verifier_meta
cd ~/verifier_meta

# 1. copy the small files from the original HF model dir
HF_MODEL_DIR=/path/to/MiniMax-M2.7-FP8
cp $HF_MODEL_DIR/config.json .
cp $HF_MODEL_DIR/tokenizer*.json .
cp $HF_MODEL_DIR/chat_template.jinja .       # if present
cp $HF_MODEL_DIR/configuration_*.py .         # MiniMax-specific
cp $HF_MODEL_DIR/modeling_*.py .              # MiniMax-specific

# 2. extract the 3 needed tensors from the GGUF and write them as one safetensors shard
#    (this script lives in scripts/build_verifier_meta_stub.py — see below)
python3 scripts/build_verifier_meta_stub.py \
    --gguf /path/to/MiniMax-M2.7-UD-IQ4_XS-00002-of-00004.gguf \
    --out  ~/verifier_meta/model.safetensors
```

The script `scripts/build_verifier_meta_stub.py` reads the GGUF, dequantizes `token_embd.weight (Q8_0) → model.embed_tokens.weight`, `output.weight (Q6_K) → lm_head.weight`, `output_norm.weight (F32) → model.norm.weight`, and writes them all to one bf16 safetensors file (~2.4 GB for MiniMax-M2.7).

Then write a matching `model.safetensors.index.json`:

```bash
python3 -c "
import json
total = 0
import safetensors.torch as st
loaded = st.load_file('/root/verifier_meta/model.safetensors')
for k, v in loaded.items():
    total += v.numel() * v.element_size()
weight_map = {k: 'model.safetensors' for k in loaded.keys()}
json.dump(
    {'metadata': {'total_size': total}, 'weight_map': weight_map},
    open('/root/verifier_meta/model.safetensors.index.json', 'w'),
    indent=2,
)
"
```

> **Cluster shortcut:** if you've already built `verifier_meta` on one host, don't redo it — `rsync -a host1:/path/to/verifier_meta/ host2:/path/to/verifier_meta/` over QSFP. 2.4 GB ships in ~8 seconds at 322 MB/s.

---

## Step 4: build a prompts dataset

You need an HF Dataset on disk with at least an `input_ids` column (Sequence of int). Optionally include `loss_mask` (Sequence of bool).

```python
# scripts/build_prompts_arrow.py
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/path/to/MiniMax-M2.7-FP8")
src = load_dataset("allenai/tulu-3-sft-mixture", split="train")

rows = []
for example in src.select(range(800_000)):                 # take ~800K prompts
    msgs = example["messages"]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    if 16 < len(ids) < 2048:
        rows.append({"input_ids": ids})

Dataset.from_list(rows).save_to_disk("~/prompts_tulu3")
```

Save to `~/prompts_tulu3/`. Workers will index into it by row number (e.g. `--rows 576034:626034`).

---

## Step 5: install speculators

```bash
cd ~
git clone https://github.com/neuralmagic/speculators
cd speculators
git checkout 67bafe6                                # or latest with the patches
pip install -e .

# apply the 4 NaN-handling patches if you have them (optional — only matters
# for legacy fp8 traces; the v3 self-describing format never has NaN)
for p in ~/dflash-llama/patches/speculators/*.patch; do git apply "$p"; done
```

After this, `~/speculators/scripts/train.py` exists. The library finds it via the `SPECULATORS_TRAIN_SCRIPT` env var (or the default `~/repos/speculators/scripts/train.py`):

```bash
export SPECULATORS_TRAIN_SCRIPT=$HOME/speculators/scripts/train.py
```

---

## Step 6: generate traces (single host)

```bash
mkdir -p ~/traces ~/state ~/logs

PYTHONPATH=$HOME/dflash-llama/src \
python3 -m dflash_llama.cli generate \
    --verifier minimax-m2.7-iq4-xs \
    --gguf-path /path/to/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
    --prompts $HOME/prompts_tulu3 \
    --rows 0:6500 \
    --out $HOME/traces \
    --state $HOME/state/gen_state.json \
    --binary $HOME/buun-llama-cpp/build/bin/llama-dump-hiddens \
    --max-seq-len 2048 \
    2>&1 | tee $HOME/logs/gen.log
```

You should see `[gen] completed=10 skipped=0 failed=0` lines streaming. ~2-3 seconds per trace on a single GB10. **Resumable** — re-running skips existing files.

The Python API equivalent (used internally by the CLI):

```python
from dflash_llama import TraceGenerator, load_verifier

verifier = load_verifier(
    "minimax-m2.7-iq4-xs",
    gguf_path="/path/to/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf",
)
gen = TraceGenerator(
    verifier=verifier,
    storage="fp8_per_tensor_scale",
    backend="llamacpp_gguf",
    backend_kwargs={
        "binary": "/path/to/llama-dump-hiddens",
        "timeout": 600,
    },
)
gen.generate(
    prompts="~/prompts_tulu3",
    output_dir="~/traces",
    rows=range(0, 6500),
    state_path="~/state/gen_state.json",
)
```

---

## Step 7: generate traces (multi-host, disjoint shards)

For N hosts, partition the row range into disjoint shards. Each host runs a worker on its own slice; output goes to a per-host directory and gets rsynced together at the end.

Example for 4 hosts, 200K total prompts:

| host    | shard | rows                | tmux session         | state file               |
|---------|-------|---------------------|----------------------|--------------------------|
| spark-1 | D     | `0:50000`           | `dflash_v3_api_D`    | `state_worker_D.json`    |
| spark-2 | B     | `50000:100000`      | `dflash_v3_api_B`    | `state_worker_B.json`    |
| spark-3 | A     | `100000:150000`     | `dflash_v3_api_A`    | `state_worker_A.json`    |
| spark-4 | C     | `150000:200000`     | `dflash_v3_api_C`    | `state_worker_C.json`    |

The library API worker `scripts/worker_api_v3.py` (in this repo) wraps `TraceGenerator.generate()` with arg parsing. Launch in tmux:

```bash
HOST=spark-1; SHARD=D; LO=0; HI=50000

ssh $HOST "tmux new-session -d -s dflash_v3_api_${SHARD} '
    PYTHONPATH=$HOME/dflash-llama/src \
    python3 $HOME/dflash-llama/scripts/worker_api_v3.py \
        --shard-id $SHARD \
        --rows ${LO}:${HI} \
        --out $HOME/traces \
        --state $HOME/state/state_worker_${SHARD}.json \
        --prompts $HOME/prompts_tulu3 \
        --gguf-path /path/to/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
        --binary $HOME/buun-llama-cpp/build/bin/llama-dump-hiddens \
        2>&1 | tee $HOME/logs/api_worker_${SHARD}_\$(date +%Y%m%d_%H%M%S).log;
    sleep 600
'"
```

Every trace file is named `hs_<global_row_idx>.safetensors`. Because shards are disjoint, you can safely rsync all hosts' `~/traces/` directories into one place without collision:

```bash
mkdir -p ~/traces_consolidated
for h in spark-1 spark-2 spark-3 spark-4; do
    rsync -a $h:~/traces/ ~/traces_consolidated/
done
```

> **Use QSFP IPs for inter-host transfers.** WiFi hostnames are ~10 MB/s; QSFP fabric is ~322 MB/s.

---

## Step 8: train

```bash
export SPECULATORS_TRAIN_SCRIPT=$HOME/speculators/scripts/train.py

PYTHONPATH=$HOME/dflash-llama/src \
python3 - <<'PY'
from dflash_llama import DFlashTrainer, load_verifier

verifier = load_verifier(
    "minimax-m2.7-iq4-xs",
    hf_path="/root/verifier_meta",         # the stub from step 3
)

trainer = DFlashTrainer(
    traces_dir="~/traces_consolidated",
    verifier=verifier,
    paired_dir="~/paired",
    num_layers=5,
    draft_vocab_size=32768,
)

# Step A — build the paired-arrow + vocab maps + hidden_states symlink farm
prep_report = trainer.prepare(force=True)
print("prepare:", prep_report)

# Step B — 90s smoke (rc=124 from `timeout` is the expected success exit)
smoke = trainer.smoke(
    timeout_sec=90,
    save_path="~/smoke_ckpt",
    log_path="~/smoke.log",
    port=29503,
)
assert smoke.ok, f"smoke failed: {smoke}"
print("smoke OK; first_loss=", smoke.first_loss, "last_loss=", smoke.last_loss)

# Step C — real training (epochs=17 is the canonical recipe)
result = trainer.train(
    save_to="~/ckpt",
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
print("train rc=0; saved to ~/ckpt")

# Step D — optional offline eval
trainer.offline_eval(checkpoint="~/ckpt/checkpoint_best", max_batches=60)
PY
```

Expected runtime on a single GB10 with ~5K traces: smoke=90s, train ~3 hours for 17 epochs at `--num-workers 1`. Output checkpoint:

```
~/ckpt/
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
t = load_trace("~/traces/hs_0.safetensors")
print("shape  :", t["hidden_states"].shape)            # (seq_len, n_layers, hidden_size)
print("dtype  :", t["hidden_states"].dtype)            # torch.bfloat16 (decoded from fp8)
print("any NaN:", torch.isnan(t["hidden_states"]).any().item())   # False
print("abs_max:", t["hidden_states"].abs().max().item())          # may be > 448 — that's fine
```

If `any NaN` is `False` and `abs_max` is reasonable (typical: 500-3000 for MiniMax-M2.7), you're done.

---

## Common gotchas

| symptom | cause | fix |
|---|---|---|
| `FileNotFoundError: 'torchrun'` | venv not on PATH | The library auto-resolves torchrun next to `sys.executable`; if you still hit this, ensure `python3 -c "import torch.distributed.run"` works in your venv. |
| `Column 'seq_len' doesn't exist` | old paired arrow without `seq_len` | Re-run `trainer.prepare(force=True)` — current library writes `seq_len`. |
| `torch.equal(): must be Tensor, not list` | paired arrow saved without `set_format("torch")` | Re-run `trainer.prepare(force=True)` — current library sets the format. |
| `Expected local file missing: model-00000-of-00130.safetensors` | speculators trying to read full sharded weights | Build the `verifier_meta` stub (Step 3); pass that path as `hf_path=` to `load_verifier`. |
| smoke `rc != 124` and no loss lines | training crashed before first log | Check `~/smoke.log` for the actual rank0 traceback. |
| `[gen] row N failed: timed out after 600s` | one prompt too long or model fell back to CPU | normal at the < 1% level; raise `--per-trace-timeout` if persistent. |

---

## What you have at the end

- `~/traces/` — `hs_<i>.safetensors` files, self-describing fp8 (zero NaN by construction)
- `~/paired/prompts/` — HF Dataset (input_ids, loss_mask, seq_len, source_name, source_row_idx) + `t2d.npy`, `d2t.npy`, `token_freq.pt`
- `~/paired/hidden_states/` — symlink farm `hs_0.safetensors` → real trace, aligned with arrow row order
- `~/ckpt/0/model.safetensors` — trained DFlash drafter, ready for `~/dflash_minimax/repos/speculators/scripts/eval_offline.py` or vLLM speculative decoding (with the `dflash` speculator type).
