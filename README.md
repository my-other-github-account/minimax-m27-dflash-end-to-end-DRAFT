# dflash-llama

Self-describing fp8 trace generation and DFlash drafter training for llama-family verifiers (MiniMax-M2, Kimi-K2.5, Qwen3, …).

This library replaces a fragile shell-script pipeline with a clean Python API. It is the canonical way to train DFlash speculative-decoding drafters in this repo.

> **🚀 First time on a Spark? Start here:** [`repro/00-spark-from-scratch.md`](repro/00-spark-from-scratch.md) — full bringup from a bare machine, including the `llama-dump-hiddens` build, the `verifier_meta` stub, the prompts arrow, the speculators install, and the multi-host shard plan. Tested end-to-end on a 4× DGX Spark cluster.

Three other walkthroughs:

| topic | doc |
|---|---|
| Generate traces (single host or multi-host) | [`repro/01-generation.md`](repro/01-generation.md) |
| Train a DFlash drafter | [`repro/02-training.md`](repro/02-training.md) |
| Run inference with the trained drafter | [`repro/03-inference.md`](repro/03-inference.md) |

## What you get

- **Self-describing fp8 traces** — every `hs_<i>.safetensors` carries `hidden_states` (saturating fp8 + per-tensor scale, **never NaN**), `token_ids`, `input_ids`, `loss_mask`, plus full provenance metadata (`schema_version`, `source_name`, `source_row_idx`, `gen_timestamp`, `layer_ids`). No more post-hoc sha256 pairing.
- **End-to-end DFlash training** — `DFlashTrainer.prepare() → .smoke() → .train() → .offline_eval()`. Wraps the [speculators](https://github.com/neuralmagic/speculators) trainer; does the data prep inside the library instead of in shell scripts.
- **Model-family abstraction** — `BaseVerifier` configs encode `hidden_size`, `vocab_size`, `mask_token_id`, layer taps, etc. Switching from MiniMax-M2.7 to Kimi-K2.5 is a single config change. Custom verifiers register at runtime via `register_verifier(name, factory)`.
- **Saturating-fp8 storage** — ~50% disk vs bfloat16 with **zero data loss for over-range values** (the bf16→fp8 cast bug that motivated this library is impossible by construction).

## Install

```bash
git clone https://github.com/my-other-github-account/minimax-m27-dflash-end-to-end-DRAFT.git dflash-llama
cd dflash-llama
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest tests/ -q                       # 45 passed in <1s
```

Then build `llama-dump-hiddens` reproducibly from a pinned upstream `llama.cpp` commit + our vendored example:

```bash
bash scripts/build_llama_dump_hiddens.sh           # outputs binary path on stdout
```

Speculators is required for `trainer.smoke()` and `trainer.train()` — install via `pip install speculators` or pin to a specific commit (see [`repro/00-spark-from-scratch.md`](repro/00-spark-from-scratch.md) §5). For trace generation alone, only torch + safetensors + datasets are needed.

## Python API quickstart

```python
from dflash_llama import DFlashTrainer, TraceGenerator, load_verifier

# Hub slugs — library downloads to ~/.cache/dflash-llama/ on first use
verifier = load_verifier(
    "minimax-m2.7-iq4-xs",
    hf_repo="MiniMaxAI/MiniMax-M2",
    gguf_repo="unsloth/MiniMax-M2-GGUF",
    gguf_quant="UD-IQ4_XS",
)

# 1. Generate traces (one safetensors file per prompt)
gen = TraceGenerator(
    verifier=verifier,
    storage="fp8_per_tensor_scale",
    backend="llamacpp_gguf",
    backend_kwargs={"binary": "build/llama.cpp-dflash/build/bin/llama-dump-hiddens"},
)
gen.generate(
    prompts="data/prompts_tulu3",
    output_dir="data/traces",
    rows=range(0, 6500),
    state_path="data/gen_state.json",
)

# 2. Train (no separate "build paired dataset" step — trainer.prepare() does it)
trainer = DFlashTrainer(
    traces_dir="data/traces",
    verifier=verifier,
    num_layers=5,
    draft_vocab_size=32768,
    paired_dir="data/paired",
)
trainer.prepare()                                          # arrow + vocab maps + hs symlinks
trainer.smoke(timeout_sec=90, save_path="data/smoke_ckpt") # 90s plumbing check
trainer.train(save_to="data/ckpt", epochs=17)              # ~3 hours on one GB10
trainer.offline_eval(checkpoint="data/ckpt/0", max_batches=60)
```

Local-path mode is also supported when you already have files staged (e.g. on a cluster):

```python
verifier = load_verifier(
    "minimax-m2.7-iq4-xs",
    hf_path="data/hf-cache/MiniMax-M2",
    gguf_path="data/models/UD-IQ4_XS/shard-00001.gguf",
)
```

## CLI quickstart

```bash
# list registered verifiers
dflash-llama info

# generate (Hub-slug mode — auto-downloads to ~/.cache/dflash-llama/)
dflash-llama generate \
  --verifier minimax-m2.7-iq4-xs \
  --hf-repo MiniMaxAI/MiniMax-M2 \
  --gguf-repo unsloth/MiniMax-M2-GGUF --gguf-quant UD-IQ4_XS \
  --binary build/llama.cpp-dflash/build/bin/llama-dump-hiddens \
  --prompts data/prompts_tulu3 \
  --rows 0:6500 \
  --out data/traces \
  --state data/gen_state.json

# 90-second smoke
dflash-llama smoke \
  --verifier minimax-m2.7-iq4-xs \
  --hf-repo MiniMaxAI/MiniMax-M2 \
  --traces data/traces \
  --timeout 90

# full training run
dflash-llama train \
  --verifier minimax-m2.7-iq4-xs \
  --hf-repo MiniMaxAI/MiniMax-M2 \
  --traces data/traces \
  --output data/ckpt \
  --epochs 17 --lr 3e-5 --max-anchors 512
```

You can substitute `--hf-path` / `--gguf-path` for the `--*-repo` flags when files are already on disk.

## Examples

| File | What it shows | Wall-clock |
|---|---|---|
| [`repro/examples/tiny_smoke.py`](repro/examples/tiny_smoke.py) | The smallest end-to-end demo — 10 traces, prepare, 90s smoke. Verify your environment in <6 minutes. | ~5 min |
| [`repro/examples/minimax_m27_full_run.py`](repro/examples/minimax_m27_full_run.py) | Canonical full pipeline — 6.5K traces → 17-epoch train → offline eval | ~6-7 hours |
| [`repro/examples/kimi_k25_full_run.py`](repro/examples/kimi_k25_full_run.py) | Same code, different verifier — proves the abstraction works for non-MiniMax models | ~6-7 hours |

## Walkthroughs

End-to-end documentation (with troubleshooting and reference numbers) lives in [`repro/`](repro/):

- [§1 — Trace generation](repro/01-generation.md)
- [§2 — Training the DFlash drafter](repro/02-training.md)
- [§3 — Inference (GGUF conversion + speculative-decode benchmark)](repro/03-inference.md) *(not yet rewired through the library — uses the legacy script in this section)*

## Registered verifiers

| Name | Verifier model | Drafter target |
|---|---|---|
| `minimax-m2.7` | MiniMax-M2.7 (FP8) | hidden=3072, layers=`[2,16,30,45,59,61]`, vocab=200064, mask=200054 |
| `minimax-m2.7-iq4-xs` | MiniMax-M2.7 (UD-IQ4_XS GGUF) | same as above; uses `llamacpp_gguf` backend |
| `kimi-k2.5` | Kimi-K2.5 | hidden=7168, layers=`[1,12,24,35,47,58]`, vocab=163840, mask=163838 |
| `deepseek-v4-flash` | DeepSeek-V4-Flash | hidden=4096, layers=`[2,11,21,32,41,42]`, vocab=129280, mask=1 (EOS) |
| `deepseek-v4-pro` | DeepSeek-V4-Pro | same shape as Flash |
| `nemotron3-super-120b` | Nemotron-3-Super-120B-A12B | hidden=4096, layers=`[3,22,44,66,86,87]`, vocab=131072, mask=0 (PAD) ⚠️ hybrid Mamba arch |
| `nemotron3-nano-30b-a3b` | Nemotron-3-Nano-30B-A3B | hidden=2688, layers=`[2,13,26,39,50,51]`, vocab=131072, mask=0 (PAD) ⚠️ hybrid Mamba arch |
| `qwen3-4b`, `qwen3-14b` | Qwen3 family | reference small-model targets |
| `generic` | **any model** | shape entirely from CLI/Python kwargs — see below |

> ⚠️ The Nemotron-3 family is **hybrid Mamba+MLP+Attention** (per-layer
> `hybrid_override_pattern` in HF config). DFlash speculative decoding has
> historically been validated on pure-transformer verifiers; treat the
> Nemotron factories as starting points and verify your loss curve before
> assuming the canonical tap schedule works.

## Adapting to a new model

Three escalating levels of customization:

### 1. Override layer taps for an existing family

The factory defaults reflect what we trained against, but you can pick any
schedule:

```python
v = load_verifier(
    "minimax-m2.7-iq4-xs",
    hf_path=..., gguf_path=...,
    layer_ids=[1, 8, 20, 35, 50, 61],   # custom tap schedule
)
```

CLI:

```bash
dflash-llama generate --verifier minimax-m2.7-iq4-xs \
  --hf-repo MiniMaxAI/MiniMax-M2 --gguf-repo unsloth/MiniMax-M2-GGUF --gguf-quant UD-IQ4_XS \
  --layer-ids "1,8,20,35,50,61" \
  --prompts data/prompts_tulu3 --out data/traces ...
```

Per-shape overrides (`--hidden-size`, `--num-hidden-layers`, `--vocab-size`,
`--mask-token-id`, `--block-size`, `--drafter-arch`, `--drafter-hidden-act`)
work the same way.

### 2. Build a fully-custom verifier inline (no Python file)

Use `--verifier generic` (or `name="generic"`) plus the four required shape
kwargs. Layer taps are mandatory unless you ask for `--num-layer-taps N`.

```python
v = load_verifier(
    "generic",
    name_override="llama-3.1-8b",
    hf_path=..., gguf_path=...,
    hidden_size=4096,
    num_hidden_layers=32,
    vocab_size=128256,
    mask_token_id=128255,
    layer_ids=[2, 8, 16, 24, 30, 31],
)
```

```bash
dflash-llama generate --verifier generic \
  --name-override llama-3.1-8b \
  --hf-repo meta-llama/Llama-3.1-8B \
  --gguf-repo bartowski/Llama-3.1-8B-GGUF --gguf-quant Q4_K_M \
  --hidden-size 4096 --num-hidden-layers 32 \
  --vocab-size 128256 --mask-token-id 128255 \
  --layer-ids "2,8,16,24,30,31" \
  --prompts data/prompts_tulu3 --out data/traces ...
```

If you don't have a known-good schedule, omit `--layer-ids` and pass
`--num-layer-taps 6` — `auto_layer_ids` will spread taps across the network
and always include the final residual. **Verify your loss curve looks sane;
auto-spread is a starting point, not gospel.**

### 3. Register a Python factory (only if you want a stable name)

For one-line reuse across scripts/tests/teammates, use `register_verifier`:

```python
from dflash_llama import BaseVerifier, register_verifier

def my_model(*, hf_path=None, gguf_path=None, **kw):
    return BaseVerifier(
        name="my-model-8b",
        hidden_size=4096, num_hidden_layers=32,
        vocab_size=131072, mask_token_id=131071,
        layer_ids=[2, 8, 16, 24, 30, 31],
        hf_path=hf_path, gguf_path=gguf_path, **kw,
    )

register_verifier("my-model-8b", my_model)
```

## Architecture

```
src/dflash_llama/
├── verifiers/                 # model-family configs (the abstraction)
│   ├── base.py                # BaseVerifier dataclass
│   ├── minimax_m2.py          # MiniMax-M2.7 + IQ4 variant
│   ├── kimi_k25.py            # Kimi-K2.5
│   ├── qwen3.py               # Qwen3-4B/14B + generic factory
│   └── auto.py                # detect from HF/GGUF config
├── generation/                # trace generation
│   ├── format.py              # save_trace, load_trace, saturating_fp8_cast
│   ├── trace_generator.py     # TraceGenerator high-level API
│   └── backends/              # verifier-execution backends
│       ├── llamacpp_gguf.py   # wraps llama-dump-hiddens binary
│       └── …                  # (transformers_hf future)
├── training/                  # drafter training
│   ├── prompts.py             # assemble_prompts_arrow (replaces v2 build_paired_dataset.py)
│   ├── vocab_maps.py          # build_vocab_maps (with v2 import + dtype bugs fixed)
│   ├── dataset.py             # SelfDescribingTraceDataset
│   ├── smoke.py               # 90s smoke runner
│   ├── trainer.py             # DFlashTrainer high-level API
│   └── eval.py                # offline_eval (§2.8 validator)
├── inference/                 # placeholders — buun-converter integration TBD
└── cli.py                     # `dflash-llama` entry point
tests/                         # 34+ unit tests, all passing on CPU
repro/                         # walkthroughs + examples
```

## Design contract

- Saturating fp8 cast is the **only** supported fp8 path. Naive `tensor.to(torch.float8_e4m3fn)` produces NaN for any value with magnitude >448. This library never emits NaN.
- Self-describing trace files mean pairing-by-hash is gone. The trainer reads `input_ids`/`loss_mask`/source metadata directly off each safetensor.
- The training shell-out goes through `torchrun … speculators/scripts/train.py`. This is intentionally pragmatic — when the speculators in-process API stabilises we will swap to a programmatic invocation without users having to change anything.

## Tests

```bash
pip install -e .[dev]
pytest -v
```

34 tests cover trace format roundtrip (no NaN with abs_max=5000), verifier configs, vocab maps (numpy/torch coercion), `SelfDescribingTraceDataset`, and end-to-end smoke on synthetic data.

## Roadmap

- GGUF export of trained drafters (currently a placeholder; uses legacy `prep_full_for_buun_converter.py` for now)
- Per-block-32 fp8 storage option (poor-man's MXFP8) — better quality for verifiers with deep-layer outliers
- HF transformers backend for trace generation (currently only llama.cpp GGUF)
- Multi-machine sharding helpers (`workers.py`)
