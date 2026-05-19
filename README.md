# dflash-llama

Self-describing fp8 trace generation, DFlash drafter training, GGUF
export, OpenAI-compatible serving, and per-position + chain-cumulative
acceptance benchmarking — for **MiniMax-M2.7** as the validated path,
plus a generic adapter for any other llama-family verifier.

This library replaces a fragile shell-script pipeline with a clean
Python API. It is the canonical way to train DFlash speculative-decoding
drafters in this repo.

> **🚀 First time on a Spark? Start here:** [`repro/00-spark-from-scratch.md`](repro/00-spark-from-scratch.md) — full bringup from a bare machine, including the `llama-dump-hiddens` build, the `verifier_meta` stub, the prompts arrow, the speculators install, and the multi-host shard plan. Tested end-to-end on a 4× DGX Spark cluster.

## What is verified end-to-end

The library has been validated through the full pipeline (gen → train →
GGUF → speculative-decode bench with measured non-zero pos-2 acceptance
matching training prediction within sample noise) for **one** family:

- **`minimax-m2.7-iq4-xs`** — MiniMax-M2.7 verifier with the Unsloth
  `UD-IQ4_XS` GGUF quant + DFlash 5L drafter (draft_vocab=32768).
  Validated 2026-04-30 / 2026-05-02 on the reference single-GPU host.

`minimax-m2.7` (FP8 quant) shares all shape metadata. Trace generation
via FP8 was retired (the `tensor.to(float8_e4m3fn)` cast silently
NaN-corrupts any |x|>448; that's the bug this library exists to make
impossible). Kept as a registered factory for documentation continuity.

Other model families have factories with reasonable shape metadata but
**have not been end-to-end validated by this library**. They live in
[`dflash_llama.verifiers.experimental`](src/dflash_llama/verifiers/experimental/)
and require explicit opt-in. See **Experimental factories** below.

## Walkthroughs

| topic | doc |
|---|---|
| §1 — Trace generation | [`repro/01-generation.md`](repro/01-generation.md) |
| §2 — Training a DFlash drafter | [`repro/02-training.md`](repro/02-training.md) |
| §3 — Inference: GGUF export, OpenAI-compat server, speculative-decode benchmark | [`repro/03-inference.md`](repro/03-inference.md) |
| §4 — Empirical tau against llama-benchy / Project Gutenberg traffic | [`repro/04-empirical-tau-llama-benchy.md`](repro/04-empirical-tau-llama-benchy.md) |
| §5 — Runtime perf notes: how the v11.1 wall-clock gate was cleared | [`repro/05-runtime-perf-notes.md`](repro/05-runtime-perf-notes.md) |
| §6 — FP8 training (Float8CurrentScaling HYBRID + fused TE LayerNormMLP) | [`repro/06-fp8-training.md`](repro/06-fp8-training.md) |
| §7 — Phase 2 training perf optimization (2.3× over bf16 baseline) | [`repro/07-perf-optimization.md`](repro/07-perf-optimization.md) |
| §8 — Persistent batched-decode trace server (2.5× faster trace generation) | [`repro/08-tracegen-server.md`](repro/08-tracegen-server.md) |

## What you get

- **Self-describing fp8 traces** — every `hs_<i>.safetensors` carries
  `hidden_states` (saturating fp8 + per-tensor scale, **never NaN**),
  `token_ids`, `input_ids`, `loss_mask`, plus full provenance metadata
  (`schema_version`, `source_name`, `source_row_idx`, `gen_timestamp`,
  `layer_ids`). No more post-hoc sha256 pairing.
- **End-to-end DFlash training** —
  `DFlashTrainer.prepare() → .smoke() → .train() → .offline_eval()`.
  Wraps the [speculators](https://github.com/neuralmagic/speculators)
  trainer; does the data prep inside the library instead of in shell
  scripts. Bf16 is the default; pass ``fp8_recipe_kind="current_fp8",
  te_use_fused=True`` to ``.train()`` for the verified-stable FP8 path
  on DGX Spark sm_121a (+18% throughput, see [§6](repro/06-fp8-training.md)).
- **GGUF export of trained drafters** — `export_to_gguf()` runs the full
  validated recipe: rebake `lm_head` from draft-vocab to target-vocab
  with the `-65504` floor (the d2t zero-row dilution fix), idempotent
  tokenizer-hash registration in buun's converter, then the converter
  itself. Verified to round-trip through `llama-speculative-simple`.
- **OpenAI-compat serving** — `LlamaServer(verifier_gguf, drafter_gguf)`
  context manager wraps `llama-server` with `--spec-type dflash`. Plug
  any OpenAI client at `http://<host>:<port>/v1`.
- **Per-position + chain-cumulative benchmark** — `benchmark()` sweeps
  `--draft-max`, parses the rejection histogram, and emits a
  `SpeculativeReport` with per-position conditional `p_k` and
  chain-cumulative `∏ p_i` with z-scores against the training prediction
  loaded from `val_metrics.json`.
- **Saturating-fp8 storage** — ~50% disk vs bfloat16 with **zero data
  loss for over-range values**.

## Install

```bash
git clone https://github.com/my-other-github-account/minimax-m27-dflash-end-to-end-DRAFT.git dflash-llama
cd dflash-llama
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest tests/ -q                       # 79 tests pass on CPU in <2s
```

Then build `llama-dump-hiddens` reproducibly from a pinned upstream
`llama.cpp` commit + our vendored example:

```bash
bash scripts/build_llama_dump_hiddens.sh           # outputs binary path on stdout
```

Speculators is required for `trainer.smoke()` and `trainer.train()` —
install via `pip install speculators` or pin to a specific commit (see
[`repro/00-spark-from-scratch.md`](repro/00-spark-from-scratch.md) §5).
For trace generation alone, only torch + safetensors + datasets are
needed.

## Python API quickstart

```python
from dflash_llama import (
    DFlashTrainer, TraceGenerator, load_verifier,
    export_to_gguf, LlamaServer, benchmark,
)

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
trainer.train(save_to="data/ckpt", epochs=14)              # writes val_metrics.json per epoch
trainer.offline_eval(checkpoint="data/ckpt/checkpoint_best", max_batches=60)

# 3. Export to GGUF
gguf = export_to_gguf(
    checkpoint="data/ckpt/checkpoint_best",
    output_path="data/drafter.gguf",
    verifier_meta_dir="data/verifier_meta",   # tokenizer source
    buun_repo="/path/to/buun-llama-cpp",
)

# 4. Serve as OpenAI-compatible endpoint
with LlamaServer(verifier_gguf=".../verifier.gguf", drafter_gguf=gguf, port=8080) as srv:
    print(srv.url)  # → http://localhost:8080/v1

# 5. Benchmark — per-position p_k + chain-cumulative ∏p_i + z-scores
report = benchmark(
    verifier_gguf=".../verifier.gguf",
    drafter_gguf=gguf,
    val_metrics="data/ckpt/checkpoint_best/val_metrics.json",
    dmax_sweep=[2, 4, 7],
)
print(report.markdown())
```

Local-path mode is also supported when you already have files staged
(e.g. on a cluster):

```python
verifier = load_verifier(
    "minimax-m2.7-iq4-xs",
    hf_path="data/hf-cache/MiniMax-M2",
    gguf_path="data/models/UD-IQ4_XS/shard-00001.gguf",
)
```

## CLI quickstart

```bash
# list registered (validated) verifiers
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
  --epochs 14 --lr 3e-5 --max-anchors 64 --save-best

# GGUF export
dflash-llama export-gguf \
  --checkpoint data/ckpt/checkpoint_best \
  --output data/drafter.gguf \
  --verifier-meta-dir data/verifier_meta \
  --verify

# OpenAI-compat server (Ctrl-C to stop)
dflash-llama serve \
  --verifier .../verifier.gguf \
  --drafter  data/drafter.gguf \
  --port 8080

# Speculative-decode benchmark sweep (per-position + chain-cumulative)
dflash-llama benchmark \
  --verifier .../verifier.gguf \
  --drafter  data/drafter.gguf \
  --val-metrics data/ckpt/checkpoint_best/val_metrics.json \
  --dmax 2,4,7
```

You can substitute `--hf-path` / `--gguf-path` for the `--*-repo` flags
when files are already on disk.

## Examples

| File | What it shows |
|---|---|
| [`repro/examples/tiny_smoke.py`](repro/examples/tiny_smoke.py) | The smallest end-to-end demo — 10 traces, prepare, 90s smoke. Verify your environment quickly. |
| [`repro/examples/minimax_m27_full_run.py`](repro/examples/minimax_m27_full_run.py) | Canonical full pipeline — traces → train → offline eval. |
| [`repro/examples/03_export_gguf.py`](repro/examples/03_export_gguf.py) | Convert a trained drafter checkpoint → buun-loadable GGUF. |
| [`repro/examples/03_serve_openai.py`](repro/examples/03_serve_openai.py) | OpenAI-compatible server with DFlash speculative decoding. |
| [`repro/examples/03_benchmark.py`](repro/examples/03_benchmark.py) | Speculative-decode benchmark sweep — per-position p_k + chained ∏p_i + z-scores against training. |

## Registered (validated) verifiers

| Name | Verifier model | Drafter target |
|---|---|---|
| `minimax-m2.7-iq4-xs` | MiniMax-M2.7 (UD-IQ4_XS GGUF) | hidden=3072, layers=`[2,16,30,45,59,61]`, vocab=200064, mask=200054 |
| `minimax-m2.7` | MiniMax-M2.7 (FP8) | same shape; FP8 trace path retired (NaN bug — see Design contract) |
| `generic` | **any model** | shape entirely from CLI/Python kwargs — see "Adapting to a new model" below |

## Experimental factories

The following factories are present in
[`dflash_llama.verifiers.experimental`](src/dflash_llama/verifiers/experimental/)
with sensible-looking shape metadata but **have not been end-to-end
validated** by this library (no full gen → train → GGUF → speculative-
decode bench cycle has been completed against them):

- `kimi_k25` — Kimi-K2.5
- `qwen3`, `qwen3_4b`, `qwen3_14b` — Qwen3 family
- `deepseek_v4_flash`, `deepseek_v4_pro` — DeepSeek-V4
- `nemotron3_super_120b`, `nemotron3_nano_30b_a3b` — Nemotron-3 (hybrid
  Mamba+MLP+Attention; DFlash is historically validated only on pure
  transformers, so even shape-correct factories may not produce
  trainable hiddens)

To use one by name, opt in explicitly:

```python
from dflash_llama import register_verifier
from dflash_llama.verifiers.experimental import kimi_k25
register_verifier("kimi-k2.5", kimi_k25)
v = load_verifier("kimi-k2.5", gguf_path=...)
```

If you successfully end-to-end-validate one, please open a PR moving the
factory back to the main `verifiers/` package along with a short report
(val_loss, chain-pos-1/-2 measured vs predicted, drafter GGUF SHA).

## Adapting to a new model

Three escalating levels of customization:

### 1. Override layer taps for an existing family

The factory defaults reflect what we trained against, but you can pick
any schedule:

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

Per-shape overrides (`--hidden-size`, `--num-hidden-layers`,
`--vocab-size`, `--mask-token-id`, `--block-size`, `--drafter-arch`,
`--drafter-hidden-act`) work the same way.

### 2. Build a fully-custom verifier inline (no Python file)

Use `--verifier generic` (or `name="generic"`) plus the four required
shape kwargs. Layer taps are mandatory unless you ask for
`--num-layer-taps N`.

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
`--num-layer-taps 6` — `auto_layer_ids` will spread taps across the
network and always include the final residual. **Verify your loss curve
looks sane; auto-spread is a starting point, not gospel.**

### 3. Register a Python factory (only if you want a stable name)

For one-line reuse across scripts/tests/teammates, use
`register_verifier`:

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
│   ├── minimax_m2.py          # MiniMax-M2.7 + IQ4 variant   ← validated
│   ├── generic.py             # generic_verifier + auto_layer_ids
│   ├── auto.py                # detect from HF/GGUF config
│   └── experimental/          # NOT end-to-end-validated; explicit opt-in
│       ├── kimi_k25.py
│       ├── qwen3.py
│       ├── deepseek_v4.py
│       └── nemotron3.py
├── generation/                # trace generation
│   ├── format.py              # save_trace, load_trace, saturating_fp8_cast
│   ├── trace_generator.py     # TraceGenerator high-level API
│   └── backends/              # verifier-execution backends
│       └── llamacpp_gguf.py   # wraps llama-dump-hiddens binary
├── training/                  # drafter training
│   ├── prompts.py             # assemble_prompts_arrow
│   ├── vocab_maps.py          # build_vocab_maps
│   ├── dataset.py             # SelfDescribingTraceDataset
│   ├── smoke.py               # 90s smoke runner
│   ├── trainer.py             # DFlashTrainer high-level API
│   └── eval.py                # offline_eval (§2.8 validator)
├── inference/                 # GGUF export, OAI server, spec-decode bench
│   ├── gguf_export.py         # export_to_gguf, prep_for_buun_converter
│   ├── server.py              # LlamaServer (OpenAI-compat)
│   ├── benchmark.py           # benchmark() with tqdm progress
│   └── analyze.py             # SpeculativeReport, log parser, z-score
└── cli.py                     # `dflash-llama` entry point
tests/                         # 79 unit tests, all passing on CPU
repro/                         # walkthroughs + examples
```

## Design contract

- Saturating fp8 cast is the **only** supported fp8 path. Naive
  `tensor.to(torch.float8_e4m3fn)` produces NaN for any value with
  magnitude >448. This library never emits NaN.
- Self-describing trace files mean pairing-by-hash is gone. The trainer
  reads `input_ids`/`loss_mask`/source metadata directly off each
  safetensor.
- The training shell-out goes through
  `torchrun … speculators/scripts/train.py`. This is intentionally
  pragmatic — when the speculators in-process API stabilises we will
  swap to a programmatic invocation without users having to change
  anything.
- The GGUF export uses an `lm_head` rebake with a `-65504` floor for
  non-mapped rows. Using `0.0` recreates a documented "47×
  accept-rate gap" bug — see the d2t zero-row dilution note in
  [`repro/03-inference.md`](repro/03-inference.md).
- Only end-to-end-validated families ship in the named registry.
  Everything else lives in `dflash_llama.verifiers.experimental` and
  requires explicit opt-in.

## Tests

```bash
pip install -e .[dev]
pytest -v
```

79 tests cover trace format roundtrip (no NaN with abs_max=5000),
verifier configs (validated + experimental), vocab maps (numpy/torch
coercion), `SelfDescribingTraceDataset`, end-to-end smoke on synthetic
data, GGUF-export module imports without buun, log-parser correctness
on real `llama-speculative-simple` output, chained-prediction math, and
the `LlamaServer` argv builder. All run on CPU; no GPU/network
dependencies.

## Roadmap

- Self-describing prompt arrows: `stage_prompts_mix()` to build a
  multi-source HF prompt dataset with per-row provenance (source dataset
  + source row idx + prompt sha + tokenizer sha) so partial-run gen
  pools are reconstructible without seed-based magic. (Audit complete;
  implementation pending.)
- Per-block-32 fp8 storage option (poor-man's MXFP8) — better quality
  for verifiers with deep-layer outliers.
- HF transformers backend for trace generation (currently only
  llama.cpp GGUF).
- More end-to-end-validated families, promoting one experimental factory
  per validation cycle into the main namespace.
