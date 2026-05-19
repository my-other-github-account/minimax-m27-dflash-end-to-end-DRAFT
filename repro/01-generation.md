# §1 — Trace Generation

End-to-end recipe to generate self-describing fp8 hidden-state traces for a llama-family verifier, using the `dflash-llama generate` CLI (or the `TraceGenerator` Python API).

## What you get

For every input prompt, one `hs_<row>.safetensors` file containing:

| key | shape | dtype | notes |
|---|---|---|---|
| `hidden_states` | `(seq, n_layers, hidden)` | `float8_e4m3fn` | saturating per-tensor scale — never NaN |
| `hidden_states_scale` | `(1,)` | `float32` | apply on load: `hs_bf16 = hs_fp8 * scale` |
| `token_ids` | `(seq,)` | `int64` | verifier-emitted token sequence |
| `input_ids` | `(seq,)` | `int64` | the prompt's tokens (for trainer pairing) |
| `loss_mask` | `(seq,)` | `bool` | anchor mask (right-side by default) |

Plus safetensor metadata: `schema_version=v3`, `source_name`, `source_row_idx`, `gen_timestamp`, `storage`, `n_layers`, `seq_len`, `hidden_size`, `layer_ids` (JSON), `abs_max`.

This is **self-describing** — there is no separate prompts arrow + sha256 pairing step. Every field needed for training is right there in the safetensor.

## Why fp8 with saturating scale

Direct `tensor.to(torch.float8_e4m3fn)` produces NaN for any value whose magnitude exceeds 448 (the fp8_e4m3fn finite range). MiniMax-M2.7 hidden states routinely peak at ±2260, so a naive cast silently turns >2% of activations into NaN. The library's `saturating_fp8_cast` divides by `max(abs_max/448, 1.0)`, clamps, casts, and stores the scale — round-trip max relative error is ~1.5%, NaN count is exactly zero.

## CLI quickstart

```bash
# 1. Generate 1k traces of MiniMax-M2.7 from a prompts arrow dataset
dflash-llama generate \
    --verifier minimax-m2.7-iq4-xs \
    --gguf-path /path/to/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
    --prompts /path/to/prompts_arrow_dir \
    --rows 0:1000 \
    --out /path/to/output/traces \
    --state /path/to/output/state.json \
    --max-seq-len 2048
```

The run is **resumable**: re-running with the same `--out` skips rows whose `hs_<i>.safetensors` already exists, and the `--state` JSON tracks completed/failed counts atomically.

To use a different verifier:

```bash
dflash-llama info  # list registered verifier names
# ... or describe an arbitrary model with --verifier generic:
dflash-llama generate --verifier generic \
  --name-override my-model-8b \
  --hf-path /models/my-model --gguf-path /models/my-model.gguf \
  --hidden-size 4096 --num-hidden-layers 32 \
  --vocab-size 128256 --mask-token-id 128255 \
  --layer-ids "2,8,16,24,30,31" ...
```

## Python API quickstart

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
)
gen.generate(
    prompts="/path/to/prompts_arrow_dir",
    output_dir="/path/to/output/traces",
    rows=range(0, 1000),
    state_path="/path/to/output/state.json",
    max_seq_len=2048,
)
```

## Verifier-specific defaults (validated families)

| verifier | hidden | n_layers | layer_ids | vocab | mask |
|---|---|---|---|---|---|
| `minimax-m2.7` / `…-iq4-xs` | 3072 | 62 | `[2, 16, 30, 45, 59, 61]` | 200064 | 200054 |

For experimental factories (`kimi_k25`, `qwen3*`, `deepseek_v4_*`,
`nemotron3_*`) see `dflash_llama.verifiers.experimental`. They have
plausible shape metadata but have NOT been end-to-end validated by this
library — opt in explicitly with `register_verifier(...)`.

The `layer_ids` list is what the generator passes to `llama-dump-hiddens`. The trainer auto-appends a final tap, so the speculators `--target-layer-ids` flag receives `layer_ids[:-1]`.

## Backend: `llamacpp_gguf`

> **🚀 For production trace generation, prefer the persistent batched-decode trace server (~2.5× faster, same outputs).** See [§8 — Persistent batched-decode trace server](08-tracegen-server.md). The `llamacpp_gguf` backend documented below is kept as the simple/portable path for one-off jobs and CI.

Wraps the `llama-dump-hiddens` binary (a forked `llama.cpp` build that captures hidden states). The binary is invoked once per row with `TOKENS_BIN`/`OUT_BIN`/`CAPTURE_LAYERS` env vars and a fixed model file.

Default arguments (override via `backend_kwargs={...}`):

```
binary  = "llama-dump-hiddens"
ctx     = 4096
ngl     = 99
extras  = ("-ot", "exps=CPU")
timeout = 600  # per row
```

If the binary lives elsewhere:

```bash
dflash-llama generate ... --binary /home/user/iq4_tracegen/buun-llama-cpp/build/bin/llama-dump-hiddens
```

## Verifying a trace

```python
from dflash_llama.generation import load_trace, validate_trace

meta = validate_trace("/path/to/traces/hs_0.safetensors")
# raises ValueError if NaN, schema mismatch, or missing required fields

d = load_trace("/path/to/traces/hs_0.safetensors")
# d["hidden_states"]: (seq, n_layers, hidden) bf16 — scale already applied
# d["token_ids"], d["input_ids"], d["loss_mask"]
```

## Next step

Once you have a directory of traces, jump to [`02-training.md`](02-training.md). There is **no pairing/sha-matching step** — the trainer reads metadata directly off the trace files.
