# dflash-llama

Self-describing fp8 trace generation and DFlash drafter training for llama-family verifiers (MiniMax-M2, Kimi-K2.5, Qwen3, …).

## Quickstart (3 lines)

```bash
pip install -e .
dflash-llama generate --verifier minimax-m2.7-iq4-xs --gguf-path /path/to/UD-IQ4_XS --prompts /path/to/prompts_arrow --rows 0:1000 --out ./traces
dflash-llama train --traces ./traces --verifier minimax-m2.7-iq4-xs --output ./checkpoint
```

See `repro/01-generation.md` and `repro/02-training.md` for end-to-end walkthroughs.

## What this library is for

- **Generate self-describing fp8 traces** — every safetensor file contains hidden states (saturating fp8 + per-tensor scale, never NaN), token_ids, input_ids, loss_mask, plus full provenance metadata (source name, source row index, generation timestamp, schema version). No more post-hoc sha256 pairing.
- **Train DFlash drafters end-to-end** — wraps the [speculators](https://github.com/neuralmagic/speculators) trainer, but does the prep (prompts arrow assembly, vocab maps) inside the library instead of in a fragile shell-script chain.
- **Model-family abstractions** — `BaseVerifier` configs encode `hidden_size`, `vocab_size`, `mask_token_id`, layer taps, etc. Picking a new family means adding one ~30-line config file.

## Design contract

- Saturating fp8 cast is the **only** supported fp8 path. Direct `bf16 → fp8_e4m3fn` casts produce NaN for any value > ±448 — this library never emits NaN.
- Self-describing trace files mean pairing-by-hash is gone. The trainer reads metadata directly off the safetensor.
- The training shell-out goes through `torchrun … speculators/scripts/train.py` and is intentionally pragmatic — when the speculators in-process API stabilises we will swap to a programmatic invocation.

See `src/dflash_llama/` for the package source and `tests/` for full coverage.
