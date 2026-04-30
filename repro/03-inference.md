# Section 3 — Inference (DFlash Speculative Decoding)

> **⚠️ DRAFT — STUB.** This section is intentionally light pending a focused pass on inference reproducibility. The Generation section (§1) is fleshed out; Training (§2) and this section contain pointers + last-known-good config so the 3-section structure is intact.

End-to-end reproduction of `llama.cpp`-side DFlash speculative decoding using the MiniMax-M2.7 target + DFlash drafter trained in §2.

## 3.1 The "47× gap" lesson — read first

The training-side metrics report **per-position conditional accept** (assumes all earlier hiddens correct). The runtime measures **chain-cumulative accept** (each chain position requires all earlier positions to have accepted). These are different quantities.

Predicted chain-gated accept rates from the training-side per-position conditionals:

| | per-pos conditional | chain-cumulative |
|---|---|---|
| pos 1 | 0.2888 | **0.2888** |
| pos 2 | 0.2017 | **0.0583** |
| pos 3 | 0.1480 | **0.0080** |
| pos 7 | 0.1161 | **0.0000004** |

Verified clean-baseline measurements on the Fibonacci prompt (n=95, dmax=2):
- chain-pos-1 = **31.6 %** (z = +0.59σ vs predicted 28.88 %)
- chain-pos-2 = **4.2 %** (z = -0.67σ vs predicted 5.83 %)

**All within sample noise. The drafter is performing as designed.** The earlier "47× apparent gap" was a metric-definition mismatch, not a bug.

## 3.2 Build path

The DFlash inference path lives in PR #22105 (`ruixiang63/llama.cpp@dflash`) plus the 4 patches in this repo's `patches/llama.cpp/`. See the top-level README's "Building the patched llama.cpp" section.

Tested base: PR #22105 tip @ `67cb0d507`. Repo fork with our patches applied: `my-other-github-account/llama.cpp@dflash-minimax-m2`.

## 3.3 Last-known-good inference invocation

```bash
./build/bin/llama-server \
  --model /path/to/MiniMax-M2.7-UD-IQ4_XS.gguf \
  --model-draft /path/to/MiniMax-M2.7-DFlash.gguf \
  --dflash \
  --draft-max 7 \
  --port 8011

# Or smoke-test via:
./build/bin/llama-speculative-simple \
  --spec-type dflash -n 128 -ngl 99 -ngld 99 \
  -ot exps=CPU -devd CUDA0 \
  --model /path/to/MiniMax-M2.7-UD-IQ4_XS.gguf \
  --model-draft /path/to/MiniMax-M2.7-DFlash.gguf \
  -c 8192 --temp 0 \
  -p "<prompt>"
```

## 3.4 GGUF metadata that must match training

The drafter GGUF must encode:
- `target_layer_ids = [2, 16, 30, 45, 59]` (matches §1 generation taps and §2 training)
- `block_size = 8`
- `mask_token_id = 200054`

`target_layer_ids` must match the **integer indices** of the layer taps used at trace generation. **Do not** apply an off-by-one shift like `[1, 15, 29, 44, 58]` — the prior-loop assumption that "buun layer = vLLM index − 1" was wrong. The verified clean-baseline run uses `[2, 16, 30, 45, 59]` and matches the chain-gated math.

## 3.5 Reading the rejection histogram

`llama-speculative-simple`'s output table reports, per chain position, how many drafted tokens were accepted vs rejected. To compute chain-cumulative accept:

```python
# accepts[i] = number of times chain-position-i accepted
# n_drafts   = total number of drafting rounds
chain_accept_pos_i = accepts[i] / n_drafts
```

The `dflash #acc drafts = 0` counter in some buun builds is **stale and never increments** — ignore it. Use the per-position rejection histogram from speculative-simple instead.

## 3.6 What this section needs (pending pass)

- Embed the actual `clean_smoke_dmax{2,4,7}.log` artifacts from the verified run (currently sitting in `${REPRO_WORKSPACE}/logs/` on node1)
- Add a runnable `chain_gated_eval.py` script that takes a `speculative-simple` log and produces the side-by-side table with z-scores
- Document the path from speculators checkpoint → patched-converter input → GGUF inline rather than by pointer
- Document throughput (currently verify-cost-bound at 97-98 % — DFlash is not yet beating AR baseline of 4.4 t/s on this drafter)

Back to [Section 1 — Generation](01-generation.md) or [Section 2 — Training](02-training.md).
