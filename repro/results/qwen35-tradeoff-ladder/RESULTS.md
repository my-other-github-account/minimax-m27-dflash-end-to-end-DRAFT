# Qwen3.6-35B-A3B DFlash — Draft-Length Trade-off Ladder

Measured end-to-end on a single GB10 host (Spark-class, unified memory), DFlash
speculative decoding with the public 5-layer drafter (`draft_vocab=32768`).
All widths run on **one** server binary held byte-identical before/after every
sweep point.

## Provenance / reproducibility pins

- Verifier: `Qwen3.6-35B-A3B` Unsloth `UD-IQ4_XS` GGUF.
- Drafter: DFlash 5L safetensors.
- Server binary md5 (held constant across the whole sweep): `49d9a23d4eb2a655c4f35a18a7a1905a`
- Prompts: 50 fixed eval prompts (256 emitted-token budget).
- Regime (identical across **all** widths and the AR baseline):
  `max_tokens=512, think_max_tokens=256, max_ctx=8192, temperature=0,
  thinking enabled, prefix_cache_slots=0`.
- Width set via `DFLASH_VERIFY_Q_LEN_OVERRIDE` on the same binary.
- Honesty gate: every row had 50/50 non-empty visible-text responses. (The
  server response schema reported `reasoning_tokens_total=0`; rows are validated
  by the visible-text-nonempty fallback gate, not voided.)

`AL_true = n_pred / (n_pred - n_acc)`.

## The ladder

| width | AL_true | wall_speedup_vs_w0 | decode_tok_s | ms/step | accept_eff |
|------:|--------:|-------------------:|-------------:|--------:|-----------:|
| 0 (AR) | 1.000 | 1.000 | 42.2 | —    | —     |
| 1      | 1.000 | 0.677 | 28.2 | —    | —     |
| 2      | 1.848 | 1.168 | 49.6 | 37.3 | 0.924 |
| 4      | 3.005 | 1.658 | 71.9 | 41.8 | 0.751 |
| **8**  | **4.171** | **1.945 ← peak** | **85.4** | 48.8 | 0.521 |
| 12     | 5.344 | 1.591 | 68.9 | 77.6 | 0.445 |
| 16     | 6.166 | 1.776 | 77.5 | 79.6 | 0.385 |

## Findings

1. **Acceptance scales cleanly to width 16.** `AL_true` rises monotonically
   (3.0 → 4.2 → 5.3 → 6.2). The per-position acceptance curve is smooth with
   **no pos≥3 cliff** — e.g. at width 8: `[0.700, 0.569, 0.488, 0.426, 0.381,
   0.339, 0.319]`. This matches the published expectation that wider drafts keep
   accepting more tokens per step.

2. **Full-wall speedup peaks at width 8 (1.95×), then falls.** This is *not* an
   acceptance failure — it is a per-step **verify-cost** effect.

3. **Root cause = a super-linear per-step cost jump between width 8 and 12.**
   `ms/step` jumps **48.8 → 77.6 (+60%)** while drafted tokens grow only ~17%
   (24,552 → 28,740). The extra accepted tokens at w≥12 don't pay for the larger
   verify batch. The leading hypothesis is the on-GPU MoE expert-matmul
   (`ggml_mul_mat_id`) batch-cost regime crossing a threshold around
   `q_len ≈ 8→12` — i.e. a kernel/verify-batch economics effect, not a model or
   acceptance effect. Published implementations that keep net speedup rising to
   width 16 use grouped-GEMM batched-expert kernels where the wider verify batch
   stays cheap.

## Takeaway

- **Best operating point on this host/kernel: width 8 → 1.95× wall speedup.**
- **Acceptance matches the published width-16 benefit** (AL_true 6.17 @ w16); the
  gap to a width-16 *wall-clock* optimum is localized to the MoE verify-batch
  cost cliff at `q_len 8→12`. Fixing that kernel cost is the lever that would
  move the wall-speedup peak from 8 toward 16.

## Files

- `ladder_metrics.json` — full sweep (manifest + per-width raw rows).
- `wN_metrics.json` — per-width detail (AL, per_pos, drafted/accepted, timings).
- `run_ladder.py` — the reproduction harness.
- `REPORT.md` — machine-generated summary emitted by the harness.

Paths in artifacts are scrubbed to `$WORK/`; hostnames to `<host>`; model
weights referenced by md5 only.
