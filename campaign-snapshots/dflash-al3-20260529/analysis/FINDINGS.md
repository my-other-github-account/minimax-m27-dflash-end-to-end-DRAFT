# FINDINGS — dflash-al3 session analysis (2026-05-29)

This file captures reasoning developed in the driver session that is not in any single task report.

## 1. What the "best" numbers actually are

| result | model | step | from-scratch? | prompts | dmax | n | measured AL | reproducible? |
|---|---|---|---|---|---|---|---|---|
| T72 repro | OUR step500 (093736) | 500 | yes | GSM8K-ish | 13 | 5 | 2.433 | yes (REPRO_WINDOW) |
| T72 probe | OUR step500 (093736) | 500 | yes | GSM8K-ish | 15 | 3 | 2.922 | thin (optional_probe) |
| T81 | OLD step2500 export | 2500 | n/a | GSM8K-ish | 17 | 3 | 3.368 | reproduced @dmax17 |

- The OUR-adapter best is a **step-500 from-scratch** drafter (cold init: global_step=0 loss=10.75 full_acc=0).
  Run dir: `c15_iq4v18b_NAS_T3_spark3_val500_20260529T093736Z`.
- The "3.39" is **NOT our adapter** — T81's preferred step500 GGUF was missing on spark-1, so it silently
  fell back to `/home/dnola/exports/c15_iq4v18_step2500_dflash_mm_tok.gguf` (older, 5x more trained).

## 2. Train-side AL bounds vs runtime (the red flag)

step500 train per-position accept @ step500: [0.302,0.223,0.163,0.145,0.131,0.118,0.122]
- AL_LO (strict chain, 1+Σ∏p) = **1.38**
- AL_HI (independent, 1+Σp)    = **2.20**

The trainer logs **no AL_LO/AL_HI lines**; these were computed from per-position accept accs.

**Runtime AL (2.43–2.92) EXCEEDS AL_HI (2.20).** A correctly-measured verified speculative-decode AL
cannot exceed the teacher-forced ceiling. Exceeding it means one of:
  (a) runtime accept rule looser than teacher-forced eval (temp/top_k/criterion mismatch),
  (b) runtime prompts OOD-easy vs the val distribution, or
  (c) wrong AL denominator.

## 3. Prime suspect: the AL denominator + draft_max self-feed

- Reported AL formula (T72/T81): `AL = n_predict / (n_drafted / draft_max)`. draft_max is in the
  denominator. **Correct AL = accepted_tokens / target_verify_steps.**
- Evidence it is an artifact: same model + same prompts, only draft_max changed 15→17, and **every
  prompt's AL rose in lockstep** (p1 3.43→3.98, p2 2.46→2.79; mean 2.94→3.39). A real AL saturates with
  draft budget; it does not scale with it.
- We trained **only 7 draft positions**. draft_max=13/15/17 means the drafter is **self-feeding past its
  trained depth** (positions 8–17 are out-of-training-distribution autoregression).
- Third red flag: **top_k=1 and top_k=0 produced byte-identical AL** (2.944955 == 2.944955;
  3.388706 == 3.388706). If the verifier truly sampled, these could not match — suggests the accept
  decision may not depend on the verifier distribution (possible unverified/free-running draft).

## 4. Distribution mismatch

Consolidated train/val mix = **86% prompts_tulu3** (170,305 rows) + 14% preprocessed; seq_len median
529, p90 1664. NOT short grade-school math. GSM8K runtime prompts are plausibly OOD-easy → inflated
runtime AL relative to the Tulu3-measured per-position accs. Hence the in-dist measurement task.

## 5. Open verification plan (as of snapshot)

- `t_e6dde2c8` (spark-1): in-dist Tulu3 runtime AL + **denominator audit (recompute AL_true =
  emitted/target_verify_steps)** + lossless-output check (target-only greedy MUST equal target+draft
  greedy, token-for-token) + use OUR step500 GGUF only + recover step2500 train-side [AL_LO,AL_HI] for
  a fair comparison + explain top_k invariance + sweep dmax {7,13,15,17} (AL_true should be flat if real).
- `t_d5077e99` (spark-1, child): measured wall-clock tokens/sec speedup vs no-draft baseline, same host,
  warmed server, same prompts. Reports break-even accept_frac/AL if speedup <= 1.0x.

## 6. The single result that settles it

**AL_true (accepted ÷ target-verify-steps) on OUR step500, with token-identical output verified.**
If AL_true >= 3 AND output is lossless → real. If AL_true flattens near the 7-trained-head depth and
the 2.9/3.4 collapse → the headline numbers were denominator/self-feed mirages.

## Infra notes carried in
- One actor per spark host (prevents the spark-2 OOM-from-contention wedge). Workers block SPARK*_BUSY.
- buun-llama-cpp runner: `llama-speculative-simple --spec-type dflash` at
  `/home/dnola/buun-llama-cpp/build/bin/`. Target = MiniMax-M2.7-UD-IQ4_XS GGUF.
- Watchdog cron auto-flags: WRONG_MODEL, DENOMINATOR_ARTIFACT, draft_max>7, TOP_K_INVARIANT,
  NOT_LOSSLESS, EOS short-stop.
