# LUCE-MINIMAX-REVERT-ROPE report (t_77088062)

Host: spark-3 (`<USER>@<SPARK_HOST>`). Target GGUF: `/home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf`. Drafter: `/home/<USER>/t_5567a42d/golden_step_00020000/model_lucebox_layout.safetensors`, md5 `30c27f646590b562608774bee82d2108`.

## Step A: revert to best measured pre_K_no_mask

Reverted the measured regressions:
- `src/qwen35/qwen35_backend.cpp:1341-1343`: restored `pos_k[i]=i` / 0-based full K span for draft graph.
- `src/common/dflash_spec_decode.cpp:138-140`: restored same 0-based full K span for common decode path.
- Restored pre-mask graph files from `*.pre-draftmaskfix-t_d81951a4`: `src/draft/draft_graph.{h,cpp}`, `src/common/step_graph.h`, `src/common/dflash_draft_graph.cpp`, `src/common/dflash_spec_decode.cpp`; full layers are back to `mask=nullptr` and only SWA can receive `causal_mask_swa`.

Reverted/baseline binary md5: `fa08acfdb72b0df2463b934161b75caa`.
Previous bad K+mask md5 was `d930bc1f81cb6876a379a881ccb82c20`.

Baseline remeasure, same 7 prompts, free-running greedy (`temperature=0`, `top_k=1`, `max_tokens=64`):
- AL_true = 3.527559055118
- p1..p7 argmax = [0.677165, 0.503937, 0.354331, 0.196850, 0.228346, 0.212599, 0.173228]
- prefix p1..p7 = [0.677165, 0.464567, 0.259842, 0.125984, 0.078740, 0.015748, 0.000000]

This exactly reproduces the approved best baseline (`AL_true≈3.53`, p1≈0.677), so the mechanical revert is complete.

Artifacts:
- `/home/<USER>/t_77088062/measure_revert_7prompts_summary.json`
- `/home/<USER>/t_77088062/raw_completions_revert_7prompts.json`
- `/home/<USER>/t_77088062/server_8097_revert.latest`

## Step B: RoPE/K-Q convention probe and one fix attempt

Named divergence:
- Runtime lucebox best-baseline feeds synthetic draft Q/K RoPE positions starting at `draft_ctx`: `src/qwen35/qwen35_backend.cpp:1341-1343` and `src/common/dflash_spec_decode.cpp:138-140`.
- `draft_graph.cpp:111-125` applies `ggml_rope_ext` directly to Q using `in.positions_q` and to K using `pk`/`in.positions_k`, so this source position convention is exactly the post-RoPE convention.
- The z-lab/upstream training audit in `/home/<USER>/ref_posdiff_t_91fe2ebc/report.md:8-20` shows default train `position_ids` are 1-based and draft block RoPE rows are anchor+1..anchor+block. For a runtime `draft_ctx=10`, z-lab synthetic Q rows are `[11..18]`, while best-baseline lucebox uses `[10..17]`.

Cosine probe verdict (analytic RoPE rows, dim=128, theta=10000, representative `draft_ctx=10`):
- best-baseline Q positions: [10, 11, 12, 13, 14, 15, 16, 17]
- z-lab/reference Q positions: [11, 12, 13, 14, 15, 16, 17, 18]
- row cosine baseline-vs-zlab per q0..q7: [0.970214, 0.970214, 0.970214, 0.970214, 0.970214, 0.970214, 0.970214, 0.970214]
- attempted +1 fix Q positions: [11, 12, 13, 14, 15, 16, 17, 18]
- row cosine +1-vs-zlab per q0..q7: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]

One reference-grounded fix tried:
- Changed synthetic draft Q/K-noise positions to `draft_ctx+i+1` while keeping base K at the measured-best `0..ctx-1`.
- Rebuilt binary md5: `2be5094b16c5d18fce61c11b67c72120`.
- Server probe confirmed e.g. `pos_q_first=11 pos_q_last=18 pos_k_base_first=0 pos_k_base_last=9 pos_k_noise_first=11 pos_k_noise_last=18`.

Remeasure of that fix, same 7 prompts:
- AL_true = 2.461538461538
- p1..p7 argmax = [0.313187, 0.302198, 0.269231, 0.230769, 0.181319, 0.236264, 0.126374]
- prefix p1..p7 = [0.313187, 0.126374, 0.027473, 0.016484, 0.010989, 0.000000, 0.000000]

Verdict: the +1 synthetic Q/K-noise RoPE convention made rotary rows match the z-lab default-position audit, but it strongly regressed acceptance (AL_true 3.5276 -> 2.4615; p1 0.6772 -> 0.3132). Therefore it is NOT kept.

## Final kept config

Kept final binary md5: `fa08acfdb72b0df2463b934161b75caa` (the reverted pre_K_no_mask best baseline). The on-disk source has been restored to that baseline after the failed +1 attempt and rebuilt.

Final runtime vs train target:
- train p1..p7 target = [0.723000, 0.593000, 0.501000, 0.437000, 0.383000, 0.334000, 0.290000]
- kept p1..p7 = [0.677165, 0.503937, 0.354331, 0.196850, 0.228346, 0.212599, 0.173228]
- attempted +1 p1..p7 = [0.313187, 0.302198, 0.269231, 0.230769, 0.181319, 0.236264, 0.126374]

Not reconciled. AL_true is inside the calc band but p2..p7 are not within 20% of train. Next named file:line to attack should NOT be another blind position-origin shift; the empirical result says the clean best runtime uses zero-based synthetic RoPE despite the train default audit. Next probe should capture real post-RoPE Q/K tensors from `src/draft/draft_graph.cpp:118-125` (not only position rows) and compare against a Python/torch forward loaded from the exported drafter for the same prompt/hidden state.
