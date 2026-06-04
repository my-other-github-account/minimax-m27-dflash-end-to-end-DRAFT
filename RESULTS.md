# RESULTS: MiniMax-M2 DFlash lucebox reproduction (draft)

Status: reproduced AL 3.53; publication draft, no GitHub push from this worker.

## Primary reproduced result: R1 on spark-3

Source artifacts:

- Report: `spark-3:/home/<USER>/t_77088062/NOT_RECONCILED_MINIMAX_REVERT_ROPE.md`
- Summary JSON: `spark-3:/home/<USER>/t_77088062/measure_revert_7prompts_summary.json`
- Raw completions: `spark-3:/home/<USER>/t_77088062/raw_completions_revert_7prompts.json`
- Server log: `spark-3:/home/<USER>/t_77088062/server_8097_revert_20260604T130925Z.log`
- Server command: `spark-3:/home/<USER>/t_77088062/server_cmd_revert.sh`

Measured summary:

- model: `minimax-luce-golden-revert`
- prompts: 7
- max_tokens: 64
- temperature: 0
- top_k: 1
- total_tokens: 448
- total_steps: 127
- AL_true_mean_commit: `3.52755905511811`
- accepted/draft_total: `326/1016`
- accept_rate: `0.32086614173228345`
- per_pos_argmax_match weighted p1..p7:
  `[0.6771653464566929, 0.5039369921259843, 0.35433083464566933, 0.19685030708661416, 0.22834636220472443, 0.21259853543307086, 0.17322849606299215]`
- per_pos_prefix_accept weighted p1..p7:
  `[0.6771653464566929, 0.46456685826771654, 0.2598424566929134, 0.1259842283464567, 0.07874012598425197, 0.015748023622047243, 0.0]`

Binary/source evidence:

- Reverted/baseline binary md5 recorded in report: `fa08acfdb72b0df2463b934161b75caa`
- Target GGUF shard 00001 md5: `019759eeec4be4931592eeefbd54d2ba`
- Drafter lucebox layout md5: `30c27f646590b562608774bee82d2108`

## Independent reproduction: R2 on spark-6

Source artifacts:

- Summary JSON: `spark-6:/home/<USER>/t_1d157fda/measure_repro6_7prompts_summary.json`
- Result report: `spark-6:/home/<USER>/t_1d157fda/REPRO6_RESULT.md`
- Server command: `spark-6:/home/<USER>/t_1d157fda/server_cmd_repro6.sh`
- Binary md5 file: `spark-6:/home/<USER>/t_1d157fda/binary_md5.txt`

Measured summary:

- model: `minimax-luce-golden-repro6`
- prompts: 7
- total_tokens: 448
- total_steps: 127
- AL_true_mean_commit: `3.52755905511811`
- accepted/draft_total: `326/1016`
- accept_rate: `0.32086614173228345`
- per_pos_argmax_match weighted p1..p7:
  `[0.6771653464566929, 0.5039369921259843, 0.35433083464566933, 0.19685030708661416, 0.22834636220472443, 0.21259853543307086, 0.17322849606299215]`
- per_pos_prefix_accept weighted p1..p7:
  `[0.6771653464566929, 0.46456685826771654, 0.2598424566929134, 0.1259842283464567, 0.07874012598425197, 0.015748023622047243, 0.0]`

Binary evidence:

- `7e27408f4d1db727777f9ff7a03c0ed9  build-sm121-local/dflash_server`

## Original / approved baseline

The spark-3 report states that Step A "exactly reproduces the approved best baseline (`AL_true≈3.53`, p1≈0.677)" and names the kept baseline as the reverted `pre_K_no_mask` state. The same report lists the measured values:

- AL_true: `3.527559055118`
- p1..p7 argmax: `[0.677165, 0.503937, 0.354331, 0.196850, 0.228346, 0.212599, 0.173228]`
- prefix p1..p7: `[0.677165, 0.464567, 0.259842, 0.125984, 0.078740, 0.015748, 0.000000]`

The separate R2 spark-6 run matches R1 exactly on AL, tokens/steps, accepted/draft_total, and weighted per-position vectors.

## Known issues / pending work

- 256-token server stability remains pending (R5 status); this draft reports the verified 64-token, 7-prompt measurement only.
- Result is front-loaded: p1-p4 drive most of AL, with weaker later positions.
- Speedup-vs-AR baseline remains pending (R6).
- Accept-verify details remain pending (R4).
- Runtime-vs-train p2..p7 are not reconciled in the spark-3 report; it explicitly says not to keep the attempted +1 RoPE shift because it regressed AL from 3.5276 to 2.4615.
