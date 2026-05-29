# T69: T66 training-side vs runtime gap audit

## Scope
- Host: spark-3 only.
- Checkpoint: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T3_spark3_val500_20260529T093736Z/step_00000500`.
- Repo/env used: `/home/dnola/speculators-c15-api`, `/home/dnola/venvs/te/bin/python`.
- No training was launched.

## Offline eval attempt
I attempted a 5-batch read-only teacher-forced eval with the exact training repo/env and T66/T3 launcher knobs. The checkpoint load only missed the frozen verifier head/norm (`verifier_lm_head.weight`, `verifier_norm.weight`) and otherwise matched after enabling the same fused TE wrapper.

The eval could not complete a real val batch because the cached hidden-state symlinks under `iq4_v17_consolidated/hidden_states_nas` currently point at missing `/home/dnola/nas_traces/by_hash/...` targets on spark-3, so the dataloader produced empty batches and DFlash raised `No valid anchors were selected for this batch`. Evidence:
- `/home/dnola/iq4_full_run/iq4_v17_consolidated/hidden_states_nas/hs_0.safetensors -> /home/dnola/nas_traces/by_hash/00/00/...safetensors`
- `/home/dnola/nas_traces/by_hash` is absent on spark-3 at audit time.
- Eval log: `/home/dnola/v18_fix/T69_fast_eval.log`.

Given the FAST artifact requirement, I used the source training log from the same checkpoint run as the training-side metric source:
`/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T3_spark3_val500_20260529T093736Z/train.log`.

## Numeric result from training-side log
Source slice: last 20 logged training batches before the checkpoint, global_step 481..500.

Per-position marginal argmax acc vector, positions 1..7 (anchor pos0 excluded):
`[0.306000, 0.213850, 0.156100, 0.130150, 0.117450, 0.108600, 0.101650]`

Other metrics:
- slice size: 20 logged training batches
- first parsed training-side loss_0: 9.750 at global_step 21
- current step loss at checkpoint step 500: 5.219
- last-20 mean loss: 5.115750
- last-20 mean full_acc: 0.162050

Bounds:
- AL_LO = 1 + sum cumulative products = 1.383157
- AL_HI = 1 + sum marginals = 2.133800

For reference, the single checkpoint step 500 row alone was:
- marginals: `[0.302, 0.223, 0.163, 0.145, 0.131, 0.118, 0.122]`
- AL_LO: 1.382151
- AL_HI: 2.204000
- loss: 5.219

## Runtime comparison
- T66 raw runtime AL baseline: 1.623
- T66 raw runtime AL chat: 1.846
- Training-side AL_LO from current-window metrics: 1.383, below runtime.
- Training-side AL_HI from current-window metrics: 2.134, below the campaign target of 3.0.

## Verdict
MODEL_LOW.

Rationale: even the optimistic training-side upper bound from current checkpoint-window marginal accuracies is below 3.0, while the lower bound is near/below the observed runtime accept lengths. This points at the T66 checkpoint/model quality rather than a large runtime underperformance gap. The offline val path should be retried only after restoring the missing NAS hidden-state targets on spark-3.
