# T54 SPARK3 STEP700 SMOKE

Status: SMOKE_BLOCKED
Timestamp: 2026-05-29T11:16:55.799251+00:00
Host: spark-3

## Verdict
No new +100/+200 checkpoint was produced. The mandatory contention preflight passed, and the trainer resumed from the harvested step_00000500 checkpoint, but every attempted post-resume training batch raised `[ANCHOR-SKIP][train]: No valid anchors were selected for this batch`, so the trainer advanced `global_step` counters without optimizer steps, never hit the successful-forward validation/checkpoint branch, and exited rc=0 at the smoke guard.

This is a launcher/trainer-data blocker, not a CKPT_READY result. I did not fabricate a step700 checkpoint by copying step500.

## Preflight evidence
```text
pgrep -af [s]cripts/train.py: <none>
nvidia-smi compute apps: <none>
free -g Mem: Mem:             119           5          98           0          17         114
TORCHINDUCTOR_COMPILE_THREADS=4 and OMP_NUM_THREADS=8 were exported in /tmp/t54_launch.sh before launch.
```

## Starting checkpoint
/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T3_spark3_val500_20260529T093736Z/step_00000500

## Config sanity
- draft_vocab_size=32768 (expected 32768)
- transformer_layer_config.vocab_size=200064
- mask_token_id=200054
- block_size=8, max_anchors=3072, aux_hidden_state_layer_ids=[2, 16, 30, 45, 59]
- effective vocab mapping appears sane in config: draft_vocab_size is present and target vocab remains 200064.

## Smoke attempts
### c15_iq4v18b_NAS_T54_spark3_step700_20260529T111128Z
- save_path: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T54_spark3_step700_20260529T111128Z`
- train_log: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T54_spark3_step700_20260529T111128Z/train.log`
- result: rc=0 (observed by launcher)
- step dirs present: step_00000500
- train/loss lines after resume: 0
- train anchor-skip warnings: 200
- in-epoch validation/checkpoint triggers: 0
- stop line: `INFO     Stopping training early at global_step=700    trainer.py:399`

### c15_iq4v18b_NAS_T54_spark3_probe1200_20260529T111329Z
- save_path: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T54_spark3_probe1200_20260529T111329Z`
- train_log: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T54_spark3_probe1200_20260529T111329Z/train.log`
- result: rc=0 (observed by launcher)
- step dirs present: step_00000500
- train/loss lines after resume: 0
- train anchor-skip warnings: 700
- in-epoch validation/checkpoint triggers: 0
- stop line: `INFO     Stopping training early at global_step=1200   trainer.py:399`

## Key blocker excerpt
```text
[04:12:04] WARNING  [ANCHOR-SKIP][train] step=500: No valid anchors were selected for this batch — skipping batch
...
[04:15:16] WARNING  [ANCHOR-SKIP][train] step=1199: No valid anchors were selected for this batch — skipping batch
INFO     Stopping training early at global_step=1200 due to max_train_steps=1200
```

## Cleanup / final state
No live `scripts/train.py` remains after the smoke attempts. No GPU compute apps were left running.
