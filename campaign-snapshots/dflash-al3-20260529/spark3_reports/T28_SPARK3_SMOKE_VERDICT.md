# T28 spark-3 v18b bounded smoke verdict

Verdict: SPARK3_TRAINING_VIABLE

UTC written: 2026-05-29T03:10:28Z
Host: spark-3
Launcher: /home/dnola/iq4_full_run/launch_c15_v18b_NAS_spark3.sh
Train log: /home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T28_spark3_smoke_20260529T022305Z/train.log
Run dir: /home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T28_spark3_smoke_20260529T022305Z
Checkpoint/save path: /home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T28_spark3_smoke_20260529T022305Z

## Smoke result
- MAX_TRAIN_STEPS=100 was respected.
- Last logged train metric: global_step=99; trainer then stopped cleanly at global_step=100 due to max_train_steps=100.
- Coverage gate: PASS (97.286%, 198486 rows/symlinks).
- TE FP8 wrap: PASS; HYBRID Float8CurrentScaling; fprop/dgrad/wgrad use_split_accumulator=True; coverage_after unfused=0.
- Locked shape used: max_anchors=3072, total_seq_len=8192, block_size=8, num_layers=6, hidden=3072, intermediate=4096, draft_vocab_size=32768.
- Dataloader: num_workers=2, prefetch_factor=2, val_num_workers=0, on_missing=skip, NasFS mounted.
- No spark-3 host reset during smoke; host uptime remained continuous after run.
- Journal warning scan for OOM/NVRM/Xid/memory/panic/reset/segfault/killed since run start: CLEAN.

## Loss / accuracy trajectory
- Initial train/loss: 10.75 at first metric.
- Final train/loss: 6.656 at last metric.
- Initial full_acc: 0.0.
- Final full_acc: 0.086.
- Observed trajectory: loss descended from ~10.7 to ~6.6 by step 99; full_acc rose from 0 to ~0.086. This is a healthy bounded smoke trajectory, not the T24 host-reset signature.

## Memory / host pattern
- nvidia-smi on GB10 reports utilization but memory used/total as N/A in this environment, so GPU-memory stability was assessed by run survival + absence of kernel/journal memory-pressure reset evidence.
- The training process exited cleanly via max_train_steps, and only the long-running nasfs tmux session remained.

## Journal evidence
```text
no matching warning/error lines
```

## Recommendation
spark-3 is viable as an alternate single-Spark training host for the locked v18b shape. T3 can be relaunched on spark-3 with this launcher/path; spark-2 remains blocked by T25 as a host-specific hardware/pressure gate.
