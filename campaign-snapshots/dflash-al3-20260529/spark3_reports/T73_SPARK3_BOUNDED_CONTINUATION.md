# T73: spark-3 bounded continuation after HS restore

Verdict: CKPT_READY
Timestamp: 2026-05-29T13:12Z
Host scope: spark-3 only

## Preflight
- Contention: no pre-existing trainer found before launch; post-run no live train.py remains.
- GPU: no compute apps before launch; training used spark-3 GB10 only.
- Memory: preflight `free -g` showed 119 GiB total, 111 GiB available (>=90 GiB requirement; free column low due buff/cache).
- Hidden-state visibility gate over `/home/dnola/iq4_full_run/iq4_v17_consolidated/hidden_states_nas`: 1000/1000 Path.exists() sample, 100.0% coverage, 39.843 sec, bad_sample=[].

## Patch scope
Used only the T71-approved live scoped optimizer-resume patch in:

`/home/dnola/speculators-c15-api/src/speculators/train/trainer.py`

Backup remains:

`/home/dnola/v18_fix/trainer.py.pre_t71_foreach_patch`

Patch behavior: AdamW constructed with foreach=False and resumed optimizer param_groups foreach/fused reset false. No broader source/export mutation was made.

## Launch
Primary CKPT-producing run:

- RUN_LABEL: `c15_iq4v18b_NAS_T73_spark3_bounded_step600_ckpt50_20260529T130019Z`
- SAVE: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T73_spark3_bounded_step600_ckpt50_20260529T130019Z`
- TRAIN_LOG: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T73_spark3_bounded_step600_ckpt50_20260529T130019Z/train.log`
- Source checkpoint: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T3_spark3_val500_20260529T093736Z/step_00000500`
- MAX_TRAIN_STEPS=600
- VAL_EVERY_STEPS=50
- VAL_IN_EPOCH_MAX_BATCHES=20
- TORCHINDUCTOR_COMPILE_THREADS=4
- OMP_NUM_THREADS=8
- Locked C15/v18b knobs preserved: max_anchors=3072, block_size=8, hidden=3072, intermediate=4096, layers=6, seq_len=8192, single-host/no FSDP.

Note: an initial bounded run with VAL_EVERY_STEPS=100 reached global_step=600, but the trainer's max_train_steps stop check fired before writing a fresh step600 checkpoint. I reran with VAL_EVERY_STEPS=50 so a fresh checkpoint was written at step550 while still stopping at global_step=600.

## CKPT_READY
Fresh candidate checkpoint:

`/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T73_spark3_bounded_step600_ckpt50_20260529T130019Z/step_00000550`

Convenience symlink:

`/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T73_spark3_bounded_step600_ckpt50_20260529T130019Z/checkpoint_last_inepoch -> step_00000550`

Run reached global_step=600 and stopped cleanly; no trainer left running.

Checkpoint contents verified:

- config.json
- config.py
- model.safetensors
- optimizer_state_dict.pt
- scheduler_state_dict.pt
- trainer_state.json (global_step=550, checkpoint_tag=step_00000550)

## Numeric signal
Training log parsed 100 train records from global_step 500 through 599.

Train metric start:

- global_step=500: loss=6.781, full_acc=0.078

Train metric end before stop:

- global_step=599: loss=5.094, full_acc=0.182

In-epoch validation metrics:

- step_00000500: loss_epoch=6.421875, full_acc_epoch=0.11304893046617508
- step_00000550: loss_epoch=5.39375, full_acc_epoch=0.1450233481824398

Step550 position accs:

- position 1 acc_epoch=0.2734439313411713
- position 2 acc_epoch=0.1901867777109146
- position 3 acc_epoch=0.13993567749857902
- position 4 acc_epoch=0.11738023385405541
- position 5 acc_epoch=0.10579918362200261
- position 6 acc_epoch=0.09612232223153114
- position 7 acc_epoch=0.09161320254206658

No training-side AL_LO/AL_HI lines were logged. No ANCHOR-SKIP lines were logged.

## Artifacts
- Report: `/home/dnola/v18_fix/T73_SPARK3_BOUNDED_CONTINUATION.md`
- Launch env: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T73_spark3_bounded_step600_ckpt50_20260529T130019Z/t73_launch_env.txt`
- Train log: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T73_spark3_bounded_step600_ckpt50_20260529T130019Z/train.log`
