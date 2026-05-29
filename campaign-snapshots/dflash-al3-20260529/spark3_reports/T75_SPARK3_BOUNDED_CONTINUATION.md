# T75: spark-3 bounded continuation from T73 step550

Verdict: CKPT_READY
Timestamp: 2026-05-29T13:37Z
Host scope: spark-3 only

## Preflight
- Contention: no pre-existing `scripts/train.py` trainer found before launch. `tmux ls` showed only `nasfs_t71` (NasFS mount helper), not a trainer session.
- GPU: no compute apps before launch; no compute apps after completion.
- Memory: preflight `free -g` showed 119 GiB total, 96 GiB available (>=90 GiB requirement; free column low due buff/cache).
- Source checkpoint existed: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T73_spark3_bounded_step600_ckpt50_20260529T130019Z/step_00000550`.
- Hidden-state visibility gate over `/home/dnola/iq4_full_run/iq4_v17_consolidated/hidden_states_nas`: bounded `test -e` sample for `hs_0.safetensors` through `hs_999.safetensors` passed 1000/1000, 100.0% coverage, 177 sec, no missing/timeouts. A first Python `Path.exists()` loop without per-file timeout exceeded 120 sec and was killed; the bounded per-entry gate passed before training.

## Patch scope
Used only the existing T71-approved optimizer-resume patch in:

`/home/dnola/speculators-c15-api/src/speculators/train/trainer.py`

Backup remains:

`/home/dnola/v18_fix/trainer.py.pre_t71_foreach_patch`

No additional source patch was made.

## Launch
- RUN_LABEL: `c15_iq4v18b_NAS_T75_spark3_bounded_step650_ckpt50_20260529T132624Z`
- SAVE: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T75_spark3_bounded_step650_ckpt50_20260529T132624Z`
- TRAIN_LOG: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T75_spark3_bounded_step650_ckpt50_20260529T132624Z/train.log`
- Launch env: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T75_spark3_bounded_step650_ckpt50_20260529T132624Z/t75_launch_env.txt`
- Source checkpoint: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T73_spark3_bounded_step600_ckpt50_20260529T130019Z/step_00000550`
- Source checkpoint was symlinked into the new save dir as `step_00000550` for resume discovery.
- MAX_TRAIN_STEPS=650
- VAL_EVERY_STEPS=50
- VAL_IN_EPOCH_MAX_BATCHES=20
- TORCHINDUCTOR_COMPILE_THREADS=4
- OMP_NUM_THREADS=8
- Locked C15/v18b knobs preserved by `/home/dnola/iq4_full_run/launch_c15_v18b_NAS_spark3.sh`: max_anchors=3072, block_size=8, hidden=3072, intermediate=4096, layers=6, seq_len=8192, single-host/no FSDP.

## CKPT_READY
Fresh candidate checkpoint:

`/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T75_spark3_bounded_step650_ckpt50_20260529T132624Z/step_00000600`

Convenience symlink:

`/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T75_spark3_bounded_step650_ckpt50_20260529T132624Z/checkpoint_last_inepoch -> step_00000600`

Run resumed at global_step=550, trained 100 records through global_step=649, stopped cleanly at global_step=650 due to `max_train_steps=650`, and left no trainer running.

Checkpoint contents verified:

- config.json
- config.py
- model.safetensors
- optimizer_state_dict.pt
- scheduler_state_dict.pt
- trainer_state.json (`global_step=600`, `checkpoint_tag=step_00000600`)

## Numeric signal
Training log parsed 100 train records from global_step 550 through 649.

Train metric start:

- global_step=550: loss=5.531, full_acc=0.105

Train metric end before stop:

- global_step=649: loss=5.000, full_acc=0.184

In-epoch validation metrics:

- step_00000550: loss_epoch=5.64375, full_acc_epoch=0.12678863406181334
- step_00000600: loss_epoch=5.328125, full_acc_epoch=0.15694108754396438

Step600 position accs:

- position 1 acc_epoch=0.3021322160959244
- position 2 acc_epoch=0.20864173173904418
- position 3 acc_epoch=0.15145491622388363
- position 4 acc_epoch=0.12806084789335728
- position 5 acc_epoch=0.11284168250858784
- position 6 acc_epoch=0.1007454063743353
- position 7 acc_epoch=0.09393191412091255

No training-side AL_LO/AL_HI lines were logged. No ANCHOR-SKIP lines were logged.

## Artifacts
- Report: `/home/dnola/v18_fix/T75_SPARK3_BOUNDED_CONTINUATION.md`
- Run label file: `/home/dnola/v18_fix/T75_RUN_LABEL.txt`
- Launch wrapper: `/home/dnola/iq4_full_run/launch_T75_spark3_bounded.sh`
- Launch env: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T75_spark3_bounded_step650_ckpt50_20260529T132624Z/t75_launch_env.txt`
- Train log: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T75_spark3_bounded_step650_ckpt50_20260529T132624Z/train.log`
