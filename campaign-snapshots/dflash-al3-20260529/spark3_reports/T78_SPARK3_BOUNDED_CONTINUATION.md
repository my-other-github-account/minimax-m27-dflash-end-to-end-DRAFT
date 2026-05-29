# T78: spark-3 bounded continuation from T75 step600

Verdict: CKPT_READY
Timestamp: 2026-05-29T14:07Z
Host scope: spark-3 only

## Preflight
- Contention: no pre-existing `scripts/train.py` / trainer process found before launch; `tmux ls` showed only `nasfs_t71` (NasFS mount helper), not a trainer session.
- GPU: no compute apps before launch; no compute apps after completion.
- Memory: preflight `free -g` showed 119 GiB total and 96 GiB available (>=90 GiB requirement; free column was low due buff/cache).
- Source checkpoint existed: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T75_spark3_bounded_step650_ckpt50_20260529T132624Z/step_00000600`.
- Hidden-state visibility gate over `/home/dnola/iq4_full_run/iq4_v17_consolidated/hidden_states_nas`: bounded 1000-entry `test -e` sample for `hs_0.safetensors` through `hs_999.safetensors` passed 1000/1000, 100.0% coverage, 50.6 sec, no missing/timeouts.

## Patch scope
Used only the existing T71-approved optimizer-resume patch in:

`/home/dnola/speculators-c15-api/src/speculators/train/trainer.py`

Backup remains:

`/home/dnola/v18_fix/trainer.py.pre_t71_foreach_patch`

No additional source patch was made.

## Launch
- RUN_LABEL: `c15_iq4v18b_NAS_T78_spark3_bounded_step700_ckpt50_20260529T135637Z`
- SAVE: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T78_spark3_bounded_step700_ckpt50_20260529T135637Z`
- TRAIN_LOG: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T78_spark3_bounded_step700_ckpt50_20260529T135637Z/train.log`
- Launch env: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T78_spark3_bounded_step700_ckpt50_20260529T135637Z/t78_launch_env.txt`
- Source checkpoint: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T75_spark3_bounded_step650_ckpt50_20260529T132624Z/step_00000600`
- Source checkpoint was symlinked into the new save dir as `step_00000600` for resume discovery.
- MAX_TRAIN_STEPS=700
- VAL_EVERY_STEPS=50
- VAL_IN_EPOCH_MAX_BATCHES=20
- TORCHINDUCTOR_COMPILE_THREADS=4
- OMP_NUM_THREADS=8
- Locked C15/v18b knobs preserved by `/home/dnola/iq4_full_run/launch_c15_v18b_NAS_spark3.sh`: max_anchors=3072, block_size=8, hidden=3072, intermediate=4096, layers=6, seq_len=8192, single-host/no FSDP.

## CKPT_READY
Fresh candidate checkpoint:

`/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T78_spark3_bounded_step700_ckpt50_20260529T135637Z/step_00000650`

Convenience symlink:

`/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T78_spark3_bounded_step700_ckpt50_20260529T135637Z/checkpoint_last_inepoch -> step_00000650`

Run resumed at global_step=600, trained 100 records through global_step=699, stopped cleanly at global_step=700 due to `max_train_steps=700`, and left no trainer running.

Checkpoint contents verified:

- /home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T78_spark3_bounded_step700_ckpt50_20260529T135637Z/step_00000600 (source symlink, trainer_state global_step=600, tag=step_00000600)
  - files: config.json, config.py, model.safetensors, optimizer_state_dict.pt, scheduler_state_dict.pt, trainer_state.json
- /home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T78_spark3_bounded_step700_ckpt50_20260529T135637Z/step_00000650 (fresh, trainer_state global_step=650, tag=step_00000650)
  - files: config.json, config.py, model.safetensors, optimizer_state_dict.pt, scheduler_state_dict.pt, trainer_state.json

## Numeric signal
Training log parsed 100 train records from global_step 600 through 699.

Train metric start:

- global_step=600: loss=5.375, full_acc=0.119

Train metric end before stop:

- global_step=699: loss=4.875, full_acc=0.196

In-epoch validation metrics:

- step_00000600: loss_epoch=5.4109375, full_acc_epoch=0.14426071494817733
  - position accs: p1=0.2646817214787006, p2=0.1836784802377224, p3=0.13962815925478936, p4=0.119141785800457, p5=0.1078032098710537, p6=0.09970380179584026, p7=0.09453188218176364
- step_00000650: loss_epoch=5.125, full_acc_epoch=0.16470656618475915
  - position accs: p1=0.3273520298302174, p2=0.22142985314130784, p3=0.1607556752860546, p4=0.12757223844528198, p5=0.11547689065337181, p6=0.10318727307021618, p7=0.09630763530731201

No training-side AL_LO/AL_HI lines were logged. No ANCHOR-SKIP lines were logged.

## Post-run verification
- `nvidia-smi --query-compute-apps` after completion: empty.
- `ps` search for `scripts/train.py` / `train.py` after completion: empty.

## Artifacts
- Report: `/home/dnola/v18_fix/T78_SPARK3_BOUNDED_CONTINUATION.md`
- Parse JSON: `/home/dnola/v18_fix/T78_PARSE.json`
- Run label file: `/home/dnola/v18_fix/T78_RUN_LABEL.txt`
- Launch wrapper: `/home/dnola/iq4_full_run/launch_T78_spark3_bounded.sh`
- Launch env: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T78_spark3_bounded_step700_ckpt50_20260529T135637Z/t78_launch_env.txt`
- Train log: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T78_spark3_bounded_step700_ckpt50_20260529T135637Z/train.log`
