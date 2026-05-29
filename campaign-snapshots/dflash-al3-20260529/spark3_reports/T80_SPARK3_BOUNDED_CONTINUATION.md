# T80: spark-3 bounded continuation from T78 step650

Verdict: CKPT_READY
Timestamp: 2026-05-29T14:37Z
Host scope: spark-3 only

## Preflight

- Contention: no pre-existing trainer process found before launch; `tmux ls` showed only `nasfs_t71` (NasFS mount helper), not a trainer session.
- GPU: no compute apps before launch; no compute apps after completion.
- Memory: `free -g` showed 119 GiB total, 49 GiB free, 97 GiB available (>=90 GiB usable; free column low due buff/cache).
- Source checkpoint existed: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T78_spark3_bounded_step700_ckpt50_20260529T135637Z/step_00000650`.
- Hidden-state visibility gate over `/home/dnola/iq4_full_run/iq4_v17_consolidated/hidden_states_nas`: 1000-entry `test -e` sample with per-entry timeout passed 978/1000, 97.8% coverage, elapsed 29 sec, meeting >=95% requirement.

## Patch scope

Used only the existing T71-approved optimizer-resume patch in `/home/dnola/speculators-c15-api/src/speculators/train/trainer.py` (AdamW foreach/fused disabled on resume). No additional source patch was made.

## Launch

- RUN_LABEL: `c15_iq4v18b_NAS_T80_spark3_bounded_step750_ckpt50_20260529T142824Z`
- SAVE: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T80_spark3_bounded_step750_ckpt50_20260529T142824Z`
- TRAIN_LOG: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T80_spark3_bounded_step750_ckpt50_20260529T142824Z/train.log`
- Launch wrapper: `/home/dnola/iq4_full_run/launch_T80_spark3_bounded.sh`
- Launch env: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T80_spark3_bounded_step750_ckpt50_20260529T142824Z/t80_launch_env.txt`
- Source checkpoint was symlinked into the new save dir as `step_00000650` for resume discovery.
- MAX_TRAIN_STEPS=750
- VAL_EVERY_STEPS=50
- VAL_IN_EPOCH_MAX_BATCHES=20
- TORCHINDUCTOR_COMPILE_THREADS=4
- OMP_NUM_THREADS=8
- Locked C15/v18b knobs preserved by `/home/dnola/iq4_full_run/launch_c15_v18b_NAS_spark3.sh`: max_anchors=3072, block_size=8, hidden=3072, intermediate=4096, layers=6, seq_len=8192, draft_vocab_size=32768, single-host/no FSDP.

## CKPT_READY

Fresh candidate checkpoint:

`/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T80_spark3_bounded_step750_ckpt50_20260529T142824Z/step_00000700`

Convenience symlink:

`/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T80_spark3_bounded_step750_ckpt50_20260529T142824Z/checkpoint_last_inepoch -> step_00000700`

Run resumed at global_step=650, trained 100 records through global_step=749, stopped cleanly at global_step=750 due to `max_train_steps=750`, and left no trainer running.

Checkpoint contents verified:

- /home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T80_spark3_bounded_step750_ckpt50_20260529T142824Z/step_00000650 (source symlink, trainer_state global_step=650, tag=step_00000650)
  - files: config.json, config.py, model.safetensors, optimizer_state_dict.pt, scheduler_state_dict.pt, trainer_state.json
- /home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T80_spark3_bounded_step750_ckpt50_20260529T142824Z/step_00000700 (fresh, trainer_state global_step=700, tag=step_00000700)
  - files: config.json, config.py, model.safetensors, optimizer_state_dict.pt, scheduler_state_dict.pt, trainer_state.json

## Numeric signal

Training log parsed 100 train records from global_step 650 through 749.

Train metric start:

- global_step=650: loss=5.0, full_acc=0.134

Train metric end before stop:

- global_step=749: loss=4.781, full_acc=0.203

In-epoch validation metrics:

- step_00000650: loss_epoch=5.2609375, full_acc_epoch=0.14919113852083682, AL_LO=1.354830, AL_HI=2.043590
  - position accs: p1=0.29051066264510156, p2=0.19072631895542144, p3=0.14181663915514947, p4=0.11992547363042831, p5=0.1058699294924736, p6=0.0999649479985237, p7=0.09477596171200275
- step_00000700: loss_epoch=5.1375, full_acc_epoch=0.17352997474372386, AL_LO=1.446152, AL_HI=2.213771
  - position accs: p1=0.34621629044413565, p2=0.24161419793963432, p3=0.16917904391884803, p4=0.13299551159143447, p5=0.11915787048637867, p6=0.1062010195106268, p7=0.09840726740658283

No explicit AL_LO/AL_HI lines were logged; brackets above were computed from validation position accs using LO=1+sum cumulative products and HI=1+sum marginal accs. ANCHOR-SKIP lines: 0.

## Post-run verification

- `nvidia-smi --query-compute-apps` after completion: empty.
- `ps` search for `scripts/train.py` / `train.py` after completion: empty.

## Artifacts

- Report: `/home/dnola/v18_fix/T80_SPARK3_BOUNDED_CONTINUATION.md`
- Parse JSON: `/home/dnola/v18_fix/T80_PARSE.json`
- Run label file: `/home/dnola/v18_fix/T80_RUN_LABEL.txt`
- Launch wrapper: `/home/dnola/iq4_full_run/launch_T80_spark3_bounded.sh`
- Train log: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T80_spark3_bounded_step750_ckpt50_20260529T142824Z/train.log`
