# T83: spark-3 bounded continuation from T80 step700

Verdict: CKPT_READY
Timestamp: 2026-05-29T15:04Z
Host scope: spark-3 only

## Preflight

- Contention: no pre-existing trainer process found before launch; `tmux ls` showed only `nasfs_t71` (NasFS mount helper), not a trainer session.
- GPU: no compute apps before launch; no compute apps after completion.
- Memory: `free -g` showed 119 GiB total and 98 GiB available before launch (>=90 GiB usable); after completion available was 97 GiB.
- Source checkpoint existed: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T80_spark3_bounded_step750_ckpt50_20260529T142824Z/step_00000700`.
- Hidden-state visibility gate over `/home/dnola/iq4_full_run/iq4_v17_consolidated/hidden_states_nas`: 1000-entry `test -e` / Path.exists-equivalent sample passed 1000/1000, 100.0% coverage, meeting >=95% requirement.

## Patch scope

Used only the existing T71-approved optimizer-resume patch in `/home/dnola/speculators-c15-api/src/speculators/train/trainer.py` (AdamW foreach/fused disabled on resume). No additional source patch was made.

## Launch

- RUN_LABEL: `c15_iq4v18b_NAS_T83_spark3_bounded_step800_ckpt50_20260529T145314Z`
- SAVE: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T83_spark3_bounded_step800_ckpt50_20260529T145314Z`
- TRAIN_LOG: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T83_spark3_bounded_step800_ckpt50_20260529T145314Z/train.log`
- Launch wrapper: `/home/dnola/iq4_full_run/launch_T83_spark3_bounded.sh`
- Launch env: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T83_spark3_bounded_step800_ckpt50_20260529T145314Z/t83_launch_env.txt`
- Source checkpoint was symlinked into the new save dir as `step_00000700` for resume discovery.
- MAX_TRAIN_STEPS=800 (+100 optimizer records from checkpoint state 700; stopped at global_step=800)
- VAL_EVERY_STEPS=50
- VAL_IN_EPOCH_MAX_BATCHES=20
- TORCHINDUCTOR_COMPILE_THREADS=4
- OMP_NUM_THREADS=8
- Locked C15/v18b knobs preserved by `/home/dnola/iq4_full_run/launch_c15_v18b_NAS_spark3.sh`: max_anchors=3072, block_size=8, hidden=3072, intermediate=4096, layers=6, seq_len=8192, draft_vocab_size=32768, single-host/no FSDP.

## CKPT_READY

Fresh candidate checkpoint:

`/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T83_spark3_bounded_step800_ckpt50_20260529T145314Z/step_00000750`

Convenience symlink:

`/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T83_spark3_bounded_step800_ckpt50_20260529T145314Z/checkpoint_last_inepoch -> step_00000750`

Run resumed from checkpoint tag step_00000700 at global_step=700, trained 100 records through global_step=799, stopped cleanly at global_step=800 due to `max_train_steps=800`, and left no trainer running.

Checkpoint contents verified:

- `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T83_spark3_bounded_step800_ckpt50_20260529T145314Z/step_00000700` (symlink to /home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T80_spark3_bounded_step750_ckpt50_20260529T142824Z/step_00000700, trainer_state global_step=700)
- `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T83_spark3_bounded_step800_ckpt50_20260529T145314Z/step_00000750` (fresh directory, trainer_state global_step=750)

## Numeric signal

Training log parsed 100 train records from global_step 700 through 799.

Train metric start:

- global_step=700: loss=4.625, full_acc=0.158

Train metric end before stop:

- global_step=799: loss=4.719, full_acc=0.207

In-epoch validation metrics:

- step_00000700: loss_epoch=5.1390625, full_acc_epoch=0.161855913, AL_LO=1.396280, AL_HI=2.132152
  - position accs: p1=0.316826121, p2=0.213334484, p3=0.153197389, p4=0.129375948, p5=0.113623498, p6=0.105494003, p7=0.100300631
- step_00000750: loss_epoch=5.1218750, full_acc_epoch=0.182724073, AL_LO=1.469839, AL_HI=2.278093
  - position accs: p1=0.359512553, p2=0.253352751, p3=0.181758465, p4=0.142470723, p5=0.124372286, p6=0.111567877, p7=0.105057926

## Post-run evidence

- Trainer process count: 0
- GPU compute app count: 0
- `free -g` after completion:

```text
total        used        free      shared  buff/cache   available
Mem:             119          21          39          16          76          97
Swap:             15           0          15
```

## Artifacts

- Parse JSON: `/home/dnola/v18_fix/T83_PARSE.json`
- Run label: `/home/dnola/v18_fix/T83_RUN_LABEL.txt`
- Launch wrapper: `/home/dnola/iq4_full_run/launch_T83_spark3_bounded.sh`
- Train log: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T83_spark3_bounded_step800_ckpt50_20260529T145314Z/train.log`
