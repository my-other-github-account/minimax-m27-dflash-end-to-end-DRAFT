# T86: spark-3 bounded continuation from T83 step750

Verdict: CKPT_READY
Timestamp: 2026-05-29T15:36Z
Host scope: spark-3 only

## Preflight

- Contention: no pre-existing trainer process found before launch; `tmux ls` showed only `nasfs_t71` (NasFS mount helper), not a trainer session.
- GPU: no compute apps before launch; no compute apps after completion.
- Memory: `free -g` showed 119 GiB total and 97 GiB available before launch (>=90 GiB); after completion available was 98 GiB.
- Source checkpoint existed: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T83_spark3_bounded_step800_ckpt50_20260529T145314Z/step_00000750`.
- Hidden-state visibility gate over `/home/dnola/iq4_full_run/iq4_v17_consolidated/hidden_states_nas`: 1000/1000 = 100.0% (Path.exists sample; elapsed 222.63s), meeting >=95% requirement.

## Patch scope

Used only the existing T71-approved optimizer-resume patch in `/home/dnola/speculators-c15-api/src/speculators/train/trainer.py` (AdamW foreach/fused disabled on resume). No additional source patch was made.

## Launch

- RUN_LABEL: `c15_iq4v18b_NAS_T86_spark3_bounded_step850_ckpt50_20260529T152740Z`
- SAVE: `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T86_spark3_bounded_step850_ckpt50_20260529T152740Z`
- TRAIN_LOG: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T86_spark3_bounded_step850_ckpt50_20260529T152740Z/train.log`
- Launch wrapper: `/home/dnola/iq4_full_run/launch_T86_spark3_bounded.sh`
- Source checkpoint symlinked into new save dir as `step_00000750` for resume discovery.
- MAX_TRAIN_STEPS=850 (+100 optimizer records from checkpoint state 750; stopped at global_step=850)
- VAL_EVERY_STEPS=50
- VAL_IN_EPOCH_MAX_BATCHES=20
- TORCHINDUCTOR_COMPILE_THREADS=4
- OMP_NUM_THREADS=8
- Locked C15/v18b knobs preserved by `/home/dnola/iq4_full_run/launch_c15_v18b_NAS_spark3.sh`: max_anchors=3072, block_size=8, hidden=3072, intermediate=4096, layers=6, seq_len=8192, draft_vocab_size=32768, single-host/no FSDP.

## CKPT_READY

Fresh candidate checkpoint:

`/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T86_spark3_bounded_step850_ckpt50_20260529T152740Z/step_00000800`

Convenience symlink:

`/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T86_spark3_bounded_step850_ckpt50_20260529T152740Z/checkpoint_last_inepoch -> step_00000800`

Run resumed from checkpoint tag step_00000750 at global_step=750, trained 100 records through global_step=849, stopped cleanly at global_step=850 due to `max_train_steps=850`, and left no trainer running.

Checkpoint contents verified:

- `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T86_spark3_bounded_step850_ckpt50_20260529T152740Z/step_00000750` (symlink to /home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T83_spark3_bounded_step800_ckpt50_20260529T145314Z/step_00000750, trainer_state global_step=750)
- `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T86_spark3_bounded_step850_ckpt50_20260529T152740Z/step_00000800` (fresh directory, trainer_state global_step=800)

## Numeric signal

Training log parsed 100 train records from global_step 750 through 849.

Train metric start:

- global_step=750: loss=4.469, full_acc=0.172

Train metric end before stop:

- global_step=849: loss=4.688, full_acc=0.21

In-epoch validation metrics:

- step_00000750: loss_epoch=5.0625000, full_acc_epoch=0.174943630, AL_LO=1.438781, AL_HI=2.223684
  - position accs: p1=0.343599495, p2=0.231802861, p3=0.168597367, p4=0.137924994, p5=0.121772271, p6=0.113538484, p7=0.106448526
- step_00000800: loss_epoch=5.0312500, full_acc_epoch=0.188035661, AL_LO=1.486921, AL_HI=2.315252
  - position accs: p1=0.369564420, p2=0.261068808, p3=0.185172077, p4=0.147009632, p5=0.129392223, p6=0.115072382, p7=0.107972924

## Post-run evidence

- Trainer process count: 0
- GPU compute app count: 0
- `free -g` after completion:

```text
total        used        free      shared  buff/cache   available
Mem:             119          21          35          16          80          98
Swap:             15           0          15
```

## Artifacts

- `/home/dnola/v18_fix/T86_PARSE.json`
- `/home/dnola/v18_fix/T86_RUN_LABEL.txt`
- `/home/dnola/iq4_full_run/launch_T86_spark3_bounded.sh`
- `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T86_spark3_bounded_step850_ckpt50_20260529T152740Z/train.log`
