# T65 spark-3 lease cleanup / orphan trainer harvest

Timestamp: 2026-05-29T03:55-07:00
Host: spark-3

## Verdict
SPARK3_CLEANED candidate: the live `scripts/train.py` was from the T3 spark-3 cron monitor (`c15_train_NAS_v18b_s3`, run `c15_iq4v18b_NAS_T3_spark3_val500_20260529T104916Z`) while T3/T54 were blocked and no running dflash-al3 Kanban task owned spark-3. It was therefore an orphan/lease leak for this cleanup task.

## Ownership cross-check
- Board running tasks at cleanup time: T65 (this cleanup), T64 (MiniMax AL prompt/BOS A-B), T5 (spark-1 background AL). No running spark-3 training task.
- T3 is blocked; its local cron job `148d9a1051e0` (`T3 v18b spark-3 monitor + AL harvest`) was still enabled and last ran at 03:50, relaunching the live trainer. T65 paused that cron before process termination.
- T54 is blocked on `SPARK3_BUSY` from the same family of PIDs.
- T41 is blocked on spark-2, T49/T59 are done.

## Read-only evidence before cleanup
- `pgrep -af [s]cripts/train.py`:
  - PID 2343189 parent python `scripts/train.py`, run-name/save/log `c15_iq4v18b_NAS_T3_spark3_val500_20260529T104916Z`
  - PIDs 2343566, 2343568 worker children with same args
- `tmux ls`: `c15_train_NAS_v18b_s3`, `nasfs`
- `free -g`: 119G total, 73G used, 1G free, 82G buff/cache, 45G available
- `nvidia-smi --query-compute-apps`: PID 2343189 using 25228 MiB

## Harvested progress/checkpoints
- Current orphan run: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T3_spark3_val500_20260529T104916Z/train.log`
  - advanced only to `global_step=42`; last observed loss=9.062 at 03:54:09; no checkpoint directory entries under `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T3_spark3_val500_20260529T104916Z/`.
- New useful complete checkpoint harvested from the immediately previous same-family run:
  - `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T3_spark3_val500_20260529T093736Z/step_00000500`
  - verified complete: `config.json` 1588 bytes, `model.safetensors` 2280441827 bytes, `trainer_state.json` 67 bytes
  - log: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T3_spark3_val500_20260529T093736Z/train.log`
  - reached `global_step=500`, `train/loss=5.219`, saved pre-validation checkpoint at 03:28:51, then hit torch._dynamo recompile warnings at 03:29:03.
- Older complete step500 checkpoints also exist:
  - `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T3_spark3_val500_20260529T074947Z/step_00000500`
  - `/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T3_spark3_val500_20260529T064306Z/step_00000500`

## Action taken
Paused local cron job `148d9a1051e0` in `/Users/macmini/.hermes/profiles/atlasdflash3/cron/jobs.json` before stopping the trainer, to prevent immediate relaunch.
Stopped the scoped spark-3 `c15_train_NAS_v18b_s3` trainer/session and its `scripts/train.py` PIDs first. Subsequent memory recovery found the T3 `nasfs` tmux/cache still consuming `/dev/shm`; with no running spark-3 Kanban owner, T65 stopped/cleared those stale T3 NASFS resources too.

## Additional memory recovery
Post-trainer cleanup still showed only 77G available because the T3 `nasfs` tmux had left `/dev/shm/nasfs-cache-s3` at 38G. No `nasfs.py` process or tmux session remained, so T65 removed that stale cache directory.

Final verification after cache cleanup:
- no `scripts/train.py` remains
- no `nasfs.py` remains
- no tmux sessions remain on spark-3
- `/dev/shm`: 60G size, 11M used
- `free -g`: 119G total, 37G used, 111G free, 4G buff/cache, 114G available
- `nvidia-smi --query-compute-apps`: no compute apps
