# T71: spark-3 hidden-state restore + one-step proof

Verdict: HS_RESTORED_ONE_STEP_OK
Timestamp: 2026-05-29T12:23Z
Host scope: spark-3 only

## Contention preflight
- `ps ... scripts/train.py`: none before launch.
- `nvidia-smi --query-compute-apps`: no compute apps before launch.
- `free -h` before launch: 119Gi total, 113Gi available (free column low because buff/cache, available >=90Gi).

## Restore action
Restarted the existing lightweight NasFS path in tmux, without copying the dataset:

```text
tmux session: nasfs_t71
command: /home/dnola/venvs/nas-fuse/bin/python /home/dnola/nasfs.py \
  --host nas \
  --remote-root /volume1/dnola/traces \
  --mount-point /home/dnola/nas_traces \
  --cache-dir /dev/shm/nasfs-cache-t71 \
  --cache-bytes 34359738368 \
  --foreground
mount: NasFS on /home/dnola/nas_traces type fuse (ro,...)
```

This restores the expected symlink target root: `/home/dnola/nas_traces/by_hash`.

## Symlink-target coverage gate
Ran a `Path.exists()` gate over the trainer-visible symlink farm before any proof launch:

```json
{
  "sample": 1000,
  "exists": 1000,
  "coverage_pct": 100.0,
  "seconds": 27.540523290634155,
  "bad_sample": []
}
```

The launch script's existing state gate also printed:

```text
[T28] coverage gate PASS: coverage=97.286 rows=198486 symlinks=198486
```

## One-step proof
Seed checkpoint:

```text
/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T3_spark3_val500_20260529T093736Z/step_00000500
```

Successful proof run:

```text
RUN_LABEL=c15_iq4v18b_NAS_T71_spark3_onestep_optpatch_20260529T122119Z
SAVE=/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T71_spark3_onestep_optpatch_20260529T122119Z
TRAIN_LOG=/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T71_spark3_onestep_optpatch_20260529T122119Z/train.log
MAX_TRAIN_STEPS=501
TORCHINDUCTOR_COMPILE_THREADS=4
OMP_NUM_THREADS=8
```

Proof log lines:

```text
[05:21:56] INFO     Training epoch 1/4 started                    trainer.py:638
[05:22:10] INFO     train/loss=6.781, train/full_acc=0.078,       trainer.py:447
                    train/position 1 acc=0.131, train/position 2
                    acc=0.094, train/position 3 acc=0.076,
                    train/position 4 acc=0.068, train/position 5
                    acc=0.064, train/position 6 acc=0.060,
                    train/position 7 acc=0.057, epoch=0,
                    lr=3.00e-07, global_step=500
           INFO     Stopping training early at global_step=501    trainer.py:524
                    due to max_train_steps=501
```

No `[ANCHOR-SKIP]` lines occurred in the successful proof log. No checkpoint beyond the seeded step500 checkpoint was expected or written because this was a one-step bounded proof with `VAL_EVERY_STEPS=999999` and `MAX_TRAIN_STEPS=501`.

## Additional blocker found and patched for proof
The first restored-data proof attempt reached the optimizer step but failed in PyTorch Adam foreach grouping after loading the step500 TE optimizer state:

```text
RuntimeError: Tensors of the same index must be on the same device and the same dtype except `step` tensors that can be CPU and float32/64 notwithstanding
```

Failed log:

```text
/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T71_spark3_onestep_20260529T121800Z/train.log
```

For the successful proof, I applied a small live patch on spark-3 to force resumed AdamW param groups off foreach/fused mode after optimizer-state load:

```diff
--- /home/dnola/v18_fix/trainer.py.pre_t71_foreach_patch
+++ /home/dnola/speculators-c15-api/src/speculators/train/trainer.py
@@
-        self.opt = torch.optim.AdamW(self.model.named_parameters(), lr=self.config.lr)
+        self.opt = torch.optim.AdamW(self.model.named_parameters(), lr=self.config.lr, foreach=False)
@@
             self.checkpointer.load_optimizer_state_dict(self.model, self.opt)
+            for _g in self.opt.param_groups:
+                _g["foreach"] = False
+                _g["fused"] = False
```

Backup of the pre-patch trainer is at:

```text
/home/dnola/v18_fix/trainer.py.pre_t71_foreach_patch
```

## Final state
- Hidden-state symlink visibility restored on spark-3 via NasFS.
- One real optimizer step from the step500 checkpoint completed with numeric `train/loss`.
- `scripts/train.py`: no live process after proof.
- GPU compute apps: none after proof.
- NasFS tmux `nasfs_t71` remains mounted so the restored hidden-state path stays visible for the next spark-3 action.
