# T70: ANCHOR-SKIP root cause on spark-3

Status: ANCHOR_SKIP_ROOT_CAUSE_ONLY
Timestamp: 2026-05-29T11:49Z
Host: spark-3 only

## Verdict
The T54 post-resume `[ANCHOR-SKIP][train] No valid anchors` flood is not a checkpoint/optimizer/sampler-offset failure. It is caused by the trainer-visible hidden-state cache being 100% broken symlinks on spark-3: `iq4_v17_consolidated/hidden_states_nas/hs_*.safetensors` points into `/home/dnola/nas_traces/by_hash/...`, but `/home/dnola/nas_traces/by_hash` is absent on spark-3 now. With `--on-missing skip`, every sample in each packed batch is dropped; `create_collate_fn()` substitutes an empty padded sample (`lengths=[0]`, `loss_mask_sum=0`), so DFlash has no legal anchors and the trainer skip patch increments `global_step` without any optimizer step.

## Contention preflight
```text
pgrep -af [s]cripts/train.py: <none>
free -h: 119Gi total, 97Gi free, 114Gi available
nvidia-smi: No running processes found, GPU util 0%
```

## Evidence
T54 resumed correctly from the step500 checkpoint:
```text
Found checkpoint at .../T54_spark3_probe1200.../step_00000500.
Resuming training from checkpoint tag=step_00000500 epoch=0 global_step=500.
```
Then it had zero post-resume train/loss rows and 700/700 train-anchor skips:
```text
[04:14:07] WARNING [ANCHOR-SKIP][train] step=500: No valid anchors were selected for this batch — skipping batch
...
[04:15:16] WARNING [ANCHOR-SKIP][train] step=1199: No valid anchors were selected for this batch — skipping batch
INFO Stopping training early at global_step=1200 due to max_train_steps=1200
```

Hidden-state path audit on spark-3:
```json
{
  "links": 198486,
  "existing_targets": 0,
  "broken_symlinks": 198486,
  "sample_broken": [
    "/home/dnola/iq4_full_run/iq4_v17_consolidated/hidden_states_nas/hs_11757.safetensors -> /home/dnola/nas_traces/by_hash/0f/46/0f46e8e2c6a7605b77c1baca691946c54e3b2fc454cc3899a645361eb4b69823.safetensors"
  ]
}
```
`/home/dnola/nas_traces` exists only as an empty directory on spark-3; there is no mounted/available `by_hash` tree.

Direct dataloader proof, using the exact T54 data arguments (`--data-path /home/dnola/iq4_full_run/iq4_v17_consolidated/prompts`, `--hidden-states-path /home/dnola/iq4_full_run/iq4_v17_consolidated/hidden_states_nas`, `--on-missing skip`, train split 0.9):
```text
dataset_len 178637 batches 17322
{"batch": 0, "lengths": [0], "loss_mask_sum": 0, "input_nonzero": 0}
{"batch": 1, "lengths": [0], "loss_mask_sum": 0, "input_nonzero": 0}
{"batch": 2, "lengths": [0], "loss_mask_sum": 0, "input_nonzero": 0}
{"batch": 3, "lengths": [0], "loss_mask_sum": 0, "input_nonzero": 0}
{"batch": 4, "lengths": [0], "loss_mask_sum": 0, "input_nonzero": 0}
{"batch": 5, "lengths": [0], "loss_mask_sum": 0, "input_nonzero": 0}
{"batch": 6, "lengths": [0], "loss_mask_sum": 0, "input_nonzero": 0}
{"batch": 7, "lengths": [0], "loss_mask_sum": 0, "input_nonzero": 0}
```
This exactly explains `select_anchors(loss_mask, ...) -> no valid anchors` for every batch.

By contrast, the original source run that produced the starting checkpoint did have real optimizer steps from the same recipe before the NAS targets disappeared. Example source log: `/home/dnola/iq4_full_run/logs/c15_iq4v18b_NAS_T3_spark3_val500_20260529T093736Z/train.log` contains train/loss rows through step 500, ending with:
```text
[03:28:30] INFO train/loss=5.219, train/full_acc=0.172, ... global_step=500
```

## Discriminated hypotheses
- resume sampler offset: rejected. Resume state loads `epoch=0 global_step=500`, but the current first eight loader batches are empty before any sampler-offset issue matters.
- first700 local-data slice exhaustion: rejected. The train split has 178637 rows and 17322 packed batches; the rows are present, the hidden-state targets are not.
- `--on-missing skip` + hidden_states path mismatch/missing targets: confirmed.
- anchor coverage filter/model config/mapping mismatch: not the primary T54 blocker; empty `loss_mask` reaches the anchor selector because every hidden-state file is missing.

## Why no one-step optimizer proof was run
A real one-step proof with the locked v17 NAS mapping requires restoring the missing hidden-state targets. Running against the unrelated local pools (`/home/dnola/c15_local_pool_6500` or `/home/dnola/mxfp8_spike/data`) would change the prompt/vocab mapping (`d2t.npy`/`t2d.npy` differ from `iq4_v17_consolidated`) and would not be a valid production proof for this checkpoint. I did not launch a fake/mismatched training step.

## Exact next one-line fix
Restore a real `/home/dnola/nas_traces/by_hash` tree on spark-3 (or rebuild `/home/dnola/iq4_full_run/iq4_v17_consolidated/hidden_states_nas` to point at an actually local copy of the same v17 hidden-state files), then rerun the tiny T54-style resume probe with a coverage gate that checks `Path.exists()` for symlink targets, not just symlink count/state.json.

Recommended preflight gate before any rerun:
```bash
python - <<'PY'
from pathlib import Path
hs=Path('/home/dnola/iq4_full_run/iq4_v17_consolidated/hidden_states_nas')
links=list(hs.glob('hs_*.safetensors'))
ok=sum(1 for p in links if p.exists())
print(f'hidden-state existing targets: {ok}/{len(links)}')
assert ok >= int(0.95*len(links)), 'hidden-state target coverage below 95%; refusing launch'
PY
```

Once that passes, rerun the same bounded proof from the step500 checkpoint; the success criterion is at least one post-resume `train/loss=...` row before any `max_train_steps` stop.
