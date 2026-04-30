# Reproduction Guide — DFlash NaN Cache Bug

End-to-end repro of the verifier-side bug, validation that the patches fix it, and how to use the resulting clean cache for downstream training.

## Hardware Assumed

- 2 GB10-class GPU nodes connected by a fast point-to-point link (we used QSFP 200 Gbps, MTU 9000). Single GPU per node.
- Reachable as `node-rank0` (head) and `node-rank1` (worker) over a private subnet (we used <NODE2_QSFP_IP> + <NODE3_QSFP_IP> on `<QSFP_NIC>`).
- Both nodes share an installed Python 3.12 venv with vLLM nightly + speculators (see versions above).
- The verifier model file (~120 GB for MiniMax-M2.7-NVFP4) at `$MODEL_PATH` on **both** nodes.
- A preprocessed dataset at `$DATA_PATH` on the rank-0 node (we used `combined_48k`, ~48k tokenized chat samples).

## 1. Confirm the bug exists (without patches)

On rank 0 only:
```bash
# Launch verifier (TP=2, no-Ray, plain TCP NCCL)
bash repro/vllm_tp2_clean.sh
# (also run on rank 1 simultaneously — script auto-detects rank by NIC IP)

# Wait ~3 min for /v1/models endpoint to come up
curl http://127.0.0.1:8000/v1/models

# Launch data-gen client (will write to ./hs_staging/)
bash repro/data_gen.sh
```

After ~30 minutes (>5,000 generated samples) check:
```python
import os
from safetensors.torch import load_file
D = "./hs_staging"
files = sorted(os.listdir(D), key=lambda f: os.path.getmtime(os.path.join(D, f)))
# Pre-bug: first 4000 are clean. Post-bug: every file from ~5000 onward has NaN on layer indices 2 & 3.
nan_per_layer = [0,0,0,0]
for f in files[-100:]:
    hs = load_file(os.path.join(D, f))['hidden_states']
    for l in range(4):
        if hs[:, l, :].isnan().any().item():
            nan_per_layer[l] += 1
print(nan_per_layer)  # Expect: [0, 0, ~100, ~100] = bug present
```

## 2. Apply patches

```bash
VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
SPEC_DIR=$(python3 -c "import speculators, os; print(os.path.dirname(speculators.__file__))")

# vLLM
cd "$VLLM_DIR/.." && patch -p1 < /path/to/patches/vllm/01-interfaces-aux-overflow-fix.patch
cd "$VLLM_DIR/.." && patch -p1 < /path/to/patches/vllm/02-extract-hidden-states-buffer-zero.patch

# Speculators (installed-package edits — use the data.py and trainer.py installed in the venv)
cd "$SPEC_DIR/.." && patch -p2 < /path/to/patches/speculators/03-data-empty-sample-dtypes.patch
cd "$SPEC_DIR/.." && patch -p2 < /path/to/patches/speculators/04-trainer-nan-guard-and-midepoch-ckpt.patch
# The eagle3/core.py and train.py patches apply to a checked-out speculators repo:
cd /path/to/speculators-repo && git apply /path/to/patches/speculators/01-eagle3-core-dtype-fixes.patch
cd /path/to/speculators-repo && git apply /path/to/patches/speculators/02-train-script-dtype-cast.patch
```

Verify all 6 patches landed:
```bash
grep -c 'R33' "$VLLM_DIR/model_executor/models/interfaces.py" "$VLLM_DIR/v1/spec_decode/extract_hidden_states.py"
grep -c 'DFLASH_R' "$SPEC_DIR/train/data.py" "$SPEC_DIR/train/trainer.py"
# Expect: 3, 1, 1, 2 respectively
```

## 3. Restart verifier and validate fix

Stop verifier on both ranks, kill any data-gen, **delete the staging dir** (otherwise the new run will keep producing into the old buffer state):
```bash
# both ranks
systemctl --user stop vllm-tp2-verifier 2>/dev/null
pkill -f vllm.entrypoints
rm -rf ./hs_staging/* ./hs_clean_pool/* ./hs_quarantine/*
```

Relaunch with same scripts as Step 1. Start the validator daemon:
```bash
python3 repro/validator_daemon.py &
```

Let it run for **~30 min** and accumulate ~1,500 samples. Then audit:
```bash
python3 repro/deep_audit.py
```

Expected output (post-fix):
```
DEEP AUDIT: examining 150 random recent samples
Problems found: 0 / 150
Per-layer std stats:
  L0 target=2 (shallow): mean_std=~0.6
  L1 target=31 (mid):    mean_std=~4-9
  L2 target=60 (deep):   mean_std=~7-12   <- previously 100% NaN
  L3 target=62 (last):   mean_std=~13-30  <- previously 100% NaN
```

A "**hard-gate test**" used in the original investigation: monitor the validator log
and require **2,500 consecutive clean samples** with zero NaN before declaring the
fix successful. Even one NaN in any layer of any file resets the counter.

## 4. Use the cache for DFlash drafter training

```bash
torchrun --master_port=29501 --nproc-per-node=1 train.py \
    --speculator-type dflash \
    --verifier-name-or-path "$MODEL_PATH" \
    --data-path /path/to/preprocessed_dataset \
    --hidden-states-path ./hs_clean_pool \
    --save-path ./checkpoints/dflash-drafter \
    --epochs 15 \
    --total-seq-len 4096 \
    --max-anchors 64 \
    --num-workers 1 --prefetch-factor 2 \
    --on-missing skip \
    --target-layer-ids 2 31 60 \
    --draft-arch qwen3 \
    --draft-hidden-act silu \
    --mask-token-id 200054 \
    --block-size 8 \
    --hidden-states-dtype bfloat16 \
    --num-layers 5 \
    --draft-vocab-size 32768 \
    --save-best \
    --log-freq 5
```

Expected: monotonic loss decrease, val_acc rising from ~0.04 to >0.10 within 15 epochs on a small (~2.5k sample) dataset.

## Pitfalls

- **Don't use TP=1 as a "workaround".** TP=2 is the supported topology on multi-Spark; TP=1 has its own KV-extraction quirks that are not addressed here.
- **Do NOT use Ray** for distributed-executor-backend on this combo. Use `--nnodes 2 --node-rank N --master-addr ... --master-port ...` (torchrun-style multi-node) instead. Ray-based launches were observed to deadlock during EngineCore init on this topology.
- **Don't set `TORCHDYNAMO_DISABLE=1`** in the trainer if using DFlash — it forces flex_attention into an unfused materialize-`[seq, seq]`-scores fallback that wedges the GPU.
- **`--enable-chunked-prefill False`** is recommended (we used it) but does NOT fix the bug on its own — overflow happens regardless of chunked-prefill. The patches are required.
- After applying patches, you can remove the conservative `--enable-chunked-prefill False` flag if performance matters; the bug should not recur.
- The `--target-layer-ids` arg passed to `launch_vllm.py` auto-appends `num_hidden_layers` (the "last layer") — so for MiniMax-M2.7 (62 layers), passing `--target-layer-ids 2 31 60` results in vLLM extracting layers `[2, 31, 60, 62]`. Make sure your training script uses the same 4 layer IDs.
