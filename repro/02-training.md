# Section 2 — Training the DFlash Drafter

> **⚠️ DRAFT — STUB.** This section is intentionally light pending a focused pass on training reproducibility. The Generation section (§1) is fleshed out; Training and Inference contain pointers + last-known-good config so the 3-section structure is intact.

End-to-end reproduction of the DFlash drafter checkpoint that pairs with `MiniMax-M2.7-FP8` as the verifier. The output is a PyTorch checkpoint convertible to GGUF for `llama.cpp` inference.

## 2.1 Inputs

- Hidden-state pool from §1: `${DATA_ROOT}/preprocessed_5L_FP8/hs_clean_pool/` (≥ 1000 files recommended, >2500 better)
- Verifier model: `${MODELS}/MiniMax-M2.7-FP8` (same path as §1)
- Speculators repo: `${WORKSPACE}/repos/speculators` @ `67bafe6`

## 2.2 Last-known-good training config

The drafter `MiniMax-M2.7-DFlash.gguf` (MD5 `785c5b5a6bcf8eecb545a1bebb75eb4e`) currently in production was trained with:

```bash
torchrun --master_port=29501 --nproc-per-node=1 \
    ${WORKSPACE}/repos/speculators/scripts/train.py \
    --speculator-type dflash \
    --verifier-name-or-path ${MODELS}/MiniMax-M2.7-FP8 \
    --data-path <prompt-source-preprocessed-dir> \
    --hidden-states-path ${DATA_ROOT}/preprocessed_5L_FP8/hs_clean_pool \
    --save-path ./checkpoints/dflash-drafter \
    --epochs 17 \
    --total-seq-len 4096 \
    --max-anchors 64 \
    --num-workers 1 --prefetch-factor 2 \
    --on-missing skip \
    --target-layer-ids 2 16 30 45 59 \
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

Critical details:
- `--num-layers 5` for the DFlash adapter (5-layer drafter, distinct from the 6 layer taps in the trace data — the 6th tap is the verifier last hidden, used as the verifier-side input to the cross-attention)
- `--target-layer-ids` must match `[2, 16, 30, 45, 59]` from §1 (the auto-appended last-hidden 62 is not passed here)
- `--block-size 8` (DFlash chunk length)
- `--mask-token-id 200054` (MiniMax-M2.7 mask token)
- `--draft-vocab-size 32768` — drafter operates on a reduced vocab via `d2t` map (drafter_token → target_token)
- `--hidden-states-dtype bfloat16` — matches what §1 produced

## 2.3 Last-known-good metrics

From `val/full_acc_epoch=0.143` of the verified production run (2026-04-29 21:16 PT):
- pos 1: 0.218
- pos 2: 0.199
- pos 3: 0.131
- pos 7: 0.116
- prefix ≥ pos 1: 0.218
- prefix ≥ pos 2: 0.009 (chain-gated collapses fast — see §3 on metric framing)

## 2.4 Patches required

The original NaN-bug patches in `patches/speculators/` (R27/R28/R29/R30/R31/R32) are still required for training stability **on training-side dtype/NaN issues** even though the vLLM-side patches were reverted. Apply them per the top-level README's instructions before training. These remain valid — they fix bugs in the trainer itself, not in the trace pipeline.

## 2.5 GGUF conversion

After training, convert the `checkpoint_best/` to GGUF for `llama.cpp` inference. See `repro/legacy/prep_for_pr22105_converter.py` for the speculators-checkpoint → PR-#22105-converter-input transformation, and the top-level README §"Building the patched llama.cpp" for the converter invocation.

## 2.6 What this section needs (pending pass)

- Capture and embed the actual training log (loss curve + val-acc-per-epoch evidence)
- Document the GGUF conversion step inline rather than by pointer
- Capture the exact prompt-source split used for training data vs. trace generation (currently both sides used the same `combined_48k` + `mixed_*` pool)
- Add a sanity-check script that verifies a fresh checkpoint matches the published per-position accuracies within tolerance

Continue to [Section 3 — Inference](03-inference.md), or back to [Section 1 — Generation](01-generation.md).
