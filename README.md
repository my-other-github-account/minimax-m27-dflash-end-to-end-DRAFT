# DFlash Hidden-State Extraction Bug Fixes for vLLM + Speculators, plus llama.cpp PR #22105 MiniMax-M2 Port

**Issue:** When generating offline hidden-state caches for DFlash/EAGLE3 drafter training using vLLM's `extract_hidden_states` speculative-decoding method on large MoE models with deep layers (e.g. MiniMax-M2.7 in NVFP4 quantization, 62 layers), **deep-layer hidden states (target_layer ≥ 60 + verifier_last_hidden) are corrupted with NaN / Inf** after the first ~5,000 samples. The shallow layers (e.g. target_layer=2, 31) remain clean.

**Symptom:** Dataset built from `data_generation_offline.py` is unusable for training — drafter loss is NaN from step 0. Filtering produces only ~10% usable samples. Subsequent verifier runs reproduce the corruption deterministically.

**Plus:** Once a clean drafter has been trained, deploying it for actual inference speedup against MiniMax-M2 family models requires a tested upstream path. **PR #22105 (`ruixiang63/llama.cpp@dflash`)** provides the only generic, model-agnostic DFlash speculative-decoding implementation in mainline `llama-cli` / `llama-server`. To target MiniMax-M2.x, three small additional patches are required (this bundle).

**Root cause:** Two compounding bugs in vLLM:
1. `_maybe_add_hidden_state` in `vllm/model_executor/models/interfaces.py` does `hidden_states + residual` in bf16. At deep layers of a quantized MoE, residual stream magnitudes routinely reach 65,536 (the NVFP4 saturation ceiling), and bf16 sum overflows to Inf/NaN.
2. `ExtractHiddenStatesProposer` in `vllm/v1/spec_decode/extract_hidden_states.py` uses a persistent `self.hidden_states` buffer allocated once with `torch.zeros`. When the buffer is partially overwritten by a smaller request and then read back via padded `slot_mapping`, stale values from a prior NaN-poisoned request are read, propagating corruption.

**Together**, once any single request produces a NaN at depth 60+, the persistent buffer poisons every subsequent request that aliases to the same KV-cache slot. This explains the "first 5K clean, then 100% NaN forever" pattern observed before the fix.

## Verification

Deep audit of 150 random samples produced *after* the fix on MiniMax-M2.7-NVFP4-GB10:
- **0 NaN / 0 Inf** across all 4 hidden-state layers (target=2, 31, 60, 62)
- Layer std distributions monotonic with depth (0.6, 4.6, 6.7, 15.5) — no dead layers
- Pool grew **+1,525 clean samples in 26 minutes** (~58/min throughput, TP=2 on 2× DGX Spark GB10)
- Streak monitor at scan #51: 1,525 consecutive clean, 0 NaN. Continuing to climb.

## Bundled Patches

This bundle contains 6 patches across 2 upstream repos:

### vLLM (`vllm-project/vllm`)

| File | Patch | Issue |
|---|---|---|
| `vllm/model_executor/models/interfaces.py` | `patches/vllm/01-interfaces-aux-overflow-fix.patch` | bf16 overflow in `_maybe_add_hidden_state` at deep layers; fp32 sum + NaN-clamp guard |
| `vllm/v1/spec_decode/extract_hidden_states.py` | `patches/vllm/02-extract-hidden-states-buffer-zero.patch` | Persistent `self.hidden_states` buffer not zeroed between requests, causing stale-NaN propagation |

### Speculators (`vllm-project/speculators`)

| File | Patch | Issue |
|---|---|---|
| `src/speculators/models/eagle3/core.py` | `patches/speculators/01-eagle3-core-dtype-fixes.patch` | (R28+R29) `fc` Linear missing `dtype=bfloat16`, plus norm output dtype mismatch with `verifier_lm_head` |
| `scripts/train.py` | `patches/speculators/02-train-script-dtype-cast.patch` | (R27) `draft_model.to(hidden_states_dtype)` cast required before training; comment about not forcing eager attention |
| `src/speculators/train/data.py` | `patches/speculators/03-data-empty-sample-dtypes.patch` | (R30) Empty sample tensors missing `dtype=long`/`dtype=bool`, causing `torch.cat` in collate to upcast all real samples to float, crashing `embed_tokens` |
| `src/speculators/train/trainer.py` | `patches/speculators/04-trainer-nan-guard-and-midepoch-ckpt.patch` | (R31+R32) NaN-loss skip guard (don't backprop NaN gradients), plus opt-in mid-epoch checkpointing via `MIDEPOCH_CHECKPOINT_FREQ` env var |

### llama.cpp (`my-other-github-account/llama.cpp@dflash-minimax-m2`, on top of PR #22105 base)

PR #22105 (`ruixiang63/llama.cpp@dflash`) is the upstream generic DFlash speculative-decoding implementation for `llama-cli` / `llama-server`. It is model-agnostic via a `cb()`-callback pattern that captures hidden states at configured `target_layer_ids`. To make a MiniMax-M2 family model usable as a DFlash *target*, AND to load drafters with non-5 target-layer counts (the PR's reference drafter uses 5; ours uses 3), three patches are needed:

| File | Patch | Issue |
|---|---|---|
| `src/models/minimax-m2.cpp` | `patches/llama.cpp/01-minimax-m2-cb-hooks.patch` | Add the `cb()` injection hooks for `eagle3_extract_<idx>` and `dflash_extract_<idx>` at each `target_layer_ids[i]`, mirroring the pattern in `qwen3.cpp`. ~25 LOC. |
| `src/llama-hparams.h`, `src/llama-model.cpp`, `src/llama-context.cpp`, `src/llama-model-loader.cpp`, `src/models/dflash.cpp` | `patches/llama.cpp/02-variable-length-target-layer-ids.patch` | PR #22105 hardcodes `dflash_target_layer_ids` as `std::array<int, 5>` matching its reference drafter (z-lab/Qwen3-8B-DFlash-b16). Switches to `std::array<int, 16>` + a separate `n_dflash_target_layer_ids` count, preserving `llama_hparams`'s `is_trivially_copyable` static_assert (which forbids `std::vector`). Lets PR load any DFlash drafter with 1..16 target layers without re-training. |
| `convert_hf_to_gguf.py` | `patches/llama.cpp/03-converter-drop-drafter-only-tensors.patch` | The PR's `DFlashModel.modify_tensors()` does not drop drafter-only tensors (`d2t`, `t2d`, `lm_head.weight`, `embed_tokens.weight`) which are not in the `MODEL_ARCH.DFLASH` whitelist (PR uses target's embeddings + lm_head at runtime). Without this drop, conversion fails with `Can not map tensor 'model.d2t'`. |

## Applying

### From a fresh vLLM nightly + speculators main installation:

```bash
# Determine your installed paths
VLLM_DIR=$(python3 -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")
SPEC_DIR=$(python3 -c "import speculators; import os; print(os.path.dirname(speculators.__file__))")

# vLLM patches
patch -p1 -d "$VLLM_DIR/.." < patches/vllm/01-interfaces-aux-overflow-fix.patch
patch -p1 -d "$VLLM_DIR/.." < patches/vllm/02-extract-hidden-states-buffer-zero.patch

# Speculators patches (repo or installed package)
# If you have the speculators repo cloned, apply with `git apply`:
cd /path/to/speculators && git apply /path/to/patches/speculators/01-eagle3-core-dtype-fixes.patch
cd /path/to/speculators && git apply /path/to/patches/speculators/02-train-script-dtype-cast.patch
# For installed-package patches (data.py, trainer.py), use plain `patch`:
patch -p2 -d "$SPEC_DIR/.." < patches/speculators/03-data-empty-sample-dtypes.patch
patch -p2 -d "$SPEC_DIR/.." < patches/speculators/04-trainer-nan-guard-and-midepoch-ckpt.patch
```

### Building the patched llama.cpp:

```bash
# 1. Clone PR #22105's base branch
git clone -b dflash https://github.com/ruixiang63/llama.cpp llama.cpp-pr22105
cd llama.cpp-pr22105
git checkout 67cb0d507  # PR #22105 tip as of late April 2026

# 2. Apply our 3 MiniMax + variable-length + converter patches
git am /path/to/patches/llama.cpp/01-minimax-m2-cb-hooks.patch
git am /path/to/patches/llama.cpp/02-variable-length-target-layer-ids.patch
git apply /path/to/patches/llama.cpp/03-converter-drop-drafter-only-tensors.patch

# 3. Build (CUDA 13 + Blackwell SM 12.1a on DGX Spark GB10 example)
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="121-real" \
  -DGGML_NATIVE=ON
cmake --build build --target llama-server llama-cli llama-quantize -j

# 4. Convert your speculators-trained drafter to PR-compatible GGUF
python repro/prep_for_pr22105_converter.py \
  --in  /path/to/speculators-checkpoint_best \
  --out /tmp/drafter-prepped
python convert_hf_to_gguf.py /tmp/drafter-prepped \
  --outfile /path/to/drafter.gguf --outtype f16 \
  --target-model-dir /path/to/target-model-with-tokenizer

# 5. Run llama-server with DFlash speculative decoding
./build/bin/llama-server \
  --model /path/to/MiniMax-M2.7-UD-IQ4_XS.gguf \
  --model-draft /path/to/drafter.gguf \
  --dflash \
  --draft-max 7 \
  --port 8011
```

## Tested Environment

- **Hardware:** DGX Spark GB10 × 2, QSFP 200 Gbps interconnect, MTU 9000
- **CUDA:** 13.x (Blackwell SM 12.1a)
- **vLLM:** 0.20.1rc1.dev23+gde3da0b97 (nightly main, late April 2026)
- **Speculators:** main @ 67bafe6 ("Dflash verifier targets" PR #477)
- **llama.cpp base:** PR #22105 tip @ 67cb0d507 (`ruixiang63/llama.cpp@dflash`, "dflash: enable llama-cli & llama-server with np=1")
- **llama.cpp fork (this bundle):** `my-other-github-account/llama.cpp@dflash-minimax-m2` @ 2c32f36fc
- **Model:** MiniMax-M2.7 in NVFP4 quantization (`MiniMax-M2.7-NVFP4-GB10`, 62 layers, MoE, 256 experts) for hidden-state extraction; MiniMax-M2.7 GGUF (UD-IQ4_XS quant) for llama.cpp inference
- **Topology:** TP=2 (rank 0 + rank 1) across 2 nodes, no Ray, plain TCP NCCL over QSFP

## Reproduction

See [`repro/REPRODUCE.md`](repro/REPRODUCE.md) for end-to-end setup, vLLM launch, data-gen, validation, and training commands.

## Files in this bundle

```
patches/
  vllm/
    01-interfaces-aux-overflow-fix.patch
    02-extract-hidden-states-buffer-zero.patch
  speculators/
    01-eagle3-core-dtype-fixes.patch
    02-train-script-dtype-cast.patch
    03-data-empty-sample-dtypes.patch
    04-trainer-nan-guard-and-midepoch-ckpt.patch
  llama.cpp/
    01-minimax-m2-cb-hooks.patch                      # add cb() hooks for hidden-state capture
    02-variable-length-target-layer-ids.patch         # accept drafters with !=5 target layers
    03-converter-drop-drafter-only-tensors.patch      # let converter handle d2t/t2d/lm_head/embed
repro/
  REPRODUCE.md                          # full end-to-end reproduction instructions
  vllm_tp2_clean.sh                     # verifier launch script (auto-rank from NIC IP)
  data_gen.sh                           # data-gen client (feeds verifier)
  validator_daemon.py                   # NaN-validator (staging → clean_pool / quarantine)
  deep_audit.py                         # external audit script for confirming pool integrity
  prep_for_pr22105_converter.py         # transform speculators ckpt -> PR #22105 converter input
README.md                               # this file
```
