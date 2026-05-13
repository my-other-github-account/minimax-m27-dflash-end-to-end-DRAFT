#!/usr/bin/env bash
# launch_full_fp8.sh — Production DFlash training run with FP8 (Float8CurrentScaling HYBRID)
# + fused TE LayerNormMLP on a single DGX Spark (GB10, sm_121a).
#
# This is the FP8 sibling of launch_full.sh. See repro/06-fp8-training.md for the
# full recipe rationale, including the two bugs (silent-bf16 trap + split-accumulator
# NaN cliff) that this script's invocation deliberately avoids.
#
# Required env (or edit defaults below):
#   DATA_ROOT     — contains the paired pool (prompts arrow + hidden_states symlink farm)
#   WORKSPACE     — contains repos/speculators
#   MODELS        — contains MiniMax-M2.7 verifier metadata
#   CHECKPOINTS   — directory to write checkpoints (will be created)
#   TE_VENV       — venv with TransformerEngine source-built for sm_121a
#                   (must include dflash_llama on PYTHONPATH; see §6.2 in the doc)
#   VLLM_VENV     — venv with speculators + datasets + transformers
#
# Optional env:
#   PORT          — kept for compatibility with launch_full.sh, NOT used here
#                   (FP8 path bypasses torchrun, see Bug A in repro/06-fp8-training.md)
#   EPOCHS        — training epochs (default 15)
#   LOG_FREQ      — log every N steps (default 5)
#   FP8_RECIPE    — recipe kind (default current_fp8 — the production-stable choice)
#   NUM_LAYERS    — drafter depth (default 6 for the v12 baseline; v11 used 5)
#
# Critical launch-mode notes:
#   - We invoke train.py DIRECTLY (no torchrun). torchrun --nproc-per-node=1 sets
#     RANK/WORLD_SIZE, which routes the trainer through the FSDP branch where the
#     FP8 wrap is silently skipped. Result: a clean-looking bf16 run with the wrong
#     throughput. See repro/06-fp8-training.md §6.1 Bug A.
#   - We set NVTE_FUSED_ATTN=0 to avoid sm_120 fused-attn driver issues; the vanilla
#     attention path is what the validated v12-stable run used.
#   - We disable torch.compile (TORCHDYNAMO_DISABLE=1) to match the v11 baseline —
#     flex_attention + inductor pattern-matcher has known bugs on Qwen3.

set -eo pipefail

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT (paired pool dir, contains prompts/ and hidden_states/)}"
WORKSPACE="${WORKSPACE:?set WORKSPACE (contains repos/speculators)}"
MODELS="${MODELS:?set MODELS (contains the verifier_meta directory)}"
CHECKPOINTS="${CHECKPOINTS:-${WORKSPACE}/checkpoints}"
TE_VENV="${TE_VENV:?set TE_VENV (path to venv with TransformerEngine sm_121a built)}"
VLLM_VENV="${VLLM_VENV:?set VLLM_VENV (path to venv with speculators + datasets)}"

EPOCHS="${EPOCHS:-15}"
LOG_FREQ="${LOG_FREQ:-5}"
FP8_RECIPE="${FP8_RECIPE:-current_fp8}"
NUM_LAYERS="${NUM_LAYERS:-6}"

PAIRED="${DATA_ROOT}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="v12_fp8_${FP8_RECIPE}_${TS}"
SAVE_DIR="${CHECKPOINTS}/${RUN_NAME}"
LOG_DIR="${WORKSPACE}/logs"
LOG="${LOG_DIR}/${RUN_NAME}.log"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "FP8 run: $RUN_NAME"
echo "  paired:        $PAIRED"
echo "  save:          $SAVE_DIR"
echo "  log:           $LOG"
echo "  epochs:        $EPOCHS"
echo "  fp8 recipe:    $FP8_RECIPE"
echo "  num layers:    $NUM_LAYERS"
echo "  launch mode:   direct python (no torchrun — FP8 single-GPU path)"

# Sanity checks
test -d "${PAIRED}/prompts"            || { echo "missing ${PAIRED}/prompts"; exit 1; }
test -d "${PAIRED}/hidden_states"      || { echo "missing ${PAIRED}/hidden_states"; exit 1; }
test -f "${PAIRED}/prompts/d2t.npy"    || { echo "missing ${PAIRED}/prompts/d2t.npy"; exit 1; }
test -f "${PAIRED}/prompts/t2d.npy"    || { echo "missing ${PAIRED}/prompts/t2d.npy"; exit 1; }
test -f "${PAIRED}/prompts/token_freq.pt" || { echo "missing ${PAIRED}/prompts/token_freq.pt"; exit 1; }
test -d "${TE_VENV}"                   || { echo "missing TE_VENV=$TE_VENV"; exit 1; }
test -d "${VLLM_VENV}"                 || { echo "missing VLLM_VENV=$VLLM_VENV"; exit 1; }

# PYTHONPATH layering: TE venv first (wins on `import transformer_engine`),
# vllm venv second (speculators + datasets + transformers + dflash_llama).
PY_VER="$(${TE_VENV}/bin/python -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
export PYTHONPATH="${TE_VENV}/lib/${PY_VER}/site-packages:${VLLM_VENV}/lib/${PY_VER}/site-packages:${PYTHONPATH:-}"
PY="${TE_VENV}/bin/python"

# CUDA / TE knobs
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export NVTE_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Defensive: ensure no leftover distributed env vars from a parent shell route us
# into the FSDP branch (Bug A from repro/06-fp8-training.md).
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID 2>/dev/null || true

cd "${WORKSPACE}/repos/speculators"

# Direct python launch — NOT torchrun. See header comment.
"$PY" \
    scripts/train.py \
    --speculator-type        dflash \
    --verifier-name-or-path  "${MODELS}/verifier_meta" \
    --data-path              "${PAIRED}/prompts" \
    --hidden-states-path     "${PAIRED}/hidden_states" \
    --save-path              "$SAVE_DIR" \
    --epochs                 "$EPOCHS" \
    --total-seq-len          2048 \
    --max-anchors            1024 \
    --num-workers            1 --prefetch-factor 1 \
    --on-missing             skip \
    --target-layer-ids       2 16 30 45 59 \
    --draft-arch             qwen3 \
    --draft-hidden-act       silu \
    --mask-token-id          200054 \
    --block-size             8 \
    --hidden-states-dtype    bfloat16 \
    --num-layers             "$NUM_LAYERS" \
    --draft-vocab-size       32768 \
    --lr                     3e-4 \
    --scheduler-warmup-steps 100 \
    --noise-std              0.05 \
    --save-best \
    --log-freq               "$LOG_FREQ" \
    --val-every-steps        145 \
    --val-in-epoch-max-batches 80 \
    --save-every-n-vals      1 \
    --fp8-recipe-kind        "$FP8_RECIPE" \
    --te-use-fused \
  2>&1 | tee "$LOG"

echo "FP8 run done: $RUN_NAME"
echo
echo "Post-launch verification (run within 60s of launch):"
echo "  grep '\\[FP8\\]' $LOG | head -3      # must show TE_VERSION + recipe + te_layernorm_mlp count"
echo "  grep -oE 'use_split_accumulator=(True|False)' $LOG | sort -u   # must all be True"
echo "  grep -c 'train/loss=nan' $LOG       # must be 0"
echo "  grep -c 'NaN-SKIP' $LOG             # must be 0 in steady state"
