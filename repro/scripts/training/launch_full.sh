#!/usr/bin/env bash
# launch_full.sh — Production DFlash 5L training run on a single Spark.
#
# Hyperparameters match the production drafter (MiniMax-M2.7-DFlash.gguf,
# md5 785c5b5a6bcf8eecb545a1bebb75eb4e). Designed to be invoked under
# `systemd-run --user --unit=dflash-full --collect bash launch_full.sh`
# so it survives any SSH session disconnect.
#
# Required env (or edit defaults below):
#   DATA_ROOT     — contains preprocessed_5L_FP8/train_all_paired
#   WORKSPACE     — contains repos/speculators and venvs/vllm
#   MODELS        — contains MiniMax-M2.7-FP8
#   CHECKPOINTS   — directory to write checkpoints (will be created)
#
# Optional env:
#   PORT          — torchrun master_port (default 29502)
#   EPOCHS        — training epochs (default 17)
#   LOG_FREQ      — log every N steps (default 5)

set -eo pipefail

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
WORKSPACE="${WORKSPACE:?set WORKSPACE}"
MODELS="${MODELS:?set MODELS}"
CHECKPOINTS="${CHECKPOINTS:-${WORKSPACE}/dflash_minimax/checkpoints}"
PORT="${PORT:-29502}"
EPOCHS="${EPOCHS:-17}"
LOG_FREQ="${LOG_FREQ:-5}"

PAIRED="${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="full_5L_paired_${TS}"
SAVE_DIR="${CHECKPOINTS}/${RUN_NAME}"
LOG_DIR="${WORKSPACE}/dflash_minimax/logs"
LOG="${LOG_DIR}/${RUN_NAME}.log"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "FULL run: $RUN_NAME"
echo "  paired: $PAIRED"
echo "  save:   $SAVE_DIR"
echo "  log:    $LOG"
echo "  epochs: $EPOCHS"

# Sanity-check the paired dataset is shaped correctly
test -d "${PAIRED}/prompts"        || { echo "missing ${PAIRED}/prompts";        exit 1; }
test -d "${PAIRED}/hidden_states"  || { echo "missing ${PAIRED}/hidden_states";  exit 1; }
test -f "${PAIRED}/prompts/d2t.npy" || { echo "missing ${PAIRED}/prompts/d2t.npy — run build_vocab_maps.py first"; exit 1; }
test -f "${PAIRED}/prompts/t2d.npy" || { echo "missing ${PAIRED}/prompts/t2d.npy — run build_vocab_maps.py first"; exit 1; }
test -f "${PAIRED}/prompts/token_freq.pt" || { echo "missing ${PAIRED}/prompts/token_freq.pt — run build_vocab_maps.py first"; exit 1; }

# Activate venv
if [ -f "${WORKSPACE}/venvs/vllm/bin/activate" ]; then
    source "${WORKSPACE}/venvs/vllm/bin/activate"
elif [ -f "${WORKSPACE}/dflash_minimax/venv/bin/activate" ]; then
    source "${WORKSPACE}/dflash_minimax/venv/bin/activate"
else
    echo "no venv found — install speculators + torch into ${WORKSPACE}/venvs/vllm" >&2
    exit 1
fi

cd "${WORKSPACE}/repos/speculators"

# Production hyperparameters (matches MiniMax-M2.7-DFlash.gguf reference run)
torchrun --master_port="$PORT" --nproc-per-node=1 \
    scripts/train.py \
    --speculator-type dflash \
    --verifier-name-or-path "${MODELS}/MiniMax-M2.7-FP8" \
    --data-path "${PAIRED}/prompts" \
    --hidden-states-path "${PAIRED}/hidden_states" \
    --save-path "$SAVE_DIR" \
    --epochs "$EPOCHS" \
    --total-seq-len 2048 \
    --max-anchors 512 \
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
    --lr 3e-5 \
    --scheduler-warmup-steps 100 \
    --save-best \
    --log-freq "$LOG_FREQ" \
  2>&1 | tee "$LOG"

echo "FULL done: $RUN_NAME"
