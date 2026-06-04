#!/usr/bin/env bash
set -euo pipefail

# Directly derived from spark-3:/home/<USER>/t_77088062/server_cmd_revert.sh.
# Override paths if your local checkout/artifacts live elsewhere.
LUCEBOX_SERVER_DIR="${LUCEBOX_SERVER_DIR:-/home/<USER>/lucebox-latest-20260603/server}"
TARGET_GGUF="${TARGET_GGUF:-/home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf}"
DRAFTER="${DRAFTER:-/home/<USER>/t_5567a42d/golden_step_00020000/model_lucebox_layout.safetensors}"
PORT="${PORT:-8097}"
HOST="${HOST:-127.0.0.1}"
MODEL_NAME="${MODEL_NAME:-minimax-luce-golden-revert}"

cd "$LUCEBOX_SERVER_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DFLASH_PERPOS_PROBE="${DFLASH_PERPOS_PROBE:-1}"
export DFLASH_DRAFTATTN_PROBE="${DFLASH_DRAFTATTN_PROBE:-1}"

exec stdbuf -oL -eL ./build-sm121/dflash_server "$TARGET_GGUF" \
  --draft "$DRAFTER" \
  --port "$PORT" \
  --host "$HOST" \
  --max-ctx 2048 \
  --default-max-tokens 128 \
  --hard-limit-reply-budget 0 \
  --model-name "$MODEL_NAME"
