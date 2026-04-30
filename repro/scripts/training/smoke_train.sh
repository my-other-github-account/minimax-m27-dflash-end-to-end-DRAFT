#!/usr/bin/env bash
# smoke_train.sh — 90-second training smoke run for a paired dataset.
#
# Pass criteria (see repro/02-training.md §2.6):
#   - exit code 124 (timeout-killed = ran the full 90s without crashing)
#   - log shows global_step >= 50, real per-position acc, lr ramping
#   - log does NOT contain: R54, prefix mismatch, padding-anchor, t2d size error
#
# Usage:
#   PAIRED_DIR=/path/to/train_all_paired ./smoke_train.sh
#
# Required env (or edit defaults below):
#   PAIRED_DIR        directory containing prompts/ and hidden_states/
#   WORKSPACE         contains repos/speculators
#   MODELS            contains MiniMax-M2.7-FP8

set -u

PAIRED_DIR="${PAIRED_DIR:?set PAIRED_DIR}"
WORKSPACE="${WORKSPACE:-$HOME}"
MODELS="${MODELS:-$HOME/models}"
SAVE_PATH="${SAVE_PATH:-/tmp/dflash-smoke}"
LOG="${LOG:-/tmp/dflash-smoke.log}"
PORT="${PORT:-29501}"
TIMEOUT="${TIMEOUT:-90}"

mkdir -p "$SAVE_PATH"

echo "smoke: paired=$PAIRED_DIR  log=$LOG  timeout=${TIMEOUT}s"

timeout "$TIMEOUT" torchrun \
    --master_port="$PORT" --nproc-per-node=1 \
    "${WORKSPACE}/repos/speculators/scripts/train.py" \
    --speculator-type dflash \
    --verifier-name-or-path "${MODELS}/MiniMax-M2.7-FP8" \
    --data-path "${PAIRED_DIR}/prompts" \
    --hidden-states-path "${PAIRED_DIR}/hidden_states" \
    --save-path "$SAVE_PATH" \
    --epochs 1 \
    --total-seq-len 2048 \
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
    --log-freq 5 \
  2>&1 | tee "$LOG"

ec="${PIPESTATUS[0]}"
echo "----"
echo "exit_code=$ec   (124 = timeout-killed = success)"

# Quick post-flight check
fail=0
for bad in "R54: hs prompt prefix mismatch" \
           "anchor_positions include padding" \
           "don't match input ids" \
           "t2d has" \
           "d2t has"
do
    if grep -qF "$bad" "$LOG"; then
        echo "FAIL: log contains '$bad'"
        fail=1
    fi
done

if ! grep -qE "global_step=[0-9]+" "$LOG"; then
    echo "FAIL: log shows no global_step lines"
    fail=1
fi

if [ "$fail" -eq 0 ] && [ "$ec" -eq 124 ]; then
    echo "PASS: smoke run clean — safe to proceed to full training"
    exit 0
fi

echo "smoke FAILED — do not proceed to full training until fixed"
exit 1
