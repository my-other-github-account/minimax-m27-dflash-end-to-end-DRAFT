#!/bin/bash
# Generic verifier launch / data-gen script (from DFlash NaN-fix bundle).
# REQUIRED env vars: MODEL_PATH (path to verifier model on each node).
# OPTIONAL: HEAD_IP, WORKER_IP, ETH_IF, MISSION_DIR, VENV_DIR.
# Auto-detects rank by NIC IP. Run on BOTH nodes simultaneously.

set -euo pipefail
LOG=${MISSION_DIR:-./dflash_minimax}/logs/data-gen-tick33.log
mkdir -p "$(dirname $LOG)"

cd ${MISSION_DIR:-./dflash_minimax}/repos/speculators
source ${VENV_DIR:-./venvs/vllm}/bin/activate
export PATH=${HOME_DIR:-$HOME}/.local/bin:$PATH

echo "[$(date)] TICK33 data-gen starting; waiting for verifier endpoint..." | tee -a "$LOG"

# Poll for verifier readiness (max 10 min)
for i in $(seq 1 60); do
    if curl -sf --max-time 5 http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
        echo "[$(date)] Verifier UP after $i polls (${i}0s)" | tee -a "$LOG"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "[$(date)] FAIL: verifier did not come up in 10 min" | tee -a "$LOG"
        exit 1
    fi
    sleep 10
done

mkdir -p ${MISSION_DIR:-./dflash_minimax}/cache/hs_staging

echo "[$(date)] Launching data_generation_offline.py from combined_48k" | tee -a "$LOG"
exec python3 scripts/data_generation_offline.py \
    --model ${MODEL_PATH} \
    --endpoint http://127.0.0.1:8000/v1 \
    --preprocessed-data ${MISSION_DIR:-./dflash_minimax}/cache/bonus/preprocessed/combined_48k \
    --output ${MISSION_DIR:-./dflash_minimax}/cache/hs_staging \
    --max-samples 48000 \
    --concurrency 4 \
    --max-consecutive-errors 50 >> "$LOG" 2>&1
