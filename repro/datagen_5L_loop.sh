#!/bin/bash
# 5-layer DFlash data-gen client — drives prompts at the vLLM endpoint
# Uses Trap 13 wrapper (HS_POOL_DIR / HS_QUARANTINE_DIR) to skip already-processed indices.
# Endless loop: when --max-samples is exhausted, restarts to re-attempt failed indices and
# (if dataset exhausted) waits for prompt expansion.
set -euo pipefail

LOG=/home/user/dflash_minimax/logs/datagen-5L-$(date +%Y%m%d-%H%M%S).log
mkdir -p "$(dirname $LOG)"

source /home/user/venvs/vllm/bin/activate

# Wait for vLLM endpoint up to 15min
echo "[$(date)] Waiting for /v1/models on 127.0.0.1:8000..." | tee -a "$LOG"
for i in $(seq 1 90); do
  if curl -sf --max-time 5 http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
    echo "[$(date)] endpoint UP after ${i}×10s polls" | tee -a "$LOG"
    break
  fi
  if [ $i -eq 90 ]; then
    echo "[$(date)] FAIL: endpoint not up after 15min" | tee -a "$LOG"
    exit 1
  fi
  sleep 10
done

# Also wait for hidden_states extraction service to be ready (smoketest 1 sample)
sleep 20

cd /home/user/dflash_minimax/repos/speculators

# Loop: keep driving until killed
ITER=0
while true; do
  ITER=$((ITER+1))
  echo "[$(date)] iteration $ITER" | tee -a "$LOG"
  HS_POOL_DIR=/home/user/dflash_minimax/data/preprocessed_5L/hs_clean_pool \
  HS_QUARANTINE_DIR=/home/user/dflash_minimax/data/preprocessed_5L/hs_quarantine \
  HS_REDO_QUARANTINE=1 \
  python3 /home/user/dflash_minimax/scripts/datagen_skip_existing_wrapper.py \
      --model /home/user/models/MiniMax-M2.7-NVFP4-GB10 \
      --endpoint http://127.0.0.1:8000/v1 \
      --preprocessed-data /home/user/dflash_minimax/data/preprocessed \
      --output /home/user/dflash_minimax/data/preprocessed_5L/hs_staging \
      --max-samples 12000 \
      --concurrency 2 \
      --max-consecutive-errors 50 2>&1 | tee -a "$LOG" || \
      echo "[$(date)] iteration $ITER exited (likely dataset complete or transient err)" | tee -a "$LOG"
  echo "[$(date)] sleep 60s before next iteration" | tee -a "$LOG"
  sleep 60
done
