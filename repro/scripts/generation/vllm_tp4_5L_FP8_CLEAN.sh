#!/usr/bin/env bash
# CLEAN FP8 TP=4 launch across node1/2/3/4.
# Rank map: node2=head/API rank0; node1=rank1; node3=rank2; node4=rank3.
set -euo pipefail

ETH_IF=<HIGH_BW_NIC>
NODE_IP=$(ip -4 -o addr show "$ETH_IF" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
HEAD_IP=<HEAD_IP>
MASTER_PORT=29504
MODEL=${MODELS}/MiniMax-M2.7-FP8
HS_PATH=${DATA_ROOT}/preprocessed_5L_FP8/hs_staging
LOG_DIR=${WORKSPACE}/logs
LOG="$LOG_DIR/vllm-fp8-tp4-clean-$(hostname)-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$HS_PATH" "$LOG_DIR"

export VLLM_HOST_IP="$NODE_IP"
export GLOO_SOCKET_IFNAME="$ETH_IF"
export NCCL_SOCKET_IFNAME="$ETH_IF"
export NCCL_IB_DISABLE=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export FLASHINFER_CUDA_ARCH_LIST=12.1a
export TORCH_CUDA_ARCH_LIST=12.1a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export HF_HUB_OFFLINE=1
export DFLASH_TP4_FP8=1

source ${VENV}/bin/activate

echo "[$(date)] host=$(hostname) NODE_IP=$NODE_IP HEAD_IP=$HEAD_IP" | tee -a "$LOG"
python3 - <<'PY' | tee -a "$LOG"
import vllm, torch
from vllm.config import SpeculativeConfig
print('vllm', vllm.__version__, 'torch', torch.__version__, 'deep_import_ok')
PY

if [ ! -f "$MODEL/config.json" ] || [ ! -f "$MODEL/model.safetensors.index.json" ]; then
  echo "[$(date)] MISSING_MODEL_FILES at $MODEL" | tee -a "$LOG"
  exit 2
fi

GPU_USERS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
if [ -n "$GPU_USERS" ]; then
  echo "[$(date)] PRE_FLIGHT_ABORT: GPU already has compute users" | tee -a "$LOG"
  echo "$GPU_USERS" | tee -a "$LOG"
  exit 1
fi
EXISTING=$(pgrep -af "(vllm serve|EngineCore|RayWorkerWrapper|launch_vllm\.py)" 2>/dev/null | grep -v pgrep | grep -v "vllm_tp4_5L_FP8_CLEAN" || true)
if [ -n "$EXISTING" ]; then
  echo "[$(date)] PRE_FLIGHT_ABORT: existing vLLM workers" | tee -a "$LOG"
  echo "$EXISTING" | tee -a "$LOG"
  exit 1
fi

case "$(hostname)" in
  node2) NODE_RANK=0; EXTRA_FLAGS=""; ROLE="rank0-head-api" ;;
  node1) NODE_RANK=1; EXTRA_FLAGS="--headless"; ROLE="rank1-worker" ;;
  node3) NODE_RANK=2; EXTRA_FLAGS="--headless"; ROLE="rank2-worker" ;;
  node4) NODE_RANK=3; EXTRA_FLAGS="--headless"; ROLE="rank3-worker" ;;
  *) echo "Unknown hostname $(hostname)" | tee -a "$LOG"; exit 2 ;;
esac

echo "[$(date)] Role: $ROLE rank=$NODE_RANK hs_path=$HS_PATH" | tee -a "$LOG"
cd ${WORKSPACE}/repos/speculators

exec python3 scripts/launch_vllm.py "$MODEL" \
  --hidden-states-path "$HS_PATH" \
  --target-layer-ids 2 16 30 45 59 \
  -- \
  --tensor-parallel-size 4 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --no-enable-flashinfer-autotune \
  --no-enable-chunked-prefill \
  --max-num-batched-tokens 2048 \
  --kv-cache-dtype auto \
  --max-num-seqs 1 \
  --load-format fastsafetensors \
  --trust-remote-code \
  --port 8000 --host 0.0.0.0 \
  --nnodes 4 --node-rank "$NODE_RANK" \
  --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" \
  $EXTRA_FLAGS \
  >> "$LOG" 2>&1
