#!/bin/bash
# Generic verifier launch / data-gen script (from DFlash NaN-fix bundle).
# REQUIRED env vars: MODEL_PATH (path to verifier model on each node).
# OPTIONAL: HEAD_IP, WORKER_IP, ETH_IF, MISSION_DIR, VENV_DIR.
# Auto-detects rank by NIC IP. Run on BOTH nodes simultaneously.


set -euo pipefail

LOG_DIR=${MISSION_DIR:-./dflash_minimax}/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/vllm-tp2-clean-tick33-$(hostname).log"
exec > >(tee -a "$LOG") 2>&1

echo "=================================================="
echo "TICK 33 TP=2 CLEAN GEN (no-chunked-prefill fix) at $(date -Iseconds) on $(hostname)"
echo "=================================================="

# Pre-flight: refuse to launch if any other vllm/EngineCore is alive
EXISTING=$(pgrep -af "(vllm serve|EngineCore|RayWorkerWrapper)" 2>/dev/null | grep -v pgrep | grep -v rsync | grep -v "DFLASH_TICK_OWNER=tick32" || true)
if [ -n "$EXISTING" ]; then
  echo "[$(date)] PRE_FLIGHT_ABORT: existing vLLM workers:"
  echo "$EXISTING"
  exit 0
fi
GPU_USERS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | head -3 || true)
if [ -n "$GPU_USERS" ]; then
  echo "[$(date)] PRE_FLIGHT_ABORT: GPU compute apps already running:"
  echo "$GPU_USERS"
  exit 0
fi

NODE_IP=$(ip -4 -o addr show ${ETH_IF:-<QSFP_NIC>} | awk '{print $4}' | cut -d/ -f1)
ETH_IF=${ETH_IF:-<QSFP_NIC>}
HEAD_IP=${HEAD_IP:-<NODE2_QSFP_IP>}
MASTER_PORT=29501

# vLLM identity
export VLLM_HOST_IP=$NODE_IP
export DFLASH_TICK_OWNER="tick33-tp2-clean-r33-patched"

# Multi-node NCCL via QSFP (plain TCP, no IB; per try-7 lesson)
export GLOO_SOCKET_IFNAME=$ETH_IF
export NCCL_SOCKET_IFNAME=$ETH_IF
export NCCL_IB_DISABLE=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=WARN

# GB10 / SM 12.1a golden env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export FLASHINFER_CUDA_ARCH_LIST=12.1a
export TORCH_CUDA_ARCH_LIST=12.1a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_NVFP4_GEMM_BACKEND=cutlass

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "NODE_IP=$NODE_IP HEAD_IP=$HEAD_IP ETH_IF=$ETH_IF"

source ${VENV_DIR:-./venvs/vllm}/bin/activate

if [[ "$NODE_IP" == "$HEAD_IP" ]]; then
    NODE_RANK=0
    EXTRA_FLAGS=""
    echo "Role: RANK 0 (head/API server)"
else
    NODE_RANK=1
    EXTRA_FLAGS="--headless"
    echo "Role: RANK 1 (worker, headless)"
fi

MODEL=${MODEL_PATH}
HS_PATH=${MISSION_DIR:-./dflash_minimax}/cache/hs_staging
mkdir -p "$HS_PATH"

cd ${MISSION_DIR:-./dflash_minimax}/repos/speculators

echo "[$(date)] Launching launch_vllm.py — STAGING path: $HS_PATH"

exec python3 scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HS_PATH" \
    --target-layer-ids 2 31 60 \
    -- \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --enforce-eager \
    --no-enable-chunked-prefill \
    --no-enable-flashinfer-autotune \
    --kv-cache-dtype auto \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --load-format fastsafetensors \
    --trust-remote-code \
    --port 8000 --host 0.0.0.0 \
    --nnodes 2 --node-rank $NODE_RANK \
    --master-addr $HEAD_IP --master-port $MASTER_PORT \
    $EXTRA_FLAGS
