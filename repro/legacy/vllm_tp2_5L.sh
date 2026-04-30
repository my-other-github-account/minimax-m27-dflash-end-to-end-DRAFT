#!/bin/bash
# 5-layer DFlash hidden state extraction for MiniMax-M2.7-NVFP4
# Per DFlash paper §5.1 + Table 6: 5 target hidden features (uniformly between layer 2 and N-3=59)
# Verifier: MiniMax-M2.7-NVFP4-GB10 (62 layers). Extract layers [2, 16, 30, 45, 59] + auto-appended layer 62 = 6 layers total.
#
# Patches required (verify before launch):
#   - R33 in interfaces.py (×2) — fp32 sum + finite-clamp guard
#   - R33 in extract_hidden_states.py (×1) — re-zero buffer between requests
#   - R34 in example_hidden_states_connector.py (×1) — upcast hidden_states to fp32 before save
#
# Same script runs on BOTH spark-2 (rank 0) and spark-3 (rank 1). Auto-detects role.
set -euo pipefail

NODE_IP=$(ip -4 -o addr show enp1s0f1np1 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
ETH_IF=enp1s0f1np1
HEAD_IP=192.168.200.2
MASTER_PORT=29501

MODEL=/home/user/models/MiniMax-M2.7-NVFP4-GB10
HS_PATH=/home/user/dflash_minimax/data/preprocessed_5L/hs_staging
LOG=/home/user/dflash_minimax/logs/vllm-tp2-5L-$(hostname)-$(date +%Y%m%d-%H%M%S).log
mkdir -p "$HS_PATH" "$(dirname $LOG)"

# CRITICAL: vLLM identity (no-Ray pattern still needs this)
export VLLM_HOST_IP=$NODE_IP

# Multi-node NCCL via QSFP — plain TCP, no IB
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

source /home/user/venvs/vllm/bin/activate

# Pre-flight: verify all 4 patch markers are in place
PATCH_INTERFACES=$(grep -c R33 /home/user/venvs/vllm/lib/python3.12/site-packages/vllm/model_executor/models/interfaces.py)
PATCH_EXTRACT=$(grep -c R33 /home/user/venvs/vllm/lib/python3.12/site-packages/vllm/v1/spec_decode/extract_hidden_states.py)
PATCH_CONNECTOR=$(grep -c R34_UPCAST_HS /home/user/venvs/vllm/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py)
echo "[$(date)] PATCH_CHECK: R33-interfaces=$PATCH_INTERFACES (need 2), R33-extract=$PATCH_EXTRACT (need 1), R34-connector=$PATCH_CONNECTOR (need 1)" | tee -a "$LOG"
if [ "$PATCH_INTERFACES" -lt 2 ] || [ "$PATCH_EXTRACT" -lt 1 ] || [ "$PATCH_CONNECTOR" -lt 1 ]; then
  echo "[$(date)] FATAL: missing patches. Refuse to launch." | tee -a "$LOG"
  exit 1
fi

# Pre-flight: refuse if existing vllm workers
EXISTING=$(pgrep -af "(vllm serve|EngineCore|RayWorkerWrapper)" 2>/dev/null | grep -v pgrep || true)
if [ -n "$EXISTING" ]; then
  echo "[$(date)] PRE_FLIGHT_ABORT: existing vLLM workers" | tee -a "$LOG"
  echo "$EXISTING" | tee -a "$LOG"
  exit 0
fi

# Auto-detect rank
if [[ "$NODE_IP" == "$HEAD_IP" ]]; then
    NODE_RANK=0
    EXTRA_FLAGS=""
    echo "[$(date)] Role: RANK 0 (head/API server)" | tee -a "$LOG"
else
    NODE_RANK=1
    EXTRA_FLAGS="--headless"
    echo "[$(date)] Role: RANK 1 (worker, headless)" | tee -a "$LOG"
fi

cd /home/user/dflash_minimax/repos/speculators

echo "[$(date)] Launching: target_layer_ids=[2,16,30,45,59], hs_path=$HS_PATH" | tee -a "$LOG"

exec python3 scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HS_PATH" \
    --target-layer-ids 2 16 30 45 59 \
    -- \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --enforce-eager \
    --no-enable-flashinfer-autotune \
    --no-enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype auto \
    --max-num-seqs 8 \
    --load-format fastsafetensors \
    --trust-remote-code \
    --port 8000 --host 0.0.0.0 \
    --nnodes 2 --node-rank $NODE_RANK \
    --master-addr $HEAD_IP --master-port $MASTER_PORT \
    $EXTRA_FLAGS \
    >> "$LOG" 2>&1
