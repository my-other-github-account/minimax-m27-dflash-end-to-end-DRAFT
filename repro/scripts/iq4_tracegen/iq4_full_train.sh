#!/bin/bash
# IQ4 (GGUF-only) FULL training launcher on spark-1 — resumable, kill-safe, regular checkpoints.
#
# DATA PROVENANCE (zero FP8 in any path):
#   - hidden_states: pooled GGUF-generated traces from 3 workers
#                    (verifier=MiniMax-M2.7-GGUF/UD-IQ4_XS)
#   - prompts:       same dense Arrow dataset that paired with the GGUF traces,
#                    SHUFFLED before save (see filter_dataset_v2.py)
#   - verifier-name-or-path: meta-only dir (config.json + tokenizer + 3 GGUF-derived
#                            bridge tensors). NO FP8 weight files.
#
# CHECKPOINTING:
#   --save-best emits checkpoint_best/ on every val-loss improvement.
#   Per-epoch durable checkpoints via the speculators trainer.
#   State.json + atomic writes per repro/plan/00-resumability-doctrine.md.
#
# Resume: re-run; if checkpoint_best/ exists, speculators loads and continues.

set -e
set -o pipefail

WORK=/home/user/iq4_full_run
DATA_PROMPTS=$WORK/prompts_dense_v2
DATA_HS=$WORK/traces_dense_v2/hidden_states
LOGS=$WORK/logs
CKPT=$WORK/checkpoints/iq4_full_gguf_only_v2
mkdir -p $LOGS $CKPT

# Meta-only dir (config + tokenizer + 3 GGUF-derived bridge tensors)
VERIFIER_META=$WORK/verifier_meta

# Same recipe as FP8 FULL run, scaled to current data:
EPOCHS=${EPOCHS:-10}
MAX_ANCHORS=${MAX_ANCHORS:-64}
LR=${LR:-3e-5}
WARMUP=${WARMUP:-100}

LOG=$LOGS/iq4_full_train_$(date +%Y%m%d_%H%M%S).log

echo "[launcher] === IQ4 GGUF-ONLY FULL TRAINING ===" | tee -a $LOG
echo "[launcher] data path (prompts):       $DATA_PROMPTS" | tee -a $LOG
echo "[launcher] hidden_states (GGUF):      $DATA_HS" | tee -a $LOG
echo "[launcher]   trace count: $(ls $DATA_HS | wc -l)" | tee -a $LOG
echo "[launcher] verifier meta (no weights): $VERIFIER_META" | tee -a $LOG
echo "[launcher] checkpoints out:           $CKPT" | tee -a $LOG
echo "[launcher] epochs=$EPOCHS  max_anchors=$MAX_ANCHORS  lr=$LR  warmup=$WARMUP" | tee -a $LOG
echo "[launcher] log:                       $LOG" | tee -a $LOG

cd /home/user/dflash_minimax/repos/speculators
source /home/user/venvs/vllm/bin/activate

exec stdbuf -oL torchrun --nproc-per-node=1 --standalone scripts/train.py \
    --speculator-type dflash \
    --verifier-name-or-path $VERIFIER_META \
    --data-path $DATA_PROMPTS \
    --hidden-states-path $DATA_HS \
    --save-path $CKPT \
    --epochs $EPOCHS \
    --total-seq-len 2048 \
    --max-anchors $MAX_ANCHORS \
    --num-workers 2 --prefetch-factor 2 --on-missing skip \
    --target-layer-ids 2 16 30 45 59 \
    --draft-arch qwen3 --draft-hidden-act silu \
    --mask-token-id 200054 --block-size 8 \
    --hidden-states-dtype bfloat16 \
    --num-layers 5 --draft-vocab-size 32768 \
    --lr $LR --scheduler-warmup-steps $WARMUP \
    --save-best --log-freq 5 \
    2>&1 | tee -a $LOG
