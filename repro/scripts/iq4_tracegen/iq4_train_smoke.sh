#!/bin/bash
# IQ4 SMOKE training launcher — resumable, kill-safe.
# Per repro/plan/00-resumability-doctrine.md:
#   - Line-buffered output via tee
#   - speculators auto-saves checkpoint_best after each new-best val (durable per-epoch)
#   - SIGTERM → torchrun → trainer signal handler will save and exit
#
# Stoppable by: `tmux kill-session -t iq4_smoke_train` or pkill -TERM -f train.py
# Resume by: re-run this script (speculators will load checkpoint_best and continue)
#
# Output dir is timestamp-suffixed so re-runs don't clobber. To truly resume an
# interrupted run, pass RESUME_DIR=/path/to/existing/run as env.

set -e
set -o pipefail

WORK=/home/user/iq4_train
DATA=$WORK/dataset_iq4
LOGS=$WORK/logs
mkdir -p $LOGS

VERIFIER=/home/user/models/MiniMax-M2.7-FP8

# Output dir — fixed name for this experiment to enable resume
OUT=${RESUME_DIR:-$WORK/checkpoints/iq4_smoke_500_3ep}
mkdir -p $OUT

# Same recipe as FP8 SMOKE for apples-to-apples comparison:
# 3 epochs, max_anchors=64, lr=3e-5, warmup=100
EPOCHS=${EPOCHS:-3}
MAX_ANCHORS=${MAX_ANCHORS:-64}
LR=${LR:-3e-5}

LOG=$LOGS/iq4_smoke_$(date +%Y%m%d_%H%M%S).log

echo "[launcher] output dir: $OUT" | tee -a $LOG
echo "[launcher] data: $DATA" | tee -a $LOG
echo "[launcher] verifier (config/tokenizer only): $VERIFIER" | tee -a $LOG
echo "[launcher] epochs=$EPOCHS max_anchors=$MAX_ANCHORS lr=$LR" | tee -a $LOG
echo "[launcher] log: $LOG" | tee -a $LOG

cd /home/user/dflash_minimax/repos/speculators

source /home/user/venvs/vllm/bin/activate

# stdbuf -oL to keep tee line-buffered all the way down
exec stdbuf -oL torchrun --nproc-per-node=1 --standalone scripts/train.py \
    --speculator-type dflash \
    --verifier-name-or-path $VERIFIER \
    --data-path $DATA/prompts \
    --hidden-states-path $DATA/hidden_states \
    --save-path $OUT \
    --epochs $EPOCHS \
    --total-seq-len 2048 \
    --max-anchors $MAX_ANCHORS \
    --num-workers 1 --prefetch-factor 2 --on-missing skip \
    --target-layer-ids 2 16 30 45 59 \
    --draft-arch qwen3 --draft-hidden-act silu \
    --mask-token-id 200054 --block-size 8 \
    --hidden-states-dtype bfloat16 \
    --num-layers 5 --draft-vocab-size 32768 \
    --lr $LR --scheduler-warmup-steps 100 \
    --save-best --log-freq 5 \
    2>&1 | tee -a $LOG
