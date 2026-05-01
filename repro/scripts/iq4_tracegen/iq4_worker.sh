#!/bin/bash
# IQ4 trace generation worker — runs on any spark with the IQ4 verifier + buun build.
# Per repro/plan/00-resumability-doctrine.md: line-buffered, atomic, skip-existing,
# state.json with FM27 hash-verify, SIGTERM-safe (current trace finishes, exits clean).
#
# Each worker processes a disjoint index range. Multiple workers can run in parallel
# on different machines; they MUST have non-overlapping --start/--end ranges OR rely
# on the skip-existing race (which is safe because writes are atomic and skip checks
# happen on file-existence, but pointless work is avoided by disjoint ranges).
#
# Usage on a machine:
#   WORKER_ID=A START=500 END=2500 ./iq4_worker.sh
#   (will run gen_traces_v2.py in tmux, named iq4_worker_$WORKER_ID)
#
# Stop:   tmux kill-session -t iq4_worker_<ID>
# Status: tmux capture-pane -t iq4_worker_<ID> -p | tail -30
# Resume: just re-run with the same WORKER_ID/START/END.

set -e
WORKER_ID=${WORKER_ID:?"set WORKER_ID (e.g. A, B, C)"}
START=${START:?"set START (lower bound, inclusive)"}
END=${END:?"set END (upper bound, exclusive)"}

# Per-machine paths (defaults work on spark-3; override on others)
WORK=${WORK:-/home/user/iq4_tracegen}
PROMPTS=${PROMPTS:-$WORK/prompts_fp8}
TRACES=${TRACES:-$WORK/traces/hidden_states}
LOGS=${LOGS:-$WORK/logs}
SCRIPTS=${SCRIPTS:-$WORK/scripts}
BINARY=${BINARY:-$WORK/buun-llama-cpp/build/bin/llama-dump-hiddens}
MODEL=${MODEL:-/home/user/clawd/iq4_models/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf}
CTX=${CTX:-4096}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}

mkdir -p $TRACES $LOGS

LOG=$LOGS/worker_${WORKER_ID}_$(date +%Y%m%d_%H%M%S).log
SESSION=iq4_worker_$WORKER_ID

echo "[worker $WORKER_ID] range=[$START,$END)" | tee -a $LOG
echo "[worker $WORKER_ID] log=$LOG" | tee -a $LOG
echo "[worker $WORKER_ID] tmux session=$SESSION" | tee -a $LOG

# Kill any existing session of the same id (idempotent re-launch)
tmux kill-session -t $SESSION 2>/dev/null || true

# Launch in tmux
tmux new-session -d -s $SESSION \
    "source /home/user/venvs/vllm/bin/activate && \
     cd $WORK && \
     stdbuf -oL python3 $SCRIPTS/gen_traces_v2.py \
        --prompts $PROMPTS \
        --out $TRACES \
        --binary $BINARY \
        --model $MODEL \
        --start $START --end $END \
        --ctx $CTX --max_seq_len $MAX_SEQ_LEN \
        --state $WORK/state_worker_${WORKER_ID}.json \
        2>&1 | tee -a $LOG; \
     touch /tmp/iq4_worker_${WORKER_ID}.end"

sleep 2
tmux ls
echo "[worker $WORKER_ID] launched. tail -f $LOG to watch."
