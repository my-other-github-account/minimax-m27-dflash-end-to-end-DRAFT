#!/usr/bin/env bash
# Launch v14 BATCHED tracegen using the fast generate_many() API.
#
# Survives ssh logout via systemd-run --user --scope.
#
# All paths are configurable via env vars; defaults match the spark-4
# production layout.
#
# Required env (or default present):
#   POOL_DIR      output dir root        (default: $HOME/iq4_tracegen_v14_pool)
#   WORKER_BIN    llama-dump-hiddens     (default: rebuilt path under repo)
#   PROMPTS_DIR   tulu3 prompts dataset  (default: $HOME/iq4_tracegen/prompts_tulu3)
#   HF_DIR        verifier meta dir      (default: $HOME/verifier_meta)
#   GGUF_PATH     minimax IQ4_XS GGUF    (default: under $HOME/clawd/iq4_models/...)
#   PYTHON        python interpreter     (default: $HOME/venvs/vllm/bin/python3)
#
# Tunables:
#   SHARD            scope/shard id        (default: V14_S4_BATCHED)
#   SAMPLE_SEED      deterministic perm    (default: 28)
#   SAMPLE_TARGET    stop after N unique   (default: 200000)
#   BATCH_WIDTH      seqs per batch        (default: 8)
#   MAX_BATCH_TOK    n_batch in C worker   (default: 8192)
#   LENGTH_BUCKET    pad-up bucket size    (default: 128)
#   PREWARM          set to 1 to prewarm   (default: 1)
set -euo pipefail

SHARD="${SHARD:-V14_S4_BATCHED}"
SAMPLE_SEED="${SAMPLE_SEED:-28}"
SAMPLE_TARGET="${SAMPLE_TARGET:-200000}"
BATCH_WIDTH="${BATCH_WIDTH:-8}"
MAX_BATCH_TOK="${MAX_BATCH_TOK:-8192}"
LENGTH_BUCKET="${LENGTH_BUCKET:-128}"
PREWARM="${PREWARM:-1}"

POOL_DIR="${POOL_DIR:-$HOME/iq4_tracegen_v14_pool}"
PROMPTS_DIR="${PROMPTS_DIR:-$HOME/iq4_tracegen/prompts_tulu3}"
HF_DIR="${HF_DIR:-$HOME/verifier_meta}"
GGUF_PATH="${GGUF_PATH:-$HOME/clawd/iq4_models/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf}"
PYTHON="${PYTHON:-$HOME/venvs/vllm/bin/python3}"

# Worker script: prefer in-repo copy if this script lives in repro/scripts/iq4_tracegen/
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${WORKER:-$SCRIPT_DIR/v14_batched_worker.py}"

# Worker binary: default to the locally-built rebuild path.
WORKER_BIN="${WORKER_BIN:-$SCRIPT_DIR/../../../build/llama.cpp-dflash/build/bin/llama-dump-hiddens-worker}"

SESSION="v14_batched_${SHARD}"
LOG="$POOL_DIR/logs/batched_worker_${SHARD}_$(date +%Y%m%d_%H%M%S).log"
SERVER_LOG="$POOL_DIR/logs/batched_server_${SHARD}_$(date +%Y%m%d_%H%M%S).log"
SOCK="/tmp/dflash_v14_batched_${SHARD}.sock"

mkdir -p "$POOL_DIR/logs" "$POOL_DIR/traces"

if [ ! -x "$WORKER_BIN" ]; then
  echo "[launcher] FATAL: worker binary $WORKER_BIN not found or not executable" >&2
  exit 1
fi
if ! strings "$WORKER_BIN" 2>/dev/null | grep -q "n_seq_max"; then
  echo "[launcher] WARNING: worker binary at $WORKER_BIN missing n_seq_max signature -- did you rebuild after patching dump_hiddens_worker.cpp?" >&2
fi
if [ ! -f "$WORKER" ]; then
  echo "[launcher] FATAL: worker script $WORKER not found" >&2
  exit 1
fi

CMD=(
  "$PYTHON" "$WORKER"
  --shard-id "$SHARD"
  --sample-seed "$SAMPLE_SEED"
  --sample-target "$SAMPLE_TARGET"
  --batch-width "$BATCH_WIDTH"
  --max-batch-tokens "$MAX_BATCH_TOK"
  --length-bucket "$LENGTH_BUCKET"
  --out "$POOL_DIR/traces"
  --state "$POOL_DIR/state_worker_${SHARD}.json"
  --prompts "$PROMPTS_DIR"
  --binary "$WORKER_BIN"
  --verifier-name minimax-m2.7-iq4-xs
  --hf-path "$HF_DIR"
  --gguf-path "$GGUF_PATH"
  --layer-ids "2,16,30,45,59,61"
  --max-seq-len 2048
  --skip-hashes "$POOL_DIR/cluster_union_hashes.pkl"
  --new-hashes-out "$POOL_DIR/traces/new_hashes_batched.pkl"
  --hash-flush-every 10
  --log-every 50
  --socket "unix://$SOCK"
  --ctx 16384
  --ngl 99
  --override-tensor "exps=CPU"
  --server-log "$SERVER_LOG"
)
if [ "$PREWARM" = "1" ]; then
  CMD+=( --prewarm )
fi

INNER="cd $POOL_DIR && while true; do \
echo \"--- launching $SHARD at \$(date -Iseconds) ---\"; \
rm -f $SOCK; \
${CMD[*]} 2>&1 | tee -a $LOG; \
ec=\$?; \
echo \"--- worker $SHARD exited rc=\$ec at \$(date -Iseconds) ---\" | tee -a $LOG; \
if [ \$ec -eq 0 ]; then echo \"--- clean exit, stopping --\" | tee -a $LOG; break; fi; \
echo \"--- restart in 30s ---\" | tee -a $LOG; \
sleep 30; \
done"

systemd-run --user --scope --unit="$SESSION" -- bash -lc "$INNER" >/dev/null 2>&1 &
SCOPE_PID=$!
disown $SCOPE_PID 2>/dev/null || true

sleep 3

if systemctl --user is-active "$SESSION.scope" >/dev/null 2>&1; then
  echo "[launcher] OK: $SESSION.scope is active"
  echo "[launcher] log:         $LOG"
  echo "[launcher] server log:  $SERVER_LOG"
  echo "[launcher] socket:      $SOCK"
  echo "[launcher] target:      $POOL_DIR/traces (target=$SAMPLE_TARGET unique)"
  echo "[launcher] batch_width=$BATCH_WIDTH max_batch_tok=$MAX_BATCH_TOK length_bucket=$LENGTH_BUCKET prewarm=$PREWARM"
  echo "[launcher] check with:  systemctl --user status $SESSION.scope"
  echo "[launcher] tail with:   tail -f $LOG"
else
  echo "[launcher] ERROR: $SESSION.scope did not become active" >&2
  systemctl --user status "$SESSION.scope" 2>&1 | head -20
  exit 1
fi
