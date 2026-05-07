#!/usr/bin/env bash
# launch_server_ar.sh — same llama-server, no speculation. Used as the
# autoregressive baseline for the wall-clock comparison in §4.6.
#
# Required env:
#   VERIFIER_GGUF   path to MiniMax-M2.7-FP8 verifier GGUF
# Optional env:
#   PORT            default 8080
#   CTX             default 8192
#   LLAMA_SERVER    default $(command -v llama-server)
#   LOG             default ./server_ar.log
set -euo pipefail

: "${VERIFIER_GGUF:?set VERIFIER_GGUF to the verifier path}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"
LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server || true)}"
LOG="${LOG:-./server_ar.log}"

if [[ -z "${LLAMA_SERVER}" || ! -x "${LLAMA_SERVER}" ]]; then
    echo "error: llama-server binary not found; set LLAMA_SERVER" >&2
    exit 2
fi
if [[ ! -f "${VERIFIER_GGUF}" ]]; then
    echo "error: VERIFIER_GGUF=${VERIFIER_GGUF} not found" >&2
    exit 2
fi

echo "[launch_server_ar] VERIFIER_GGUF=${VERIFIER_GGUF}"
echo "[launch_server_ar] PORT=${PORT} CTX=${CTX}"
echo "[launch_server_ar] log -> ${LOG} (no -md, no --draft-max)"

exec "${LLAMA_SERVER}" \
    -m "${VERIFIER_GGUF}" \
    --top-k 1 --temp 0.0 \
    --ctx-size "${CTX}" \
    --host 127.0.0.1 --port "${PORT}" \
    2>&1 | tee "${LOG}"
