#!/usr/bin/env bash
# launch_server_dflash.sh — wrap llama-server with the env vars and flags
# required for empirical-tau capture (see repro/04-empirical-tau-llama-benchy.md
# §4.2). The legacy target-lm-head GGUF is mandatory; the compact
# d2t-i32 GGUF collapses tau to ~1.0 at runtime (see §4.1).
#
# Required env:
#   VERIFIER_GGUF   path to MiniMax-M2.7-FP8 verifier GGUF
#   DRAFTER_GGUF    path to *legacy-targethead* drafter GGUF
# Optional env:
#   PORT            default 8080
#   CTX             default 8192
#   LLAMA_SERVER    default $(command -v llama-server)
#   LOG             default ./server_dflash.log
set -euo pipefail

: "${VERIFIER_GGUF:?set VERIFIER_GGUF to the verifier path}"
: "${DRAFTER_GGUF:?set DRAFTER_GGUF to the legacy-targethead drafter path}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"
LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server || true)}"
LOG="${LOG:-./server_dflash.log}"

if [[ -z "${LLAMA_SERVER}" || ! -x "${LLAMA_SERVER}" ]]; then
    echo "error: llama-server binary not found; set LLAMA_SERVER" >&2
    exit 2
fi
if [[ ! -f "${VERIFIER_GGUF}" ]]; then
    echo "error: VERIFIER_GGUF=${VERIFIER_GGUF} not found" >&2
    exit 2
fi
if [[ ! -f "${DRAFTER_GGUF}" ]]; then
    echo "error: DRAFTER_GGUF=${DRAFTER_GGUF} not found" >&2
    exit 2
fi

# DFlash runtime knobs — all three are mandatory.
export DFLASH_BLOCK_INCLUDES_ANCHOR=1
export DFLASH_RAW_TOKENS=1
export DFLASH_VERIFIER_KV_TRIM_ON_REJECT=1

echo "[launch_server_dflash] VERIFIER_GGUF=${VERIFIER_GGUF}"
echo "[launch_server_dflash] DRAFTER_GGUF=${DRAFTER_GGUF}"
echo "[launch_server_dflash] PORT=${PORT} CTX=${CTX}"
echo "[launch_server_dflash] DFLASH_BLOCK_INCLUDES_ANCHOR=${DFLASH_BLOCK_INCLUDES_ANCHOR}"
echo "[launch_server_dflash] DFLASH_RAW_TOKENS=${DFLASH_RAW_TOKENS}"
echo "[launch_server_dflash] DFLASH_VERIFIER_KV_TRIM_ON_REJECT=${DFLASH_VERIFIER_KV_TRIM_ON_REJECT}"
echo "[launch_server_dflash] log -> ${LOG}"

exec "${LLAMA_SERVER}" \
    -m "${VERIFIER_GGUF}" \
    -md "${DRAFTER_GGUF}" \
    --draft-max 7 \
    --top-k 1 --temp 0.0 \
    --ctx-size "${CTX}" \
    --host 127.0.0.1 --port "${PORT}" \
    2>&1 | tee "${LOG}"
