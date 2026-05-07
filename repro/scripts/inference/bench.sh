#!/usr/bin/env bash
# bench.sh — drive eugr/llama-benchy through the proxy on $PORT.
#
# Optional env:
#   PORT  default 8081  (the tau_capture_proxy port; switch to 8080 to
#                       hit llama-server directly and skip capture)
#   OUT   default ./llama_benchy_$(date +%s).json
set -euo pipefail

PORT="${PORT:-8081}"
OUT="${OUT:-./llama_benchy_$(date +%s).json}"

echo "[bench] base-url=http://127.0.0.1:${PORT}/v1  out=${OUT}"

exec uvx --from git+https://github.com/eugr/llama-benchy llama-benchy \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --pp 256 1024 4096 \
    --tg 128 \
    --depth 0 2048 8192 \
    --runs 5 \
    --latency-mode generation \
    --enable-prefix-caching \
    --save-result "${OUT}" \
    --format json
