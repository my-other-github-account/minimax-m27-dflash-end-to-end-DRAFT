#!/usr/bin/env bash
# wait_for_server.sh — poll /v1/models until 200, max 60s.
#
# Optional env:
#   PORT     default 8080
#   TIMEOUT  default 60
set -euo pipefail

PORT="${PORT:-8080}"
TIMEOUT="${TIMEOUT:-60}"
URL="http://127.0.0.1:${PORT}/v1/models"

echo "[wait_for_server] polling ${URL} (timeout=${TIMEOUT}s)"
deadline=$(( $(date +%s) + TIMEOUT ))
while (( $(date +%s) < deadline )); do
    if curl -fsS -o /dev/null -m 2 "${URL}"; then
        echo "[wait_for_server] ready"
        exit 0
    fi
    sleep 1
done

echo "[wait_for_server] TIMEOUT after ${TIMEOUT}s waiting on ${URL}" >&2
exit 1
