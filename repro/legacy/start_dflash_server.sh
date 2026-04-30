#!/bin/bash
# Launch llama-server with DFlash speculative decoding for MiniMax-M2.7 + drafter.
# Critical: do NOT pass -ngl/--n-gpu-layers. The 101 GB UD-IQ4_XS quant cannot fit
# in the GB10's GPU-accessible VRAM (~120 GB unified memory shared with system),
# so we let llama.cpp use the default tensor placement (CPU-side mmap'd weights
# with on-the-fly GPU compute). This matches the working AR-baseline + ngram
# speculative-decode launch pattern on the same hardware.
set -euo pipefail

LLAMA=${LLAMA:-${WORKSPACE}/llama.cpp-pr22105/build/bin/llama-server}
TARGET=${TARGET:-${WORKSPACE}/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf}
DRAFTER=${DRAFTER:-${WORKSPACE}/models/r4-best-pr22105.gguf}
PORT=${PORT:-8013}
LOG=${LOG:-/tmp/llama-dflash-server.log}
DRAFT_MAX=${DRAFT_MAX:-7}

echo "[$(date)] Starting llama-server with DFlash"
echo "  target:    $TARGET"
echo "  drafter:   $DRAFTER"
echo "  draft-max: $DRAFT_MAX"
echo "  port:      $PORT"
echo "  log:       $LOG"

exec "$LLAMA" \
  --jinja -fa on --no-warmup \
  -t 20 \
  -c 32768 \
  -ctk q8_0 -ctv q8_0 \
  -np 1 \
  -m "$TARGET" \
  -md "$DRAFTER" \
  --dflash \
  --draft-max "$DRAFT_MAX" \
  --host 127.0.0.1 --port "$PORT" \
  2>&1 | tee "$LOG"
