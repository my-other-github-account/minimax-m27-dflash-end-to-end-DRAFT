#!/bin/bash
# Run llama-speculative-simple sweep on a DFlash drafter GGUF.
# Same recipe as tick-9 of dflash-clean-repro-loop, parameterized by env vars.
#
# Required env:
#   DRAFTER  - path to drafter GGUF (e.g. /home/user/models/MiniMax-M2.7-DFlash-FULL-epoch5.gguf)
#   TAG      - tag for log filenames (e.g. "FULL-epoch5")
#
# Optional env:
#   BUILD    - build dir (default /home/user/dflash_clean_repro/build_clean)
#   TARGET   - verifier first shard (default UD-IQ4_XS shard 1)
#   LOGDIR   - log directory (default /home/user/dflash_clean_repro/logs)
#   PROMPT   - prompt text (default Fibonacci)
#   N        - generation length (default 256)
#   DMAXES   - space-separated dmax values (default "2 4 7")

set -e
BUILD=${BUILD:-/home/user/dflash_clean_repro/build_clean}
TARGET=${TARGET:-/home/user/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf}
LOGDIR=${LOGDIR:-/home/user/dflash_clean_repro/logs}
PROMPT=${PROMPT:-'Write a Python function that computes the nth Fibonacci number iteratively. Then explain step by step what your code does.'}
N=${N:-256}
DMAXES=${DMAXES:-"2 4 7"}

if [ -z "$DRAFTER" ] || [ -z "$TAG" ]; then
  echo "ERROR: DRAFTER and TAG env vars must be set" >&2
  exit 1
fi

mkdir -p "$LOGDIR"

for DMAX in $DMAXES; do
  LOG="$LOGDIR/${TAG}_dmax${DMAX}.log"
  echo "=== running dmax=$DMAX ==="
  "$BUILD/bin/llama-speculative-simple" \
    -m "$TARGET" \
    -md "$DRAFTER" \
    --spec-type dflash \
    --draft-max "$DMAX" \
    -p "$PROMPT" \
    -n "$N" \
    -ngl 99 -ngld 99 \
    -ot exps=CPU \
    -devd CUDA0 \
    -c 8192 --temp 0 \
    > "$LOG" 2>&1
  echo "--- DONE dmax=$DMAX ---"
  echo "exit=$?"
done
echo "ALL DONE"
