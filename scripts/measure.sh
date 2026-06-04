#!/usr/bin/env bash
set -euo pipefail

# Directly derived from spark-3:/home/<USER>/t_77088062/run_measure_7.py
# and the reproduced run environment used for measure_revert_7prompts_summary.json.
PORT="${PORT:-8097}"
MODEL="${MODEL:-minimax-luce-golden-revert}"
TAG="${TAG:-revert}"
OUTDIR="${OUTDIR:-$PWD/results}"
LUCE_LOG="${LUCE_LOG:?set LUCE_LOG to the dflash_server log file, e.g. /home/<USER>/t_77088062/server_8097_revert.latest}"

mkdir -p "$OUTDIR"
export PORT MODEL TAG OUTDIR LUCE_LOG
exec python3 "$(dirname "$0")/run_measure_7.py"
