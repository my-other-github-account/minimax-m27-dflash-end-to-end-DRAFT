#!/usr/bin/env python3
"""Wrapper for data_generation_offline.py — skip indices already in pool.

Per speculators-data-pipeline skill Trap 13: data_generation_offline.py only
scans --output dir for existing samples, but the validator daemon DRAINS that
dir continuously. Result: verifier wastes GPU on indices already in clean_pool.

Fix: monkey-patch get_existing_hidden_state_indices to UNION
(staging-scan ∪ pool ∪ quarantine).

Env vars:
  HS_POOL_DIR        — clean pool dir (always added to skip set)
  HS_QUARANTINE_DIR  — quarantine dir
  HS_REDO_QUARANTINE — "1" → DON'T add quarantine to skip (re-attempt those)
                      "0" → add quarantine to skip
"""
import os, sys, re
from pathlib import Path

POOL_DIR  = Path(os.environ.get("HS_POOL_DIR", ""))
QUAR_DIR  = Path(os.environ.get("HS_QUARANTINE_DIR", ""))
REDO_QUAR = os.environ.get("HS_REDO_QUARANTINE", "0") == "1"

def collect_idx(d: Path) -> set:
    out = set()
    if not d.exists(): return out
    for f in d.iterdir():
        m = re.match(r"hs_(\d+)\.safetensors$", f.name)
        if m: out.add(int(m.group(1)))
    return out

POOL_IDX = collect_idx(POOL_DIR) if POOL_DIR.name else set()
QUAR_IDX = collect_idx(QUAR_DIR) if QUAR_DIR.name else set()
SKIP_SET = set(POOL_IDX)
if not REDO_QUAR:
    SKIP_SET |= QUAR_IDX

print(f"[wrapper] POOL: {len(POOL_IDX)} | QUAR: {len(QUAR_IDX)} | "
      f"REDO_QUAR={REDO_QUAR} | skip_set={len(SKIP_SET)}", flush=True)

# Locate the script and monkey-patch
SCRIPT_DIR = "${WORKSPACE}/dflash_minimax/repos/speculators/scripts"
sys.path.insert(0, SCRIPT_DIR)
import data_generation_offline as dg

_orig = dg.get_existing_hidden_state_indices

def patched(output_path):
    staging_idx = set(_orig(output_path))
    combined = sorted(staging_idx | SKIP_SET)
    print(f"[wrapper] existing-indices: staging={len(staging_idx)} "
          f"pool={len(POOL_IDX)} total_skip={len(combined)}", flush=True)
    return combined

dg.get_existing_hidden_state_indices = patched

# Re-invoke main() — argv is already set up
dg.main()
