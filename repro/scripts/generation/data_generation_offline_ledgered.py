#!/usr/bin/env python3
"""
Ledger-aware drop-in for data_generation_offline.py.

Monkey-patches `get_existing_hidden_state_indices` to also count indices
already present in pool/quarantine via a persistent ledger. After each run,
appends ALL indices it tried to the ledger so subsequent runs skip them.

Run via: PYTHONPATH=<scripts_dir> python3 this_file.py <orig args>
"""
import os
import sys
import json
from pathlib import Path

LEDGER_PATH = Path('${WORKSPACE}/cache/datagen_ledger.json')
POOL = Path('${WORKSPACE}/cache/hs_clean_pool')
QUAR = Path('${WORKSPACE}/cache/hs_quarantine')


def load_ledger():
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH) as f:
            return set(json.load(f))
    return set()


def save_ledger(seen):
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, 'w') as f:
        json.dump(sorted(seen), f)


def seed_ledger_from_disk():
    """Initialize ledger from existing pool + quarantine if not yet seeded."""
    seen = load_ledger()
    if seen:
        return seen
    # Cold start: assume all FIRST N indices were tried, where N = pool + quarantine count
    # (pool was renumbered monotonically, so we can't recover original indices, but the
    # validator only ran on first ~15K samples sequentially, so we can mark 0..N-1 as seen)
    pool_count = len(list(POOL.glob('hs_*.safetensors'))) if POOL.exists() else 0
    quar_count = len([f for f in QUAR.iterdir() if f.suffix == '.safetensors']) if QUAR.exists() else 0
    upper = pool_count + quar_count
    # Estimate: assume the first `upper` indices have been processed (data-gen advances 0..N).
    # This may overcount slightly (some in [0, upper) might have errored without producing
    # any artifact), but those will simply be re-tried — which is what we want.
    seen = set(range(upper))
    save_ledger(seen)
    print(f"[ledger] seeded ledger with {upper} indices (pool={pool_count} + quar={quar_count})", flush=True)
    return seen


# Monkey-patch BEFORE importing main module
import data_generation_offline as _dgo

_orig_get_existing = _dgo.get_existing_hidden_state_indices


def _patched_get_existing(output_path):
    seen = seed_ledger_from_disk()
    # Also include any actual hs_*.safetensors currently in staging (rare race)
    staging_seen = set(_orig_get_existing(output_path))
    combined = sorted(seen | staging_seen)
    print(f"[ledger] get_existing returns {len(combined)} indices "
          f"(ledger={len(seen)}, staging={len(staging_seen)})", flush=True)
    return combined


_dgo.get_existing_hidden_state_indices = _patched_get_existing


# Also patch main to update ledger after run
_orig_main = _dgo.main


def _patched_main():
    args = _dgo.parse_args()
    ledger_before = load_ledger()

    # Predict what indices will be processed in this run
    from datasets import load_from_disk
    dataset = load_from_disk(args.preprocessed_data)
    existing = _patched_get_existing(Path(args.output))
    to_process = _dgo.get_indices_to_process(len(dataset), args.max_samples, existing)
    print(f"[ledger] this run plans to process {len(to_process)} indices "
          f"(first 5: {to_process[:5]}, last 5: {to_process[-5:] if to_process else []})",
          flush=True)

    # Update ledger optimistically — every index we try is "seen" regardless of outcome
    new_seen = ledger_before | set(to_process)
    save_ledger(new_seen)
    print(f"[ledger] saved new ledger with {len(new_seen)} indices "
          f"(was {len(ledger_before)})", flush=True)

    # Now invoke the original main (which has its own argparse re-call internally — let
    # it do that; sys.argv stays the same)
    return _orig_main()


_dgo.main = _patched_main


if __name__ == '__main__':
    sys.exit(_dgo.main() or 0)
