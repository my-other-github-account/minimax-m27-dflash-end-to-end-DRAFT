#!/usr/bin/env python3
"""
build_vocab_maps.py — Generate canonical d2t.npy / t2d.npy / token_freq.pt
from a paired dataset's prompts/ arrow file.

Outputs (all written into the dataset's prompts/ subdir, where the trainer
looks for them):

  token_freq.pt   dict[int, int]                token-id → loss-mask freq count
  t2d.npy         (verifier_vocab_size,) bool   True where in draft vocab
                                                (sum() == draft_vocab_size)
  d2t.npy         (draft_vocab_size,) int64     OFFSET table:
                                                verifier_token = draft_id + d2t[draft_id]

CRITICAL FORMAT NOTES (see repro/02-training.md §2.4):
  - t2d is a BOOL MASK, not an index map.
  - d2t stores OFFSETS, not target token ids.
  Get either of these wrong and the trainer rejects the maps with
  "t2d has N non-zero values, expected M" or similar.

This script delegates to speculators' canonical helper
`build_vocab_mappings_from_distribution` to avoid format drift.

Usage:
    python build_vocab_maps.py \\
        --paired-dir ${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired \\
        --verifier-vocab-size 200064 \\
        --draft-vocab-size 32768
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

# Imported lazily so an import failure surfaces a clear error.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-dir", required=True, type=Path,
                    help="Directory containing prompts/data-*.arrow")
    ap.add_argument("--verifier-vocab-size", type=int, default=200064,
                    help="MiniMax-M2.7 = 200064")
    ap.add_argument("--draft-vocab-size", type=int, default=32768)
    ap.add_argument("--prompts-subdir", default="prompts")
    args = ap.parse_args()

    prompts_dir = args.paired_dir / args.prompts_subdir
    if not prompts_dir.exists():
        raise SystemExit(f"prompts dir not found: {prompts_dir}")

    print(f"[1/3] Loading {prompts_dir}")
    ds = load_from_disk(str(prompts_dir))
    if "input_ids" not in ds.column_names or "loss_mask" not in ds.column_names:
        raise SystemExit("dataset must have input_ids and loss_mask columns")
    print(f"      {len(ds)} rows")

    print("[2/3] Counting token frequencies under loss mask")
    counter: Counter[int] = Counter()
    total_masked = 0
    for row in ds:
        ids = row["input_ids"]
        mask = row["loss_mask"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        for tok, m in zip(ids, mask):
            if m:
                counter[int(tok)] += 1
                total_masked += 1
    print(f"      {total_masked} loss-mask tokens, {len(counter)} unique")

    # Coverage of top-K
    top = counter.most_common(args.draft_vocab_size)
    top_count = sum(c for _, c in top)
    coverage = 100.0 * top_count / max(1, total_masked)
    print(f"      top-{args.draft_vocab_size} coverage: {coverage:.2f}%")
    if coverage < 95.0:
        print(f"      WARN: coverage <95%; dataset may be too narrow")

    print("[3/3] Building canonical vocab maps via speculators")
    try:
        from speculators.train.utils.vocab_mapping import (
            build_vocab_mappings_from_distribution,
        )
    except ImportError as e:
        raise SystemExit(
            "speculators not importable; install or PYTHONPATH it. "
            f"Error: {e}"
        )

    freq_dict = dict(counter)
    d2t, t2d = build_vocab_mappings_from_distribution(
        freq_dict,
        target_vocab_size=args.verifier_vocab_size,
        draft_vocab_size=args.draft_vocab_size,
    )

    # Validate canonical format
    assert t2d.dtype == np.bool_, f"t2d must be bool, got {t2d.dtype}"
    assert t2d.shape == (args.verifier_vocab_size,)
    assert int(t2d.sum()) == args.draft_vocab_size, (
        f"t2d.sum()={int(t2d.sum())} != draft_vocab_size={args.draft_vocab_size}"
    )
    assert d2t.dtype == np.int64, f"d2t must be int64, got {d2t.dtype}"
    assert d2t.shape == (args.draft_vocab_size,)

    np.save(prompts_dir / "t2d.npy", t2d)
    np.save(prompts_dir / "d2t.npy", d2t)
    torch.save(freq_dict, prompts_dir / "token_freq.pt")

    print(f"  wrote t2d.npy   shape={t2d.shape}  dtype={t2d.dtype}  sum={int(t2d.sum())}")
    print(f"  wrote d2t.npy   shape={d2t.shape}  dtype={d2t.dtype}  unique={len(np.unique(d2t))}")
    print(f"  wrote token_freq.pt   {len(freq_dict)} entries")

    # Stash a small report
    report = {
        "n_rows": len(ds),
        "total_loss_mask_tokens": total_masked,
        "unique_tokens_seen": len(counter),
        "draft_vocab_size": args.draft_vocab_size,
        "verifier_vocab_size": args.verifier_vocab_size,
        "top_k_coverage_pct": round(coverage, 4),
        "d2t_unique_offsets": int(len(np.unique(d2t))),
    }
    (args.paired_dir / "vocab_maps_report.json").write_text(json.dumps(report, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
