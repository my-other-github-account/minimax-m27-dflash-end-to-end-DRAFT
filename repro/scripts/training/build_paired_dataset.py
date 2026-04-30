#!/usr/bin/env python3
"""
build_paired_dataset.py — Re-pair a heterogeneous hidden-state pool against
known prompt sources by content hash.

Use case: an autonomous trace-generation loop accumulates `hs_<N>.safetensors`
files across multiple prompt sources, with monotonic file renaming. The
trainer requires a 1:1 index pairing between hidden-state files and prompts
(see repro/02-training.md §2.2). This script rebuilds that pairing by hashing
the `token_ids` saved inside each safetensor and matching against hashed
prompt input_ids across all candidate sources.

Outputs a paired dataset directory ready for training:

    <output>/
    ├── prompts/data-00000-of-00001.arrow   (HF Dataset, paired rows in pool order)
    ├── prompts/dataset_info.json
    ├── prompts/state.json
    ├── hidden_states/hs_<i>.safetensors    (symlinks back to pool)
    ├── pairing_report.json
    └── match_table.jsonl                   (per-file provenance)

Vocab maps (d2t.npy, t2d.npy, token_freq.pt) are NOT generated here — run
build_vocab_maps.py against the output afterwards.

Usage:
    python build_paired_dataset.py \\
        --pool ${DATA_ROOT}/preprocessed_5L_FP8/hs_clean_pool \\
        --prompt-source name1=${DATA_ROOT}/preprocessed/source1/prompts \\
        --prompt-source name2=${DATA_ROOT}/preprocessed/source2/prompts \\
        --output ${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, load_from_disk
from safetensors.torch import safe_open


def hash_token_ids(token_ids) -> str:
    """Stable sha256 over a token id sequence (list, tensor, or array)."""
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return hashlib.sha256(json.dumps(list(token_ids)).encode()).hexdigest()


def hash_pool_files(pool_dir: Path) -> dict[int, str]:
    """Return {pool_index: sha256_of_token_ids} for every hs_<N>.safetensors."""
    hashes: dict[int, str] = {}
    files = sorted(pool_dir.glob("hs_*.safetensors"),
                   key=lambda p: int(p.stem.split("_")[1]))
    for path in files:
        idx = int(path.stem.split("_")[1])
        with safe_open(str(path), framework="pt") as f:
            if "token_ids" not in f.keys():
                print(f"WARN: {path.name} missing token_ids; skipping", file=sys.stderr)
                continue
            tok = f.get_tensor("token_ids")
        hashes[idx] = hash_token_ids(tok)
    return hashes


def hash_prompt_source(prompt_dir: Path) -> dict[str, int]:
    """Return {sha256: row_idx} for the first occurrence of each input_ids hash."""
    ds: Dataset = load_from_disk(str(prompt_dir))
    if "input_ids" not in ds.column_names:
        raise ValueError(f"{prompt_dir} has no input_ids column")
    out: dict[str, int] = {}
    for i, row in enumerate(ds):
        h = hash_token_ids(row["input_ids"])
        if h not in out:
            out[h] = i
    return out, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path,
                    help="Directory of hs_<N>.safetensors files")
    ap.add_argument("--prompt-source", action="append", required=True,
                    metavar="NAME=PATH",
                    help="Repeatable. e.g. combined_48k=/path/to/prompts")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    sources: dict[str, Path] = {}
    for spec in args.prompt_source:
        if "=" not in spec:
            ap.error(f"--prompt-source must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        sources[name] = Path(path)

    print(f"[1/4] Hashing pool: {args.pool}")
    pool_hashes = hash_pool_files(args.pool)
    print(f"      {len(pool_hashes)} pool files hashed")

    print(f"[2/4] Hashing {len(sources)} prompt sources")
    src_hash_maps: dict[str, dict[str, int]] = {}
    src_datasets: dict[str, Dataset] = {}
    for name, path in sources.items():
        h, ds = hash_prompt_source(path)
        src_hash_maps[name] = h
        src_datasets[name] = ds
        print(f"      {name}: {len(ds)} rows, {len(h)} unique hashes")

    print("[3/4] Matching pool → sources")
    matched_rows = []                                # list[dict] for arrow
    match_table = []                                 # provenance
    by_source = {n: 0 for n in sources}
    unmatched: list[int] = []

    pool_indices = sorted(pool_hashes.keys())
    for pool_idx in pool_indices:
        h = pool_hashes[pool_idx]
        chosen = None
        for src_name, hmap in src_hash_maps.items():
            if h in hmap:
                chosen = (src_name, hmap[h])
                break
        if chosen is None:
            unmatched.append(pool_idx)
            continue
        src_name, src_row_idx = chosen
        by_source[src_name] += 1
        row = dict(src_datasets[src_name][src_row_idx])
        matched_rows.append(row)
        match_table.append({
            "out_index": len(matched_rows) - 1,
            "pool_index": pool_idx,
            "source": src_name,
            "source_row": src_row_idx,
            "sha256": h,
        })

    print(f"      matched: {len(matched_rows)} / {len(pool_hashes)}"
          f" ({100*len(matched_rows)/max(1,len(pool_hashes)):.2f}%)")
    print(f"      unmatched: {len(unmatched)}")
    for n, c in by_source.items():
        print(f"        {n}: {c}")

    print(f"[4/4] Writing paired dataset to {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    prompts_dir = args.output / "prompts"
    hs_dir = args.output / "hidden_states"
    prompts_dir.mkdir(exist_ok=True)
    hs_dir.mkdir(exist_ok=True)

    out_ds = Dataset.from_list(matched_rows)
    out_ds.save_to_disk(str(prompts_dir))

    for entry in match_table:
        src = args.pool / f"hs_{entry['pool_index']}.safetensors"
        dst = hs_dir / f"hs_{entry['out_index']}.safetensors"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

    report = {
        "pool_dir": str(args.pool),
        "dest_dir": str(args.output),
        "n_pool_files": len(pool_hashes),
        "n_matched": len(matched_rows),
        "n_unmatched": len(unmatched),
        "match_rate_pct": round(100 * len(matched_rows) / max(1, len(pool_hashes)), 2),
        "by_source": by_source,
        "unmatched_first_20": unmatched[:20],
    }
    (args.output / "pairing_report.json").write_text(json.dumps(report, indent=2))

    with (args.output / "match_table.jsonl").open("w") as f:
        for entry in match_table:
            f.write(json.dumps(entry) + "\n")

    print("Done. Next: run build_vocab_maps.py against this directory.")


if __name__ == "__main__":
    main()
