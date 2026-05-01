#!/usr/bin/env python3
"""
Build a filtered training dataset that only contains rows for which we have IQ4 traces.

Reads:
  --src-prompts: original prompts arrow dataset (e.g. ~/iq4_train/prompts)
  --src-traces:  dir of IQ4 hs_<i>.safetensors files (sparse indices)

Writes:
  --out-prompts: new prompts arrow dataset (densely indexed 0..N-1)
  --out-traces:  new traces dir with hs_0..hs_{N-1}.safetensors (renamed from sparse)

The mapping is: for each i in sorted(existing_indices), row i_new = original_row[i_old].
Vocab maps (d2t.npy, t2d.npy, token_freq.pt) are copied verbatim to --out-prompts.
"""
import sys, os, json, shutil, argparse
from pathlib import Path
from datasets import load_from_disk, Dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-prompts", required=True)
    ap.add_argument("--src-traces", required=True)
    ap.add_argument("--out-prompts", required=True)
    ap.add_argument("--out-traces", required=True)
    args = ap.parse_args()

    src_prompts = Path(args.src_prompts)
    src_traces = Path(args.src_traces)
    out_prompts = Path(args.out_prompts)
    out_traces = Path(args.out_traces)

    print(f"[filter] loading prompts from {src_prompts}", flush=True)
    ds = load_from_disk(str(src_prompts))
    print(f"[filter] {len(ds)} source rows", flush=True)

    # Find existing trace indices
    indices = sorted(int(p.stem.split("_")[1]) for p in src_traces.glob("hs_*.safetensors"))
    print(f"[filter] {len(indices)} existing IQ4 traces, range {indices[0]}..{indices[-1]}", flush=True)

    # Build filtered dataset
    filtered = ds.select(indices)
    print(f"[filter] filtered to {len(filtered)} rows", flush=True)

    # Save dataset to out-prompts
    out_prompts.mkdir(parents=True, exist_ok=True)
    filtered.save_to_disk(str(out_prompts))

    # Copy vocab maps
    for name in ("d2t.npy", "t2d.npy", "token_freq.pt"):
        src = src_prompts / name
        if src.exists():
            shutil.copy2(src, out_prompts / name)
            print(f"[filter] copied {name}", flush=True)

    # Build dense traces dir
    out_traces.mkdir(parents=True, exist_ok=True)
    for new_i, old_i in enumerate(indices):
        src_f = src_traces / f"hs_{old_i}.safetensors"
        dst_f = out_traces / f"hs_{new_i}.safetensors"
        if dst_f.exists():
            continue  # idempotent re-run
        # Hardlink if same fs, else copy
        try:
            os.link(src_f, dst_f)
        except OSError:
            shutil.copy2(src_f, dst_f)
    print(f"[filter] wrote {len(indices)} traces to {out_traces} (densely indexed 0..{len(indices)-1})", flush=True)

    # Write a manifest of the index mapping for traceability
    manifest = {
        "source_prompts": str(src_prompts),
        "source_traces": str(src_traces),
        "n_filtered": len(indices),
        "old_to_new": {str(old_i): new_i for new_i, old_i in enumerate(indices)},
    }
    (out_prompts / "filter_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[filter] wrote manifest to {out_prompts}/filter_manifest.json", flush=True)

if __name__ == "__main__":
    main()
