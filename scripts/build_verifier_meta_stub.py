"""Build a 3-tensor verifier_meta stub from a GGUF for the speculators trainer.

Speculators reads `model.safetensors.index.json` and tries to load every shard
listed in `weight_map`. For a 130-shard model this would read 200 GB the
trainer doesn't actually use — only `lm_head.weight`, `model.embed_tokens.weight`,
and `model.norm.weight` are consumed.

This script extracts those 3 tensors from a GGUF, dequantizes them to bf16,
and writes them as a single safetensors file (~2.4 GB for MiniMax-M2.7).
Pair it with a trivial `model.safetensors.index.json` pointing all 3 keys at
the single shard.

Usage:
    python3 build_verifier_meta_stub.py \\
        --gguf  /path/to/MiniMax-M2.7-UD-IQ4_XS-00002-of-00004.gguf \\
        --out   ~/verifier_meta/model.safetensors

After running, also write the index.json (see repro/00-spark-from-scratch.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import safetensors.torch as st


GGUF_TO_HF = {
    "token_embd.weight":   "model.embed_tokens.weight",
    "output.weight":       "lm_head.weight",
    "output_norm.weight":  "model.norm.weight",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True,
                    help="Path to a GGUF shard that contains the 3 bridge tensors")
    ap.add_argument("--out", required=True,
                    help="Output safetensors path (e.g. ~/verifier_meta/model.safetensors)")
    ap.add_argument("--also-write-index", action="store_true",
                    help="Also write model.safetensors.index.json next to --out")
    args = ap.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # gguf-py is the canonical reader, but it's heavy. llama-cpp-python's
    # gguf reader works fine and ships with `pip install gguf`. We use the
    # standalone `gguf` package directly to avoid adding a torch dependency
    # we don't otherwise need.
    try:
        import gguf
    except ImportError:
        print("ERROR: pip install gguf  (the upstream llama.cpp reader)", file=sys.stderr)
        return 1

    print(f"[stub] reading {args.gguf}", flush=True)
    reader = gguf.GGUFReader(args.gguf, "r")

    found: dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        if tensor.name in GGUF_TO_HF:
            # GGUF tensor -> dequantized fp32 -> bf16
            arr = tensor.data        # numpy array, possibly already dequant
            if arr.dtype.name not in ("float32", "float16", "bfloat16"):
                # Heavy dequant path — defer to gguf's own helper.
                arr = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
            t = torch.from_numpy(arr.copy()).to(torch.bfloat16)
            # GGUF stores transposed for some quants — leave as-is, the
            # speculators trainer only reads `.shape` for sanity-checks.
            hf_name = GGUF_TO_HF[tensor.name]
            found[hf_name] = t
            print(f"[stub] {tensor.name:25s} -> {hf_name:35s} {tuple(t.shape)} {t.dtype}",
                  flush=True)

    missing = [hf for gn, hf in GGUF_TO_HF.items() if hf not in found]
    if missing:
        print(f"ERROR: missing tensors after scan: {missing}", file=sys.stderr)
        print(f"       (tip: try a different GGUF shard — these are typically in shard 2)",
              file=sys.stderr)
        return 1

    print(f"[stub] writing {out_path}", flush=True)
    st.save_file(found, str(out_path))
    print(f"[stub] wrote {out_path.stat().st_size / 1e9:.2f} GB", flush=True)

    if args.also_write_index:
        idx_path = out_path.parent / "model.safetensors.index.json"
        total = sum(v.numel() * v.element_size() for v in found.values())
        weight_map = {k: out_path.name for k in found}
        idx_path.write_text(json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map},
            indent=2,
        ))
        print(f"[stub] wrote {idx_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
