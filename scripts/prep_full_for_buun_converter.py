#!/usr/bin/env python3
"""Prep a speculators-format DFlash checkpoint for buun-llama-cpp's
``convert_hf_to_gguf.py`` DFlashDraftModel converter.

Steps:
1. Read d2t (draft->target offsets) and lm_head ``[draft_V, hidden]``
2. Build new lm_head ``[target_V, hidden]`` = ``full(-65504)`` then scatter
   draft rows to target_idx (the -65504 floor is the d2t-zero-row-dilution fix
   from the ``dflash-gguf-conversion`` skill — without it, runtime chain-pos-2
   measures ~5x lower than the training prediction)
3. Drop d2t and t2d from output
4. Flatten config: hoist ``transformer_layer_config.*`` keys to top level,
   rename ``aux_hidden_state_layer_ids`` → ``target_layer_ids``,
   set ``draft_vocab_size`` = ``target_vocab_size``
5. Copy tokenizer files from the verifier directory

Usage::

  python3 scripts/prep_full_for_buun_converter.py SRC_DIR OUT_DIR \\
      [--verifier-meta-dir DIR] [--rebake-floor -65504.0]

Library equivalent::

  from dflash_llama.inference import prep_for_buun_converter
  prep_for_buun_converter(src_dir, out_dir, verifier_meta_dir=...)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def prep_for_buun_converter(
    src_dir: str | Path,
    out_dir: str | Path,
    verifier_meta_dir: Optional[str | Path] = None,
    rebake_floor: float = -65504.0,
    verbose: bool = True,
) -> Path:
    """Run the prep recipe in-process.

    Parameters
    ----------
    src_dir : path
        Speculators-format checkpoint (must contain ``config.json``,
        ``model.safetensors``, optionally ``val_metrics.json``).
    out_dir : path
        Where to write the prepped checkpoint.
    verifier_meta_dir : path, optional
        Directory containing tokenizer files (``tokenizer.json``,
        ``tokenizer_config.json``, etc). If omitted, falls back to
        ``cfg["speculators_config"]["verifier"]["name_or_path"]`` then
        ``/home/user/models/MiniMax-M2.7-FP8``.
    rebake_floor : float
        Value used for non-mapped rows of the rebaked lm_head. ``-65504.0``
        (largest-magnitude finite bf16 negative) is the documented correct
        value; do not change without reading the d2t-rebake skill.
    """
    src = Path(src_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def log(*a):
        if verbose:
            print(*a, flush=True)

    # 1. Read config
    with open(src / "config.json") as f:
        cfg = json.load(f)

    target_vocab = cfg["transformer_layer_config"]["vocab_size"]
    draft_vocab = cfg["draft_vocab_size"]
    hidden_size = cfg["transformer_layer_config"]["hidden_size"]
    target_layer_ids = cfg["aux_hidden_state_layer_ids"]
    log(f"target_vocab={target_vocab}, draft_vocab={draft_vocab}, hidden={hidden_size}")
    log(f"target_layer_ids={target_layer_ids}")

    # 2. Load tensors, rebake lm_head, drop d2t/t2d
    src_st = src / "model.safetensors"
    new_tensors = {}
    with safe_open(str(src_st), "pt") as f:
        keys = list(f.keys())
        d2t = f.get_tensor("d2t") if "d2t" in keys else None
        t2d = f.get_tensor("t2d") if "t2d" in keys else None
        lm_head = f.get_tensor("lm_head.weight") if "lm_head.weight" in keys else None

        if d2t is not None and lm_head is not None and lm_head.shape[0] == draft_vocab:
            log(f"REBAKE: lm_head {list(lm_head.shape)} -> [{target_vocab}, {hidden_size}] "
                f"with {rebake_floor} floor")
            if t2d is not None:
                t_true = int(t2d.sum().item())
                ok = "OK" if t_true == draft_vocab else "MISMATCH"
                log(f"  t2d.sum() = {t_true} (expected {draft_vocab}: {ok})")

            target_ids = torch.arange(draft_vocab, dtype=torch.long) + d2t.to(torch.long)
            assert target_ids.min() >= 0, f"negative target id {target_ids.min()}"
            assert target_ids.max() < target_vocab, (
                f"target_id {target_ids.max()} >= vocab {target_vocab}"
            )

            new_lm_head = torch.full(
                (target_vocab, hidden_size),
                float(rebake_floor),
                dtype=lm_head.dtype,
            )
            new_lm_head[target_ids] = lm_head
            new_tensors["lm_head.weight"] = new_lm_head
            log(f"  rebaked: scattered {draft_vocab} rows into {target_vocab}-row tensor")
        elif lm_head is not None:
            log(f"lm_head already at target shape {list(lm_head.shape)} - no rebake")
            new_tensors["lm_head.weight"] = lm_head

        # Copy other tensors except d2t/t2d/lm_head (already handled)
        for k in keys:
            if k in ("d2t", "t2d", "lm_head.weight"):
                continue
            new_tensors[k] = f.get_tensor(k)

    log(f"Output tensors: {len(new_tensors)} (was {len(keys)})")

    # 3. Save flattened safetensors
    save_file(new_tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    log(f"Wrote {out / 'model.safetensors'}")

    # 4. Flatten config for buun converter
    new_cfg = {"architectures": ["DFlashDraftModel"]}
    tlc = cfg.get("transformer_layer_config", {})
    for k, v in tlc.items():
        new_cfg[k] = v
    new_cfg["block_size"] = cfg["block_size"]
    new_cfg["mask_token_id"] = cfg["mask_token_id"]
    new_cfg["target_layer_ids"] = target_layer_ids
    new_cfg["aux_hidden_state_layer_ids"] = target_layer_ids
    new_cfg["draft_vocab_size"] = target_vocab  # after rebake
    new_cfg["model_type"] = "qwen3"
    new_cfg["max_anchors"] = cfg.get("max_anchors", 512)

    with open(out / "config.json", "w") as f:
        json.dump(new_cfg, f, indent=2)
    log(f"Wrote {out / 'config.json'}")

    # Also copy val_metrics.json if present (downstream eval reads it)
    if (src / "val_metrics.json").exists():
        shutil.copy2(src / "val_metrics.json", out / "val_metrics.json")
        log("  copied val_metrics.json")

    # 5. Resolve tokenizer source and copy
    if verifier_meta_dir is not None:
        verifier_path = Path(verifier_meta_dir)
    else:
        candidate = cfg.get("speculators_config", {}).get("verifier", {}).get("name_or_path")
        verifier_path = Path(candidate) if candidate else Path("")
        if not verifier_path.exists():
            for fallback in ("/home/user/models/MiniMax-M2.7-FP8",
                             "/home/user/models/MiniMax-M2.7"):
                if Path(fallback).exists():
                    verifier_path = Path(fallback)
                    break

    log(f"Verifier path for tokenizer: {verifier_path}")
    copied = []
    for tk in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "vocab.json", "merges.txt", "tokenizer.model"):
        src_tk = verifier_path / tk
        if src_tk.exists():
            shutil.copy2(src_tk, out / tk)
            log(f"  copied {tk}")
            copied.append(tk)
    if not copied:
        raise RuntimeError(
            f"No tokenizer files found at {verifier_path}. "
            f"Pass --verifier-meta-dir to override."
        )

    log(f"\nDone. Run buun converter on: {out}")
    return out


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--verifier-meta-dir", default=None,
                    help="Directory holding tokenizer.json etc (default: read from config)")
    ap.add_argument("--rebake-floor", type=float, default=-65504.0,
                    help="Floor value for non-mapped rows in rebaked lm_head (default: -65504.0)")
    args = ap.parse_args()
    prep_for_buun_converter(args.src_dir, args.out_dir,
                             verifier_meta_dir=args.verifier_meta_dir,
                             rebake_floor=args.rebake_floor)


if __name__ == "__main__":
    _cli()
