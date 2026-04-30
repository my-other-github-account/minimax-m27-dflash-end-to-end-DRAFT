#!/usr/bin/env python3
"""Transform a speculators-trained DFlash drafter checkpoint into the
input schema expected by the PR #22105 fork of convert_hf_to_gguf.py,
WITH proper d2t rebake of lm_head into the target vocab.

Why this exists: PR #22105's convert_hf_to_gguf.py DFlashModel class doesn't
know about d2t/t2d/lm_head — it falls back to Qwen3 logic that drops
draft-vocab `lm_head.weight` and silently relies on tied embed_tokens for
output projection. Result: the trained lm_head is lost, and at runtime the
drafter's output logits are projected via embed_tokens (= input embedding),
producing dramatically lower per-position accept rates than training measured.

The fix (per the dflash-gguf-conversion skill, "d2t rebake into target-vocab
lm_head"): construct a target-vocab-shaped lm_head by scattering the trained
draft-vocab lm_head into rows indexed by `d2t[i] + i`, with all non-mapped
rows set to a very-negative finite floor (NOT zero) so they can never win
argmax. Use -65504 (BF16 most-negative) for safety.

Speculators stores hyperparameters under nested 'transformer_layer_config'
plus a 'speculators_config' section, with target-layer ids stashed as
'aux_hidden_state_layer_ids' and the mask token id at the top level.

PR #22105's converter expects a flat Qwen3-style config (since
DFlashModel inherits from Qwen3Model) plus a 'dflash_config' nested object
containing 'target_layer_ids' and 'mask_token_id', and a top-level 'block_size'.

Usage:
    python prep_for_pr22105_v2_with_d2t_rebake.py \\
        --in   /path/to/speculators-checkpoint_best \\
        --out  /path/to/prepped-output-dir

Reads <in>/config.json + <in>/model.safetensors, writes
<out>/config.json (transformed) + <out>/model.safetensors (with rebaked
lm_head, d2t/t2d dropped).
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


NEG_FLOOR = -65504.0  # BF16 most-negative value; survives any common quant


def transform_config(spec_cfg: dict) -> dict:
    tlc = spec_cfg["transformer_layer_config"]
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "vocab_size": tlc["vocab_size"],
        "hidden_size": tlc["hidden_size"],
        "intermediate_size": tlc["intermediate_size"],
        "num_hidden_layers": tlc["num_hidden_layers"],
        "num_attention_heads": tlc["num_attention_heads"],
        "num_key_value_heads": tlc["num_key_value_heads"],
        "head_dim": tlc["head_dim"],
        "hidden_act": tlc["hidden_act"],
        "max_position_embeddings": tlc["max_position_embeddings"],
        "rms_norm_eps": tlc["rms_norm_eps"],
        "rope_theta": tlc["rope_parameters"]["rope_theta"],
        "tie_word_embeddings": tlc.get("tie_word_embeddings", False),
        "torch_dtype": spec_cfg.get("dtype", "bfloat16"),
        "block_size": spec_cfg["block_size"],
        "dflash_config": {
            "target_layer_ids": spec_cfg["aux_hidden_state_layer_ids"],
            "mask_token_id":    spec_cfg["mask_token_id"],
        },
    }


def rebake_lm_head(lm_head_draft: torch.Tensor,
                   d2t: torch.Tensor,
                   target_vocab_size: int) -> torch.Tensor:
    """Scatter draft-vocab lm_head [draft_V, hidden] into target-vocab shape
    [target_V, hidden] using d2t offsets, with non-mapped rows = NEG_FLOOR.

    Per speculators convention: target_token_id = i + d2t[i] for draft index i.
    """
    draft_V, hidden = lm_head_draft.shape
    assert d2t.shape == (draft_V,), f"d2t shape {d2t.shape} != ({draft_V},)"
    assert d2t.dtype in (torch.int32, torch.int64), f"d2t dtype {d2t.dtype} not int"

    out_dtype = lm_head_draft.dtype
    expanded = torch.full(
        (target_vocab_size, hidden),
        NEG_FLOOR,
        dtype=out_dtype,
    )
    # target_id = i + d2t[i]
    indices = torch.arange(draft_V, dtype=torch.int64) + d2t.to(torch.int64)
    if indices.min().item() < 0 or indices.max().item() >= target_vocab_size:
        bad = ((indices < 0) | (indices >= target_vocab_size)).sum().item()
        raise ValueError(
            f"d2t produces {bad} out-of-range target ids "
            f"(min={indices.min().item()}, max={indices.max().item()}, "
            f"target_V={target_vocab_size})"
        )
    expanded[indices] = lm_head_draft
    return expanded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("--no-rebake", action="store_true",
                    help="Skip lm_head rebake (testing only; produces broken drafter)")
    a = ap.parse_args()

    src_cfg = json.loads((a.src / "config.json").read_text())
    out_cfg = transform_config(src_cfg)
    target_V = out_cfg["vocab_size"]
    a.dst.mkdir(parents=True, exist_ok=True)
    (a.dst / "config.json").write_text(json.dumps(out_cfg, indent=2) + "\n")
    print(f"[1/3] wrote config.json (target_vocab={target_V})")

    # Read source safetensors
    src_st = a.src / "model.safetensors"
    print(f"[2/3] reading {src_st}")
    tensors = {}
    with safe_open(str(src_st), framework="pt") as f:
        keys = list(f.keys())
        print(f"      {len(keys)} source tensors")
        for k in keys:
            tensors[k] = f.get_tensor(k)

    # Drop t2d (not used by runtime), keep d2t for rebake
    skip = {"t2d"}
    if "d2t" not in tensors:
        raise RuntimeError("d2t tensor missing from source — is this really a "
                           "speculators-trained DFlash checkpoint with reduced draft vocab?")
    if "lm_head.weight" not in tensors:
        raise RuntimeError("lm_head.weight missing — cannot rebake")

    if not a.no_rebake:
        print("[3/3] rebaking lm_head [draft_V] -> [target_V] with NEG_FLOOR=-65504")
        lh_draft = tensors["lm_head.weight"]
        d2t = tensors["d2t"]
        print(f"      lm_head_draft shape: {tuple(lh_draft.shape)}")
        print(f"      d2t shape: {tuple(d2t.shape)}, dtype: {d2t.dtype}")
        lh_target = rebake_lm_head(lh_draft, d2t, target_V)
        non_neg = (lh_target != NEG_FLOOR).any(dim=1).sum().item()
        print(f"      rebaked lm_head shape: {tuple(lh_target.shape)}, "
              f"non-floor rows: {non_neg} (expected: {lh_draft.shape[0]})")
        tensors["lm_head.weight"] = lh_target

    # Drop d2t and t2d — runtime doesn't honor them, and the rebake replaces their function
    drop = {"d2t", "t2d"}
    out_tensors = {k: v for k, v in tensors.items() if k not in drop}
    print(f"      writing {len(out_tensors)} tensors (dropped: {sorted(drop)})")

    save_file(out_tensors, str(a.dst / "model.safetensors"))
    print(f"\nWrote {a.dst}/config.json + {a.dst}/model.safetensors")
    print("Now run convert_hf_to_gguf.py against this directory.")


if __name__ == "__main__":
    main()
