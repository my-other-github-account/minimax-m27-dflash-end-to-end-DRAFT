#!/usr/bin/env python3
"""Transform a speculators-trained DFlash drafter checkpoint into the
input schema expected by the PR #22105 fork of convert_hf_to_gguf.py.

Speculators stores hyperparameters under nested 'transformer_layer_config'
plus a 'speculators_config' section, with target-layer ids stashed as
'aux_hidden_state_layer_ids' and the mask token id at the top level.

PR #22105's converter expects a flat Qwen3-style config (since
DFlashModel inherits from Qwen3Model) plus a 'dflash_config' nested object
containing 'target_layer_ids' and 'mask_token_id', and a top-level 'block_size'.

Usage:
    python prep_for_pr22105_converter.py \\
        --in   /path/to/speculators-checkpoint_best \\
        --out  /path/to/prepped-output-dir

Reads <in>/config.json + <in>/model.safetensors, writes
<out>/config.json (transformed) and copies model.safetensors verbatim.
"""
import argparse, json, shutil
from pathlib import Path

def transform(spec_cfg: dict) -> dict:
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    a = ap.parse_args()

    src_cfg = json.loads((a.src / "config.json").read_text())
    out_cfg = transform(src_cfg)
    a.dst.mkdir(parents=True, exist_ok=True)
    (a.dst / "config.json").write_text(json.dumps(out_cfg, indent=2) + "\n")

    # Copy / symlink the safetensors verbatim — the converter reads it directly.
    src_st = a.src / "model.safetensors"
    dst_st = a.dst / "model.safetensors"
    if dst_st.exists():
        dst_st.unlink()
    # Use hardlink if same FS, else fallback to copy
    try:
        dst_st.hardlink_to(src_st)
    except OSError:
        shutil.copy2(src_st, dst_st)

    print(f"Wrote {a.dst}/config.json")
    print(f"Linked {a.dst}/model.safetensors -> {src_st}")
    print(f"Now run:")
    print(f"  python convert_hf_to_gguf.py {a.dst} \\")
    print(f"    --outfile <out>.gguf --outtype f16 \\")
    print(f"    --target-model-dir <path-to-target-tokenizer-dir>")

if __name__ == "__main__":
    main()
