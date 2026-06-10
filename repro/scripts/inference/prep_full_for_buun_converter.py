#!/usr/bin/env python3
"""
Prep a speculators-format DFlash checkpoint (with draft_vocab_size < target_vocab_size)
for buun-llama-cpp's convert_hf_to_gguf.py DFlashDraftModel converter.

Steps:
1. Read d2t (draft->target offsets) and lm_head [draft_V, hidden]
2. Build new lm_head [target_V, hidden] = full(-65504) then scatter draft rows to target_idx
   (the -65504 floor is the dflash-gguf-conversion skill fix for zero-row dilution)
3. Drop d2t and t2d from output
4. Copy verifier-owned embed_tokens/lm_head tensors when verifier_meta has a
   tiny model.safetensors, restoring the 71-tensor served-drafter GGUF gate
5. Flatten config: hoist transformer_layer_config keys to top level, rename
   aux_hidden_state_layer_ids -> target_layer_ids, set draft_vocab_size = target_vocab_size
6. Copy tokenizer files from the verifier path

Usage:
  python3 prep_full_for_buun_converter.py SRC_DIR OUT_DIR
"""
import sys, json, shutil, os, argparse
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("src_dir")
ap.add_argument("out_dir")
ap.add_argument("--verifier-meta-dir", default=None)
ap.add_argument("--d2t-path", default=None,
                help="draft-to-target vocab map .npy for sparse served-drafter output head")
args = ap.parse_args()

SRC = Path(args.src_dir)
OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

# 1. Read config
with open(SRC / "config.json") as f:
    cfg = json.load(f)

target_vocab = cfg["transformer_layer_config"]["vocab_size"]
draft_vocab = cfg["draft_vocab_size"]
hidden_size = cfg["transformer_layer_config"]["hidden_size"]
target_layer_ids = cfg["aux_hidden_state_layer_ids"]
print(f"target_vocab={target_vocab}, draft_vocab={draft_vocab}, hidden={hidden_size}")
print(f"target_layer_ids={target_layer_ids}")

verifier_path = Path(args.verifier_meta_dir) if args.verifier_meta_dir else Path(cfg["speculators_config"]["verifier"]["name_or_path"])

# 2. Load tensors, rebake lm_head, drop d2t/t2d
src_st = SRC / "model.safetensors"
if not src_st.exists():
    lucebox_st = SRC / "model_lucebox_layout.safetensors"
    if lucebox_st.exists():
        src_st = lucebox_st
    else:
        raise FileNotFoundError(f"expected {SRC / 'model.safetensors'} or {lucebox_st}")
out = {}
with safe_open(str(src_st), "pt") as f:
    keys = list(f.keys())
    d2t = f.get_tensor("d2t") if "d2t" in keys else None
    t2d = f.get_tensor("t2d") if "t2d" in keys else None
    lm_head = f.get_tensor("lm_head.weight") if "lm_head.weight" in keys else None

    if d2t is not None and lm_head is not None and lm_head.shape[0] == draft_vocab:
        print(f"REBAKE: lm_head {list(lm_head.shape)} -> [{target_vocab}, {hidden_size}] with -65504 floor")
        if t2d is not None:
            t_true = int(t2d.sum().item())
            print(f"  t2d.sum() = {t_true} (expected {draft_vocab}: {'OK' if t_true == draft_vocab else 'MISMATCH'})")

        # d2t[i] = OFFSET: target_id = draft_idx + d2t[draft_idx]
        target_ids = torch.arange(draft_vocab, dtype=torch.long) + d2t.to(torch.long)
        assert target_ids.min() >= 0
        assert target_ids.max() < target_vocab

        # -65504 = largest-magnitude finite bf16 negative; ensures softmax over target vocab
        # effectively zeros out non-mapped rows (per dflash-gguf-conversion skill)
        new_lm_head = torch.full((target_vocab, hidden_size), -65504.0, dtype=lm_head.dtype)
        new_lm_head[target_ids] = lm_head
        out["lm_head.weight"] = new_lm_head
        print(f"  rebaked: scattered {draft_vocab} rows into {target_vocab}-row tensor")
    elif lm_head is not None:
        print(f"lm_head already at target shape {list(lm_head.shape)} - no rebake")
        out["lm_head.weight"] = lm_head

    # Copy all other tensors except d2t, t2d, lm_head (handled above)
    for k in keys:
        if k in ("d2t", "t2d", "lm_head.weight"):
            continue
        out[k] = f.get_tensor(k)

verifier_st = verifier_path / "model.safetensors"
if verifier_st.exists():
    with safe_open(str(verifier_st), "pt") as vf:
        vkeys = set(vf.keys())
        for src_key, dst_key in [("model.embed_tokens.weight", "embed_tokens.weight"),
                                 ("lm_head.weight", "lm_head.weight")]:
            if dst_key in out:
                print(f"  keeping checkpoint-provided {dst_key}")
                continue
            if src_key not in vkeys:
                print(f"  verifier tensor absent: {src_key}")
                continue
            tensor = vf.get_tensor(src_key)
            if list(tensor.shape) != [target_vocab, hidden_size]:
                raise ValueError(f"{src_key} shape {list(tensor.shape)} != [{target_vocab}, {hidden_size}]")
            if dst_key == "lm_head.weight":
                d2t_path = Path(args.d2t_path) if args.d2t_path else verifier_path.parent / "iq4_v17_consolidated" / "prompts" / "d2t.npy"
                if d2t_path.exists():
                    d2t_np = np.load(d2t_path)
                    if d2t_np.shape != (draft_vocab,):
                        raise ValueError(f"{d2t_path} shape {d2t_np.shape} != ({draft_vocab},)")
                    d2t = torch.from_numpy(d2t_np.astype(np.int64, copy=False)).to(torch.long)
                    target_ids = torch.arange(draft_vocab, dtype=torch.long) + d2t
                    if int(target_ids.min()) < 0 or int(target_ids.max()) >= target_vocab:
                        target_ids = d2t
                    if int(target_ids.min()) < 0 or int(target_ids.max()) >= target_vocab:
                        raise ValueError(f"{d2t_path} does not map into target vocab {target_vocab}")
                    sparse = torch.full((target_vocab, hidden_size), -65504.0, dtype=tensor.dtype)
                    sparse[target_ids] = tensor[target_ids]
                    tensor = sparse
                    print(f"  rebaked verifier {src_key} -> {dst_key} using {d2t_path}")
                else:
                    print(f"  copied full verifier {src_key} -> {dst_key} (no d2t sidecar)")
            out[dst_key] = tensor
            if dst_key != "lm_head.weight":
                print(f"  copied verifier {src_key} -> {dst_key} {list(tensor.shape)}")
else:
    print(f"Verifier shared tensor file absent: {verifier_st}")

print(f"Output tensors: {len(out)} (was {len(keys)})")

save_file(out, str(OUT / "model.safetensors"), metadata={"format": "pt"})
print(f"Wrote {OUT / 'model.safetensors'}")

# 3. Flatten config for buun converter
new_cfg = {}
new_cfg["architectures"] = ["DFlashDraftModel"]
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

with open(OUT / "config.json", "w") as f:
    json.dump(new_cfg, f, indent=2)
print(f"Wrote {OUT / 'config.json'}")

# 4. Copy tokenizer files from the verifier
print(f"Verifier path for tokenizer: {verifier_path}")
for tk in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model"]:
    src_tk = verifier_path / tk
    if src_tk.exists():
        shutil.copy2(src_tk, OUT / tk)
        print(f"  copied {tk}")

print("\nDone. Run buun converter on:", OUT)
