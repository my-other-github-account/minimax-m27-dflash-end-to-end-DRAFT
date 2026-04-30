#!/usr/bin/env python3
"""
Prep a speculators-format DFlash checkpoint (with draft_vocab_size < target_vocab_size)
for buun-llama-cpp's convert_hf_to_gguf.py DFlashDraftModel converter.

Steps:
1. Read d2t (draft->target offsets) and lm_head [draft_V, hidden]
2. Build new lm_head [target_V, hidden] = full(-65504) then scatter draft rows to target_idx
   (the -65504 floor is the dflash-gguf-conversion skill fix for zero-row dilution)
3. Drop d2t and t2d from output
4. Flatten config: hoist transformer_layer_config keys to top level, rename
   aux_hidden_state_layer_ids -> target_layer_ids, set draft_vocab_size = target_vocab_size
5. Copy tokenizer files from the verifier path

Usage:
  python3 prep_full_for_buun_converter.py SRC_DIR OUT_DIR
"""
import sys, json, shutil, os
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
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

# 2. Load tensors, rebake lm_head, drop d2t/t2d
src_st = SRC / "model.safetensors"
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
verifier_path = Path(cfg["speculators_config"]["verifier"]["name_or_path"])
if not verifier_path.exists():
    for cand in ["/home/user/models/MiniMax-M2.7-FP8", "/home/user/models/MiniMax-M2.7"]:
        if Path(cand).exists():
            verifier_path = Path(cand)
            break
print(f"Verifier path for tokenizer: {verifier_path}")
for tk in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model"]:
    src_tk = verifier_path / tk
    if src_tk.exists():
        shutil.copy2(src_tk, OUT / tk)
        print(f"  copied {tk}")

print("\nDone. Run buun converter on:", OUT)
