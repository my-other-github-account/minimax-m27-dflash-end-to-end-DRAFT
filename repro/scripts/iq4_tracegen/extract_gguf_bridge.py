#!/usr/bin/env python3
"""
Extract bridge tensors from a GGUF UD-IQ4_XS verifier and write a HuggingFace-format
single-shard safetensors model directory consumable by speculators training.

This is the **GGUF-only path**: NO FP8 weights are touched. The 3 tensors needed by
speculators.load_verifier_weights() — embed_tokens, lm_head, model.norm — are
dequantized from the GGUF (Q8_0/Q6_K/F32 → bf16) and saved as a single safetensors
shard alongside the existing config.json and tokenizer files.

Tensor mapping:
  GGUF token_embd.weight   (Q8_0)   → model.embed_tokens.weight (bf16)
  GGUF output.weight       (Q6_K)   → lm_head.weight             (bf16)
  GGUF output_norm.weight  (F32)    → model.norm.weight          (bf16)

Pitfall: GGUF.dequantize() returns HF layout [vocab, hidden] natively for embeddings.
Do NOT transpose. Earlier draft transposed to [hidden, vocab] and produced garbage
embeddings, which manifested deep in attention as
"ValueError: No valid (non-padding) positions in document_ids".
"""
import os, json, sys
import torch
from safetensors.torch import save_file
from gguf import GGUFReader, quants

GGUF_DIR = os.environ.get(
    "GGUF_DIR",
    "/home/user/models/MiniMax-M2.7-GGUF/UD-IQ4_XS",
)
OUT_DIR = os.environ.get(
    "OUT_DIR",
    "/home/user/iq4_full_run/verifier_meta",
)

# Bridge tensors all live in shard 2 of MiniMax-M2.7-UD-IQ4_XS-{1..4}-of-00004.gguf.
# Adjust if the GGUF layout changes.
BRIDGE_SHARD = os.path.join(GGUF_DIR, "MiniMax-M2.7-UD-IQ4_XS-00002-of-00004.gguf")

needed = {
    "token_embd.weight":    "model.embed_tokens.weight",
    "output.weight":        "lm_head.weight",
    "output_norm.weight":   "model.norm.weight",
}

print(f"[load] reading {BRIDGE_SHARD}")
r = GGUFReader(BRIDGE_SHARD)

out_state = {}
for t in r.tensors:
    if t.name in needed:
        target_name = needed[t.name]
        print(f"[extract] {t.name}  type={t.tensor_type.name}  shape={list(t.shape)}  →  {target_name}")
        arr = quants.dequantize(t.data, t.tensor_type)  # returns HF layout [vocab, hidden]
        tensor = torch.from_numpy(arr.copy()).to(torch.bfloat16)
        out_state[target_name] = tensor
        print(f"           saved as bf16 shape={list(tensor.shape)}")

for tgt in needed.values():
    assert tgt in out_state, f"missing {tgt}"

os.makedirs(OUT_DIR, exist_ok=True)
out_file = os.path.join(OUT_DIR, "model.safetensors")
print(f"[save] writing {out_file}")
save_file(out_state, out_file)

idx = {
    "metadata": {"total_size": sum(t.numel() * t.element_size() for t in out_state.values())},
    "weight_map": {name: "model.safetensors" for name in out_state},
}
with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(idx, f, indent=2)
print("[save] wrote model.safetensors.index.json (single-shard)")

with open(os.path.join(OUT_DIR, "GGUF_PROVENANCE.txt"), "w") as f:
    f.write(f"""Bridge tensors extracted from GGUF UD-IQ4_XS (shard 2).
Source: {BRIDGE_SHARD}
Quantization → bf16 dequantization:
  token_embd.weight  (Q8_0)   → model.embed_tokens.weight
  output.weight      (Q6_K)   → lm_head.weight
  output_norm.weight (F32)    → model.norm.weight

NO FP8 WEIGHTS USED. Pure GGUF-only training path.
""")
print("DONE")
