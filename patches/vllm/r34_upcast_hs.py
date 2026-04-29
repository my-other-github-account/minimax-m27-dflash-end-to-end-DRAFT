#!/usr/bin/env python3
"""R34: Upcast hidden_states to float32 before saving to disk.

Rationale: NVFP4 verifier produces hidden states with a wide dynamic range at deep
layers (>4096 mantissa). Storing as bf16 truncates ~5 significant bits below the
exponent threshold. Drafter training quality depends on the precision of the
verifier_lm_head(verifier_norm(verifier_last)) target signal — bf16 rounding noise
contaminates this. Storing as fp32 doubles disk cost (~150KB → ~300KB per sample,
240GB total for 800K samples — well within disk budget) but fully preserves the
verifier's signal. Trainer downcasts to bf16 at load time, but at least the cached
target signal is faithful.

Marker: R34_UPCAST_HS
"""
import sys
p = sys.argv[1]
src = open(p).read()
if "R34_UPCAST_HS" in src:
    print(f"already patched: {p}")
    sys.exit(0)
OLD = '''            hidden_states = extract_from_kv_cache(
                kv_layer, request.slot_mapping, request.token_ids.shape[0]
            )
            tensors = {
                "hidden_states": hidden_states.detach().cpu(),
                "token_ids": request.token_ids.detach().cpu(),
            }
            safetensors.torch.save_file(tensors, request.filename)'''
NEW = '''            hidden_states = extract_from_kv_cache(
                kv_layer, request.slot_mapping, request.token_ids.shape[0]
            )
            # R34_UPCAST_HS: store hidden states as fp32 for max precision in training targets
            # NVFP4 verifier saturates bf16 mantissa at deep layers; fp32 preserves signal
            tensors = {
                "hidden_states": hidden_states.detach().to(torch.float32).cpu(),
                "token_ids": request.token_ids.detach().cpu(),
            }
            safetensors.torch.save_file(tensors, request.filename)'''
assert OLD in src, "OLD block not found"
# also need 'import torch' at top — check
if "import torch" not in src:
    src = "import torch\n" + src
src = src.replace(OLD, NEW)
open(p, "w").write(src)
print(f"patched: {p}")
