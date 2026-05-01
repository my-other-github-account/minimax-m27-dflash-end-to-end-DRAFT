#!/usr/bin/env python3
"""Apples-to-apples cosine + norm comparison: IQ4 vs FP8 reference for matched indices."""
import sys
from pathlib import Path
import torch
from safetensors import safe_open

IQ4 = Path("/home/user/iq4_tracegen/traces/hidden_states")
FP8 = Path("/home/user/iq4_tracegen/fp8_ref_traces")

def load(path):
    with safe_open(str(path), "pt") as f:
        return f.get_tensor("hidden_states"), f.get_tensor("token_ids")

iq4_idx = sorted(int(p.stem.split("_")[1]) for p in IQ4.glob("hs_*.safetensors"))
fp8_idx = sorted(int(p.stem.split("_")[1]) for p in FP8.glob("hs_*.safetensors"))
common = sorted(set(iq4_idx) & set(fp8_idx))
print(f"Comparing {len(common)} matched indices: {common}", flush=True)
print(f"\n{'idx':>5} {'ntok':>5}  per-layer cosine [L0..L5]              tok_match  per-layer norm-ratio iq4/fp8", flush=True)
print("-" * 110, flush=True)

cos_acc = [0.0]*6
ct = 0
norm_acc_iq4 = [0.0]*6
norm_acc_fp8 = [0.0]*6
seq_match = 0
for i in common:
    iq4_hs, iq4_tk = load(IQ4 / f"hs_{i}.safetensors")
    fp8_hs, fp8_tk = load(FP8 / f"hs_{i}.safetensors")
    
    # Token match check
    n_min = min(iq4_tk.shape[0], fp8_tk.shape[0])
    tok_eq_prefix = int((iq4_tk[:n_min] == fp8_tk[:n_min]).all().item())
    seq_match += int(iq4_tk.shape[0] == fp8_tk.shape[0])
    if tok_eq_prefix == 0:
        print(f"  hs_{i}: TOKEN MISMATCH iq4_seq={iq4_tk.shape[0]} fp8_seq={fp8_tk.shape[0]}", flush=True)
        continue
    
    # Truncate to common length
    n = n_min
    iq4_hs_t = iq4_hs[:n].float()
    fp8_hs_t = fp8_hs[:n].float()
    
    # Per-layer cosine averaged over tokens
    cos_per_layer = []
    for L in range(6):
        a = iq4_hs_t[:, L, :]
        b = fp8_hs_t[:, L, :]
        num = (a * b).sum(dim=-1)
        den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-9
        c = (num / den).mean().item()
        cos_per_layer.append(c)
        cos_acc[L] += c
    
    # Per-layer mean norm
    iq4_norms = [iq4_hs_t[:, L, :].norm(dim=-1).mean().item() for L in range(6)]
    fp8_norms = [fp8_hs_t[:, L, :].norm(dim=-1).mean().item() for L in range(6)]
    for L in range(6):
        norm_acc_iq4[L] += iq4_norms[L]
        norm_acc_fp8[L] += fp8_norms[L]
    ratios = [iq4_norms[L]/fp8_norms[L] for L in range(6)]
    ct += 1
    cos_str = " ".join(f"{c:.3f}" for c in cos_per_layer)
    rat_str = " ".join(f"{r:.2f}" for r in ratios)
    print(f"  {i:>4} {n:>5}  [{cos_str}]  prefix={tok_eq_prefix}  [{rat_str}]", flush=True)

if ct:
    cos_mean = [c/ct for c in cos_acc]
    iq4_n_mean = [n/ct for n in norm_acc_iq4]
    fp8_n_mean = [n/ct for n in norm_acc_fp8]
    print(f"\n--- AGGREGATE over {ct} samples ---", flush=True)
    print(f"  Mean per-layer cosine: {[f'{c:.4f}' for c in cos_mean]}", flush=True)
    print(f"  Mean IQ4 norm:         {[f'{v:.1f}' for v in iq4_n_mean]}", flush=True)
    print(f"  Mean FP8 norm:         {[f'{v:.1f}' for v in fp8_n_mean]}", flush=True)
    print(f"  Mean ratio iq4/fp8:    {[f'{a/b:.3f}' for a,b in zip(iq4_n_mean, fp8_n_mean)]}", flush=True)
    print(f"  Sequence-length matches: {seq_match}/{ct}", flush=True)
    
    # Verdict per layer
    print(f"\n--- Verdict per layer ---", flush=True)
    for L in range(6):
        c = cos_mean[L]
        nr = iq4_n_mean[L]/fp8_n_mean[L]
        verdict = "✅" if c >= 0.95 else ("⚠️ " if c >= 0.85 else "❌")
        print(f"  L{L} (target_layer={[2,16,30,45,59,'LAST'][L]}): cos={c:.4f} norm_ratio={nr:.3f} {verdict}", flush=True)
