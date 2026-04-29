#!/usr/bin/env python3
"""Deep-audit the clean pool: pick recent files (post-R33), verify they really are clean."""
import os, time, random, statistics
from safetensors.torch import load_file

import os
POOL = os.environ.get("CLEAN_POOL_DIR", "./hs_clean_pool")
all_files = sorted([f for f in os.listdir(POOL) if f.endswith(".safetensors")],
                   key=lambda f: os.path.getmtime(os.path.join(POOL, f)))
total = len(all_files)
print(f"Total files in pool: {total}")

now = time.time()
recent = [f for f in all_files if now - os.path.getmtime(os.path.join(POOL, f)) < 3600]
print(f"Files added in last hour (post-R33): {len(recent)}")
if recent:
    print(f"  Oldest in last hour: {recent[0]} ({(now - os.path.getmtime(os.path.join(POOL, recent[0])))/60:.1f} min ago)")
    print(f"  Newest:              {recent[-1]} ({(now - os.path.getmtime(os.path.join(POOL, recent[-1])))/60:.1f} min ago)")

# Deep audit
sample = random.sample(recent, min(150, len(recent)))
print(f"\nDEEP AUDIT: examining {len(sample)} random recent samples")
problems = []
per_layer_stats = [[],[],[],[]]
shape_examples = []

for f in sample:
    p = os.path.join(POOL, f)
    try:
        hs = load_file(p)['hidden_states']
        if hs.dim() != 3 or hs.shape[1] != 4:
            problems.append((f, f'odd shape {hs.shape}'))
            continue
        shape_examples.append(tuple(hs.shape))
        for l in range(4):
            slice_ = hs[:, l, :]
            if slice_.isnan().any().item():
                problems.append((f, f'layer{l} NaN'))
                break
            if slice_.isinf().any().item():
                problems.append((f, f'layer{l} Inf'))
                break
            per_layer_stats[l].append(slice_.float().std().item())
    except Exception as e:
        problems.append((f, f'load error: {e}'))

print(f"\nProblems found: {len(problems)} / {len(sample)}")
for f, reason in problems[:10]:
    print(f"  {f}: {reason}")

print(f"\nShape examples (seq_len varies): first 5 = {shape_examples[:5]}")

print(f"\nPer-layer std stats:")
labels = ["target=2 (shallow)", "target=31 (mid)", "target=60 (deep)", "target=62 (last)"]
for l in range(4):
    s = per_layer_stats[l]
    if s:
        print(f"  L{l} {labels[l]}: count={len(s)}, mean_std={statistics.mean(s):.3f}, min={min(s):.3f}, max={max(s):.3f}")
