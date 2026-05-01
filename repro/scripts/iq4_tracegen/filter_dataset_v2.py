#!/usr/bin/env python3
"""
Filter prompts to rows that have a corresponding trace, then SHUFFLE before saving.

This fixes the v1 bug where the saved Arrow dataset preserved worker-shard order
(A's traces first, then B's, then C's), causing speculators' 90/10 train/val split
to land disproportionately on worker C — a different ntok distribution than train.

After this fix, both train and val see proportional samples from each worker shard,
restoring proper validation-set representativeness.
"""
import os, random
from datasets import Dataset

PROMPTS_IN = os.environ.get("PROMPTS_IN", "/home/user/iq4_full_run/prompts")
TRACES = os.environ.get("TRACES", "/home/user/iq4_full_run/traces/hidden_states")
PROMPTS_OUT = os.environ.get("PROMPTS_OUT", "/home/user/iq4_full_run/prompts_dense_v2")
DENSE_TRACES = os.environ.get("DENSE_TRACES", "/home/user/iq4_full_run/traces_dense_v2/hidden_states")

random.seed(42)  # deterministic shuffle for reproducibility

# Available trace indices on disk
have = set()
for f in os.listdir(TRACES):
    if f.startswith("hs_") and f.endswith(".safetensors"):
        try:
            have.add(int(f[3:-len(".safetensors")]))
        except ValueError:
            pass
print(f"[have] {len(have)} traces on disk")

ds = Dataset.load_from_disk(PROMPTS_IN)
print(f"[in] dataset: {len(ds)} rows")

kept_orig = [i for i in range(len(ds)) if i in have]
print(f"[kept] {len(kept_orig)} rows")

# Shuffle with fixed seed
shuffled = kept_orig[:]
random.shuffle(shuffled)

filtered = ds.select(shuffled)
print(f"[shuffled] {len(filtered)} rows")

os.makedirs(PROMPTS_OUT, exist_ok=True)
filtered.save_to_disk(PROMPTS_OUT)

# Carry forward auxiliary files (vocab maps + token_freq) — Dataset.save_to_disk() does not
import shutil
for f in ["d2t.npy", "t2d.npy", "token_freq.pt"]:
    src = os.path.join(PROMPTS_IN, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(PROMPTS_OUT, f))
print(f"[save] wrote {PROMPTS_OUT}")

# Symlink farm: dense_idx_in_dataset → original trace
os.makedirs(DENSE_TRACES, exist_ok=True)
for f in os.listdir(DENSE_TRACES):
    p = os.path.join(DENSE_TRACES, f)
    if os.path.islink(p):
        os.unlink(p)

n_links = 0
for dense_idx, orig_idx in enumerate(shuffled):
    src = f"{TRACES}/hs_{orig_idx}.safetensors"
    dst = f"{DENSE_TRACES}/hs_{dense_idx}.safetensors"
    if os.path.exists(src):
        os.symlink(src, dst)
        n_links += 1
print(f"[symlinks] {n_links} dense trace links → {DENSE_TRACES}")

# Sanity: report the train/val split's worker composition
def worker_for(idx):
    if idx < 2500: return "A"
    if idx < 4500: return "B"
    return "C"

n = len(shuffled)
train_end = int(n * 0.9)
from collections import Counter
train_workers = Counter(worker_for(i) for i in shuffled[:train_end])
val_workers   = Counter(worker_for(i) for i in shuffled[train_end:])
print(f"[split] train(n={train_end}): {dict(train_workers)}")
print(f"[split] val(n={n-train_end}):  {dict(val_workers)}")
print("DONE")
