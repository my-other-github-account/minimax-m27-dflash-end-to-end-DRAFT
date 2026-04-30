#!/usr/bin/env python3
"""
Verify a DFlash drafter GGUF has the expected metadata.

Usage:
  python3 verify_gguf.py /path/to/drafter.gguf
"""
import sys
from gguf import GGUFReader

REQUIRED = {
    "general.architecture": "dflash-draft",
    "tokenizer.ggml.pre": "minimax-m2",
    "tokenizer.ggml.model": "gpt2",
}
INT_REQUIRED = {
    "dflash-draft.dflash.block_size": 8,
    "dflash-draft.dflash.mask_token_id": 200054,
    "dflash-draft.dflash.n_target_features": 15360,
}
ARRAY_REQUIRED = {
    "dflash-draft.dflash.target_layer_ids": [2, 16, 30, 45, 59],
}

def get_str(field):
    return field.parts[field.data[0]].tobytes().decode()

def get_int(field):
    raw = field.parts[field.data[0]]
    return int(raw[0]) if hasattr(raw, "__getitem__") else int(raw)

def get_int_array(field):
    return [int(field.parts[i][0]) for i in field.data]

def main():
    path = sys.argv[1]
    r = GGUFReader(path)
    print(f"Inspecting: {path}")
    failures = []

    for k, expected in REQUIRED.items():
        f = r.get_field(k)
        if f is None:
            print(f"  ❌ MISSING {k}")
            failures.append(k)
            continue
        actual = get_str(f)
        ok = actual == expected
        print(f"  {'✅' if ok else '❌'} {k}: {actual!r} (expected {expected!r})")
        if not ok: failures.append(k)

    for k, expected in INT_REQUIRED.items():
        f = r.get_field(k)
        if f is None:
            print(f"  ❌ MISSING {k}")
            failures.append(k)
            continue
        actual = get_int(f)
        ok = actual == expected
        print(f"  {'✅' if ok else '❌'} {k}: {actual} (expected {expected})")
        if not ok: failures.append(k)

    for k, expected in ARRAY_REQUIRED.items():
        f = r.get_field(k)
        if f is None:
            print(f"  ❌ MISSING {k}")
            failures.append(k)
            continue
        actual = get_int_array(f)
        ok = actual == expected
        print(f"  {'✅' if ok else '❌'} {k}: {actual} (expected {expected})")
        if not ok: failures.append(k)

    n_tensors = len(r.tensors)
    ok = n_tensors == 60
    print(f"  {'✅' if ok else '❌'} tensor count: {n_tensors} (expected 60)")
    if not ok: failures.append("n_tensors")

    if failures:
        print(f"\n❌ FAILED: {len(failures)} mismatches: {failures}")
        sys.exit(1)
    else:
        print("\n✅ All metadata checks passed.")

if __name__ == "__main__":
    main()
