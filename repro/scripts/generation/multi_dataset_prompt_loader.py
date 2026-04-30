#!/usr/bin/env python3
"""
DFlash tracegen multi-dataset prompt mixer.

Streams from 4 unblocked HF datasets (Nemotron + lmsys gated until HF_TOKEN provided),
normalizes to ShareGPT-style {"conversations":[...]} JSONL,
length-filters at character level (rough, prepare_data does final tokenization),
hash-dedups against persistent seen_prompt_hashes.json,
mixes proportionally with random seeded sampling.

Usage:
    python3 multi_dataset_prompt_loader.py \
        --output ${WORKSPACE}/cache/mixed_<ts>/raw.jsonl \
        --total 10000 \
        --seen-set ${LOOP_WORKSPACE}/state/seen_prompt_hashes.json
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from datasets import load_dataset

# Pivoted proportions (Nemotron 55% + lmsys 7% redistributed; gated, need HF_TOKEN)
DATASETS = [
    {
        "id": "theblackcat102/evol-codealpaca-v1",
        "split": "train",
        "proportion": 0.40,
        "kind": "instruction_output",
        "fields": ("instruction", "output"),
    },
    {
        "id": "Open-Orca/OpenOrca",
        "split": "train",
        "proportion": 0.25,
        "kind": "openorca",
    },
    {
        "id": "teknium/OpenHermes-2.5",
        "split": "train",
        "proportion": 0.25,
        "kind": "conversations",
    },
    {
        "id": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "proportion": 0.10,
        "kind": "messages",
    },
]


def normalize(sample, kind, dataset_id):
    """Return ShareGPT-style {"conversations":[{"from":..., "value":...}]} or None."""
    if kind == "instruction_output":
        instr = sample.get("instruction", "")
        out = sample.get("output", "")
        if not instr or not out:
            return None
        return [
            {"from": "human", "value": instr},
            {"from": "gpt", "value": out},
        ]
    if kind == "openorca":
        # OpenOrca: id, system_prompt, question, response
        q = sample.get("question", "")
        r = sample.get("response", "")
        if not q or not r:
            return None
        sysp = sample.get("system_prompt", "")
        convs = []
        if sysp:
            convs.append({"from": "system", "value": sysp})
        convs.append({"from": "human", "value": q})
        convs.append({"from": "gpt", "value": r})
        return convs
    if kind == "conversations":
        # OpenHermes-2.5: list of {"from": "human|gpt|system", "value": "..."}
        # May have extra fields like "weight" — strip to just from/value for schema consistency
        convs = sample.get("conversations")
        if not convs or not isinstance(convs, list):
            return None
        cleaned = []
        for t in convs:
            f = t.get("from", "")
            v = t.get("value", "")
            if f and v:
                cleaned.append({"from": f, "value": v})
        return cleaned if cleaned else None
    if kind == "messages":
        # ultrachat: [{"role": "user|assistant", "content": "..."}]
        msgs = sample.get("messages")
        if not msgs:
            return None
        out = []
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if not content:
                continue
            from_role = "human" if role == "user" else "gpt" if role == "assistant" else role
            out.append({"from": from_role, "value": content})
        return out if out else None
    return None


def first_user_text(convs):
    for t in convs:
        if t.get("from") in ("human", "user"):
            return t.get("value", "")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--total", type=int, default=10000)
    ap.add_argument("--seen-set", required=True)
    ap.add_argument("--min-chars", type=int, default=100)  # ~50 tokens lower bound
    ap.add_argument("--max-chars", type=int, default=8000)  # ~1800 tokens upper bound
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    seed = args.seed or int(time.time())
    random.seed(seed)
    print(f"[mixer] seed={seed}", file=sys.stderr)

    # Load seen-set
    seen_path = Path(args.seen_set)
    seen_path.parent.mkdir(parents=True, exist_ok=True)
    if seen_path.exists():
        seen = set(json.loads(seen_path.read_text()))
        print(f"[mixer] loaded {len(seen)} seen hashes from {seen_path}", file=sys.stderr)
    else:
        seen = set()
        print(f"[mixer] starting fresh seen-set", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute target counts per dataset
    targets = {d["id"]: max(1, int(args.total * d["proportion"])) for d in DATASETS}
    written = {d["id"]: 0 for d in DATASETS}
    rejected = {d["id"]: {"len": 0, "dup": 0, "norm": 0} for d in DATASETS}

    out_f = out_path.open("w")
    new_hashes = set()
    total_written = 0

    for d in DATASETS:
        target = targets[d["id"]]
        print(f"[mixer] {d['id']}: target {target}", file=sys.stderr)

        try:
            ds_iter = load_dataset(d["id"], split=d["split"], streaming=True)
            ds_iter = ds_iter.shuffle(seed=seed, buffer_size=2000)
        except Exception as e:
            print(f"[mixer] FAIL load {d['id']}: {e}", file=sys.stderr)
            continue

        seen_local = 0
        for sample in ds_iter:
            seen_local += 1
            if written[d["id"]] >= target:
                break
            if seen_local > target * 50:  # safety: don't iterate forever
                print(f"[mixer] {d['id']}: bailing at {seen_local} samples seen", file=sys.stderr)
                break

            convs = normalize(sample, d["kind"], d["id"])
            if not convs:
                rejected[d["id"]]["norm"] += 1
                continue

            user_text = first_user_text(convs)
            if not (args.min_chars <= len(user_text) <= args.max_chars):
                rejected[d["id"]]["len"] += 1
                continue

            h = hashlib.sha256(user_text.encode("utf-8", errors="replace")).hexdigest()
            if h in seen or h in new_hashes:
                rejected[d["id"]]["dup"] += 1
                continue

            new_hashes.add(h)
            row = {"conversations": convs, "_source_dataset": d["id"], "_prompt_hash": h}
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written[d["id"]] += 1
            total_written += 1

    out_f.close()

    # Update seen-set
    seen.update(new_hashes)
    seen_path.write_text(json.dumps(sorted(seen)))
    print(f"[mixer] wrote {total_written} samples to {out_path}", file=sys.stderr)
    print(f"[mixer] updated seen-set: {len(seen)} total hashes", file=sys.stderr)
    print(f"[mixer] per-dataset written: {written}", file=sys.stderr)
    print(f"[mixer] per-dataset rejected: {rejected}", file=sys.stderr)


if __name__ == "__main__":
    main()
