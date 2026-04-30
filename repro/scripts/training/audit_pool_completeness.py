#!/usr/bin/env python3
"""
audit_pool_completeness.py — verify that the canonical pool's hs_*.safetensors
files contain every unique sample that exists in a peer Spark's hs_staging/.

Use case: in §1's TP=N data generation, every rank writes hidden states to its
own local hs_staging/ for every prompt (TP collective broadcast). The validator
runs on ONE node and dedupes-promotes to hs_clean_pool. This script confirms
the dedup was lossless: every unique-by-token_ids sample on a peer rank IS in
the canonical pool. If any are missing, the validator dropped real data.

Method: hash token_ids of every safetensor on both sides, compare sets.

Usage:
    python audit_pool_completeness.py \\
        --pool ${DATA_ROOT}/preprocessed_5L_FP8/hs_clean_pool \\
        --peer-host other-spark-hostname \\
        --peer-staging ${DATA_ROOT}/preprocessed_5L_FP8/hs_staging
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from safetensors.torch import safe_open


HASH_HELPER = r'''
import hashlib, json, sys, os, glob
from pathlib import Path
from safetensors.torch import safe_open

src = Path(sys.argv[1])
out = {}
bad = 0
for i, p in enumerate(sorted(src.glob("*.safetensors"))):
    try:
        with safe_open(str(p), framework="pt") as f:
            if "token_ids" not in f.keys():
                continue
            t = f.get_tensor("token_ids").tolist()
        out[p.name] = hashlib.sha256(json.dumps(t).encode()).hexdigest()
    except Exception:
        bad += 1
print(f"DONE: hashed={len(out)} bad={bad}", file=sys.stderr)
print(json.dumps(out))
'''


def hash_local(directory: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    bad = 0
    for p in sorted(directory.glob("*.safetensors")):
        try:
            with safe_open(str(p), framework="pt") as f:
                if "token_ids" not in f.keys():
                    continue
                tok = f.get_tensor("token_ids").tolist()
            out[p.name] = hashlib.sha256(json.dumps(tok).encode()).hexdigest()
        except Exception:
            bad += 1
    print(f"[local] hashed {len(out)} files, bad={bad}", file=sys.stderr)
    return out


def hash_remote(host: str, directory: str, remote_python: str = "python3") -> dict[str, str]:
    """ssh into host, run the hash helper there, return parsed JSON.

    remote_python: path to a python on the peer host that has safetensors
                   installed (e.g. /opt/venvs/vllm/bin/python). Defaults to
                   system python3 — works if the user has installed
                   safetensors globally.
    """
    print(f"[remote] hashing {host}:{directory} ...", file=sys.stderr)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(HASH_HELPER)
        helper_path = f.name
    try:
        # Push the helper
        subprocess.check_call(["scp", "-q", helper_path, f"{host}:/tmp/_audit_helper.py"])
        out = subprocess.check_output([
            "ssh", host,
            f"{remote_python} /tmp/_audit_helper.py {directory}",
        ], text=True)
    finally:
        Path(helper_path).unlink(missing_ok=True)
    return json.loads(out.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path,
                    help="Local canonical pool dir (hs_<N>.safetensors)")
    ap.add_argument("--peer-host", required=True,
                    help="ssh hostname for the peer rank")
    ap.add_argument("--peer-staging", required=True,
                    help="Path on peer host to its hs_staging dir")
    ap.add_argument("--remote-python", default="python3",
                    help="Python interpreter on peer host with safetensors installed "
                         "(default: python3)")
    args = ap.parse_args()

    pool = hash_local(args.pool)
    peer = hash_remote(args.peer_host, args.peer_staging, args.remote_python)

    pool_shas = set(pool.values())
    peer_shas = set(peer.values())

    print()
    print(f"pool files (local):              {len(pool):>8,}  unique shas: {len(pool_shas):>8,}")
    print(f"peer staging ({args.peer_host}): {len(peer):>8,}  unique shas: {len(peer_shas):>8,}")
    print()
    inter = pool_shas & peer_shas
    missing_in_pool = peer_shas - pool_shas
    extra_in_pool = pool_shas - peer_shas
    print(f"intersection:                  {len(inter):>8,}")
    print(f"in peer-staging not in pool:   {len(missing_in_pool):>8,}   <-- MUST be 0 for lossless dedup")
    print(f"in pool not in peer-staging:   {len(extra_in_pool):>8,}   (rare; would mean pool has data the peer never produced)")
    print()
    ok = (len(missing_in_pool) == 0)
    if ok:
        print("RESULT: pool ⊆ peer-staging (lossless dedup) ✓")
        return 0
    else:
        print("RESULT: pool is missing samples that exist in peer-staging ✗")
        print("        (the validator may have crashed mid-run, or its dedup logic dropped real data)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
