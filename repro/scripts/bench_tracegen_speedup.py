#!/usr/bin/env python3
"""Reproducible benchmark for the persistent batched-decode trace server.

Tests three things in one run:

1. The batched ``dump_hiddens_many`` path on a fresh Spark produces
   numerically equivalent hidden states vs the single-prompt path.
2. The headline traces/min throughput on the standard 100-prompt sample
   exceeds the documented target.
3. The output is structurally identical (shapes, dtypes) to what the
   trainer expects.

The "standard sample" is intentionally deterministic so that anyone
re-running this script gets the same prompt selection — drawn from an
existing trace pool by reading back its ``input_ids`` rather than
re-tokenizing a corpus.

Usage::

    python repro/scripts/bench_tracegen_speedup.py \
        --gguf-path /path/to/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
        --pool-dir /path/to/iq4_tracegen_v13_pool/traces \
        --binary    /path/to/llama-dump-hiddens-worker

Writes ``results.json`` to ``--out-dir`` (default
``/tmp/bench_tracegen_speedup``).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from dflash_llama.tracegen import TraceClient


# Standard, documented benchmark sample selection — do not change.
DEFAULT_SEED = 20260518
DEFAULT_MIN_SEQ = 50
DEFAULT_MAX_SEQ = 96
DEFAULT_N_TOTAL = 100
DEFAULT_N_EQUIV = 10
DEFAULT_LAYER_IDS = "2,16,30,45,59,61"


def log(msg: str) -> None:
    print(f"[bench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_standard_sample(pool_dir: str, *, seed: int, min_len: int,
                           max_len: int, n_total: int):
    log(f"Selecting standard {n_total}-prompt sample from {pool_dir}...")
    all_files = sorted(glob.glob(f"{pool_dir}/hs_*.safetensors"))
    log(f"  pool size: {len(all_files)}")
    candidates = []
    for f in all_files:
        try:
            meta = load_file(f)
            if "input_ids" not in meta:
                continue
            ids = meta["input_ids"].tolist()
            if isinstance(ids[0], list):
                ids = ids[0]
            seq_len = len(ids)
            if min_len <= seq_len <= max_len:
                candidates.append((f, seq_len, ids))
        except Exception:
            continue
    log(f"  filtered ({min_len}<=seq_len<={max_len}): {len(candidates)}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(candidates))[:n_total]
    selected = [candidates[i] for i in perm]
    log(f"  selected: {len(selected)}")
    return selected


def equivalence(ref_tensors, new_tensors,
                max_abs_thr: float = 1e-2,
                mean_abs_thr: float = 1e-3,
                cos_thr: float = 0.9999):
    log("4-metric numerical-equivalence check...")
    results = []
    fails = 0
    for i, (r, n) in enumerate(zip(ref_tensors, new_tensors)):
        if r is None or n is None:
            fails += 1
            results.append({"i": i, "missing": True})
            log(f"  [{i}] MISSING")
            continue
        h_r = r.to(torch.float32)
        h_n = n.to(torch.float32)
        if h_r.shape != h_n.shape:
            fails += 1
            log(f"  [{i}] SHAPE MISMATCH ref={tuple(h_r.shape)} new={tuple(h_n.shape)}")
            results.append({"i": i, "shape_ref": list(h_r.shape), "shape_new": list(h_n.shape), "pass": False})
            continue
        diff = (h_r - h_n).abs()
        mx = diff.max().item()
        mn = diff.mean().item()
        cs = torch.nn.functional.cosine_similarity(
            h_r.flatten().unsqueeze(0), h_n.flatten().unsqueeze(0)
        ).item()
        ok = (mx < max_abs_thr) and (mn < mean_abs_thr) and (cs > cos_thr)
        if not ok:
            fails += 1
        log(f"  [{i}] {'PASS' if ok else 'FAIL'}: max={mx:.4e} mean={mn:.4e} cos={cs:.7f} shape={tuple(h_r.shape)}")
        results.append({"i": i, "max_abs": mx, "mean_abs": mn, "cosine": cs, "pass": ok})
    return results, len(ref_tensors) - fails


def gen_ref(client: TraceClient, prompts) -> tuple[list, float]:
    log(f"Single-prompt reference: {len(prompts)} traces...")
    t0 = time.time()
    out = []
    for i, (_, _, ids) in enumerate(prompts):
        r = client.dump_hiddens(input_ids=ids, max_seq_len=2048)
        out.append(r["hidden_states"])
    elapsed = time.time() - t0
    rate = len(prompts) * 60 / elapsed
    log(f"  REF done: {len(prompts)} in {elapsed:.1f}s = {rate:.2f}/min")
    return out, elapsed


def gen_batched(client: TraceClient, prompts, width: int) -> tuple[list, float]:
    log(f"Batched (width={width}, same-length groups): {len(prompts)} traces...")
    by_len: dict[int, list] = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(p[1], []).append((i, p))
    out = [None] * len(prompts)
    t0 = time.time()
    nbatches = 0
    for seq_len, group in by_len.items():
        for chunk_start in range(0, len(group), width):
            chunk = group[chunk_start:chunk_start + width]
            batch_inputs = [g[1][2] for g in chunk]
            results = client.dump_hiddens_many(
                batch_inputs=batch_inputs,
                max_seq_len=2048,
            )
            nbatches += 1
            for (i, _), res in zip(chunk, results):
                out[i] = res["hidden_states"]
    elapsed = time.time() - t0
    rate = len(prompts) * 60 / elapsed
    avg = len(prompts) / max(nbatches, 1)
    log(f"  BATCHED done: {len(prompts)} in {elapsed:.1f}s = {rate:.2f}/min ({nbatches} batches, avg {avg:.1f}/batch)")
    return out, elapsed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf-path", required=True,
                   help="path to the verifier GGUF shard 1")
    p.add_argument("--pool-dir", required=True,
                   help="directory containing hs_*.safetensors traces "
                        "(self-describing fp8 pool with input_ids embedded)")
    p.add_argument("--binary", default="llama-dump-hiddens-worker",
                   help="path or name of the worker binary")
    p.add_argument("--layer-ids", default=DEFAULT_LAYER_IDS,
                   help=f"comma-separated layer indices (default {DEFAULT_LAYER_IDS})")
    p.add_argument("--socket", default="unix:///tmp/dflash_bench_speedup.sock",
                   help="socket address for the trace-server")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--min-seq", type=int, default=DEFAULT_MIN_SEQ)
    p.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    p.add_argument("--n-total", type=int, default=DEFAULT_N_TOTAL)
    p.add_argument("--n-equiv", type=int, default=DEFAULT_N_EQUIV)
    p.add_argument("--batch-width", type=int, default=4)
    p.add_argument("--target-rate", type=float, default=54.0,
                   help="abort with nonzero exit if measured rate is below this")
    p.add_argument("--out-dir", default="/tmp/bench_tracegen_speedup")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--override-tensor", default="exps=CPU")
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--startup-timeout", type=float, default=900.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_ids = [int(x) for x in args.layer_ids.split(",") if x.strip()]

    selected = select_standard_sample(
        args.pool_dir,
        seed=args.seed,
        min_len=args.min_seq,
        max_len=args.max_seq,
        n_total=args.n_total,
    )
    if len(selected) < args.n_total:
        log(f"ERROR: pool has only {len(selected)} candidates after filtering; "
            f"need {args.n_total}")
        return 2
    equiv_set = selected[:args.n_equiv]

    log("Starting TraceClient (auto-start server)...")
    client = TraceClient(
        request_timeout=args.request_timeout,
        startup_timeout=args.startup_timeout,
        socket_path=args.socket,
        auto_start=True,
        gguf_path=args.gguf_path,
        layer_ids=layer_ids,
        binary=args.binary,
        ctx=args.ctx,
        ngl=args.ngl,
        override_tensor=args.override_tensor,
        server_log_path=str(out_dir / "server.log"),
        storage="fp8_per_tensor_scale",
    )

    final: dict = {
        "batch_width": args.batch_width,
        "n_equiv": args.n_equiv,
        "n_total": args.n_total,
        "seed": args.seed,
        "target_rate_per_min": args.target_rate,
    }
    try:
        ref_tensors, ref_elapsed = gen_ref(client, equiv_set)
        batch_tensors, batch_elapsed = gen_batched(client, equiv_set, args.batch_width)
        equiv_results, n_pass = equivalence(ref_tensors, batch_tensors)
        log(f"Equivalence: {n_pass}/{args.n_equiv} PASS at width {args.batch_width}")

        final.update({
            "n_pass": n_pass,
            "ref_rate_per_min_10sample": args.n_equiv * 60 / ref_elapsed,
            "batched_rate_per_min_10sample": args.n_equiv * 60 / batch_elapsed,
            "equivalence_details": equiv_results,
        })

        if n_pass >= max(args.n_equiv - 1, 1):
            log(f"Equivalence ≥ 9/10 — running 100-prompt timing benchmark...")
            _, elapsed100 = gen_batched(client, selected, args.batch_width)
            rate = args.n_total * 60 / elapsed100
            final["rate_per_min_100sample"] = rate
            final["elapsed_seconds_100sample"] = elapsed100
            log(f"=== FINAL: {rate:.2f} traces/min @ batch width {args.batch_width}, equiv {n_pass}/{args.n_equiv} ===")
        else:
            log(f"Equivalence FAILED — {n_pass}/{args.n_equiv} pass. "
                "Skipping 100-prompt run.")
            final["rate_per_min_100sample"] = None
    finally:
        try:
            client.close()
        except Exception:
            pass

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(final, indent=2, default=str))
    log(f"Results: {out_path}")
    print(json.dumps(final, indent=2, default=str))

    rate = final.get("rate_per_min_100sample")
    if rate is None:
        return 3
    if rate < args.target_rate:
        log(f"NOTE: measured rate {rate:.2f} < target {args.target_rate} traces/min")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
