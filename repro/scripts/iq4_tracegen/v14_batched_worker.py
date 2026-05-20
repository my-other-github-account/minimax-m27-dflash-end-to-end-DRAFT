#!/usr/bin/env python3
"""V14 trace-pool worker — BATCHED fast path.

Differences from v14_fast_worker.py (the previous "fast" worker that
actually wasn't):
  1. Uses ``TraceGenerator.generate_many(...)`` against the persistent
     trace-server with same-length groups of size ``--batch-width``.
     The bench (repro/scripts/bench_tracegen_speedup.py) measured 63.8
     traces/min on a 100-prompt sample at batch_width=4; at batch_width=8
     the rate should be comparable or better.
  2. Same hash-prefilter logic as before: walks a deterministic
     permutation of the prompt corpus, skips any row whose input_ids
     content-hash is already in the cluster_union skip-set.

Atomic. Resumable. Signal-safe.

Output filename = hs_<source_row_idx>.safetensors (positional, NOT
renumbered). Each trace's metadata.source_row_idx = the prompt row in
the source corpus. Each new hash is appended to a per-run hashes.pkl
on every Nth completion.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from safetensors import safe_open

from dflash_llama import TraceGenerator, load_verifier


def hash_input_ids(arr) -> str:
    a = np.asarray(arr, dtype=np.int32)
    return hashlib.sha256(a.tobytes()).hexdigest()


def load_hash_pickle(path: str) -> set:
    if not path or not os.path.exists(path):
        return set()
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        return set(data)
    if isinstance(data, set):
        return data
    if isinstance(data, dict):
        if "shas" in data and isinstance(data["shas"], (list, set)):
            return set(data["shas"])
        if all(isinstance(k, str) and len(k) == 64 for k in data.keys()):
            return set(data.keys())
    raise ValueError(f"unknown hash pickle format in {path}: {type(data)}")


def atomic_save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", required=True)
    ap.add_argument("--sample-seed", type=int, default=14)
    ap.add_argument("--sample-target", type=int, default=100000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--verifier-name", default="minimax-m2.7-iq4-xs")
    ap.add_argument("--hf-path", default=None)
    ap.add_argument("--gguf-path", required=True)
    ap.add_argument("--layer-ids", default="2,16,30,45,59,61")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--per-trace-timeout", type=int, default=600)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--skip-hashes", default=None)
    ap.add_argument("--new-hashes-out", default=None)
    ap.add_argument("--hash-flush-every", type=int, default=10)
    ap.add_argument("--socket", default="unix:///tmp/dflash_v14_batched.sock")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--override-tensor", default="exps=CPU")
    ap.add_argument("--server-log", default=None)
    # NEW: batched-decode knobs
    ap.add_argument("--batch-width", type=int, default=8,
                    help="prompts per run_many call (must be <= worker n_seq_max)")
    ap.add_argument("--max-batch-tokens", type=int, default=8192,
                    help="cap on total tokens per batched decode (must be <= worker n_batch). "
                         "Effective batch = min(batch_width, max_batch_tokens // padded_seq_len). "
                         "Binary compiled with n_batch=8192 -> 8192 here.")
    ap.add_argument("--length-bucket", type=int, default=512,
                    help="round seq_len UP to multiple of this when bucketing. "
                         "Bigger bucket = more batch collisions = more width=8 batches, at the "
                         "cost of small padding overhead per prompt. 512 gives 4 unique shapes "
                         "(512, 1024, 1536, 2048) which is memory-safe.")
    ap.add_argument("--pad-token-id", type=int, default=200004,
                    help="pad token id (MiniMax: 200004 = <fim_pad>)")
    ap.add_argument("--flush-after-rows", type=int, default=4096,
                    help="force-flush all length buckets after this many examined rows, "
                         "so a length that never reaches batch_width still gets written")
    ap.add_argument("--prewarm", action="store_true",
                    help="pre-warm all shape buckets at startup with dummy batches (throwaway traces)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    new_hashes_path = Path(args.new_hashes_out) if args.new_hashes_out else (out / "new_hashes.pkl")

    print(f"[v14-batched {args.shard_id}] EVEN-SAMPLED seed={args.sample_seed} "
          f"target={args.sample_target} batch_width={args.batch_width} out={out}",
          flush=True)

    skip_hashes = load_hash_pickle(args.skip_hashes) if args.skip_hashes else set()
    print(f"[v14-batched] loaded {len(skip_hashes)} existing-pool hashes from {args.skip_hashes}",
          flush=True)

    prior_new = load_hash_pickle(str(new_hashes_path)) if new_hashes_path.exists() else set()
    print(f"[v14-batched] loaded {len(prior_new)} prior new-pool hashes from {new_hashes_path}",
          flush=True)
    new_hashes_set = set(prior_new)
    union = skip_hashes | new_hashes_set
    print(f"[v14-batched] starting union dedup size: {len(union)}", flush=True)

    ds = load_from_disk(str(args.prompts))
    total = len(ds)
    print(f"[v14-batched] prompts {args.prompts} has {total} rows; col=input_ids", flush=True)
    rng = np.random.default_rng(args.sample_seed)
    perm = rng.permutation(total).astype(np.int64)
    print(f"[v14-batched] built deterministic permutation seed={args.sample_seed} "
          f"len={len(perm)} (first 5 = {perm[:5].tolist()})", flush=True)

    layer_ids = [int(x.strip()) for x in args.layer_ids.split(",") if x.strip()]
    verifier = load_verifier(
        args.verifier_name,
        hf_path=args.hf_path,
        gguf_path=args.gguf_path,
        layer_ids=layer_ids,
    )
    print(f"[v14-batched] verifier name={verifier.name} layers={tuple(verifier.layer_ids)}",
          flush=True)

    gen = TraceGenerator(
        verifier=verifier,
        storage="fp8_per_tensor_scale",
        backend="tracegen_client",
        backend_kwargs={
            "binary": args.binary,
            "socket_path": args.socket,
            "auto_start": True,
            "request_timeout": float(args.per_trace_timeout),
            "startup_timeout": 900.0,
            "ctx": args.ctx,
            "ngl": args.ngl,
            "override_tensor": args.override_tensor,
            "server_log_path": args.server_log,
        },
    )

    # Signal handling — finish current batch before exiting.
    stop = {"flag": False}

    def _sig(s, f):
        print(f"[v14-batched] signal {s}, finishing current batch", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    src_name = Path(args.prompts).name

    # Counters
    completed = pre_skipped = post_skipped = failed = oversize = 0
    rows_examined = 0
    last_log_completed = 0
    t0 = time.time()

    # Per-length buckets of pending work
    buckets: dict[int, list] = defaultdict(list)

    def _effective_k(seq_len: int) -> int:
        if seq_len <= 0:
            return args.batch_width
        cap = max(1, args.max_batch_tokens // seq_len)
        return max(1, min(args.batch_width, cap))

    def _flush_bucket(seq_len: int, group=None) -> None:
        nonlocal completed, failed
        if group is None:
            group = buckets.pop(seq_len, [])
        if not group:
            return
        try:
            gen.generate_many(
                batch_inputs=[g["input_ids"] for g in group],
                output_paths=[out / f"hs_{g['i']}.safetensors" for g in group],
                source_names=[src_name] * len(group),
                source_row_ids=[g["i"] for g in group],
                max_seq_len=args.max_seq_len,
                loss_masks=[g["loss_mask"] for g in group],
                extra_metadatas=[{
                    "v14_pool": "true",
                    "dedup_against": "v11_v12_v13_pools",
                    "sample_seed": str(args.sample_seed),
                    "batch_width_at_gen": str(len(group)),
                    "padded_seq_len": str(seq_len),
                    "raw_seq_len": str(g["raw_len"]),
                    "length_bucket": str(args.length_bucket),
                } for g in group],
            )
        except Exception as e:
            print(f"[v14-batched] batch (seq_len={seq_len}, n={len(group)}) FAILED: {e}",
                  file=sys.stderr, flush=True)
            failed += len(group)
            return
        for g in group:
            new_hashes_set.add(g["hash"])
            union.add(g["hash"])
        completed += len(group)
        if completed // args.hash_flush_every > last_log_completed // args.hash_flush_every:
            atomic_save_pickle(new_hashes_path, sorted(new_hashes_set))

    def _flush_full_buckets() -> None:
        for k in list(buckets.keys()):
            ek = _effective_k(k)
            while len(buckets.get(k, [])) >= ek:
                group = buckets[k][:ek]
                buckets[k] = buckets[k][ek:]
                if not buckets[k]:
                    del buckets[k]
                _flush_bucket(k, group=group)

    def _flush_all_buckets() -> None:
        for k in list(buckets.keys()):
            ek = _effective_k(k)
            while buckets.get(k):
                group = buckets[k][:ek]
                buckets[k] = buckets[k][ek:]
                if not buckets[k]:
                    del buckets[k]
                _flush_bucket(k, group=group)

    # Optional: pre-warm all shape buckets at startup so sched_reserve and
    # graph caches are populated BEFORE real generation starts. This
    # eliminates the long warmup ramp at the cost of a few minutes of
    # dummy work. See ``TraceGenerator.prewarm_shapes``.
    if args.prewarm:
        shapes = TraceGenerator.plan_prewarm_shapes(
            length_bucket=args.length_bucket,
            max_seq_len=args.max_seq_len,
            batch_width=args.batch_width,
            max_batch_tokens=args.max_batch_tokens,
        )
        print(f"[v14-batched] prewarm: {len(shapes)} shapes={shapes} pad_id={args.pad_token_id}",
              flush=True)
        pw_t0 = time.time()
        gen.prewarm_shapes(
            shapes=shapes,
            pad_token_id=args.pad_token_id,
            max_seq_len=args.max_seq_len,
            log_fn=lambda m: print(f"[v14-batched] {m}", flush=True),
        )
        print(f"[v14-batched] prewarm complete in {time.time()-pw_t0:.1f}s, starting real work",
              flush=True)

    for i in perm.tolist():
        if stop["flag"]:
            break
        if completed >= args.sample_target:
            print(f"[v14-batched] reached --sample-target={args.sample_target}, "
                  f"flushing and exiting", flush=True)
            break
        if i >= total:
            continue
        out_path = out / f"hs_{i}.safetensors"
        if out_path.exists():
            # already produced — count its hash into the running set
            try:
                with safe_open(str(out_path), framework="numpy") as sf:
                    keys = sf.keys()
                    arr = sf.get_tensor("input_ids" if "input_ids" in keys else "token_ids")
                h = hash_input_ids(arr)
                if h not in union:
                    union.add(h)
                    new_hashes_set.add(h)
            except Exception:
                pass
            post_skipped += 1
            continue

        row = ds[int(i)]
        input_ids = list(row["input_ids"])
        if len(input_ids) > args.max_seq_len:
            oversize += 1
            continue

        # PRE-FILTER: hash before any expensive dump
        h = hash_input_ids(input_ids)
        if h in union:
            pre_skipped += 1
            continue

        loss_mask = row.get("loss_mask", None)
        raw_len = len(input_ids)
        # Round UP to length_bucket multiple so prompts of nearby length share a batch.
        # Cap at max_seq_len so we never exceed configured context.
        if args.length_bucket > 1:
            padded_len = min(
                args.max_seq_len,
                ((raw_len + args.length_bucket - 1) // args.length_bucket) * args.length_bucket,
            )
        else:
            padded_len = raw_len
        if padded_len > raw_len:
            pad_n = padded_len - raw_len
            input_ids = input_ids + [args.pad_token_id] * pad_n
            if loss_mask is not None:
                loss_mask = list(loss_mask) + [False] * pad_n
        buckets[padded_len].append({
            "i": int(i),
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "hash": h,
            "raw_len": raw_len,
        })
        rows_examined += 1

        _flush_full_buckets()

        if rows_examined % args.flush_after_rows == 0:
            _flush_all_buckets()

        if completed - last_log_completed >= args.log_every:
            elapsed = time.time() - t0
            rate = completed / max(elapsed, 1e-9) * 60.0
            pending = sum(len(v) for v in buckets.values())
            print(f"[v14-batched] examined={rows_examined} completed={completed} "
                  f"pre_skipped={pre_skipped} post_skipped={post_skipped} "
                  f"failed={failed} oversize={oversize} pending={pending} "
                  f"rate={rate:.1f}/min union={len(union)}", flush=True)
            last_log_completed = completed

    # Final flush
    _flush_all_buckets()
    atomic_save_pickle(new_hashes_path, sorted(new_hashes_set))

    elapsed = time.time() - t0
    rate = completed / max(elapsed, 1e-9) * 60.0
    print(f"[v14-batched] DONE in {elapsed:.1f}s  completed={completed} "
          f"pre_skipped={pre_skipped} post_skipped={post_skipped} "
          f"failed={failed} oversize={oversize} union={len(union)} "
          f"rate={rate:.1f}/min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
