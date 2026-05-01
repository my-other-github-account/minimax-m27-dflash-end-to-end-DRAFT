"""
v3 trace-generation worker — uses dflash_llama high-level Python API directly.

Imports `TraceGenerator` and calls `.generate()` once with a row range. The
library handles resume, state tracking, signal handling, and per-row error
recovery internally — this script just wires arguments to the API call.

Usage:
    PYTHONPATH=/home/user/dflash-llama/src \\
    python3 worker_api_v3.py \\
        --shard-id D \\
        --rows 576034:626034 \\
        --out /home/user/iq4_v3_tracegen/traces \\
        --state /home/user/iq4_v3_tracegen/state_worker_D.json
"""

from __future__ import annotations

import argparse
import sys
import time

# === HIGH-LEVEL LIBRARY API ===
from dflash_llama import TraceGenerator, load_verifier


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", required=True, help="A, B, C, D, ...")
    ap.add_argument("--rows", required=True, help="lo:hi prompt row range")
    ap.add_argument("--out", required=True, help="Trace output directory")
    ap.add_argument("--state", required=True, help="State JSON path")
    ap.add_argument("--prompts", default="/home/user/iq4_tracegen/prompts_tulu3")
    ap.add_argument("--gguf-path",
                    default="/home/user/clawd/iq4_models/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf")
    ap.add_argument("--binary",
                    default="/home/user/iq4_tracegen/buun-llama-cpp/build/bin/llama-dump-hiddens")
    ap.add_argument("--verifier-name", default="minimax-m2.7-iq4-xs")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--per-trace-timeout", type=int, default=600)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    lo, hi = args.rows.split(":")
    lo, hi = int(lo), int(hi)
    print(f"[api-worker {args.shard_id}] rows={lo}:{hi}  out={args.out}", flush=True)
    print(f"[api-worker {args.shard_id}] verifier={args.verifier_name}", flush=True)
    print(f"[api-worker {args.shard_id}] gguf={args.gguf_path}", flush=True)

    # === HIGH-LEVEL API: load verifier with concrete GGUF path ===
    verifier = load_verifier(args.verifier_name, gguf_path=args.gguf_path)
    print(f"[api-worker {args.shard_id}] verifier loaded "
          f"hidden={verifier.hidden_size} layers={tuple(verifier.layer_ids)}",
          flush=True)

    # === HIGH-LEVEL API: TraceGenerator wraps backend + saturating fp8 + I/O ===
    gen = TraceGenerator(
        verifier=verifier,
        storage="fp8_per_tensor_scale",
        backend="llamacpp_gguf",
        backend_kwargs={
            "binary": args.binary,
            "timeout": args.per_trace_timeout,
        },
    )
    print(f"[api-worker {args.shard_id}] TraceGenerator constructed", flush=True)

    # === HIGH-LEVEL API: drive the generation loop. Library handles state, resume,
    # SIGTERM/SIGINT, per-row failure isolation. We just give it the row range. ===
    t0 = time.time()
    summary = gen.generate(
        prompts=args.prompts,
        output_dir=args.out,
        rows=range(lo, hi),
        state_path=args.state,
        max_seq_len=args.max_seq_len,
        skip_existing=True,
        log_every=args.log_every,
    )
    elapsed = time.time() - t0
    print(f"[api-worker {args.shard_id}] DONE in {elapsed:.1f}s  summary={summary}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
