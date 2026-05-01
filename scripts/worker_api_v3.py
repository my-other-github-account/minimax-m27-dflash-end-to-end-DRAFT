"""
v3 trace-generation worker — uses dflash_llama high-level Python API.

Imports `TraceGenerator` and calls `.generate()` once with a row range. The
library handles resume, state tracking, signal handling, and per-row error
recovery internally — this script just wires arguments to the API call.

Two ways to point at the model:
  - --hf-repo / --gguf-repo / --gguf-quant   (auto-download from Hub)
  - --hf-path / --gguf-path                   (local files)

Usage:
    python3 scripts/worker_api_v3.py \\
        --shard-id D --rows 0:50000 \\
        --out data/traces \\
        --state data/state/state_worker_D.json \\
        --prompts data/prompts_tulu3 \\
        --hf-repo MiniMaxAI/MiniMax-M2 \\
        --gguf-repo unsloth/MiniMax-M2-GGUF --gguf-quant UD-IQ4_XS \\
        --binary build/llama.cpp-dflash/build/bin/llama-dump-hiddens
"""

from __future__ import annotations

import argparse
import sys
import time

from dflash_llama import TraceGenerator, load_verifier


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", required=True, help="A, B, C, D, ...")
    ap.add_argument("--rows", required=True, help="lo:hi prompt row range")
    ap.add_argument("--out", required=True, help="trace output directory")
    ap.add_argument("--state", required=True, help="state JSON path")
    ap.add_argument("--prompts", required=True, help="path to HF prompts dataset on disk")
    ap.add_argument("--binary", required=True, help="path to llama-dump-hiddens binary")

    ap.add_argument("--verifier-name", default="minimax-m2.7-iq4-xs")

    # local-path mode
    ap.add_argument("--hf-path", default=None)
    ap.add_argument("--gguf-path", default=None)

    # Hub-slug mode
    ap.add_argument("--hf-repo", default=None,
                    help="HF Hub slug for the model config + tokenizer (e.g. MiniMaxAI/MiniMax-M2)")
    ap.add_argument("--gguf-repo", default=None,
                    help="HF Hub slug for GGUF weights (e.g. unsloth/MiniMax-M2-GGUF)")
    ap.add_argument("--gguf-quant", default=None,
                    help="quant subdir within --gguf-repo (e.g. UD-IQ4_XS)")
    ap.add_argument("--revision", default=None)

    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--per-trace-timeout", type=int, default=600)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    lo, hi = args.rows.split(":")
    lo, hi = int(lo), int(hi)
    print(f"[api-worker {args.shard_id}] rows={lo}:{hi}  out={args.out}", flush=True)
    print(f"[api-worker {args.shard_id}] verifier={args.verifier_name}", flush=True)

    # === HIGH-LEVEL API: load verifier (slug-aware or local-path) ===
    verifier = load_verifier(
        args.verifier_name,
        hf_path=args.hf_path,
        gguf_path=args.gguf_path,
        hf_repo=args.hf_repo,
        gguf_repo=args.gguf_repo,
        gguf_quant=args.gguf_quant,
        revision=args.revision,
    )
    print(f"[api-worker {args.shard_id}] verifier loaded "
          f"hidden={verifier.hidden_size} layers={tuple(verifier.layer_ids)} "
          f"gguf={verifier.gguf_path}", flush=True)

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
