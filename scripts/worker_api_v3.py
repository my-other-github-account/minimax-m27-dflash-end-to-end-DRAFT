"""
v3 trace-generation worker — uses dflash_llama high-level Python API.

Imports `TraceGenerator` and calls `.generate()` once with a row range. The
library handles resume, state tracking, signal handling, and per-row error
recovery internally — this script just wires arguments to the API call.

Two ways to point at the model:
  - --hf-repo / --gguf-repo / --gguf-quant   (auto-download from Hub)
  - --hf-path / --gguf-path                   (local files)

You can override the verifier shape from the CLI:
  --layer-ids "2,16,30,45,59,61"   pick the exact taps to capture
  --num-layer-taps 6               or let auto_layer_ids spread N taps
  --hidden-size / --num-hidden-layers / --vocab-size / --mask-token-id
                                   override individual shape fields
  --verifier-name generic          fully-custom: requires the four shape
                                   fields above plus --layer-ids

Usage:
    python3 scripts/worker_api_v3.py \\
        --shard-id D --rows 0:50000 \\
        --out data/traces \\
        --state data/state/state_worker_D.json \\
        --prompts data/prompts_tulu3 \\
        --hf-repo MiniMaxAI/MiniMax-M2 \\
        --gguf-repo unsloth/MiniMax-M2-GGUF --gguf-quant UD-IQ4_XS \\
        --binary build/llama.cpp-dflash/build/bin/llama-dump-hiddens \\
        --layer-ids "2,16,30,45,59,61"
"""

from __future__ import annotations

import argparse
import sys
import time

from dflash_llama import TraceGenerator, load_verifier


def _parse_layer_ids(s: str) -> list[int]:
    """Parse '2,16,30,45,59,61' → [2,16,30,45,59,61]."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", required=True, help="A, B, C, D, ...")
    ap.add_argument("--rows", required=True, help="lo:hi prompt row range")
    ap.add_argument("--out", required=True, help="trace output directory")
    ap.add_argument("--state", required=True, help="state JSON path")
    ap.add_argument("--prompts", required=True, help="path to HF prompts dataset on disk")
    ap.add_argument("--binary", required=True, help="path to llama-dump-hiddens binary")

    ap.add_argument("--verifier-name", default="minimax-m2.7-iq4-xs",
                    help="registered verifier name, or 'generic' for a fully-custom model")

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

    # Verifier shape overrides — every factory accepts these as None=use-default
    shape = ap.add_argument_group("verifier shape overrides",
        "Override which layers DFlash taps and the verifier shape. Use these to "
        "adapt the library to a new model without writing a Python factory.")
    shape.add_argument("--layer-ids", default=None,
                       help="comma-separated layer indices to tap, e.g. '2,16,30,45,59,61'. "
                            "Overrides the factory default. Required when --verifier-name=generic.")
    shape.add_argument("--num-layer-taps", type=int, default=None,
                       help="if --layer-ids is omitted, ask the library to spread this many "
                            "taps via auto_layer_ids (final residual is always included)")
    shape.add_argument("--hidden-size", type=int, default=None)
    shape.add_argument("--num-hidden-layers", type=int, default=None)
    shape.add_argument("--vocab-size", type=int, default=None)
    shape.add_argument("--mask-token-id", type=int, default=None)
    shape.add_argument("--block-size", type=int, default=None)
    shape.add_argument("--drafter-arch", default=None)
    shape.add_argument("--drafter-hidden-act", default=None)
    shape.add_argument("--family", default=None)
    shape.add_argument("--name-override", default=None,
                       help="when --verifier-name=generic, this becomes the verifier 'name' field")

    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--per-trace-timeout", type=int, default=600)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    lo, hi = args.rows.split(":")
    lo, hi = int(lo), int(hi)
    print(f"[api-worker {args.shard_id}] rows={lo}:{hi}  out={args.out}", flush=True)
    print(f"[api-worker {args.shard_id}] verifier={args.verifier_name}", flush=True)

    # Build the override kwargs dict — only pass non-None values so factory
    # defaults still take effect for fields the user didn't override.
    overrides = {}
    if args.layer_ids is not None:
        overrides["layer_ids"] = _parse_layer_ids(args.layer_ids)
    if args.num_layer_taps is not None:
        overrides["num_layer_taps"] = args.num_layer_taps
    if args.hidden_size is not None:
        overrides["hidden_size"] = args.hidden_size
    if args.num_hidden_layers is not None:
        overrides["num_hidden_layers"] = args.num_hidden_layers
    if args.vocab_size is not None:
        overrides["vocab_size"] = args.vocab_size
    if args.mask_token_id is not None:
        overrides["mask_token_id"] = args.mask_token_id
    if args.block_size is not None:
        overrides["block_size"] = args.block_size
    if args.drafter_arch is not None:
        overrides["drafter_arch"] = args.drafter_arch
    if args.drafter_hidden_act is not None:
        overrides["drafter_hidden_act"] = args.drafter_hidden_act
    if args.family is not None:
        overrides["family"] = args.family
    if args.name_override is not None:
        overrides["name_override"] = args.name_override

    if overrides:
        print(f"[api-worker {args.shard_id}] shape overrides = {overrides}", flush=True)

    # === HIGH-LEVEL API: load verifier (slug-aware or local-path) ===
    verifier = load_verifier(
        args.verifier_name,
        hf_path=args.hf_path,
        gguf_path=args.gguf_path,
        hf_repo=args.hf_repo,
        gguf_repo=args.gguf_repo,
        gguf_quant=args.gguf_quant,
        revision=args.revision,
        **overrides,
    )
    print(f"[api-worker {args.shard_id}] verifier loaded "
          f"name={verifier.name} family={verifier.family} "
          f"hidden={verifier.hidden_size} num_layers={verifier.num_hidden_layers} "
          f"vocab={verifier.vocab_size} mask={verifier.mask_token_id} "
          f"layer_ids={tuple(verifier.layer_ids)} "
          f"gguf={verifier.gguf_path}", flush=True)

    gen = TraceGenerator(
        verifier=verifier,
        storage="fp8_per_tensor_scale",
        backend="tracegen_client",
        backend_kwargs={
            "binary": args.binary,
            "request_timeout": args.per_trace_timeout,
            "auto_start": True,
            "ctx": 16384,
            "ngl": 99,
            "override_tensor": "exps=CPU",
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
