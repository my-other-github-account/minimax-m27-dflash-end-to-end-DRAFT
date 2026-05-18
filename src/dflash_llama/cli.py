"""``dflash-llama`` CLI entry point.

Subcommands::

    generate    Run the trace generator over a prompts arrow dataset
    train       Run a full DFlash training job
    smoke       Run the 90-second torchrun smoke
    eval        Run the offline drafter eval
    prepare     assemble_prompts_arrow + build_vocab_maps only
    info        Print the verifier registry
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .verifiers import load_verifier, list_verifiers


def _parse_rows(spec: Optional[str]) -> Optional[range]:
    if spec is None:
        return None
    if ":" in spec:
        a, b = spec.split(":", 1)
        return range(int(a), int(b))
    return range(0, int(spec))


def _parse_layer_ids_arg(s: Optional[str]) -> Optional[list]:
    """Parse '2,16,30,45,59,61' → [2,16,30,45,59,61]; pass through None."""
    if s is None:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _build_verifier(args) -> "BaseVerifier":  # noqa: F821
    overrides = {}
    layer_ids = _parse_layer_ids_arg(getattr(args, "layer_ids", None))
    if layer_ids is not None:
        overrides["layer_ids"] = layer_ids
    for k in (
        "num_layer_taps", "hidden_size", "num_hidden_layers",
        "vocab_size", "mask_token_id", "block_size",
        "drafter_arch", "drafter_hidden_act", "family", "name_override",
    ):
        v = getattr(args, k, None)
        if v is not None:
            overrides[k] = v
    return load_verifier(
        args.verifier,
        gguf_path=getattr(args, "gguf_path", None),
        hf_path=getattr(args, "hf_path", None),
        gguf_repo=getattr(args, "gguf_repo", None),
        hf_repo=getattr(args, "hf_repo", None),
        gguf_quant=getattr(args, "gguf_quant", None),
        revision=getattr(args, "revision", None),
        **overrides,
    )


# ----- subcommands -----
def cmd_generate(args) -> int:
    from .generation import TraceGenerator

    verifier = _build_verifier(args)
    binary = args.binary
    if args.backend == "tracegen_client" and binary == "llama-dump-hiddens":
        binary = "llama-dump-hiddens-worker"
    if args.backend == "llamacpp_gguf":
        backend_kwargs = {
            "binary": binary,
            "ctx": args.ctx,
            "timeout": args.timeout,
        }
    else:
        backend_kwargs = {
            "socket_path": args.socket,
            "request_timeout": args.timeout,
            "connect_timeout": args.connect_timeout,
            "startup_timeout": args.startup_timeout,
            "auto_start": args.auto_start_server,
            "binary": binary,
            "ctx": args.ctx,
            "ngl": args.ngl,
            "override_tensor": args.override_tensor,
            "worker_args": args.worker_arg,
            "server_log_path": args.server_log,
        }
    gen = TraceGenerator(
        verifier=verifier,
        storage=args.storage,
        backend=args.backend,
        backend_kwargs=backend_kwargs,
    )
    rows = _parse_rows(args.rows)
    summary = gen.generate(
        prompts=args.prompts,
        output_dir=args.out,
        rows=rows,
        state_path=args.state,
        max_seq_len=args.max_seq_len,
        source_name=args.source_name,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_prepare(args) -> int:
    from .training import DFlashTrainer

    verifier = _build_verifier(args)
    trainer = DFlashTrainer(
        traces_dir=args.traces,
        verifier=verifier,
        num_layers=args.num_layers,
        draft_vocab_size=args.draft_vocab_size,
        paired_dir=args.paired_dir,
    )
    report = trainer.prepare(force=args.force)
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_train(args) -> int:
    from .training import DFlashTrainer

    verifier = _build_verifier(args)
    trainer = DFlashTrainer(
        traces_dir=args.traces,
        verifier=verifier,
        num_layers=args.num_layers,
        draft_vocab_size=args.draft_vocab_size,
        paired_dir=args.paired_dir,
    )
    if not args.skip_prepare:
        trainer.prepare()
    result = trainer.train(
        save_to=args.output,
        epochs=args.epochs,
        lr=args.lr,
        max_anchors=args.max_anchors,
        total_seq_len=args.total_seq_len,
        log_freq=args.log_freq,
        scheduler_warmup_steps=args.warmup_steps,
        save_best=args.save_best,
        port=args.port,
        speculators_train_script=args.train_script,
        log_path=args.log,
        dry_run=args.dry_run,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "cmd"}, indent=2))
    if args.dry_run:
        print("DRY-RUN cmd:", " ".join(result["cmd"]))
    return result["rc"]


def cmd_smoke(args) -> int:
    from .training import DFlashTrainer

    verifier = _build_verifier(args)
    trainer = DFlashTrainer(
        traces_dir=args.traces,
        verifier=verifier,
        num_layers=args.num_layers,
        draft_vocab_size=args.draft_vocab_size,
        paired_dir=args.paired_dir,
    )
    if not args.skip_prepare:
        trainer.prepare()
    res = trainer.smoke(
        timeout_sec=args.timeout,
        save_path=args.save_path,
        log_path=args.log,
        port=args.port,
        speculators_train_script=args.train_script,
        dry_run=args.dry_run,
    )
    print(json.dumps(res.to_dict(), indent=2))
    return 0 if res.passed else 1


def cmd_eval(args) -> int:
    from .training import offline_eval

    metrics = offline_eval(
        checkpoint=args.checkpoint,
        paired_dir=args.paired_dir,
        verifier_path=args.verifier_path,
        max_batches=args.max_batches,
        total_seq_len=args.total_seq_len,
    )
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_info(args) -> int:
    print("registered verifiers:")
    for name in list_verifiers():
        print(f"  - {name}")
    return 0


def cmd_export_gguf(args) -> int:
    from .inference import export_to_gguf, verify_gguf_metadata

    out = export_to_gguf(
        checkpoint=args.checkpoint,
        output_path=args.output,
        verifier_meta_dir=args.verifier_meta_dir,
        buun_repo=args.buun_repo,
        venv_python=args.venv_python,
        outtype=args.outtype,
        rebake_floor=args.rebake_floor,
        prepped_dir=args.prepped_dir,
        register_tokenizer_hash=not args.no_register_hash,
    )
    if args.verify:
        meta = verify_gguf_metadata(out)
        print(json.dumps(meta, indent=2))
    return 0


def cmd_serve(args) -> int:
    import time
    from .inference import LlamaServer

    server = LlamaServer(
        verifier_gguf=args.verifier,
        drafter_gguf=args.drafter,
        spec_type=args.spec_type if args.drafter else None,
        draft_max=args.draft_max,
        host=args.host,
        port=args.port,
        ctx=args.ctx,
        n_gpu_layers=args.ngl,
        n_gpu_layers_draft=args.ngld,
        override_tensor=args.override_tensor,
        draft_device=args.device_draft,
        binary=args.binary,
        parallel=args.parallel,
        log_path=args.log,
    )
    server.start()
    print(f"DFlash llama-server up: {server.url}")
    print("Endpoints: /v1/chat/completions, /v1/completions, /v1/models")
    print("Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        server.stop()
    return 0


def cmd_benchmark(args) -> int:
    from .inference import benchmark

    report = benchmark(
        verifier_gguf=args.verifier,
        drafter_gguf=args.drafter,
        val_metrics=args.val_metrics,
        prompt=args.prompt,
        dmax_sweep=[int(x) for x in args.dmax.split(",")],
        n_tokens=args.n_tokens,
        ctx=args.ctx,
        temperature=args.temperature,
        n_gpu_layers=args.ngl,
        n_gpu_layers_draft=args.ngld,
        override_tensor=args.override_tensor,
        draft_device=args.device_draft,
        binary=args.binary,
        log_dir=args.log_dir,
        drafter_label=args.label,
        progress=not args.no_progress,
    )
    if args.json:
        print(report.to_json())
    else:
        print(report.markdown())
    if args.json_out:
        report.to_json(args.json_out)
        print(f"\n[saved JSON to {args.json_out}]")
    return 0


def cmd_trace_server(args) -> int:
    from .tracegen import TraceServer

    layer_ids = _parse_layer_ids_arg(args.layer_ids)
    if not layer_ids:
        raise ValueError("--layer-ids is required for trace-server")
    server = TraceServer(
        gguf_path=args.gguf_path,
        layer_ids=layer_ids,
        bind=args.socket,
        n_ctx=args.ctx,
        n_gpu_layers=args.ngl,
        override_tensor=args.override_tensor,
        binary=args.binary,
        worker_args=args.worker_arg,
        worker_log_path=args.log,
        request_timeout=args.timeout,
        startup_timeout=args.startup_timeout,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.stop()
    return 0


# ----- arg-parser -----
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dflash-llama", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    # common verifier args
    def add_verifier_args(sp):
        sp.add_argument("--verifier", required=True,
            help="verifier name (e.g. 'minimax-m2.7-iq4-xs'). "
                 "Use 'generic' to describe a custom model via shape kwargs. "
                 "Run 'dflash-llama info' to list registered names.")
        sp.add_argument("--gguf-path", default=None, help="local path to a GGUF shard (mutually exclusive with --gguf-repo)")
        sp.add_argument("--hf-path", default=None, help="local path to an HF model directory (mutually exclusive with --hf-repo)")
        sp.add_argument("--gguf-repo", default=None, help="HF Hub slug for GGUF weights, e.g. 'unsloth/MiniMax-M2-GGUF'")
        sp.add_argument("--hf-repo", default=None, help="HF Hub slug for the model config + tokenizer, e.g. 'MiniMaxAI/MiniMax-M2'")
        sp.add_argument("--gguf-quant", default=None, help="quant subdir within --gguf-repo, e.g. 'UD-IQ4_XS'")
        sp.add_argument("--revision", default=None, help="optional Hub revision (branch, tag, or commit)")

        # Verifier shape overrides — work for ANY --verifier value. Use these to
        # adapt the library to a new model without writing a Python factory.
        shape = sp.add_argument_group(
            "verifier shape overrides",
            "Override which layers DFlash taps and the verifier shape. "
            "Required (combined) when --verifier=generic.")
        shape.add_argument("--layer-ids", default=None,
            help="comma-separated layer indices to tap, e.g. '2,16,30,45,59,61'. "
                 "Overrides the factory default. Required for --verifier=generic.")
        shape.add_argument("--num-layer-taps", type=int, default=None,
            help="if --layer-ids is omitted, ask the library to spread N taps via auto_layer_ids "
                 "(final residual is always included)")
        shape.add_argument("--hidden-size", type=int, default=None,
            help="override hidden_size (required for --verifier=generic)")
        shape.add_argument("--num-hidden-layers", type=int, default=None,
            help="override num_hidden_layers (required for --verifier=generic)")
        shape.add_argument("--vocab-size", type=int, default=None,
            help="override vocab_size (required for --verifier=generic)")
        shape.add_argument("--mask-token-id", type=int, default=None,
            help="override mask_token_id (required for --verifier=generic)")
        shape.add_argument("--block-size", type=int, default=None,
            help="DFlash block_size (default 8)")
        shape.add_argument("--drafter-arch", default=None,
            help="drafter architecture name (default 'qwen3')")
        shape.add_argument("--drafter-hidden-act", default=None,
            help="drafter hidden activation (default 'silu')")
        shape.add_argument("--family", default=None,
            help="family tag for the verifier (informational)")
        shape.add_argument("--name-override", default=None,
            help="override the verifier 'name' field (mostly relevant for --verifier=generic)")

    # generate
    sg = sub.add_parser("generate", help="generate self-describing fp8 traces")
    add_verifier_args(sg)
    sg.add_argument("--prompts", required=True, help="path to HF prompts arrow dir")
    sg.add_argument("--out", required=True, help="output dir for hs_<i>.safetensors files")
    sg.add_argument("--rows", default=None, help="row range, 'A:B' or 'N' (=0:N). Default: all rows")
    sg.add_argument("--state", default=None, help="state.json path for resumability")
    sg.add_argument("--max-seq-len", type=int, default=2048)
    sg.add_argument("--storage", default="fp8_per_tensor_scale", choices=["fp8_per_tensor_scale", "bf16"])
    sg.add_argument("--backend", default="llamacpp_gguf", choices=["llamacpp_gguf", "tracegen_client"])
    sg.add_argument("--binary", default="llama-dump-hiddens")
    sg.add_argument("--ctx", type=int, default=4096)
    sg.add_argument("--ngl", type=int, default=99)
    sg.add_argument("--timeout", type=int, default=600)
    sg.add_argument("--connect-timeout", type=float, default=5.0)
    sg.add_argument("--startup-timeout", type=float, default=900.0)
    sg.add_argument("--socket", default="unix:///tmp/dflash_tracegen.sock")
    sg.add_argument("--auto-start-server", action="store_true")
    sg.add_argument("--override-tensor", default="exps=CPU")
    sg.add_argument("--worker-arg", action="append", default=None,
                    help="extra argument to pass through to llama-dump-hiddens-worker; repeatable")
    sg.add_argument("--server-log", default=None)
    sg.add_argument("--source-name", default=None)
    sg.set_defaults(func=cmd_generate)

    # prepare
    sp = sub.add_parser("prepare", help="assemble prompts arrow + build vocab maps")
    add_verifier_args(sp)
    sp.add_argument("--traces", required=True)
    sp.add_argument("--paired-dir", default=None)
    sp.add_argument("--num-layers", type=int, default=5)
    sp.add_argument("--draft-vocab-size", type=int, default=32768)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_prepare)

    # train
    st = sub.add_parser("train", help="run a full DFlash training job")
    add_verifier_args(st)
    st.add_argument("--traces", required=True)
    st.add_argument("--paired-dir", default=None)
    st.add_argument("--output", required=True)
    st.add_argument("--num-layers", type=int, default=5)
    st.add_argument("--draft-vocab-size", type=int, default=32768)
    st.add_argument("--epochs", type=int, default=17)
    st.add_argument("--lr", type=float, default=3e-5)
    st.add_argument("--max-anchors", type=int, default=512)
    st.add_argument("--total-seq-len", type=int, default=2048)
    st.add_argument("--log-freq", type=int, default=5)
    st.add_argument("--warmup-steps", type=int, default=100)
    st.add_argument("--save-best", action="store_true", default=True)
    st.add_argument("--port", type=int, default=29502)
    st.add_argument("--train-script", default=None)
    st.add_argument("--log", default=None)
    st.add_argument("--skip-prepare", action="store_true")
    st.add_argument("--dry-run", action="store_true")
    st.set_defaults(func=cmd_train)

    # smoke
    sm = sub.add_parser("smoke", help="run a 90s torchrun smoke")
    add_verifier_args(sm)
    sm.add_argument("--traces", required=True)
    sm.add_argument("--paired-dir", default=None)
    sm.add_argument("--num-layers", type=int, default=5)
    sm.add_argument("--draft-vocab-size", type=int, default=32768)
    sm.add_argument("--timeout", type=int, default=90)
    sm.add_argument("--save-path", default="/tmp/dflash-smoke")
    sm.add_argument("--log", default="/tmp/dflash-smoke.log")
    sm.add_argument("--port", type=int, default=29501)
    sm.add_argument("--train-script", default=None)
    sm.add_argument("--skip-prepare", action="store_true")
    sm.add_argument("--dry-run", action="store_true")
    sm.set_defaults(func=cmd_smoke)

    # eval
    se = sub.add_parser("eval", help="run offline DFlash drafter eval")
    se.add_argument("--checkpoint", required=True)
    se.add_argument("--paired-dir", required=True)
    se.add_argument("--verifier-path", required=True)
    se.add_argument("--max-batches", type=int, default=60)
    se.add_argument("--total-seq-len", type=int, default=2048)
    se.set_defaults(func=cmd_eval)

    # info
    si = sub.add_parser("info", help="list registered verifiers")
    si.set_defaults(func=cmd_info)

    # export-gguf
    sx = sub.add_parser("export-gguf",
                        help="convert a DFlash drafter checkpoint to a buun-loadable GGUF")
    sx.add_argument("--checkpoint", required=True,
                    help="speculators-format checkpoint dir (config.json + model.safetensors)")
    sx.add_argument("--output", required=True, help="path to write the GGUF")
    sx.add_argument("--verifier-meta-dir", default=None,
                    help="directory holding tokenizer.json etc (default: read from config)")
    sx.add_argument("--buun-repo", default="/home/user/buun-llama-cpp",
                    help="buun-llama-cpp checkout containing convert_hf_to_gguf.py")
    sx.add_argument("--venv-python", default=None,
                    help="python interpreter to invoke buun's converter (default: autodetect)")
    sx.add_argument("--outtype", default="bf16", choices=["bf16", "f16", "f32"])
    sx.add_argument("--rebake-floor", type=float, default=-65504.0,
                    help="floor for non-mapped rows in rebaked lm_head (default: -65504)")
    sx.add_argument("--prepped-dir", default=None,
                    help="staging dir for the prepped checkpoint (default: <output>.prep)")
    sx.add_argument("--no-register-hash", action="store_true",
                    help="don't auto-whitelist the FP8 tokenizer hash in buun")
    sx.add_argument("--verify", action="store_true",
                    help="after conversion, print GGUF metadata sanity-check")
    sx.set_defaults(func=cmd_export_gguf)

    # serve (OpenAI-compat)
    ss = sub.add_parser("serve",
                        help="run llama-server with optional DFlash spec decoding (OAI-compat)")
    ss.add_argument("--verifier", required=True, help="verifier GGUF path")
    ss.add_argument("--drafter", default=None, help="drafter GGUF path (DFlash)")
    ss.add_argument("--spec-type", default="dflash", choices=["dflash", "draft"])
    ss.add_argument("--draft-max", type=int, default=7)
    ss.add_argument("--host", default="0.0.0.0")
    ss.add_argument("--port", type=int, default=8080)
    ss.add_argument("--ctx", type=int, default=8192)
    ss.add_argument("--ngl", type=int, default=99)
    ss.add_argument("--ngld", type=int, default=99)
    ss.add_argument("--override-tensor", default="exps=CPU",
                    help="--override-tensor (-ot). Default 'exps=CPU' for IQ4_XS MoE.")
    ss.add_argument("--device-draft", default="CUDA0")
    ss.add_argument("--parallel", type=int, default=1)
    ss.add_argument("--binary", default=None)
    ss.add_argument("--log", default=None, help="optional log file path")
    ss.set_defaults(func=cmd_serve)

    # benchmark (speculative-decode sweep)
    sb = sub.add_parser("benchmark",
                        help="sweep --draft-max in llama-speculative-simple, "
                             "compute per-position + chain-cumulative accept rates")
    sb.add_argument("--verifier", required=True, help="verifier GGUF")
    sb.add_argument("--drafter", required=True, help="drafter GGUF")
    sb.add_argument("--val-metrics", default=None,
                    help="training val_metrics.json for z-score baseline")
    sb.add_argument("--prompt", default=None,
                    help="benchmark prompt (default: built-in Fibonacci spec)")
    sb.add_argument("--dmax", default="2,4,7", help="comma-separated draft-max values")
    sb.add_argument("--n-tokens", type=int, default=384)
    sb.add_argument("--ctx", type=int, default=8192)
    sb.add_argument("--temperature", type=float, default=0.0)
    sb.add_argument("--ngl", type=int, default=99)
    sb.add_argument("--ngld", type=int, default=99)
    sb.add_argument("--override-tensor", default="exps=CPU")
    sb.add_argument("--device-draft", default="CUDA0")
    sb.add_argument("--binary", default=None)
    sb.add_argument("--log-dir", default="/tmp/dflash_bench")
    sb.add_argument("--label", default=None,
                    help="report label (default: drafter GGUF stem)")
    sb.add_argument("--no-progress", action="store_true",
                    help="disable tqdm progress bar")
    sb.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of markdown")
    sb.add_argument("--json-out", default=None, help="also write JSON to file")
    sb.set_defaults(func=cmd_benchmark)

    stg = sub.add_parser("trace-server", help="run the persistent hidden-state trace server")
    stg.add_argument("--gguf-path", required=True)
    stg.add_argument("--layer-ids", required=True,
                     help="comma-separated layer indices to capture, e.g. 2,16,30,45,59,61")
    stg.add_argument("--socket", default="unix:///tmp/dflash_tracegen.sock")
    stg.add_argument("--ctx", type=int, default=4096)
    stg.add_argument("--ngl", type=int, default=99)
    stg.add_argument("--override-tensor", default="exps=CPU")
    stg.add_argument("--binary", default="llama-dump-hiddens-worker")
    stg.add_argument("--worker-arg", action="append", default=None,
                     help="extra argument to pass through to llama-dump-hiddens-worker; repeatable")
    stg.add_argument("--timeout", type=float, default=900.0)
    stg.add_argument("--startup-timeout", type=float, default=900.0)
    stg.add_argument("--log", default=None, help="optional worker stderr log path")
    stg.set_defaults(func=cmd_trace_server)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
