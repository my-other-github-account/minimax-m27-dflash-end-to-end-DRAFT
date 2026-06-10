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
    gen = TraceGenerator(
        verifier=verifier,
        storage=args.storage,
        backend=args.backend,
        backend_kwargs={"binary": args.binary, "ctx": args.ctx, "timeout": args.timeout}
                      if args.backend == "llamacpp_gguf" else None,
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
        fp8_recipe=args.fp8_recipe,
        te_use_fused=not args.no_te_fused,
        drafter_intermediate_size=args.drafter_intermediate_size,
        nan_skip=not args.no_nan_skip,
        use_torchrun=args.use_torchrun,
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
    extra_env = None
    if getattr(args, "fp8_recipe", None):
        # Surface the FP8-related env to the smoke run too. The smoke runner
        # itself doesn't currently emit FP8 flags (it always shells out via
        # torchrun), but documenting the intent in extra_env is useful for
        # log archeology.
        extra_env = {"DFLASH_LLAMA_FP8_RECIPE": str(args.fp8_recipe)}
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


def cmd_check_fp8(args) -> int:
    """Print the current arch capability dict + recipe recommendation.

    Cheap, side-effect-free hardware probe. Useful when bringing up a new
    Spark / DGX / Hopper machine to confirm which FP8 recipe will actually
    work before committing to a 14-epoch training run.
    """
    from .training.fp8 import current_arch

    arch = current_arch()
    print(json.dumps(arch, indent=2))
    cap_int = arch.get("compute_capability_int")
    if cap_int in (120, 121):
        print()
        print("⚠  Spark-class arch detected (sm_120/121). Reminders:")
        print("   - kind='block_fp8' is silently NON-CONVERGENT here (TE #2382). REFUSED.")
        print("   - kind='mxfp8' via TE is BLOCKED here (cuBLAS layout, TE #2668). REFUSED.")
        print("   - kind='current_fp8' + fused te.LayerNormMLP IS the production recipe.")
        print("   - Always use FP8Recipe(use_split_accumulator=True) on all 3 GEMMs.")
        print("   - Launch via python (NOT torchrun) on single-GPU to avoid the silent-bf16 trap.")
    return 0


def cmd_export_gguf(args) -> int:
    from .inference import export_to_gguf, verify_gguf_metadata

    out = export_to_gguf(
        checkpoint=args.checkpoint,
        output_path=args.output,
        verifier_meta_dir=args.verifier_meta_dir,
        d2t_path=getattr(args, "d2t_path", None),
        buun_repo=args.buun_repo,
        venv_python=args.venv_python,
        outtype=args.outtype,
        rebake_floor=args.rebake_floor,
        prepped_dir=args.prepped_dir,
        force_block_size=getattr(args, "force_block_size", None),
        register_tokenizer_hash=not args.no_register_hash,
    )
    if args.verify:
        meta = verify_gguf_metadata(out)
        print(json.dumps(meta, indent=2))
    return 0


def cmd_export_lucebox(args) -> int:
    from .inference import export_lucebox_to_gguf, verify_gguf_metadata

    out = export_lucebox_to_gguf(
        checkpoint=args.checkpoint,
        output_path=args.output,
        verifier_meta_dir=args.verifier_meta_dir,
        d2t_path=args.d2t_path,
        buun_repo=args.buun_repo,
        venv_python=args.venv_python,
        outtype=args.outtype,
        rebake_floor=args.rebake_floor,
        prepped_dir=args.prepped_dir,
        force_block_size=args.force_block_size,
        register_tokenizer_hash=not args.no_register_hash,
    )
    if args.verify:
        meta = verify_gguf_metadata(out, expected_block_size=args.force_block_size)
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


def cmd_serve_lucebox(args) -> int:
    import time
    from .inference import LuceboxDFlashServer

    server = LuceboxDFlashServer(
        target_gguf=args.target,
        draft_gguf=args.draft,
        binary=args.binary,
        host=args.host,
        port=args.port,
        max_ctx=args.max_ctx,
        default_max_tokens=args.default_max_tokens,
        think_max_tokens=args.think_max_tokens,
        hard_limit_reply_budget=args.hard_limit_reply_budget,
        model_name=args.model_name,
        prefix_cache_slots=args.prefix_cache_slots,
        accept_mode=args.accept_mode,
        accept_eta=args.accept_eta,
        accept_topk=args.accept_topk,
        log_path=args.log,
    )
    server.start()
    print(f"Lucebox dflash_server up: {server.url}")
    print("Endpoints: /v1/chat/completions, /v1/models")
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
    sg.add_argument("--backend", default="llamacpp_gguf", choices=["llamacpp_gguf"])
    sg.add_argument("--binary", default="llama-dump-hiddens")
    sg.add_argument("--ctx", type=int, default=4096)
    sg.add_argument("--timeout", type=int, default=600)
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
    # FP8 production training (0.2.0+). See src/dflash_llama/training/fp8.py
    # for the rationale behind every default in this group.
    fp8 = st.add_argument_group(
        "FP8 training (0.2.0+)",
        "TransformerEngine FP8 + fused te.LayerNormMLP. Verified production "
        "on MiniMax-M2.7-IQ4-XS v12, Spark sm_121, 2026-05-05. See "
        "repro/04-fp8-bringup.md.",
    )
    fp8.add_argument("--fp8-recipe", default=None,
        choices=["current_fp8", "delayed_e4m3", "block_fp8", "mxfp8", "bf16", "none"],
        help="FP8 recipe kind. 'current_fp8' is the production recipe. "
             "'block_fp8' is REFUSED on sm_120/121 (silently non-convergent, TE #2382). "
             "'mxfp8' via TE is REFUSED on sm_120/121 (cuBLAS layout, TE #2668). "
             "Omit (or pass 'bf16'/'none') to run bf16.")
    fp8.add_argument("--no-te-fused", action="store_true",
        help="Disable fused te.LayerNormMLP (keep separate norm+mlp). "
             "Only do this if you're debugging fusion correctness — fusion is the "
             "lever that makes intermediate=6144 fit, not FP8 alone.")
    fp8.add_argument("--drafter-intermediate-size", type=int, default=None,
        help="Override drafter MLP intermediate size. Recipe-faithful default "
             "for MiniMax-M2.7-class drafters when FP8+fused is active is 6144 "
             "(v11 used 4096; v12 production uses 6144).")
    fp8.add_argument("--no-nan-skip", action="store_true",
        help="Disable defensive NaN-skip optimizer guard. Default ON: skips "
             "optimizer.step() when any gradient is NaN/Inf, increments "
             "global_step, continues. Cheap insurance.")
    fp8.add_argument("--use-torchrun", action="store_true",
        help="Launch via torchrun instead of direct python. Default OFF in 0.2.0+ "
             "to avoid the silent-bf16 trap on single-GPU FP8 runs (torchrun sets "
             "WORLD_SIZE=1 which routes speculators through its FSDP branch — and "
             "the TE wrap only lives in the single-GPU branch). Use this only for "
             "genuine multi-GPU runs.")
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
    # FP8 informational flag for smoke runs (smoke is plumbing-only, but we
    # accept the flag so that 'dflash-llama smoke ... --fp8-recipe current_fp8'
    # doesn't error out and so logs reflect intent).
    sm.add_argument("--fp8-recipe", default=None,
        choices=["current_fp8", "delayed_e4m3", "block_fp8", "mxfp8", "bf16", "none"],
        help="(informational) FP8 recipe kind being smoke-tested. "
             "Smoke itself runs the bf16 plumbing path; full FP8 paths "
             "are exercised by 'dflash-llama train --fp8-recipe ...'.")
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

    # check-fp8 (0.2.0+) — arch capability probe
    sc = sub.add_parser("check-fp8",
        help="probe CUDA arch + print recommended FP8 recipe (0.2.0+)")
    sc.set_defaults(func=cmd_check_fp8)

    # export-gguf
    sx = sub.add_parser("export-gguf",
                        help="convert a DFlash drafter checkpoint to a buun-loadable GGUF")
    sx.add_argument("--checkpoint", required=True,
                    help="speculators-format checkpoint dir (config.json + model.safetensors)")
    sx.add_argument("--output", required=True, help="path to write the GGUF")
    sx.add_argument("--verifier-meta-dir", default=None,
                    help="directory holding tokenizer.json etc (default: read from config)")
    sx.add_argument("--d2t-path", default=None,
                    help="draft-to-target vocab map .npy for adapter-only output-head rebake")
    sx.add_argument("--buun-repo", default="buun-llama-cpp",
                    help="buun-llama-cpp checkout containing convert_hf_to_gguf.py")
    sx.add_argument("--venv-python", default=None,
                    help="python interpreter to invoke buun's converter (default: autodetect)")
    sx.add_argument("--outtype", default="bf16", choices=["bf16", "f16", "f32"])
    sx.add_argument("--rebake-floor", type=float, default=-65504.0,
                    help="floor for non-mapped rows in rebaked lm_head (default: -65504)")
    sx.add_argument("--prepped-dir", default=None,
                    help="staging dir for the prepped checkpoint (default: <output>.prep)")
    sx.add_argument("--force-block-size", type=int, default=None,
                    help="optional block-size hint for Lucebox-compatible exports")
    sx.add_argument("--no-register-hash", action="store_true",
                    help="don't auto-whitelist the FP8 tokenizer hash in buun")
    sx.add_argument("--verify", action="store_true",
                    help="after conversion, print GGUF metadata sanity-check")
    sx.set_defaults(func=cmd_export_gguf)

    # export-lucebox
    sl = sub.add_parser("export-lucebox",
                        help="convert a Lucebox-layout DFlash adapter checkpoint to GGUF")
    sl.add_argument("--checkpoint", required=True,
                    help="checkpoint dir or model_lucebox_layout.safetensors file")
    sl.add_argument("--out", "--output", dest="output", required=True,
                    help="path to write the GGUF")
    sl.add_argument("--verifier-meta-dir", default=None,
                    help="directory holding tokenizer.json etc (default: read from config)")
    sl.add_argument("--d2t-path", default=None,
                    help="draft-to-target vocab map .npy for sparse served-drafter output head")
    sl.add_argument("--buun-repo", default="buun-llama-cpp",
                    help="buun-llama-cpp checkout containing convert_hf_to_gguf.py")
    sl.add_argument("--venv-python", default=None,
                    help="python interpreter to invoke buun's converter (default: current Python)")
    sl.add_argument("--outtype", default="bf16", choices=["bf16", "f16", "f32"])
    sl.add_argument("--rebake-floor", type=float, default=-65504.0,
                    help="floor for non-mapped rows in rebaked lm_head (default: -65504)")
    sl.add_argument("--prepped-dir", default=None,
                    help="staging dir for the prepped checkpoint (default: <output>.prep)")
    sl.add_argument("--force-block-size", type=int, default=8,
                    help="expected Lucebox DFlash block size (default: 8)")
    sl.add_argument("--no-register-hash", action="store_true",
                    help="don't auto-whitelist the FP8 tokenizer hash in buun")
    sl.add_argument("--verify", action="store_true",
                    help="after conversion, print GGUF metadata sanity-check")
    sl.set_defaults(func=cmd_export_lucebox)

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

    # serve-lucebox (native Lucebox dflash_server)
    sls = sub.add_parser("serve-lucebox",
                         help="run Lucebox dflash_server with typical/top-k accept knobs")
    sls.add_argument("--target", required=True, help="target GGUF path")
    sls.add_argument("--draft", required=True, help="drafter GGUF path")
    sls.add_argument("--binary", default=None, help="dflash_server binary (default: PATH lookup)")
    sls.add_argument("--host", default="0.0.0.0")
    sls.add_argument("--port", type=int, default=8080)
    sls.add_argument("--max-ctx", type=int, default=2304)
    sls.add_argument("--default-max-tokens", type=int, default=256)
    sls.add_argument("--think-max-tokens", type=int, default=256)
    sls.add_argument("--hard-limit-reply-budget", type=int, default=0)
    sls.add_argument("--model-name", default="dflash")
    sls.add_argument("--prefix-cache-slots", type=int, default=0)
    sls.add_argument("--accept-mode", default="strict", choices=["strict", "eta", "topk"])
    sls.add_argument("--accept-eta", type=float, default=None)
    sls.add_argument("--accept-topk", type=int, default=None)
    sls.add_argument("--log", default=None, help="optional log file path")
    sls.set_defaults(func=cmd_serve_lucebox)

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

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
