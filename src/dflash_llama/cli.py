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


def _build_verifier(args) -> "BaseVerifier":  # noqa: F821
    return load_verifier(
        args.verifier,
        gguf_path=getattr(args, "gguf_path", None),
        hf_path=getattr(args, "hf_path", None),
        gguf_repo=getattr(args, "gguf_repo", None),
        hf_repo=getattr(args, "hf_repo", None),
        gguf_quant=getattr(args, "gguf_quant", None),
        revision=getattr(args, "revision", None),
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


# ----- arg-parser -----
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dflash-llama", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    # common verifier args
    def add_verifier_args(sp):
        sp.add_argument("--verifier", required=True, help="verifier name (e.g. minimax-m2.7-iq4-xs). Use 'dflash-llama info' to list.")
        sp.add_argument("--gguf-path", default=None, help="local path to a GGUF shard (mutually exclusive with --gguf-repo)")
        sp.add_argument("--hf-path", default=None, help="local path to an HF model directory (mutually exclusive with --hf-repo)")
        sp.add_argument("--gguf-repo", default=None, help="HF Hub slug for GGUF weights, e.g. 'unsloth/MiniMax-M2-GGUF'")
        sp.add_argument("--hf-repo", default=None, help="HF Hub slug for the model config + tokenizer, e.g. 'MiniMaxAI/MiniMax-M2'")
        sp.add_argument("--gguf-quant", default=None, help="quant subdir within --gguf-repo, e.g. 'UD-IQ4_XS'")
        sp.add_argument("--revision", default=None, help="optional Hub revision (branch, tag, or commit)")

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

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
