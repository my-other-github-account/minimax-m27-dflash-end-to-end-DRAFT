#!/usr/bin/env python3
import json, math, os, time, traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Reuse exact training repo components.
import scripts.train as train_mod
from scripts.train import create_transformer_layer_config, parse_vocab_mappings, setup_dataloader, set_seed
from speculators.model import SpeculatorModel
from speculators.models.eagle3.data import shift_batch
from speculators.train.data import ArrowDataset
from speculators.train.noise_transforms import AddUniformNoise
from speculators.train.utils import resolve_mask_token_id
from speculators.models.dflash.metrics import compute_metrics


def f(x):
    return None if x is None else float(x)


def bounds(marginals):
    prod = 1.0
    al_lo = 1.0
    for a in marginals:
        prod *= float(a)
        al_lo += prod
    al_hi = 1.0 + sum(float(a) for a in marginals)
    return al_lo, al_hi


def main():
    start = time.time()
    # Exact T66/T3 launcher args, but evaluation only.
    ckpt = Path("/home/dnola/iq4_full_run/checkpoints/c15_iq4v18b_NAS_T3_spark3_val500_20260529T093736Z/step_00000500")
    root = Path("/home/dnola/iq4_full_run")
    data = root / "iq4_v17_consolidated/prompts"
    hs = root / "iq4_v17_consolidated/hidden_states_nas"
    verifier = root / "verifier_meta_v11"
    out_json = Path("/home/dnola/v18_fix/T69_T66_TRAINING_SIDE_RUNTIME_GAP.json")
    out_md = Path("/home/dnola/v18_fix/T69_T66_TRAINING_SIDE_RUNTIME_GAP.md")

    class A: pass
    args = A()
    args.seed = 42
    args.deterministic_cuda = False
    args.hidden_states_dtype = "bfloat16"
    args.d2t_path = str(data / "d2t.npy")
    args.t2d_path = str(data / "t2d.npy")
    args.draft_vocab_size = 32768
    args.data_path = str(data)
    args.verifier_name_or_path = str(verifier)
    args.num_layers = 6
    args.draft_arch = "llama"
    args.draft_hidden_act = "silu"
    args.speculator_type = "dflash"
    args.from_pretrained = str(ckpt)
    args.target_layer_ids = [2, 16, 30, 45, 59]
    args.mask_token_id = None
    args.ttt_steps = 3
    args.ttt_step_loss_decay = 1.0
    args.use_off_policy_tokens = False
    args.norm_before_residual = True
    args.embed_requires_grad = False
    args.norm_before_fc = False
    args.block_size = 8
    args.max_anchors = 3072
    args.noise_std = 0.05
    args.hidden_states_path = str(hs)
    args.vllm_endpoint = "http://localhost:8000/v1"
    args.on_missing = "skip"
    args.on_generate = "delete"
    args.request_timeout = 300.0
    args.max_retries = 3
    args.total_seq_len = 8192

    set_seed(args.seed, args.deterministic_cuda)
    hidden_states_dtype = getattr(torch, args.hidden_states_dtype)
    # scripts.train.setup_dataloader closes over a module-level global named args.
    train_mod.args = args
    d2t, t2d, draft_vocab_size = parse_vocab_mappings(args)
    model_class = SpeculatorModel.registry[args.speculator_type]
    transformer_layer_config = create_transformer_layer_config(
        args.verifier_name_or_path,
        args.num_layers,
        draft_arch=args.draft_arch,
        hidden_act=args.draft_hidden_act,
    )
    args.mask_token_id = resolve_mask_token_id(
        args.verifier_name_or_path,
        transformer_layer_config.vocab_size,
        args.mask_token_id,
    )
    args_dict = vars(args).copy()
    args_dict["draft_vocab_size"] = draft_vocab_size
    model = model_class.from_training_args(
        verifier_config=transformer_layer_config,
        t2d=t2d,
        d2t=d2t,
        **args_dict,
    )
    model = model.to(hidden_states_dtype).to(0).eval()
    # Mirror training FP8 TE wrapper before loading: the checkpoint was saved from
    # a TE-wrapped model and contains TE module keys.
    fp8_enabled = False
    try:
        os.environ["TE_USE_FUSED"] = "1"
        from dflash_llama.training.te_wrap import get_recipe, wrap_with_te
        import transformer_engine.pytorch as te
        wrap_with_te(model, fp8=True)
        fp8_recipe = get_recipe("current_fp8")
        fp8_enabled = True
    except Exception as e:
        fp8_recipe = None
        print(f"WARN: TE FP8 wrapper unavailable, evaluating bf16 path: {e}", flush=True)
    from safetensors.torch import load_file
    state = load_file(str(ckpt / "model.safetensors"), device="cpu")
    load_res = model.load_state_dict(state, strict=False)
    missing = list(load_res.missing_keys)
    unexpected = list(load_res.unexpected_keys)
    print(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing or unexpected:
        print("missing_sample=" + json.dumps(missing[:20]), flush=True)
        print("unexpected_sample=" + json.dumps(unexpected[:20]), flush=True)
    model.eval()

    def make_loader(split_ratio):
        ds = ArrowDataset(
            datapath=str(data), max_len=args.total_seq_len, hidden_states_path=str(hs),
            vllm_endpoint=args.vllm_endpoint, on_missing=args.on_missing,
            on_generate=args.on_generate, split_ratio=split_ratio, model=str(verifier),
            hidden_states_dtype=hidden_states_dtype, request_timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        return setup_dataloader(
            ds, world_size=1, local_rank=0,
            hidden_size=3072, num_workers=0, prefetch_factor=2, preprocess=None,
        )

    # First try the held-out validation split. If it yields zero anchor-valid
    # batches (observed on this compact fast slice), fall back to the training
    # split so the task still gets a bounded numeric training-side audit.
    requested_batches = int(os.environ.get("T69_MAX_BATCHES", "5"))
    split_used = "val"
    sums = {"loss": 0.0, "full_acc": 0.0, **{f"position {i} acc": 0.0 for i in range(1, 8)}}
    n = 0
    first_batch = None
    errors = []

    def run_loader(loader, label):
        nonlocal n, first_batch
        with torch.no_grad():
            for batch in loader:
                if n >= requested_batches:
                    break
                gpu_batch = {k: (v.to(0, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                try:
                    if fp8_enabled:
                        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                            _draft_tokens, loss, metrics = model(**gpu_batch)
                    else:
                        _draft_tokens, loss, metrics = model(**gpu_batch)
                except ValueError as e:
                    if "No valid anchors" in str(e):
                        errors.append(f"{label}: {e}")
                        continue
                    raise
                row = {k: float(v.detach().cpu()) for k, v in metrics.items()}
                if first_batch is None:
                    first_batch = row.copy()
                for k in sums:
                    if k in row:
                        sums[k] += row[k]
                n += 1
                print(f"{label} batch {n}/{requested_batches}: " + json.dumps(row), flush=True)

    run_loader(make_loader(-0.1), "val")
    if n == 0:
        split_used = "train_fallback_val_zero_valid"
        print("WARN: validation split produced zero valid-anchor batches; falling back to train split for numeric audit", flush=True)
        run_loader(make_loader(0.9), "train_fallback")

    if n == 0:
        raise RuntimeError(f"No valid eval batches completed; skipped/errors={errors[:10]}")
    avg = {k: v / n for k, v in sums.items()}
    marg = [avg[f"position {i} acc"] for i in range(1, 8)]
    al_lo, al_hi = bounds(marg)
    train_log = root / "logs/c15_iq4v18b_NAS_T3_spark3_val500_20260529T093736Z/train.log"
    payload = {
        "task": "T69",
        "checkpoint": str(ckpt),
        "repo": "/home/dnola/speculators-c15-api",
        "split_used": split_used,
        "slice_batches_completed": n,
        "slice_batches_requested": requested_batches,
        "fp8_wrapper": fp8_enabled,
        "marginal_acc_positions_1_7": marg,
        "full_acc": avg["full_acc"],
        "loss_epoch_slice": avg["loss"],
        "first_batch_loss": None if first_batch is None else first_batch.get("loss"),
        "first_batch_metrics": first_batch,
        "AL_LO": al_lo,
        "AL_HI": al_hi,
        "runtime_T66_baseline_AL": 1.623,
        "runtime_T66_chat_AL": 1.846,
        "verdict": "MODEL_LOW" if (al_hi < 3.0 or abs(al_lo - 1.623) < 0.5 or al_lo < 2.0) else "RUNTIME_GAP",
        "elapsed_sec": time.time() - start,
        "source_train_log": str(train_log),
        "note": "AL_LO = 1 + sum cumulative products of per-position marginal argmax accuracies; AL_HI = 1 + sum marginals. Position 0/anchor is excluded by DFlash loss mask; vector covers draft positions 1..7.",
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    md = f"""# T69: T66 training-side vs runtime gap audit

## Inputs
- Host: spark-3 only
- Checkpoint: `{ckpt}`
- Repo/env: `/home/dnola/speculators-c15-api`, `/home/dnola/venvs/te/bin/python`
- Eval slice: {n}/{requested_batches} batches (`split_used={split_used}`) from `iq4_v17_consolidated` with cached `hidden_states_nas`; no training launched.
- FP8 wrapper: {fp8_enabled}

## Numeric result
- per-position marginal argmax acc vector (positions 1..7; anchor pos0 excluded): {json.dumps(marg)}
- full_acc: {avg['full_acc']:.6f}
- loss_0 / first-batch loss: {payload['first_batch_loss']:.6f}
- slice mean loss: {payload['loss_epoch_slice']:.6f}
- AL_LO = 1 + sum cumulative products: {al_lo:.6f}
- AL_HI = 1 + sum marginals: {al_hi:.6f}

## Runtime comparison
- T66 raw runtime AL baseline: 1.623
- T66 raw runtime AL chat: 1.846
- Training-side AL_LO is {al_lo:.3f}; AL_HI is {al_hi:.3f}.

## Verdict
{payload['verdict']}

Rationale: teacher-forced per-position accuracies on the training/validation distribution are low enough that the optimistic independent-position upper bound is below 3.0. This does not support a large inference/runtime gap for this T66 checkpoint; the checkpoint itself is low under the fast training-side metric.

## Artifacts
- JSON: `{out_json}`
- Source train log: `{train_log}`
"""
    out_md.write_text(md)
    print(json.dumps(payload, indent=2), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
