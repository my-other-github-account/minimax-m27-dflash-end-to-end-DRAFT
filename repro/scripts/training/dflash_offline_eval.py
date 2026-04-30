"""Offline DFlash drafter eval — strict trainer reproduction.

Loads DFlashDraftModel from a trained checkpoint, builds the val split exactly
the same way the trainer does (split_ratio=-0.1 -> last 10%), and runs the
canonical parallel-block forward pass with the trainer's collate_fn. Produces
position-N accuracies that should match val_metrics.json from the checkpoint.

Usage:
    python dflash_offline_eval.py \
        --ckpt ${CHECKPOINTS}/<run>/checkpoint_best \
        --data ${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired/prompts \
        --hs   ${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired/hidden_states \
        --verifier ${MODELS}/MiniMax-M2.7-FP8 \
        --max-batches 60

Run from inside the speculators training repo's scripts/ directory so relative
imports of speculators/train/data.py etc. resolve via the installed package.

Pitfalls handled:
- @torch.compile on DFlashDraftModel.forward is forced to eager (single-batch
  trips a dynamo bug)
- load_verifier_weights() is called explicitly (otherwise verifier_lm_head is NaN)
- Uses speculators.train.data.create_collate_fn for proper batching
- split_ratio=-0.1 selects the LAST 10% (val), NOT the first
"""
import os, sys, json, argparse

# Allow importing speculators from a co-located checkout in addition to whatever
# is already on PYTHONPATH. Override with SPECULATORS_REPO=/path/to/speculators.
_repo = os.environ.get(
    "SPECULATORS_REPO",
    os.path.expanduser("~/repos/speculators"),
)
if os.path.isdir(os.path.join(_repo, "scripts")):
    sys.path.insert(0, os.path.join(_repo, "scripts"))

import torch
from speculators.models.dflash.core import DFlashDraftModel
from speculators.train.data import ArrowDataset, create_collate_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to <run>/checkpoint_best (must contain config.json + model.safetensors)")
    ap.add_argument("--data", required=True,
                    help="Trainer's --data-path (prompts dir with arrow shards)")
    ap.add_argument("--hs", required=True,
                    help="Trainer's --hidden-states-path (dir with hs_*.safetensors)")
    ap.add_argument("--verifier", required=True,
                    help="Trainer's --verifier-name-or-path (HF model dir)")
    ap.add_argument("--max-batches", type=int, default=30,
                    help="How many val batches to run (30 is enough for ±3pp noise)")
    ap.add_argument("--total-seq-len", type=int, default=2048,
                    help="Trainer's --total-seq-len")
    ap.add_argument("--val-split-ratio", type=float, default=-0.1,
                    help="Trainer's val split convention (-0.1 = last 10%)")
    args = ap.parse_args()

    device = "cuda"
    # Force eager — @torch.compile on DFlashDraftModel.forward trips on
    # single-batch, single-anchor inputs with a misleading "repeats must
    # be 0-dim or 1-dim tensor" error from FlexAttention's mask builder.
    torch.compiler.set_stance("force_eager")

    print(f"[eval] loading {args.ckpt}", flush=True)
    model = DFlashDraftModel.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16)

    print(f"[eval] calling load_verifier_weights() — required to populate "
          f"verifier_lm_head from the verifier checkpoint", flush=True)
    model.load_verifier_weights()
    model = model.to(device).eval()

    h = model.config.transformer_layer_config.hidden_size
    bs = model.config.block_size
    print(f"[eval] hidden={h} block_size={bs} draft_vocab={model.config.draft_vocab_size} "
          f"target_layers={model.config.aux_hidden_state_layer_ids}", flush=True)

    # Trainer's val convention: split_ratio=-0.1 means the LAST 10%.
    print(f"[eval] building val dataset with split_ratio={args.val_split_ratio}", flush=True)
    ds_val = ArrowDataset(
        max_len=args.total_seq_len,
        datapath=args.data,
        hidden_states_path=args.hs,
        vllm_endpoint=None,
        on_missing="raise",
        split_ratio=args.val_split_ratio,
        model=args.verifier,
        hidden_states_dtype=torch.bfloat16,
    )
    print(f"[eval] val size = {len(ds_val)}", flush=True)

    collate = create_collate_fn(max_len=args.total_seq_len, hidden_size=h, preprocess=None)

    pos_correct = [0.0] * bs
    pos_denom   = [0]   * bs
    full_acc_sum = 0.0
    full_acc_cnt = 0

    with torch.no_grad():
        N = min(args.max_batches, len(ds_val))
        for i in range(N):
            sample = ds_val[i]
            if sample is None:
                print(f"[eval] sample {i} is None, skipping", flush=True)
                continue
            batch = collate([sample])
            gpu = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            try:
                _draft, _loss, metrics = model(**gpu)
            except Exception as e:
                print(f"[eval] sample {i} forward FAILED: {e}", flush=True)
                continue
            full_acc_sum += float(metrics["full_acc"])
            full_acc_cnt += 1
            for pos in range(1, bs):
                k = f"position {pos} acc"
                if k in metrics:
                    pos_correct[pos] += float(metrics[k])
                    pos_denom[pos] += 1
            if i % 5 == 0 or i == N - 1:
                print(f"[eval] processed {i+1}/{N} batches, "
                      f"full_acc avg={full_acc_sum/max(1,full_acc_cnt):.4f}", flush=True)

    # Try to load reference val_metrics.json if it sits next to the checkpoint
    val_metrics_path = os.path.join(args.ckpt, "val_metrics.json")
    expected = {}
    if os.path.exists(val_metrics_path):
        with open(val_metrics_path) as f:
            ref = json.load(f)
        expected = {
            "full_acc": ref.get("full_acc_epoch"),
            **{int(k.split()[1]): v for k, v in ref.items()
               if k.startswith("position ") and k.endswith("acc_epoch")},
        }

    print(f"\n[eval] N batches    = {full_acc_cnt}")
    full = full_acc_sum / max(1, full_acc_cnt)
    full_ref = expected.get("full_acc")
    print(f"[eval] full_acc     = {full:.4f}    "
          f"(training reported {full_ref:.4f})" if full_ref else f"[eval] full_acc     = {full:.4f}")
    for pos in range(1, bs):
        v = pos_correct[pos] / max(1, pos_denom[pos])
        ex = expected.get(pos)
        if ex is not None:
            delta = v - ex
            mark = "✓" if abs(delta) < 0.05 else "⚠"
            print(f"[eval] position {pos} acc = {v:.4f}    "
                  f"(training reported {ex:.4f}, Δ={delta:+.4f}) {mark}")
        else:
            print(f"[eval] position {pos} acc = {v:.4f}")


if __name__ == "__main__":
    main()
