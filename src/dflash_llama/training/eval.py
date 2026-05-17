"""Offline DFlash drafter eval (§2.8 validator).

Reproduces the trainer's val-split forward pass on a checkpoint to compute
per-position accuracies. Requires speculators to be importable; raises a
helpful error on macmini-style boxes that don't have it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def _load_checkpoint_state_dict(checkpoint: Path) -> dict:
    from safetensors import safe_open

    from .te_wrap import fused_to_unfused_state_dict

    state_dict = {}
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.endswith("._extra_state")
    }
    if any(
        key.endswith("layer_norm_weight") or key.endswith("fc1_weight") or key.endswith("fc2_weight")
        for key in state_dict
    ):
        state_dict = fused_to_unfused_state_dict(state_dict, model_arch="qwen3")
    return state_dict


def offline_eval(
    *,
    checkpoint: str,
    paired_dir: str,
    verifier_path: str,
    max_batches: int = 60,
    total_seq_len: int = 2048,
    val_split_ratio: float = -0.1,
    speculators_repo: Optional[str] = None,
) -> dict:
    """Run the offline eval. Returns a small metrics dict.

    The val split convention matches the trainer (``-0.1`` = last 10%).
    """
    if speculators_repo is None:
        speculators_repo = os.environ.get(
            "SPECULATORS_REPO",
            os.path.expanduser("~/repos/speculators"),
        )
    scripts_dir = os.path.join(speculators_repo, "scripts")
    if os.path.isdir(scripts_dir) and scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    try:
        import torch
        from speculators.models.dflash.core import DFlashDraftModel  # noqa: F401
        from speculators.train.data import ArrowDataset, create_collate_fn  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "offline_eval needs speculators importable. Set SPECULATORS_REPO "
            f"or pip install speculators. Original error: {e}"
        ) from e

    from speculators.models.dflash.core import DFlashDraftModel
    from speculators.train.data import ArrowDataset, create_collate_fn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # @torch.compile single-batch bug — force eager
    try:
        torch.compiler.set_stance("force_eager")
    except Exception:
        pass

    checkpoint_path = Path(checkpoint)
    state_dict = _load_checkpoint_state_dict(checkpoint_path)
    t2d = state_dict.pop("t2d", None)
    d2t = state_dict.pop("d2t", None)
    config = DFlashDraftModel.config_class.from_pretrained(checkpoint)
    model = DFlashDraftModel.from_pretrained(
        None,
        config=config,
        torch_dtype=torch.bfloat16,
        state_dict=state_dict,
        t2d=t2d,
        d2t=d2t,
    )
    if hasattr(model, "load_verifier_weights"):
        try:
            model.config.speculators_config.verifier.name_or_path = verifier_path
        except Exception:
            pass
        model.load_verifier_weights()
    model = model.to(device).eval()

    paired = Path(paired_dir)
    ds = ArrowDataset(
        max_len=total_seq_len,
        datapath=str(paired / "prompts"),
        hidden_states_path=str(paired / "hidden_states"),
        split_ratio=val_split_ratio,
    )
    hidden_size = getattr(getattr(model.config, "transformer_layer_config", None), "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model.config, "hidden_size")
    collate = create_collate_fn(max_len=total_seq_len, hidden_size=hidden_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate)

    correct = [0] * 8
    total = [0] * 8
    batches_seen = 0
    for batch in loader:
        if batches_seen >= max_batches:
            break
        gpu_batch = {}
        for key, value in batch.items():
            if hasattr(value, "to"):
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    gpu_batch[key] = value.to(device=device, dtype=torch.bfloat16)
                else:
                    gpu_batch[key] = value.to(device)
            else:
                gpu_batch[key] = value
        with torch.no_grad():
            preds = model(**gpu_batch)
        # Best-effort metric extraction: look for predicted/target pairs.
        if isinstance(preds, dict) and "pred" in preds and "target" in preds:
            pred = preds["pred"]
            tgt = preds["target"]
            for i in range(min(pred.shape[1] if pred.dim() >= 2 else 1, len(correct))):
                m = (pred[:, i] == tgt[:, i]).float()
                correct[i] += int(m.sum().item())
                total[i] += int(m.numel())
        batches_seen += 1

    metrics = {f"pos_{i}_acc": (correct[i] / total[i] if total[i] else None) for i in range(8)}
    metrics["batches_evaluated"] = batches_seen
    return metrics


__all__ = ["offline_eval"]
