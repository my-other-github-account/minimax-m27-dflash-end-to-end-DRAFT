"""Vocab-map builder.

Generates the canonical ``d2t.npy`` / ``t2d.npy`` / ``token_freq.pt`` files
the speculators trainer expects in the prompts directory.

This module fixes the v2 ``build_vocab_maps.py`` bugs:

  - **Import path:** ``speculators.train.vocab_mapping`` (not
    ``speculators.train.utils.vocab_mapping``).
  - **Dtype coercion:** speculators may return torch tensors on some
    versions; we coerce to numpy with the canonical dtypes
    (``t2d`` -> ``np.bool_``, ``d2t`` -> ``np.int64``).
  - **Format:** ``t2d`` is a BOOL MASK (``sum() == draft_vocab_size``),
    ``d2t`` is an OFFSET TABLE (``verifier_token = draft_id + d2t[draft_id]``).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch
from datasets import load_from_disk


def _coerce_t2d(arr) -> np.ndarray:
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.dtype != np.bool_:
        arr = arr.astype(np.bool_)
    return arr


def _coerce_d2t(arr) -> np.ndarray:
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.dtype != np.int64:
        arr = arr.astype(np.int64)
    return arr


def count_token_frequencies(prompts_dir: Union[str, Path]) -> tuple[Counter, int]:
    """Walk a prompts arrow dir and count per-token frequencies under loss_mask."""
    ds = load_from_disk(str(prompts_dir))
    if "input_ids" not in ds.column_names or "loss_mask" not in ds.column_names:
        raise ValueError(
            f"{prompts_dir}: dataset must have input_ids and loss_mask columns"
        )
    counter: Counter = Counter()
    total = 0
    for row in ds:
        ids = row["input_ids"]
        mask = row["loss_mask"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        for tok, m in zip(ids, mask):
            if m:
                counter[int(tok)] += 1
                total += 1
    return counter, total


def build_vocab_maps_from_counts(
    counter: Counter,
    *,
    verifier_vocab_size: int,
    draft_vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce ``(t2d, d2t)`` from a token-frequency Counter.

    Delegates to ``speculators.train.vocab_mapping.build_vocab_mappings_from_distribution``
    when available; otherwise falls back to a local implementation that produces
    the same canonical format (top-K by frequency, sorted ascending by token id).
    """
    freq_dict = dict(counter)
    try:
        from speculators.train.vocab_mapping import build_vocab_mappings_from_distribution

        d2t, t2d = build_vocab_mappings_from_distribution(
            freq_dict,
            target_vocab_size=verifier_vocab_size,
            draft_vocab_size=draft_vocab_size,
        )
        t2d = _coerce_t2d(t2d)
        d2t = _coerce_d2t(d2t)
    except ImportError:
        # Local fallback: pick top-K most-frequent token ids, fill remaining
        # slots from the lowest unused token ids (so the draft vocab always
        # has exactly draft_vocab_size entries even when fewer than K tokens
        # appear in the corpus).
        # Restrict the corpus to in-vocab tokens (drop OOV)
        top = [
            (tok, c) for tok, c in counter.most_common()
            if 0 <= int(tok) < verifier_vocab_size
        ][:draft_vocab_size]
        chosen = {int(tok) for tok, _ in top}
        if len(chosen) < draft_vocab_size:
            for tok in range(verifier_vocab_size):
                if len(chosen) >= draft_vocab_size:
                    break
                if tok not in chosen:
                    chosen.add(tok)
        chosen_sorted = sorted(chosen)
        if len(chosen_sorted) != draft_vocab_size:
            raise RuntimeError(
                f"fallback vocab map produced {len(chosen_sorted)} entries, "
                f"expected {draft_vocab_size}"
            )
        t2d = np.zeros(verifier_vocab_size, dtype=np.bool_)
        for tok in chosen_sorted:
            t2d[tok] = True
        d2t = np.array(
            [tok - i for i, tok in enumerate(chosen_sorted)], dtype=np.int64
        )

    # Validate the canonical format
    if t2d.dtype != np.bool_:
        raise TypeError(f"t2d must be bool, got {t2d.dtype}")
    if t2d.shape != (verifier_vocab_size,):
        raise ValueError(f"t2d.shape {t2d.shape} != ({verifier_vocab_size},)")
    if int(t2d.sum()) != draft_vocab_size:
        raise ValueError(
            f"t2d.sum()={int(t2d.sum())} != draft_vocab_size={draft_vocab_size}"
        )
    if d2t.dtype != np.int64:
        raise TypeError(f"d2t must be int64, got {d2t.dtype}")
    if d2t.shape != (draft_vocab_size,):
        raise ValueError(f"d2t.shape {d2t.shape} != ({draft_vocab_size},)")
    return t2d, d2t


def build_vocab_maps(
    prompts_dir: Union[str, Path],
    *,
    verifier_vocab_size: int,
    draft_vocab_size: int,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """Walk a prompts arrow dir, count frequencies, write the canonical maps.

    Writes ``t2d.npy``, ``d2t.npy``, ``token_freq.pt`` into ``output_dir``
    (defaults to ``prompts_dir``). Returns a small report dict.
    """
    prompts_dir = Path(prompts_dir)
    out_dir = Path(output_dir) if output_dir is not None else prompts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    counter, total = count_token_frequencies(prompts_dir)
    top = counter.most_common(draft_vocab_size)
    top_count = sum(c for _, c in top)
    coverage = 100.0 * top_count / max(1, total)
    t2d, d2t = build_vocab_maps_from_counts(
        counter,
        verifier_vocab_size=verifier_vocab_size,
        draft_vocab_size=draft_vocab_size,
    )
    np.save(out_dir / "t2d.npy", t2d)
    np.save(out_dir / "d2t.npy", d2t)
    torch.save(dict(counter), out_dir / "token_freq.pt")

    report = {
        "prompts_dir": str(prompts_dir),
        "output_dir": str(out_dir),
        "n_rows": _row_count(prompts_dir),
        "total_loss_mask_tokens": total,
        "unique_tokens_seen": len(counter),
        "verifier_vocab_size": verifier_vocab_size,
        "draft_vocab_size": draft_vocab_size,
        "top_k_coverage_pct": round(coverage, 4),
        "d2t_unique_offsets": int(len(np.unique(d2t))),
    }
    return report


def _row_count(prompts_dir: Union[str, Path]) -> int:
    try:
        ds = load_from_disk(str(prompts_dir))
        return len(ds)
    except Exception:
        return -1


__all__ = [
    "build_vocab_maps",
    "build_vocab_maps_from_counts",
    "count_token_frequencies",
]
