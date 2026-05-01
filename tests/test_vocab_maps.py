"""Tests for vocab-map building."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset

from dflash_llama.training.vocab_maps import (
    build_vocab_maps,
    build_vocab_maps_from_counts,
    count_token_frequencies,
)


def _fake_counter(verifier_vocab=2000, n_distinct=300, seed=0):
    """Make a Counter where 300 tokens have nonzero frequency."""
    rng = np.random.default_rng(seed)
    keys = rng.choice(verifier_vocab, size=n_distinct, replace=False)
    counts = rng.integers(low=1, high=100, size=n_distinct)
    return Counter({int(k): int(c) for k, c in zip(keys, counts)})


# -----------------------------------------------------------------------
# Fallback path (speculators not required)
# -----------------------------------------------------------------------
def test_build_from_counts_fallback_dtypes():
    counter = _fake_counter()
    t2d, d2t = build_vocab_maps_from_counts(
        counter, verifier_vocab_size=2000, draft_vocab_size=128,
    )
    assert t2d.dtype == np.bool_
    assert t2d.shape == (2000,)
    assert int(t2d.sum()) == 128
    assert d2t.dtype == np.int64
    assert d2t.shape == (128,)


def test_build_from_counts_d2t_offset_semantics():
    """For each draft id i, d2t[i] should equal verifier_id - i."""
    counter = _fake_counter(verifier_vocab=500, n_distinct=64)
    t2d, d2t = build_vocab_maps_from_counts(
        counter, verifier_vocab_size=500, draft_vocab_size=64,
    )
    chosen = sorted(np.where(t2d)[0].tolist())
    assert len(chosen) == 64
    for i, verifier_tok in enumerate(chosen):
        assert int(d2t[i]) == verifier_tok - i, (
            f"d2t[{i}] should be {verifier_tok - i}, got {int(d2t[i])}"
        )


def test_build_from_counts_pads_when_few_tokens_seen():
    """If fewer than draft_vocab_size unique tokens appear, fallback should
    fill remaining slots from low ids — t2d.sum() must still equal draft_size.
    """
    counter = Counter({0: 100, 1: 50, 2: 10})
    t2d, d2t = build_vocab_maps_from_counts(
        counter, verifier_vocab_size=1000, draft_vocab_size=32,
    )
    assert int(t2d.sum()) == 32


def test_build_from_counts_torch_tensor_inputs_coerced():
    """If speculators returned torch tensors, the helper must coerce."""
    counter = _fake_counter(verifier_vocab=500, n_distinct=200)
    # Re-run the helper with a monkeypatched speculators path that returns
    # torch tensors, to verify the coercion code path. We do this by importing
    # the module and calling the coerce helpers directly.
    from dflash_llama.training.vocab_maps import _coerce_t2d, _coerce_d2t

    fake_t2d = torch.zeros(500, dtype=torch.uint8)
    fake_t2d[10:50] = 1
    fake_d2t = torch.arange(40)  # int64 by default
    t2d_n = _coerce_t2d(fake_t2d)
    d2t_n = _coerce_d2t(fake_d2t)
    assert t2d_n.dtype == np.bool_
    assert d2t_n.dtype == np.int64


# -----------------------------------------------------------------------
# End-to-end against a tiny prompts dataset
# -----------------------------------------------------------------------
def test_build_vocab_maps_e2e_on_small_dataset(tmp_path):
    rows = []
    for i in range(20):
        seq = ((i * 7 + 3) % 50) + 4
        ids = [(i * 11 + j) % 200 for j in range(seq)]
        mask = [(j % 2 == 0) for j in range(seq)]
        rows.append({"input_ids": ids, "loss_mask": mask})
    ds = Dataset.from_list(rows)
    prompts_dir = tmp_path / "prompts"
    ds.save_to_disk(str(prompts_dir))

    counter, total = count_token_frequencies(prompts_dir)
    assert total > 0
    assert all(0 <= k < 200 for k in counter)

    report = build_vocab_maps(
        prompts_dir, verifier_vocab_size=200, draft_vocab_size=64,
    )
    assert report["draft_vocab_size"] == 64
    assert report["verifier_vocab_size"] == 200
    assert (prompts_dir / "t2d.npy").exists()
    assert (prompts_dir / "d2t.npy").exists()
    assert (prompts_dir / "token_freq.pt").exists()

    t2d = np.load(prompts_dir / "t2d.npy")
    d2t = np.load(prompts_dir / "d2t.npy")
    assert t2d.dtype == np.bool_
    assert int(t2d.sum()) == 64
    assert d2t.dtype == np.int64
    assert d2t.shape == (64,)
