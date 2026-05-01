"""Shared test fixtures."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def synthetic_trace_dir(tmp_path):
    """Build a small directory of self-describing traces for testing.

    Returns a 5-row dir with random-but-deterministic hidden states whose
    abs-max sometimes exceeds 448 so we exercise the saturating path.
    """
    from dflash_llama.generation.format import save_trace

    d = tmp_path / "traces"
    d.mkdir()
    torch.manual_seed(0)
    for i in range(5):
        seq_len = 32 + i * 4
        n_layers = 6
        hidden = 64
        # Make a tensor whose abs-max sometimes exceeds 448.
        scale_factor = 100.0 + 1500.0 * (i / 4)
        hs = torch.randn(seq_len, n_layers, hidden) * scale_factor
        token_ids = torch.randint(0, 32000, (seq_len,), dtype=torch.int64)
        # Use distinct input_ids so we can verify they round-trip
        input_ids = (token_ids + 1).clone()
        loss_mask = torch.ones(seq_len, dtype=torch.bool)
        loss_mask[: seq_len // 2] = False  # right-side anchors only
        save_trace(
            d / f"hs_{i}.safetensors",
            hidden_states=hs,
            token_ids=token_ids,
            input_ids=input_ids,
            loss_mask=loss_mask,
            source_name="synthetic",
            source_row_idx=i,
            storage="fp8_per_tensor_scale",
            layer_ids=[2, 16, 30, 45, 59, 61],
        )
    return d


@pytest.fixture
def speculators_available() -> bool:
    return importlib.util.find_spec("speculators") is not None
