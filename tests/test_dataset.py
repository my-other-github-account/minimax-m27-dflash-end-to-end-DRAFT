"""Tests for SelfDescribingTraceDataset."""
from __future__ import annotations

import torch

from dflash_llama.training.dataset import SelfDescribingTraceDataset


def test_dataset_length_and_keys(synthetic_trace_dir):
    ds = SelfDescribingTraceDataset(str(synthetic_trace_dir))
    assert len(ds) == 5
    row = ds[0]
    expected = {"hidden_states", "token_ids", "input_ids", "loss_mask", "metadata", "_path"}
    assert expected.issubset(row.keys())


def test_dataset_field_dtypes(synthetic_trace_dir):
    ds = SelfDescribingTraceDataset(str(synthetic_trace_dir))
    row = ds[0]
    assert row["hidden_states"].dtype == torch.bfloat16
    assert row["hidden_states"].dim() == 3
    assert row["token_ids"].dtype == torch.int64
    assert row["input_ids"].dtype == torch.int64
    assert row["loss_mask"].dtype == torch.bool


def test_dataset_no_nan_in_any_row(synthetic_trace_dir):
    """The headline guarantee — even with abs_max well past 448, no NaN."""
    ds = SelfDescribingTraceDataset(str(synthetic_trace_dir))
    for i in range(len(ds)):
        row = ds[i]
        n_nan = int(torch.isnan(row["hidden_states"]).sum().item())
        assert n_nan == 0, f"row {i} has {n_nan} NaN in hidden_states"


def test_dataset_metadata_propagates(synthetic_trace_dir):
    ds = SelfDescribingTraceDataset(str(synthetic_trace_dir))
    seen_idxs = set()
    for i in range(len(ds)):
        meta = ds[i]["metadata"]
        assert meta["source_name"] == "synthetic"
        seen_idxs.add(int(meta["source_row_idx"]))
    assert seen_idxs == {0, 1, 2, 3, 4}


def test_dataset_input_ids_and_token_ids_distinct(synthetic_trace_dir):
    """The synthetic fixture sets input_ids = token_ids + 1; verify it round-trips."""
    ds = SelfDescribingTraceDataset(str(synthetic_trace_dir))
    for i in range(len(ds)):
        row = ds[i]
        assert torch.all(row["input_ids"] == row["token_ids"] + 1), \
            f"row {i}: input_ids/token_ids drift"


def test_dataset_files_kwarg(synthetic_trace_dir):
    """Explicit file list overrides the glob."""
    files = sorted(synthetic_trace_dir.glob("hs_*.safetensors"))[:2]
    ds = SelfDescribingTraceDataset(str(synthetic_trace_dir), files=[str(f) for f in files])
    assert len(ds) == 2
