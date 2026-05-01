"""Tests for the self-describing trace format."""
from __future__ import annotations

import json
import math

import pytest
import torch

from dflash_llama.generation.format import (
    FP8_E4M3FN_MAX,
    SCHEMA_VERSION,
    load_trace,
    save_trace,
    saturating_fp8_cast,
    saturating_fp8_recover,
    validate_trace,
)


def _make_inputs(seq_len=8, n_layers=4, hidden=16, abs_max=2.0, seed=0):
    torch.manual_seed(seed)
    hs = torch.randn(seq_len, n_layers, hidden)
    cur_max = float(hs.abs().max().item())
    if cur_max > 0:
        hs = hs * (abs_max / cur_max)
    token_ids = torch.arange(seq_len, dtype=torch.int64)
    input_ids = token_ids + 1
    loss_mask = torch.ones(seq_len, dtype=torch.bool)
    return hs, token_ids, input_ids, loss_mask


# -----------------------------------------------------------------------
# 1. saturating cast — no NaN even with abs_max=5000
# -----------------------------------------------------------------------
def test_saturating_cast_no_nan_extreme_input():
    hs, *_ = _make_inputs(abs_max=5000.0)
    fp8, scale = saturating_fp8_cast(hs)
    # scale should be > 1 (saturating)
    assert scale > 1.0, f"expected scale > 1.0 with abs_max=5000, got {scale}"
    # NaN check on the recovered tensor
    rec = saturating_fp8_recover(fp8, scale, dtype=torch.float32)
    assert not torch.isnan(rec).any(), "recovered tensor has NaN"
    # The recovered tensor's abs-max should approximately match input abs-max
    assert rec.abs().max().item() == pytest.approx(hs.abs().max().item(), rel=0.05)


def test_saturating_cast_passthrough_small_values():
    """When abs_max <= 448, scale should be exactly 1.0 (no scaling)."""
    hs, *_ = _make_inputs(abs_max=10.0)
    _, scale = saturating_fp8_cast(hs)
    assert scale == 1.0, f"expected scale=1.0 for small inputs, got {scale}"


def test_saturating_cast_at_boundary():
    hs, *_ = _make_inputs(abs_max=FP8_E4M3FN_MAX)  # exactly 448
    _, scale = saturating_fp8_cast(hs)
    assert scale == pytest.approx(1.0)


# -----------------------------------------------------------------------
# 2. roundtrip — within fp8 quantization noise (~1.5%)
# -----------------------------------------------------------------------
def test_save_load_roundtrip_no_nan(tmp_path):
    hs, tok, inp, mask = _make_inputs(abs_max=2260.0, seed=42)  # MiniMax-M2 typical max
    p = tmp_path / "hs_0.safetensors"
    meta = save_trace(
        p,
        hidden_states=hs,
        token_ids=tok,
        input_ids=inp,
        loss_mask=mask,
        source_name="test",
        source_row_idx=0,
        storage="fp8_per_tensor_scale",
        layer_ids=[2, 16, 30, 45, 59, 61],
    )
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["storage"] == "fp8_per_tensor_scale"
    d = load_trace(p)
    rec = d["hidden_states"]
    assert not torch.isnan(rec).any(), "loaded hidden_states contains NaN"
    # Compare to original within fp8 noise
    err = (rec.float() - hs).abs() / (hs.abs().clamp_min(1e-6))
    rel_err = err.median().item()
    assert rel_err < 0.05, f"median relative error {rel_err} exceeds 5%"
    # Token / mask round-trip exactly
    assert torch.equal(d["token_ids"], tok)
    assert torch.equal(d["input_ids"], inp)
    assert torch.equal(d["loss_mask"], mask)


def test_save_load_roundtrip_extreme_no_nan(tmp_path):
    """The headline test: abs_max=5000 in, no NaN out."""
    hs, tok, inp, mask = _make_inputs(abs_max=5000.0, seed=1)
    p = tmp_path / "hs_extreme.safetensors"
    save_trace(
        p,
        hidden_states=hs,
        token_ids=tok,
        input_ids=inp,
        loss_mask=mask,
        source_name="extreme",
        source_row_idx=0,
        storage="fp8_per_tensor_scale",
    )
    d = load_trace(p)
    n_nan = int(torch.isnan(d["hidden_states"]).sum().item())
    assert n_nan == 0, f"loaded trace has {n_nan} NaN values"


def test_metadata_completeness(tmp_path):
    hs, tok, inp, mask = _make_inputs()
    p = tmp_path / "hs.safetensors"
    save_trace(
        p,
        hidden_states=hs, token_ids=tok, input_ids=inp, loss_mask=mask,
        source_name="src", source_row_idx=42,
        storage="fp8_per_tensor_scale",
        layer_ids=[1, 2, 3],
        extra_metadata={"custom": "yes"},
    )
    meta = validate_trace(p)
    assert meta["source_name"] == "src"
    assert meta["source_row_idx"] == "42"
    assert meta["custom"] == "yes"
    assert json.loads(meta["layer_ids"]) == [1, 2, 3]
    assert int(meta["n_layers"]) == hs.shape[1]
    assert int(meta["seq_len"]) == hs.shape[0]
    assert int(meta["hidden_size"]) == hs.shape[2]


def test_bf16_storage_roundtrip(tmp_path):
    hs, tok, inp, mask = _make_inputs(abs_max=600.0)  # bf16 has no clamp problem
    p = tmp_path / "bf16.safetensors"
    save_trace(
        p, hidden_states=hs, token_ids=tok, input_ids=inp, loss_mask=mask,
        source_name="bf16", source_row_idx=0, storage="bf16",
    )
    d = load_trace(p)
    rec = d["hidden_states"]
    assert rec.dtype == torch.bfloat16
    assert d["hidden_states_scale"] == 1.0
    assert torch.allclose(rec.float(), hs.to(torch.bfloat16).float())


def test_validate_rejects_bad_schema(tmp_path):
    """A safetensor without dflash-llama metadata should fail validate_trace."""
    from safetensors.torch import save_file

    p = tmp_path / "bad.safetensors"
    save_file(
        {"hidden_states": torch.zeros(2, 2, 2, dtype=torch.bfloat16),
         "token_ids": torch.zeros(2, dtype=torch.int64)},
        str(p),
        metadata={"foo": "bar"},
    )
    with pytest.raises(ValueError):
        validate_trace(p)


def test_save_rejects_wrong_shapes(tmp_path):
    bad_hs = torch.zeros(8, 4, dtype=torch.float32)  # 2-D, should be 3-D
    with pytest.raises(ValueError):
        save_trace(
            tmp_path / "bad.safetensors",
            hidden_states=bad_hs,
            token_ids=torch.zeros(8, dtype=torch.int64),
            input_ids=torch.zeros(8, dtype=torch.int64),
            loss_mask=torch.zeros(8, dtype=torch.bool),
            source_name="x", source_row_idx=0,
        )
