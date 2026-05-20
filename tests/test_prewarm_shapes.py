"""Tests for ``TraceGenerator.prewarm_shapes`` and the static
``plan_prewarm_shapes`` planner.

These don't spin up the real llama.cpp worker -- they use a fake
in-process backend so we can assert the API plumbs through correctly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest
import torch

from dflash_llama.generation.backends.base import BaseBackend
from dflash_llama.generation.trace_generator import TraceGenerator
from dflash_llama import load_verifier


class _RecordingBackend(BaseBackend):
    """Fake backend that records each (n_seqs, seq_len) invocation and
    returns dummy zero hidden-states. Implements both ``run_one`` and
    ``run_many``."""

    name = "recording"

    def __init__(self, hidden_size: int = 16) -> None:
        self.hidden_size = hidden_size
        self.calls: list[tuple[int, int]] = []  # (n_seqs, seq_len)

    def run_one(
        self,
        input_ids: Sequence[int],
        *,
        layer_ids: Sequence[int],
        max_seq_len: int,
    ) -> tuple[torch.Tensor, list[int]]:
        seq_len = len(input_ids)
        self.calls.append((1, seq_len))
        hs = torch.zeros(seq_len, len(layer_ids), self.hidden_size, dtype=torch.float32)
        return hs, list(input_ids)

    def run_many(
        self,
        batch_inputs: Sequence[Sequence[int]],
        *,
        layer_ids: Sequence[int],
        max_seq_len: int,
    ) -> list[tuple[torch.Tensor, list[int]]]:
        seq_len = len(batch_inputs[0])
        self.calls.append((len(batch_inputs), seq_len))
        out = []
        for ids in batch_inputs:
            hs = torch.zeros(len(ids), len(layer_ids), self.hidden_size, dtype=torch.float32)
            out.append((hs, list(ids)))
        return out


@pytest.fixture
def fake_gen(tmp_path):
    """A TraceGenerator wired to the RecordingBackend with a real
    minimax-m2.7 verifier (used only for its layer_ids / hidden size)."""
    verifier = load_verifier(name="minimax-m2.7-iq4-xs", layer_ids=(2, 16, 30))
    backend = _RecordingBackend(hidden_size=verifier.hidden_size)
    gen = TraceGenerator(verifier=verifier, backend=backend, storage="fp8_per_tensor_scale")
    return gen, backend


def test_plan_prewarm_shapes_matches_observed_production():
    """The shape plan must match what the v14 worker observed in
    production at length_bucket=128, batch_width=8, max_batch_tok=8192,
    max_seq_len=2048. See repro/scripts/iq4_tracegen/launch_v14_batched.sh.
    """
    shapes = TraceGenerator.plan_prewarm_shapes(
        length_bucket=128,
        max_seq_len=2048,
        batch_width=8,
        max_batch_tokens=8192,
    )
    # 16 shapes total
    assert len(shapes) == 16
    # First shapes saturate at batch_width=8
    assert shapes[0] == (128, 8)
    assert shapes[7] == (1024, 8)
    # n_seqs decreases past 1024 because max_batch_tokens // seq_len caps
    assert shapes[8] == (1152, 7)
    assert shapes[9] == (1280, 6)
    assert shapes[10] == (1408, 5)
    assert shapes[11] == (1536, 5)
    assert shapes[12] == (1664, 4)
    assert shapes[15] == (2048, 4)


def test_plan_prewarm_shapes_no_bucket_returns_single():
    shapes = TraceGenerator.plan_prewarm_shapes(
        length_bucket=0,
        max_seq_len=2048,
        batch_width=8,
        max_batch_tokens=8192,
    )
    assert shapes == [(2048, 8)]


def test_plan_prewarm_shapes_skips_zero_widths_at_huge_seq():
    """If max_batch_tokens // seq_len < 1 we still produce n_seqs >= 1."""
    shapes = TraceGenerator.plan_prewarm_shapes(
        length_bucket=4096,
        max_seq_len=8192,
        batch_width=8,
        max_batch_tokens=2048,
    )
    # At seq_len=4096, max_batch_tokens=2048 → cap=0 → floor to 1
    assert all(n >= 1 for _, n in shapes)


def test_prewarm_shapes_invokes_backend_at_each_shape(fake_gen, tmp_path):
    gen, backend = fake_gen
    shapes = [(64, 4), (128, 2), (256, 1)]
    results = gen.prewarm_shapes(shapes=shapes, pad_token_id=7, max_seq_len=512)
    # All three shapes succeeded
    assert len(results) == 3
    assert all(r["ok"] for r in results)
    # Backend was called with each shape (n_seqs, seq_len)
    assert backend.calls == [(4, 64), (2, 128), (1, 256)]
    # Per-shape entries record n_seqs/seq_len/elapsed
    for r, (sl, ns) in zip(results, shapes):
        assert r["seq_len"] == sl
        assert r["n_seqs"] == ns
        assert "elapsed_s" in r and r["elapsed_s"] >= 0.0


def test_prewarm_shapes_logs_via_log_fn(fake_gen):
    gen, _ = fake_gen
    logs: list[str] = []
    gen.prewarm_shapes(
        shapes=[(32, 2), (64, 1)],
        pad_token_id=0,
        log_fn=logs.append,
    )
    assert len(logs) == 2
    assert "seq_len=32" in logs[0] and "n_seqs=2" in logs[0]
    assert "seq_len=64" in logs[1] and "n_seqs=1" in logs[1]


def test_prewarm_shapes_empty_input_is_noop(fake_gen):
    gen, backend = fake_gen
    out = gen.prewarm_shapes(shapes=[])
    assert out == []
    assert backend.calls == []


def test_prewarm_shapes_cleans_up_tmp_dir(fake_gen):
    """Dummy files written during prewarm must be deleted on exit."""
    gen, _ = fake_gen
    import glob
    before = set(glob.glob("/tmp/dflash_prewarm_*"))
    gen.prewarm_shapes(shapes=[(32, 2)], pad_token_id=0)
    after = set(glob.glob("/tmp/dflash_prewarm_*"))
    # No new prewarm dirs should remain
    assert after.issubset(before)


def test_prewarm_shapes_skips_invalid_shapes(fake_gen):
    gen, backend = fake_gen
    results = gen.prewarm_shapes(shapes=[(0, 4), (32, 0), (-1, 1), (64, 2)])
    # Only the valid (64, 2) shape should produce a result
    assert len(results) == 1
    assert results[0] == {"seq_len": 64, "n_seqs": 2, "elapsed_s": pytest.approx(results[0]["elapsed_s"]), "ok": True}
    assert backend.calls == [(2, 64)]
