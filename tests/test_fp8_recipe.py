"""Tests for FP8 recipe construction, arch detection, and trainer wiring.

These tests are CPU-runnable: every TE-touching path is gated behind
``pytest.importorskip("transformer_engine")`` and the arch refusal logic is
exercised via monkeypatching ``current_arch``.

The most important test in this file is :func:`test_silent_bf16_trap_blocked`
— it pins the guard that has saved real production runs from quietly running
bf16 while the user thought they were getting +42% throughput.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from dflash_llama import (
    FP8Recipe,
    current_arch,
    fp8_autocast_ctx,
    make_te_recipe,
)
from dflash_llama.training import fp8 as fp8_mod


# ---------------------------------------------------------------------------
# FP8Recipe dataclass
# ---------------------------------------------------------------------------
def test_fp8recipe_default_split_accumulator_is_true():
    """The hard-won default: split_accumulator MUST be True.

    With False on fprop (TE's stock default), the recipe NaN's at step ~40
    on noised DFlash drafters when LR crests 1.2e-4. Don't flip this default
    without reading src/dflash_llama/training/fp8.py module docstring.
    """
    r = FP8Recipe(kind="current_fp8")
    assert r.use_split_accumulator is True
    assert r.format == "HYBRID"


def test_fp8recipe_from_string_aliases():
    assert FP8Recipe.from_string(None).kind is None
    assert FP8Recipe.from_string("").kind is None
    assert FP8Recipe.from_string("none").kind is None
    assert FP8Recipe.from_string("bf16").kind is None
    assert FP8Recipe.from_string("off").kind is None
    assert FP8Recipe.from_string("current_fp8").kind == "current_fp8"
    assert FP8Recipe.from_string("delayed_e4m3").kind == "delayed_e4m3"
    assert FP8Recipe.from_string("block_fp8").kind == "block_fp8"
    assert FP8Recipe.from_string("mxfp8").kind == "mxfp8"


def test_fp8recipe_from_string_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown FP8 recipe kind"):
        FP8Recipe.from_string("totally-bogus-recipe")


# ---------------------------------------------------------------------------
# current_arch
# ---------------------------------------------------------------------------
def test_current_arch_schema():
    arch = current_arch()
    # Always present keys.
    for k in ("compute_capability", "torch_available", "recommends"):
        assert k in arch, f"current_arch() missing key {k!r}"
    if arch.get("cuda_available"):
        for k in ("supports_fp8cs", "supports_mxfp8_te", "supports_block_fp8",
                  "compute_capability_int", "device_name"):
            assert k in arch, f"current_arch() missing key {k!r} on CUDA host"
        assert isinstance(arch["compute_capability_int"], int)


# ---------------------------------------------------------------------------
# Refusal of broken-on-Spark recipes
# ---------------------------------------------------------------------------
def _patch_arch(monkeypatch, cap_int: int):
    """Force current_arch() to report the given compute-capability int."""
    fake = {
        "compute_capability": f"{cap_int // 10}.{cap_int % 10}",
        "compute_capability_int": cap_int,
        "device_name": f"FAKE-sm_{cap_int}",
        "torch_available": True,
        "cuda_available": True,
        "supports_fp8cs": cap_int >= 89,
        "supports_mxfp8_te": cap_int in (90, 100, 101),
        "supports_block_fp8": cap_int in (89, 90, 100, 101),
        "recommends": "current_fp8",
    }
    monkeypatch.setattr(fp8_mod, "current_arch", lambda: fake)


def test_block_fp8_refused_on_spark_sm121(monkeypatch):
    """Float8BlockScaling on sm_121 is silently non-convergent (TE #2382).

    We refuse at recipe-construction time so future-us can't waste another
    14-epoch run discovering this empirically.
    """
    _patch_arch(monkeypatch, 121)
    with pytest.raises(RuntimeError, match="silently NON-CONVERGENT"):
        make_te_recipe(FP8Recipe(kind="block_fp8"))


def test_block_fp8_refused_on_spark_sm120(monkeypatch):
    _patch_arch(monkeypatch, 120)
    with pytest.raises(RuntimeError, match="silently NON-CONVERGENT"):
        make_te_recipe(FP8Recipe(kind="block_fp8"))


def test_mxfp8_te_refused_on_spark(monkeypatch):
    """MXFP8 via TE is BLOCKED on sm_120/121 (cuBLAS layout, TE #2668)."""
    _patch_arch(monkeypatch, 121)
    with pytest.raises(RuntimeError, match="MXFP8.*BLOCKED"):
        make_te_recipe(FP8Recipe(kind="mxfp8"))


def test_block_fp8_allowed_on_hopper(monkeypatch):
    """sm_90 (Hopper) does support BlockScaling — refusal must not fire.

    The TE import will still fail on a CPU dev box, so we expect a
    RuntimeError mentioning transformer_engine, NOT the 'silently
    non-convergent' message.
    """
    _patch_arch(monkeypatch, 90)
    with pytest.raises(RuntimeError) as exc:
        make_te_recipe(FP8Recipe(kind="block_fp8"))
    assert "silently NON-CONVERGENT" not in str(exc.value)


def test_make_te_recipe_none_returns_none():
    """kind=None means bf16 — make_te_recipe is a no-op."""
    assert make_te_recipe(FP8Recipe(kind=None)) is None


def test_fp8_autocast_ctx_noop_when_kind_none():
    """No-op context manager must succeed without TE installed."""
    with fp8_autocast_ctx(FP8Recipe(kind=None)):
        pass  # explicit pass — exiting cleanly is the assertion


# ---------------------------------------------------------------------------
# wrap_with_te is import-tested only if TE present
# ---------------------------------------------------------------------------
def test_wrap_with_te_noop_on_bf16():
    """wrap_with_te is a no-op when recipe.kind is None."""
    stats = fp8_mod.wrap_with_te(model=mock.MagicMock(), recipe=FP8Recipe(kind=None))
    assert stats["linear_replaced"] == 0
    assert stats["mlp_fused"] == 0
    assert "skipped" in stats


def test_wrap_with_te_requires_te_when_kind_set():
    """When kind is set but TE isn't installed, raise a clear error."""
    pytest.importorskip("torch")
    te_spec = importlib.util.find_spec("transformer_engine")
    if te_spec is not None:
        pytest.skip("TE present; skip the no-TE error path")
    import torch.nn as nn
    model = nn.Linear(8, 8)
    with pytest.raises(RuntimeError, match="transformer_engine"):
        fp8_mod.wrap_with_te(model, FP8Recipe(kind="current_fp8"))


# ---------------------------------------------------------------------------
# CLI: check-fp8
# ---------------------------------------------------------------------------
def test_cli_check_fp8_exits_zero():
    """`dflash-llama check-fp8` must succeed and print the arch dict."""
    proc = subprocess.run(
        [sys.executable, "-m", "dflash_llama.cli", "check-fp8"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "compute_capability" in proc.stdout
    assert "recommends" in proc.stdout


# ---------------------------------------------------------------------------
# Silent-bf16 trap guard — the most important test in this file
# ---------------------------------------------------------------------------
def test_silent_bf16_trap_blocked(synthetic_trace_dir, tmp_path, monkeypatch):
    """DFlashTrainer.train(fp8_recipe='current_fp8', use_torchrun=True) on a
    single-GPU machine (WORLD_SIZE=1) MUST raise.

    History: torchrun --nproc-per-node=1 sets RANK/WORLD_SIZE which routes
    speculators through its FSDP branch. The TE wrap only lives in the
    single-GPU branch, so this configuration silently runs bf16 with no
    [FP8] log line — the user thinks they're getting +42% throughput when
    they aren't. We refuse to launch in that configuration.
    """
    from dflash_llama import DFlashTrainer, load_verifier

    monkeypatch.setenv("WORLD_SIZE", "1")

    # Use the cheap synthetic trace dir + a generic verifier so we don't
    # need network access.
    verifier = load_verifier(
        "generic",
        name_override="fake-tiny",
        hidden_size=64,
        num_hidden_layers=4,
        vocab_size=32000,
        mask_token_id=31999,
        layer_ids=[0, 1, 2, 3],
        hf_path=str(tmp_path / "fake_hf"),
    )
    (tmp_path / "fake_hf").mkdir()

    trainer = DFlashTrainer(
        traces_dir=str(synthetic_trace_dir),
        verifier=verifier,
        num_layers=4,
        draft_vocab_size=128,
        paired_dir=str(tmp_path / "paired"),
    )
    trainer.prepare()

    with pytest.raises(RuntimeError, match="Silent-bf16 trap"):
        trainer.train(
            save_to=str(tmp_path / "out"),
            fp8_recipe="current_fp8",
            use_torchrun=True,
            dry_run=True,
        )


def test_silent_bf16_trap_not_triggered_for_bf16(synthetic_trace_dir, tmp_path, monkeypatch):
    """Same launch shape but bf16 (fp8_recipe=None) must NOT raise."""
    from dflash_llama import DFlashTrainer, load_verifier

    monkeypatch.setenv("WORLD_SIZE", "1")
    verifier = load_verifier(
        "generic",
        name_override="fake-tiny",
        hidden_size=64,
        num_hidden_layers=4,
        vocab_size=32000,
        mask_token_id=31999,
        layer_ids=[0, 1, 2, 3],
        hf_path=str(tmp_path / "fake_hf"),
    )
    (tmp_path / "fake_hf").mkdir()
    trainer = DFlashTrainer(
        traces_dir=str(synthetic_trace_dir),
        verifier=verifier,
        num_layers=4,
        draft_vocab_size=128,
        paired_dir=str(tmp_path / "paired"),
    )
    trainer.prepare()
    # No raise; dry_run returns the cmd list.
    res = trainer.train(
        save_to=str(tmp_path / "out"),
        fp8_recipe=None,
        use_torchrun=True,
        dry_run=True,
    )
    assert res["dry_run"] is True
    assert isinstance(res["cmd"], list)


def test_silent_bf16_trap_not_triggered_when_use_torchrun_false(synthetic_trace_dir, tmp_path):
    """The new default (use_torchrun=False) is the safe path. No raise."""
    from dflash_llama import DFlashTrainer, load_verifier

    verifier = load_verifier(
        "generic",
        name_override="fake-tiny",
        hidden_size=64,
        num_hidden_layers=4,
        vocab_size=32000,
        mask_token_id=31999,
        layer_ids=[0, 1, 2, 3],
        hf_path=str(tmp_path / "fake_hf"),
    )
    (tmp_path / "fake_hf").mkdir()
    trainer = DFlashTrainer(
        traces_dir=str(synthetic_trace_dir),
        verifier=verifier,
        num_layers=4,
        draft_vocab_size=128,
        paired_dir=str(tmp_path / "paired"),
    )
    trainer.prepare()
    res = trainer.train(
        save_to=str(tmp_path / "out"),
        fp8_recipe="current_fp8",
        use_torchrun=False,    # the new default
        drafter_intermediate_size=6144,
        dry_run=True,
    )
    assert res["dry_run"] is True
    cmd = " ".join(res["cmd"])
    # FP8 flags are present.
    assert "--fp8-recipe-kind current_fp8" in cmd
    assert "--fp8-split-accumulator" in cmd
    assert "--te-use-fused" in cmd
    assert "--drafter-intermediate-size 6144" in cmd
    assert "--nan-skip" in cmd
    # And the launcher is python, NOT torchrun.
    assert sys.executable in res["cmd"][0]


# ---------------------------------------------------------------------------
# Train cmd shape: drafter_intermediate_size + nan_skip plumbing
# ---------------------------------------------------------------------------
def test_train_cmd_includes_intermediate_and_nan_skip(synthetic_trace_dir, tmp_path):
    from dflash_llama import DFlashTrainer, load_verifier
    verifier = load_verifier(
        "generic",
        name_override="fake-tiny",
        hidden_size=64,
        num_hidden_layers=4,
        vocab_size=32000,
        mask_token_id=31999,
        layer_ids=[0, 1, 2, 3],
        hf_path=str(tmp_path / "fake_hf"),
    )
    (tmp_path / "fake_hf").mkdir()
    trainer = DFlashTrainer(
        traces_dir=str(synthetic_trace_dir),
        verifier=verifier,
        num_layers=4,
        draft_vocab_size=128,
        paired_dir=str(tmp_path / "paired"),
    )
    trainer.prepare()
    res = trainer.train(
        save_to=str(tmp_path / "out"),
        drafter_intermediate_size=6144,
        nan_skip=True,
        dry_run=True,
    )
    cmd = res["cmd"]
    assert "--drafter-intermediate-size" in cmd
    assert "6144" in cmd
    assert "--nan-skip" in cmd


def test_train_cmd_omits_fp8_flags_when_bf16(synthetic_trace_dir, tmp_path):
    """bf16 path must NOT emit --fp8-recipe-kind (so unpatched speculators
    can still consume the bf16 cmd)."""
    from dflash_llama import DFlashTrainer, load_verifier
    verifier = load_verifier(
        "generic",
        name_override="fake-tiny",
        hidden_size=64, num_hidden_layers=4,
        vocab_size=32000, mask_token_id=31999,
        layer_ids=[0, 1, 2, 3],
        hf_path=str(tmp_path / "fake_hf"),
    )
    (tmp_path / "fake_hf").mkdir()
    trainer = DFlashTrainer(
        traces_dir=str(synthetic_trace_dir),
        verifier=verifier,
        num_layers=4, draft_vocab_size=128,
        paired_dir=str(tmp_path / "paired"),
    )
    trainer.prepare()
    res = trainer.train(save_to=str(tmp_path / "out"), dry_run=True)
    cmd = " ".join(res["cmd"])
    assert "--fp8-recipe-kind" not in cmd
    assert "--te-use-fused" not in cmd
