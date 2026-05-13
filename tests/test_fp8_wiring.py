"""Unit tests for FP8 / TransformerEngine wiring in DFlashTrainer.

These tests run on CPU without TransformerEngine installed. They verify:

  1. ``te_wrap`` module imports cleanly when TE is absent (TE_AVAILABLE = False).
  2. ``count_linears`` works on a plain torch model without TE.
  3. ``DFlashTrainer.train(dry_run=True)`` builds the correct command for the
     FP8 path: direct python launch (no torchrun), with --fp8-recipe-kind and
     --te-use-fused flags.
  4. ``DFlashTrainer.train(dry_run=True)`` defaults to torchrun + no FP8 flags
     when ``fp8_recipe_kind`` is not set (back-compat with v11 bf16 launches).
  5. ``DFlashTrainer.train`` honors ``use_torchrun=True/False`` overrides.
  6. ``run_smoke_test(dry_run=True)`` follows the same launcher-mode rules.
  7. Patches 04, 05, 06 are syntactically well-formed (unified diff parseable).

We do NOT test actual TE wrapping, FP8 forward passes, or NaN-skip guard firing
in this suite — that requires a GPU + TE install and lives in the integration
tests on a Spark host (see ``repro/06-fp8-training.md`` §6.4).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---- te_wrap module-level tests -----------------------------------------------

def test_te_wrap_imports_without_te():
    """The te_wrap module must be importable in a TE-less environment."""
    from dflash_llama.training import te_wrap
    # TE_AVAILABLE may be True or False depending on the env; both must be defined.
    assert hasattr(te_wrap, "TE_AVAILABLE")
    assert hasattr(te_wrap, "TE_VERSION")
    assert hasattr(te_wrap, "wrap_with_te")
    assert hasattr(te_wrap, "get_recipe")
    assert hasattr(te_wrap, "count_linears")
    assert hasattr(te_wrap, "fp8_context")


def test_te_wrap_count_linears_on_plain_model():
    """count_linears should work on a vanilla nn.Module without TE."""
    from dflash_llama.training.te_wrap import count_linears
    m = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 8),
    )
    counts = count_linears(m)
    assert counts["nn_linear"] == 2
    # te_linear / te_layernorm_linear / te_layernorm_mlp may or may not be present
    # depending on TE availability, but if present must be 0
    for k in ("te_linear", "te_layernorm_linear", "te_layernorm_mlp"):
        if k in counts:
            assert counts[k] == 0


def test_te_wrap_wrap_with_te_fp8_false_is_noop():
    """wrap_with_te(fp8=False) must return the model unchanged."""
    from dflash_llama.training.te_wrap import wrap_with_te
    m = torch.nn.Linear(8, 16)
    out = wrap_with_te(m, fp8=False)
    assert out is m


def test_te_wrap_get_recipe_bf16_returns_none():
    """get_recipe should return None for the 'no-FP8' kinds."""
    from dflash_llama.training.te_wrap import get_recipe
    for kind in (None, "bf16", "none"):
        assert get_recipe(kind) is None


def test_te_wrap_fp8_context_is_nullcontext_when_recipe_is_none():
    """fp8_context(None) must be a no-op context manager."""
    from dflash_llama.training.te_wrap import fp8_context
    with fp8_context(None):
        # Entering / exiting must not raise
        x = 1 + 1
    assert x == 2


# ---- DFlashTrainer._build_train_cmd / .train(dry_run=True) tests --------------

@pytest.fixture
def prepared_trainer(synthetic_trace_dir, tmp_path):
    """A DFlashTrainer with .prepare() already run on synthetic data."""
    from dflash_llama.training import DFlashTrainer
    from dflash_llama.verifiers import load_verifier
    verifier = load_verifier(
        "generic", name_override="synthetic-qwen3-tiny",
        hidden_size=64, num_hidden_layers=6,
        vocab_size=2048, mask_token_id=0,
        layer_ids=(0, 1, 2, 3, 4, 5),
        hf_path="/dummy/hf",
    )
    paired = tmp_path / "paired"
    trainer = DFlashTrainer(
        traces_dir=str(synthetic_trace_dir),
        verifier=verifier,
        num_layers=2, draft_vocab_size=128,
        paired_dir=str(paired),
    )
    trainer.prepare()
    return trainer


def test_train_dryrun_default_uses_torchrun_no_fp8_flags(prepared_trainer, tmp_path):
    """Default DFlashTrainer.train() (no fp8) must use torchrun and emit NO fp8 flags."""
    res = prepared_trainer.train(save_to=str(tmp_path / "ckpt"), dry_run=True, epochs=1)
    assert res["dry_run"] is True
    cmd = res["cmd"]
    cmd_str = " ".join(cmd)
    # Launcher: torchrun
    assert "torchrun" in cmd[0]
    # No FP8 flags
    assert "--fp8-recipe-kind" not in cmd_str
    assert "--te-use-fused" not in cmd_str
    # No in-epoch val flags by default
    assert "--val-every-steps" not in cmd_str
    assert "--save-every-n-vals" not in cmd_str


def test_train_dryrun_fp8_drops_torchrun(prepared_trainer, tmp_path):
    """Setting fp8_recipe_kind must switch the launcher to direct python."""
    import sys
    res = prepared_trainer.train(
        save_to=str(tmp_path / "ckpt"),
        dry_run=True, epochs=1,
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
    )
    cmd = res["cmd"]
    # The launcher must be the current python interpreter, NOT torchrun.
    assert cmd[0] == sys.executable, (
        f"FP8 launch must use direct python (Bug A: silent-bf16 trap), "
        f"got cmd[0]={cmd[0]!r}"
    )
    assert "torchrun" not in cmd[0]
    # FP8 flags must be present
    cmd_str = " ".join(cmd)
    assert "--fp8-recipe-kind current_fp8" in cmd_str
    assert "--te-use-fused" in cmd_str
    # No master_port (that's torchrun-only)
    assert "--master_port" not in cmd_str
    assert "--nproc-per-node" not in cmd_str


def test_train_dryrun_in_epoch_val_flags(prepared_trainer, tmp_path):
    """In-epoch val cadence flags must propagate to the command."""
    res = prepared_trainer.train(
        save_to=str(tmp_path / "ckpt"),
        dry_run=True, epochs=1,
        val_every_steps=145,
        val_in_epoch_max_batches=80,
        save_every_n_vals=1,
    )
    cmd_str = " ".join(res["cmd"])
    assert "--val-every-steps 145" in cmd_str
    assert "--val-in-epoch-max-batches 80" in cmd_str
    assert "--save-every-n-vals 1" in cmd_str


def test_train_dryrun_use_torchrun_override(prepared_trainer, tmp_path):
    """use_torchrun=True must force torchrun even when fp8 is set (and vice versa)."""
    import sys
    # Force torchrun ON despite fp8_recipe_kind (should not be done in production
    # — this is just for testing the override mechanism).
    res = prepared_trainer.train(
        save_to=str(tmp_path / "ckpt"),
        dry_run=True, epochs=1,
        fp8_recipe_kind="current_fp8",
        use_torchrun=True,
    )
    assert "torchrun" in res["cmd"][0]

    # Force direct-python OFF despite no fp8 (just to confirm the override works
    # both directions).
    res2 = prepared_trainer.train(
        save_to=str(tmp_path / "ckpt"),
        dry_run=True, epochs=1,
        use_torchrun=False,
    )
    assert res2["cmd"][0] == sys.executable


def test_train_dryrun_fp8_kind_choices(prepared_trainer, tmp_path):
    """All documented fp8_recipe_kind values should be accepted and emit correctly."""
    for kind in ("current_fp8", "delayed_e4m3", "block_fp8", "mxfp8"):
        res = prepared_trainer.train(
            save_to=str(tmp_path / "ckpt"),
            dry_run=True, epochs=1,
            fp8_recipe_kind=kind,
        )
        cmd_str = " ".join(res["cmd"])
        assert f"--fp8-recipe-kind {kind}" in cmd_str


def test_smoke_dryrun_fp8_drops_torchrun(prepared_trainer, tmp_path):
    """run_smoke_test must follow the same launcher-mode rule."""
    res = prepared_trainer.smoke(
        dry_run=True, timeout_sec=15,
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
    )
    # message starts with "dry_run=True; would run: <shlex-quoted cmd>"
    assert "torchrun" not in res.message.split("would run:", 1)[1].split()[0:2][0]
    assert "--fp8-recipe-kind current_fp8" in res.message
    assert "--te-use-fused" in res.message


# ---- Patch well-formedness tests ----------------------------------------------

@pytest.mark.parametrize("patch_name", [
    "04-trainer-nan-guard-and-midepoch-ckpt.patch",
    "05-trainer-te-fp8-wrap.patch",
    "06-train-script-fp8-flags.patch",
])
def test_patch_is_wellformed(patch_name):
    """Each patch file must be a valid unified diff with the expected header."""
    p = REPO_ROOT / "patches" / "speculators" / patch_name
    assert p.exists(), f"missing patch {p}"
    text = p.read_text()
    # Must have a diff --git header and at least one hunk
    assert "diff --git" in text, f"{patch_name}: missing 'diff --git' header"
    assert re.search(r"^@@ ", text, re.MULTILINE), f"{patch_name}: no hunks found"
    # Must reference speculators source paths
    assert "src/speculators/" in text or "scripts/train.py" in text, (
        f"{patch_name}: doesn't touch speculators paths"
    )
    # No leaked usernames or hostnames from the spark cluster
    for leak in ("operator", "spark-1", "spark-2", "spark-3", "spark-4",
                 "/home/user", "192.168.200", "10.0.0.2", "admin:"):
        assert leak not in text, f"{patch_name}: leaked identifier {leak!r}"


def test_patch_04_adds_grad_level_nan_skip():
    """Patch 04 must use the *gradient-level* NaN check, not a loss-level one.

    The loss-level check is insufficient because clip_grad_norm_ runs before
    opt.step and clip_grad_norm_(NaN) is still NaN — see repro/06-fp8-training.md.
    """
    p = REPO_ROOT / "patches" / "speculators" / "04-trainer-nan-guard-and-midepoch-ckpt.patch"
    text = p.read_text()
    # Must check parameter gradients, not just loss
    assert "p.grad is not None and not torch.isfinite(p.grad).all()" in text, (
        "patch 04 must check gradient finiteness (not just loss isnan)"
    )
    assert "NaN-SKIP" in text
    # Must skip optimizer step + continue (not just skip backward)
    assert "self.opt.zero_grad()" in text
    assert "continue" in text


def test_patch_05_uses_repo_te_wrap_import():
    """Patch 05 must import from dflash_llama.training.te_wrap, NOT a magic path."""
    p = REPO_ROOT / "patches" / "speculators" / "05-trainer-te-fp8-wrap.patch"
    text = p.read_text()
    assert "from dflash_llama.training.te_wrap import" in text, (
        "patch 05 must import te_wrap helpers from the dflash_llama package"
    )
    # Must add the FP8 forward wrappers
    assert "_fp8_train_forward" in text
    assert "_fp8_val_forward" in text
    assert "_maybe_wrap_te" in text
    # Must wrap inside the single-GPU branch (NOT inside the FSDP branch)
    # We check by ensuring _maybe_wrap_te is called after model.to(local_rank)
    # but before the FSDP `return` — the actual line ordering is verified by the
    # patch context lines.
    assert "self._maybe_wrap_te()" in text


def test_patch_06_adds_required_cli_flags():
    """Patch 06 must add --fp8-recipe-kind, --te-use-fused, and the val cadence flags."""
    p = REPO_ROOT / "patches" / "speculators" / "06-train-script-fp8-flags.patch"
    text = p.read_text()
    for flag in (
        "--fp8-recipe-kind",
        "--te-use-fused",
        "--val-every-steps",
        "--val-in-epoch-max-batches",
        "--save-every-n-vals",
    ):
        assert flag in text, f"patch 06 missing CLI flag {flag}"
    # Must also wire the args into TrainerConfig
    assert "fp8_recipe_kind=args.fp8_recipe_kind" in text
    assert "te_use_fused=args.te_use_fused" in text
