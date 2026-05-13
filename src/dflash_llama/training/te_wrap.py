"""
dflash_llama.training.te_wrap
=============================
Wrap the speculators DFlash drafter with NVIDIA TransformerEngine for FP8 / MXFP8 training.

Strategy: monkey-patch ``nn.Linear`` -> ``te.Linear`` AFTER model construction. We do NOT
fork the dflash model file. The wrapping walks the module tree and replaces every
``nn.Linear`` with a ``te.Linear`` of matching shapes, copying the weight/bias tensors
(so the initialization seed is preserved exactly).

If env ``TE_USE_FUSED=1`` is set, ALSO fuse the Qwen3 decoder layer's
``(post_attention_layernorm, mlp)`` pair into a single ``te.LayerNormMLP`` (RMSNorm + SwiGLU).
This is the high-value fusion (biggest activation-memory win at ``intermediate=6144``).
The attention block's ``input_layernorm + q/k/v/o_proj`` is left as plain RMSNorm + ``te.Linear``
because fusing it cleanly requires modifying ``Qwen3Attention.forward`` (more invasive).

Production-stable recipe (DGX Spark sm_121a, verified 2026-05-05 over 2,655+ steps with zero NaN):
``current_fp8`` -> ``Float8CurrentScaling(format=HYBRID)`` with ``use_split_accumulator=True``
on ALL three GEMMs (fprop, dgrad, wgrad). The bare ``Float8CurrentScaling()`` default has
``fprop`` set to ``False``, which silently NaNs around step ~38 once LR-warmup crosses ~1.2e-4.
See ``repro/06-fp8-training.md`` for the full failure-mode analysis.

Usage::

    from dflash_llama.training.te_wrap import wrap_with_te, get_recipe, fp8_context

    model = wrap_with_te(model, fp8=True)        # idempotent
    recipe = get_recipe(kind="current_fp8")       # or "delayed_e4m3", "mxfp8", "block_fp8", None
    with fp8_context(recipe):
        out = model(**batch)
        loss.backward()

This module is import-safe in TE-less environments: it logs a warning and ``wrap_with_te``
becomes a no-op returning the model unchanged.
"""
from __future__ import annotations

import contextlib
import logging
import os
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---- TE detection (gracefully degrades if TE unavailable) ----------------------
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as te_recipe
    TE_AVAILABLE = True
    TE_VERSION = getattr(__import__("transformer_engine"), "__version__", "?")
except ImportError as e:
    TE_AVAILABLE = False
    TE_VERSION = None
    logger.warning(f"transformer_engine not importable: {e}")


# ---- Recipe selection ---------------------------------------------------------
def list_recipes() -> dict:
    """Return a dict of available TE FP8 recipes on this build."""
    if not TE_AVAILABLE:
        return {}
    out = {}
    for name in dir(te_recipe):
        if name.startswith("_"):
            continue
        obj = getattr(te_recipe, name)
        if isinstance(obj, type):
            out[name] = obj
    return out


def get_recipe(kind: str = "mxfp8"):
    """Return a TE recipe instance for the chosen FP8 mode.

    kind:
      'mxfp8'        -> MXFP8BlockScaling   (block-scaled MX format; sm 10.x only)
      'block_fp8'    -> Float8BlockScaling  (per-block tensor scaling; CUDA>=12.9)
      'current_fp8'  -> Float8CurrentScaling(per-tensor current scaling; default for sm 12.x)
      'delayed_e4m3' -> DelayedScaling      (legacy / Hopper baseline)
      None / 'bf16'  -> None  (no recipe; te_autocast disabled)
    """
    if kind in (None, "bf16", "none"):
        return None
    if not TE_AVAILABLE:
        raise RuntimeError("transformer_engine is not importable")
    available = list_recipes()
    if kind == "mxfp8":
        for cls_name in ("MXFP8BlockScaling",):
            if cls_name in available:
                return available[cls_name]()
        raise RuntimeError(
            f"MXFP8BlockScaling not available in TE {TE_VERSION}. "
            f"Available: {sorted(available)}"
        )
    if kind == "block_fp8":
        for cls_name in ("Float8BlockScaling",):
            if cls_name in available:
                return available[cls_name]()
        raise RuntimeError("Float8BlockScaling not available")
    if kind == "current_fp8":
        # v12-stable: force split-accumulator on ALL three GEMMs (fprop+dgrad+wgrad).
        # Without split accumulation in fprop the FP32 accumulator can overflow at
        # large LR; this caused the v12 NaN at step ~38.
        from transformer_engine.common.recipe import (
            Float8CurrentScaling,
            Format,
            MMParams,
        )
        return Float8CurrentScaling(
            fp8_format=Format.HYBRID,
            fp8_gemm_fprop=MMParams(use_split_accumulator=True),
            fp8_gemm_dgrad=MMParams(use_split_accumulator=True),
            fp8_gemm_wgrad=MMParams(use_split_accumulator=True),
        )
    if kind == "nvfp4_safe":
        if "NVFP4BlockScaling" in available:
            return available["NVFP4BlockScaling"](disable_rht=True)
        raise RuntimeError("NVFP4BlockScaling not available")
    if kind == "nvfp4_rtn":
        if "NVFP4BlockScaling" in available:
            return available["NVFP4BlockScaling"](
                disable_rht=True, disable_stochastic_rounding=True
            )
        raise RuntimeError("NVFP4BlockScaling not available")
    if kind == "delayed_e4m3":
        if "DelayedScaling" in available:
            return available["DelayedScaling"](
                fp8_format=te_recipe.Format.HYBRID,
                amax_history_len=16,
                amax_compute_algo="max",
            )
    raise ValueError(f"unknown recipe kind: {kind}")


# ---- fp8_autocast context wrapper --------------------------------------------
@contextlib.contextmanager
def fp8_context(recipe):
    """Yield TE's fp8_autocast context if recipe is not None, else no-op."""
    if recipe is None or not TE_AVAILABLE:
        yield
        return
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        yield


# ---- Fused MLP swap helpers --------------------------------------------------
def _is_qwen3_mlp(m: nn.Module) -> bool:
    """Detect a Qwen3MLP-shaped module: has gate_proj, up_proj, down_proj as nn.Linear."""
    return (
        isinstance(getattr(m, "gate_proj", None), nn.Linear)
        and isinstance(getattr(m, "up_proj", None), nn.Linear)
        and isinstance(getattr(m, "down_proj", None), nn.Linear)
    )


def _is_rmsnorm(m: nn.Module) -> bool:
    """Detect Qwen3RMSNorm (has .weight tensor and .variance_epsilon attribute)."""
    return (
        hasattr(m, "weight") and isinstance(getattr(m, "weight"), torch.Tensor)
        and (hasattr(m, "variance_epsilon") or hasattr(m, "eps"))
        and m.__class__.__name__.lower().endswith("rmsnorm")
    )


def _rmsnorm_eps(norm: nn.Module) -> float:
    if hasattr(norm, "variance_epsilon"):
        return float(norm.variance_epsilon)
    if hasattr(norm, "eps"):
        return float(norm.eps)
    return 1e-6


def _build_layernorm_mlp_from(post_norm: nn.Module, mlp: nn.Module) -> "te.LayerNormMLP":
    """Construct a te.LayerNormMLP that mirrors (RMSNorm + Qwen3MLP).

    Qwen3MLP forward: down_proj(silu(gate_proj(x)) * up_proj(x))
    te.LayerNormMLP with activation='swiglu' fuses (RMSNorm + fc1 + swiglu + fc2),
    where fc1_weight = concat([gate; up], dim=0) shape (2*ffn, hidden) and
    fc2_weight = down shape (hidden, ffn).
    """
    assert TE_AVAILABLE
    hidden_size = mlp.gate_proj.in_features
    ffn_hidden = mlp.gate_proj.out_features
    assert mlp.up_proj.in_features == hidden_size
    assert mlp.up_proj.out_features == ffn_hidden
    assert mlp.down_proj.in_features == ffn_hidden
    assert mlp.down_proj.out_features == hidden_size
    has_bias = mlp.gate_proj.bias is not None
    eps = _rmsnorm_eps(post_norm)
    dtype = mlp.gate_proj.weight.dtype
    device = mlp.gate_proj.weight.device

    fused = te.LayerNormMLP(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden,
        eps=eps,
        bias=has_bias,
        normalization="RMSNorm",
        activation="swiglu",
        params_dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        fused.layer_norm_weight.copy_(post_norm.weight.detach().to(dtype))
        # fc1_weight: [gate; up] concatenated along output dim
        fused.fc1_weight.copy_(
            torch.cat([mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach()], dim=0).to(dtype)
        )
        fused.fc2_weight.copy_(mlp.down_proj.weight.detach().to(dtype))
        if has_bias:
            fused.fc1_bias.copy_(
                torch.cat([mlp.gate_proj.bias.detach(), mlp.up_proj.bias.detach()], dim=0).to(dtype)
            )
            fused.fc2_bias.copy_(mlp.down_proj.bias.detach().to(dtype))
    return fused


def _fuse_qwen3_decoder_mlp_blocks(model: nn.Module) -> int:
    """Walk the model and replace every (post_attention_layernorm + mlp) pair on a
    Qwen3 decoder layer with (Identity + te.LayerNormMLP). Returns count of fusions.

    A Qwen3 decoder layer is identified by having BOTH `post_attention_layernorm`
    (an RMSNorm-like) AND `mlp` (Qwen3MLP-shaped: gate/up/down)."""
    if not TE_AVAILABLE:
        return 0
    n = 0
    for module in model.modules():
        post_norm = getattr(module, "post_attention_layernorm", None)
        mlp = getattr(module, "mlp", None)
        if post_norm is None or mlp is None:
            continue
        if not _is_rmsnorm(post_norm) or not _is_qwen3_mlp(mlp):
            continue
        fused = _build_layernorm_mlp_from(post_norm, mlp)
        # Replace mlp first (so post_norm reference is still valid for build above);
        # then replace post_attention_layernorm with Identity.
        module.mlp = fused
        module.post_attention_layernorm = nn.Identity()
        n += 1
    return n


# ---- nn.Linear -> te.Linear monkey-patch -------------------------------------
def _replace_linear(module: nn.Module, parent_name: str = "") -> int:
    """Recursively replace nn.Linear children with te.Linear, copying weights."""
    if not TE_AVAILABLE:
        return 0
    n_replaced = 0
    for name, child in list(module.named_children()):
        full = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, nn.Linear) and not isinstance(child, te.Linear):
            in_f, out_f = child.in_features, child.out_features
            bias = child.bias is not None
            new = te.Linear(in_f, out_f, bias=bias,
                            params_dtype=child.weight.dtype,
                            device=child.weight.device)
            with torch.no_grad():
                new.weight.copy_(child.weight.detach())
                if bias:
                    new.bias.copy_(child.bias.detach())
            setattr(module, name, new)
            n_replaced += 1
        else:
            n_replaced += _replace_linear(child, full)
    return n_replaced


def wrap_with_te(model: nn.Module, fp8: bool = True) -> nn.Module:
    """In-place wrap: replace every nn.Linear in `model` with te.Linear.

    If env TE_USE_FUSED=1, also fuses Qwen3 (post_attention_layernorm + mlp)
    into te.LayerNormMLP BEFORE doing the plain nn.Linear -> te.Linear sweep.

    Returns model unchanged if fp8=False or TE unavailable.
    Idempotent: calling twice is a no-op.
    """
    if not fp8:
        return model
    if not TE_AVAILABLE:
        logger.warning("TE not available; returning model unwrapped")
        return model

    use_fused = os.environ.get("TE_USE_FUSED", "0") == "1"
    n_fused_mlp = 0
    if use_fused:
        n_fused_mlp = _fuse_qwen3_decoder_mlp_blocks(model)
        logger.info(f"[te_wrap] fused {n_fused_mlp} Qwen3 (post_norm+mlp) -> te.LayerNormMLP")
        print(f"[te_wrap] fused {n_fused_mlp} Qwen3 (post_norm+mlp) -> te.LayerNormMLP", flush=True)

    n = _replace_linear(model)
    logger.info(f"[te_wrap] replaced {n} nn.Linear with te.Linear")
    return model


def count_linears(model: nn.Module):
    """Return dict with counts of nn.Linear, te.Linear, te.LayerNormLinear, te.LayerNormMLP."""
    counts = {"nn_linear": 0, "te_linear": 0, "te_layernorm_linear": 0, "te_layernorm_mlp": 0}
    if not TE_AVAILABLE:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                counts["nn_linear"] += 1
        return counts
    for m in model.modules():
        if isinstance(m, te.LayerNormMLP):
            counts["te_layernorm_mlp"] += 1
        elif isinstance(m, te.LayerNormLinear):
            counts["te_layernorm_linear"] += 1
        elif isinstance(m, te.Linear):
            counts["te_linear"] += 1
        elif isinstance(m, nn.Linear):
            counts["nn_linear"] += 1
    return counts


if __name__ == "__main__":
    # Self-test: import, list recipes, wrap a toy MLP, run one FP8 fwd+bwd.
    # Requires CUDA + TransformerEngine. Use from the repro venv on a Spark
    # GPU host, NOT in CI (CI has no TE / no GPU).
    import sys
    print(f"TE_AVAILABLE = {TE_AVAILABLE}")
    print(f"TE_VERSION   = {TE_VERSION}")
    print(f"recipes      = {sorted(list_recipes())}")
    if not TE_AVAILABLE:
        sys.exit(0)
    m = nn.Sequential(nn.Linear(3072, 4096), nn.GELU(), nn.Linear(4096, 3072)).cuda().bfloat16()
    print(f"before: {count_linears(m)}")
    wrap_with_te(m, fp8=True)
    print(f"after:  {count_linears(m)}")
    rec = get_recipe("current_fp8")
    print(f"recipe = {rec}")
    x = torch.randn(64, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with fp8_context(rec):
        y = m(x)
        loss = y.float().pow(2).mean()
    loss.backward()
    print(f"y={y.shape} loss={loss.item():.4f} grad_ok={x.grad is not None}")


__all__ = [
    "TE_AVAILABLE",
    "TE_VERSION",
    "list_recipes",
    "get_recipe",
    "fp8_context",
    "wrap_with_te",
    "count_linears",
]
