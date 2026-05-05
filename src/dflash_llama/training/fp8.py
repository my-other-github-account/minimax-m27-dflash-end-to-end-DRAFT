"""TransformerEngine FP8 production-training helpers.

This module exists because, on 2026-05-04 / 2026-05-05, we successfully ran
FP8 production training of a DFlash drafter on the Spark cluster (sm_121)
and learned a fistful of non-obvious things that the rest of the world will
otherwise re-discover the painful way.

Highlights (in priority order — read before changing defaults):

1. **split-accumulator MUST be True on ALL THREE GEMMs** (fprop, dgrad,
   wgrad). The default in stock TransformerEngine's
   ``Float8CurrentScaling`` recipe leaves ``use_split_accumulator=False``
   on fprop, which silently NaN's a noised DFlash drafter at step ~38–40
   the moment the LR schedule crests ~1.2e-4. With ``True`` on all three
   GEMMs, the same recipe / data / LR survives past step 305+ with zero
   NaN-skip events. This is the most important hard-won default in this
   library.

2. **Float8BlockScaling is silently NON-CONVERGENT on sm_120/121**
   (TransformerEngine issue #2382). Loss stays finite but never decreases.
   We refuse the kind ``"block_fp8"`` on these arches loudly at
   recipe-construction time — better to fail than to waste a 14-epoch
   training run discovering this empirically.

3. **MXFP8 via TransformerEngine is BLOCKED on sm_120/121** (cuBLAS lacks
   the non-TN GEMM layouts; TE issue #2668). ``torchao.prototype.mx_formats``
   bypasses by re-quantizing both dim0 and dim1 so all GEMMs end up TN; if
   you want MXFP8 on Spark, build torchao with
   ``TORCH_CUDA_ARCH_LIST="12.1a"`` (the trailing ``a`` is mandatory).

4. **Fused te.LayerNormMLP is the lever**, not FP8 alone. FP8 weights/GEMM
   saved roughly zero activation memory at v11 size (4096 intermediate);
   the cliff is in the FFN backward graph (O(intermediate²) without grad
   checkpointing). Fusing ``post_attention_layernorm + mlp`` into one
   ``te.LayerNormMLP`` eliminates the intermediate norm materialization,
   which is what makes the recipe-faithful intermediate=6144 actually fit.

5. **No transformer_engine import at module load time.** TE is heavy and
   not present on macOS / dev boxes. Every TE touch in this module is
   inside a function that imports ``transformer_engine`` lazily.

The matching speculators-trainer patches that consume these helpers live
in ``scripts/patch_speculators_for_fp8.py``.
"""
from __future__ import annotations

import contextlib
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Literal, Optional


FP8Kind = Literal["current_fp8", "delayed_e4m3", "block_fp8", "mxfp8"]


# -----------------------------------------------------------------------------
# Capability detection
# -----------------------------------------------------------------------------
def current_arch() -> Dict[str, Any]:
    """Return a dict describing the current CUDA arch + recipe support matrix.

    On non-CUDA hosts (macOS, CPU-only Linux) returns ``{"compute_capability":
    None}`` and recommends bf16. We return a dict — not raise — because this
    is exactly what the ``check-fp8`` CLI subcommand prints, and a clean dict
    on a dev box is more useful than an exception.
    """
    try:
        import torch  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - torch is a hard dep
        return {
            "compute_capability": None,
            "torch_available": False,
            "error": str(exc),
            "recommends": "bf16 (no torch available)",
        }

    if not torch.cuda.is_available():
        return {
            "compute_capability": None,
            "torch_available": True,
            "cuda_available": False,
            "supports_fp8cs": False,
            "supports_mxfp8_te": False,
            "supports_block_fp8": False,
            "recommends": "bf16 (no CUDA device)",
        }

    cap = torch.cuda.get_device_capability(0)
    cap_int = cap[0] * 10 + cap[1]
    name = torch.cuda.get_device_name(0)
    # sm_89 (Ada), sm_90 (Hopper), sm_100/101 (Blackwell DC), sm_120/121 (Spark)
    supports_fp8cs = cap_int >= 89
    # MXFP8 via TE: blocked on sm_120/121 (cuBLAS layout block, TE #2668).
    # Hopper sm_90 has it via TE; sm_100/101 ship in newer cuBLAS too.
    supports_mxfp8_te = cap_int in (90, 100, 101)
    # Float8BlockScaling silently non-convergent on sm_120/121 (TE #2382).
    supports_block_fp8 = cap_int in (89, 90, 100, 101)

    if cap_int >= 89:
        recommends = "current_fp8 (Float8CurrentScaling HYBRID + fused LayerNormMLP)"
    else:
        recommends = "bf16 (compute capability < 8.9; no FP8 GEMM support)"

    return {
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "compute_capability_int": cap_int,
        "device_name": name,
        "torch_available": True,
        "cuda_available": True,
        "supports_fp8cs": supports_fp8cs,
        "supports_mxfp8_te": supports_mxfp8_te,
        "supports_block_fp8": supports_block_fp8,
        "recommends": recommends,
    }


def _is_spark_class(arch: Dict[str, Any]) -> bool:
    """Spark = sm_120/sm_121 (DGX Spark GB10). The class with the BlockScaling
    bug and the cuBLAS MXFP8 block."""
    cap = arch.get("compute_capability_int")
    return cap in (120, 121)


# -----------------------------------------------------------------------------
# Recipe dataclass + factory
# -----------------------------------------------------------------------------
@dataclass
class FP8Recipe:
    """User-facing FP8 recipe spec.

    Attributes
    ----------
    kind:
        One of ``"current_fp8"`` (Float8CurrentScaling — the production
        recipe verified on v11/v12 2026-05-04..05), ``"delayed_e4m3"``
        (DelayedScaling, classic Hopper recipe), ``"block_fp8"``
        (Float8BlockScaling — REFUSED on sm_120/121), ``"mxfp8"``
        (MXFP8BlockScaling — REFUSED on sm_120/121 via TE), or ``None`` to
        run bf16.
    use_split_accumulator:
        Default ``True``. **Do not flip this off** unless you've read the
        module docstring. With ``False`` on fprop, current-scaling NaN's at
        step ~40 on noised DFlash drafters; with ``True`` on all three
        GEMMs, the same recipe survives 305+ steps cleanly.
    format:
        ``"HYBRID"`` (E4M3 fwd / E5M2 bwd — production), ``"E4M3"``,
        or ``"E5M2"``.
    margin:
        DelayedScaling-only knob. Ignored for ``current_fp8`` /
        ``block_fp8`` / ``mxfp8``.
    amax_history_len / amax_compute_algo:
        DelayedScaling-only.
    """

    kind: Optional[FP8Kind] = None
    use_split_accumulator: bool = True  # <-- HARD-WON DEFAULT. See module docstring.
    format: str = "HYBRID"
    margin: int = 0
    amax_history_len: int = 1024
    amax_compute_algo: str = "max"
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_string(cls, s: Optional[str]) -> "FP8Recipe":
        """Accept ``"current_fp8"``, ``"bf16"`` / ``"none"`` / ``None``,
        etc., and return a recipe."""
        if s is None or s in ("", "none", "bf16", "off"):
            return cls(kind=None)
        if s in ("current_fp8", "delayed_e4m3", "block_fp8", "mxfp8"):
            return cls(kind=s)  # type: ignore[arg-type]
        raise ValueError(
            f"Unknown FP8 recipe kind: {s!r}. "
            "Valid: 'current_fp8', 'delayed_e4m3', 'block_fp8', 'mxfp8', or None/'bf16'."
        )


def _refuse_block_fp8_on_spark() -> None:
    """Refuse Float8BlockScaling on sm_120/121.

    TransformerEngine issue #2382: BlockScaling runs and produces non-NaN
    losses on Spark sm_120/121, but the loss never decreases — it's
    silently non-convergent. We've burned a full 14-epoch run learning
    this; future-us is forbidden from re-discovering it.
    """
    arch = current_arch()
    if _is_spark_class(arch):
        raise RuntimeError(
            "Float8BlockScaling (kind='block_fp8') is silently NON-CONVERGENT on "
            f"sm_{arch.get('compute_capability_int')}/Spark (TransformerEngine #2382). "
            "It runs without crashing, produces finite losses, and never trains. "
            "Use kind='current_fp8' instead — verified converging on this hardware "
            "in production 2026-05-04..05."
        )


def _refuse_mxfp8_te_on_spark() -> None:
    """Refuse MXFP8 via TE on sm_120/121.

    cuBLAS on Spark lacks the non-TN GEMM layouts MXFP8 needs (TE issue
    #2668). Symptom: cublasLt error at first matmul. Workaround:
    torchao.prototype.mx_formats with ``TORCH_CUDA_ARCH_LIST="12.1a"``.
    """
    arch = current_arch()
    if _is_spark_class(arch):
        raise RuntimeError(
            "MXFP8 via TransformerEngine (kind='mxfp8') is BLOCKED on "
            f"sm_{arch.get('compute_capability_int')}/Spark (cuBLAS missing non-TN "
            "GEMM layouts; TransformerEngine #2668). Workaround: build torchao "
            "from source with TORCH_CUDA_ARCH_LIST=\"12.1a\" (trailing 'a' is "
            "mandatory) and use torchao.prototype.mx_formats — that bypass "
            "re-quantizes dim0+dim1 so every GEMM is TN."
        )


def make_te_recipe(recipe: FP8Recipe):
    """Build a TransformerEngine recipe object from an :class:`FP8Recipe`.

    Returns a TE recipe instance ready to pass to
    ``transformer_engine.pytorch.fp8_autocast(fp8_recipe=...)``.

    Raises ``RuntimeError`` for combinations we know to be silently broken
    on the current hardware (BlockScaling on sm_120/121; MXFP8 via TE on
    sm_120/121).

    The split-accumulator setting is forced True across all three GEMMs
    (fprop, dgrad, wgrad) by default — see module docstring for why.
    Pass ``FP8Recipe(use_split_accumulator=False)`` to override at your
    own risk.
    """
    if recipe.kind is None:
        return None

    # Refuse the silently-broken combos before importing TE — saves users a
    # 30-second TE import on the unhappy path.
    if recipe.kind == "block_fp8":
        _refuse_block_fp8_on_spark()
    if recipe.kind == "mxfp8":
        _refuse_mxfp8_te_on_spark()

    try:
        from transformer_engine.common.recipe import (  # noqa: WPS433
            DelayedScaling,
            Float8CurrentScaling,
            Format,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"transformer_engine is required for FP8 recipe {recipe.kind!r} but is "
            f"not importable: {exc}. See repro/04-fp8-bringup.md for the source-build "
            "recipe (NVTE_CUDA_ARCHS=121 + LD_PRELOAD libcublasLt.so.13 trick)."
        ) from exc

    fmt = getattr(Format, recipe.format)

    # All three GEMMs share the same split-accumulator policy. The
    # MMParams dataclass exists in TE >= 1.13 (and is required for the
    # split-accumulator fprop fix to take effect on Float8CurrentScaling).
    try:
        from transformer_engine.common.recipe import MMParams  # noqa: WPS433
    except ImportError:
        MMParams = None  # type: ignore[assignment]

    def _mmparams():
        if MMParams is None:
            return None
        return MMParams(use_split_accumulator=recipe.use_split_accumulator)

    if recipe.kind == "current_fp8":
        kwargs: Dict[str, Any] = {"fp8_format": fmt}
        mm = _mmparams()
        if mm is not None:
            # CRITICAL: all three GEMMs. fprop default is False in stock TE.
            # See module docstring for the NaN-at-step-40 history.
            kwargs["fp8_gemm_fprop"] = mm
            kwargs["fp8_gemm_dgrad"] = mm
            kwargs["fp8_gemm_wgrad"] = mm
        kwargs.update(recipe.extras)
        return Float8CurrentScaling(**kwargs)

    if recipe.kind == "delayed_e4m3":
        return DelayedScaling(
            fp8_format=fmt,
            margin=recipe.margin,
            amax_history_len=recipe.amax_history_len,
            amax_compute_algo=recipe.amax_compute_algo,
            **recipe.extras,
        )

    if recipe.kind == "block_fp8":
        # Should have been refused above; defensive re-raise.
        from transformer_engine.common.recipe import Float8BlockScaling  # noqa: WPS433
        return Float8BlockScaling(fp8_format=fmt, **recipe.extras)

    if recipe.kind == "mxfp8":
        # Should have been refused above; defensive re-raise.
        from transformer_engine.common.recipe import MXFP8BlockScaling  # noqa: WPS433
        return MXFP8BlockScaling(fp8_format=fmt, **recipe.extras)

    raise ValueError(f"Unknown FP8 recipe kind: {recipe.kind!r}")


# -----------------------------------------------------------------------------
# Module wrapping (te.Linear + fused te.LayerNormMLP)
# -----------------------------------------------------------------------------
def wrap_with_te(model, recipe: FP8Recipe, fused: bool = True) -> Dict[str, int]:
    """Replace ``nn.Linear`` with ``te.Linear`` and fuse Qwen3-style
    ``post_attention_layernorm + mlp`` into ``te.LayerNormMLP``.

    Why ``fused=True`` is the default: at v11/v12 sizes the FFN backward
    activation graph (O(intermediate²)) is the memory cliff. Replacing
    each unfused ``norm → up_proj → gate → silu → down_proj`` chain with
    one ``te.LayerNormMLP`` eliminates intermediate norm materialization
    and is what makes ``intermediate=6144`` actually fit alongside FP8
    GEMM. FP8 alone, without fusion, gives roughly zero memory savings
    at these shapes.

    Returns a stats dict ``{"linear_replaced": N, "mlp_fused": M,
    "intermediate_size": ..., "skipped": ...}``. Caller can log or assert.

    This is a no-op if ``recipe.kind is None`` (bf16 mode).
    """
    if recipe.kind is None:
        return {"linear_replaced": 0, "mlp_fused": 0, "skipped": "recipe.kind is None"}

    try:
        import transformer_engine.pytorch as te  # noqa: WPS433
    except ImportError as exc:
        raise RuntimeError(
            f"transformer_engine.pytorch import failed: {exc}. See "
            "repro/04-fp8-bringup.md."
        ) from exc

    import torch.nn as nn  # noqa: WPS433

    stats = {"linear_replaced": 0, "mlp_fused": 0, "intermediate_size": None}

    # Pass 1: fuse Qwen3-style decoder MLPs (post_attention_layernorm + mlp).
    # We look for the canonical Qwen3 layer shape: a child named ``mlp`` with
    # gate_proj/up_proj/down_proj siblings, and a ``post_attention_layernorm``
    # sibling on the parent decoder layer.
    if fused:
        for parent_name, parent in list(model.named_modules()):
            mlp = getattr(parent, "mlp", None)
            ln = getattr(parent, "post_attention_layernorm", None)
            if mlp is None or ln is None:
                continue
            gate = getattr(mlp, "gate_proj", None)
            up = getattr(mlp, "up_proj", None)
            down = getattr(mlp, "down_proj", None)
            if not (isinstance(gate, nn.Linear) and isinstance(up, nn.Linear)
                    and isinstance(down, nn.Linear)):
                continue
            hidden = gate.in_features
            inter = gate.out_features
            stats["intermediate_size"] = inter
            # te.LayerNormMLP gives us LN -> [up, gate] -> silu -> down in one
            # fused module.
            fused_mlp = te.LayerNormMLP(
                hidden_size=hidden,
                ffn_hidden_size=inter,
                eps=getattr(ln, "variance_epsilon", 1e-6),
                normalization="RMSNorm",
                activation="swiglu",
                bias=False,
            )
            # Replace post_attention_layernorm + mlp with a wrapper that
            # routes through te.LayerNormMLP. We don't try to reattach it
            # under the original names; speculators forward path is the one
            # patched in ``patch_speculators_for_fp8.py``. Here we just leave
            # the fused module as ``parent.te_mlp`` and the speculators patch
            # picks it up.
            parent.te_mlp = fused_mlp
            stats["mlp_fused"] += 1

    # Pass 2: replace remaining nn.Linear with te.Linear (Q/K/V/O proj, lm_head, etc.).
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                te_lin = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                )
                # Copy weights so post-wrap state matches pre-wrap.
                with __import__("torch").no_grad():
                    te_lin.weight.copy_(child.weight)
                    if child.bias is not None:
                        te_lin.bias.copy_(child.bias)
                setattr(parent, child_name, te_lin)
                stats["linear_replaced"] += 1

    return stats


# -----------------------------------------------------------------------------
# autocast context
# -----------------------------------------------------------------------------
@contextlib.contextmanager
def fp8_autocast_ctx(recipe: FP8Recipe) -> Iterator[None]:
    """Context manager wrapping ``transformer_engine.pytorch.fp8_autocast``.

    No-op when ``recipe.kind is None``. Lazy-imports TE so the helper
    survives ``import dflash_llama`` on a non-TE host.
    """
    if recipe.kind is None:
        yield
        return

    try:
        import transformer_engine.pytorch as te  # noqa: WPS433
    except ImportError as exc:
        raise RuntimeError(
            f"transformer_engine.pytorch import failed: {exc}"
        ) from exc

    te_recipe = make_te_recipe(recipe)
    with te.fp8_autocast(enabled=True, fp8_recipe=te_recipe):
        yield


__all__ = [
    "FP8Recipe",
    "FP8Kind",
    "current_arch",
    "make_te_recipe",
    "wrap_with_te",
    "fp8_autocast_ctx",
]
