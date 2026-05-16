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


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


@contextlib.contextmanager
def _te_param_init_context(enabled: Optional[bool] = None):
    """Optionally allocate TE modules with fp8 parameter storage."""
    enabled = _env_flag("TE_FP8_PARAMS") if enabled is None else enabled
    if not TE_AVAILABLE or not enabled:
        yield
        return
    recipe = None
    try:
        recipe = get_recipe("current_fp8")
    except Exception:
        recipe = None
    with te.fp8_model_init(
        enabled=True,
        recipe=recipe,
        preserve_high_precision_init_val=True,
    ):
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


def _build_layernorm_mlp_from(
    post_norm: nn.Module,
    mlp: nn.Module,
    *,
    te_fp8_params: Optional[bool] = None,
) -> "te.LayerNormMLP":
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

    with _te_param_init_context(te_fp8_params):
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


def _build_layernorm_linear_from(
    norm: nn.Module,
    linear: nn.Module,
    *,
    return_layernorm_output: bool = False,
    te_fp8_params: Optional[bool] = None,
) -> "te.LayerNormLinear":
    """Construct a te.LayerNormLinear mirroring (RMSNorm + Linear)."""
    assert TE_AVAILABLE
    has_bias = getattr(linear, "bias", None) is not None
    eps = _rmsnorm_eps(norm)
    dtype = linear.weight.dtype
    device = linear.weight.device
    with _te_param_init_context(te_fp8_params):
        fused = te.LayerNormLinear(
            linear.in_features,
            linear.out_features,
            eps=eps,
            bias=has_bias,
            normalization="RMSNorm",
            params_dtype=dtype,
            device=device,
            return_layernorm_output=return_layernorm_output,
        )
    with torch.no_grad():
        fused.layer_norm_weight.copy_(norm.weight.detach().to(dtype))
        if hasattr(fused, "layer_norm_bias") and getattr(fused, "layer_norm_bias") is not None:
            fused.layer_norm_bias.zero_()
        fused.weight.copy_(linear.weight.detach().to(dtype))
        if has_bias:
            fused.bias.copy_(linear.bias.detach().to(dtype))
    return fused


class _Qwen3DFlashFusedQProjAttention(nn.Module):
    """DFlash attention wrapper that fuses input RMSNorm + q_proj.

    DFlash shares k/v projection weights across the verifier-hidden and
    noise-hidden paths, so those stay as standalone Linear modules. The q path
    is still fusion-eligible because it consumes only the noise hidden states.
    """

    def __init__(
        self,
        input_layernorm: nn.Module,
        attn: nn.Module,
        *,
        te_fp8_params: Optional[bool] = None,
    ):
        super().__init__()
        self.config = attn.config
        self.layer_idx = attn.layer_idx
        self.head_dim = attn.head_dim
        self.num_key_value_groups = attn.num_key_value_groups
        self.scaling = attn.scaling
        self.attention_dropout = attn.attention_dropout
        self.is_causal = attn.is_causal
        self.sliding_window = getattr(attn, "sliding_window", None)

        self.q_proj = _build_layernorm_linear_from(
            input_layernorm,
            attn.q_proj,
            return_layernorm_output=True,
            te_fp8_params=te_fp8_params,
        )
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm

        g = attn.forward.__globals__
        self._apply_rotary_pos_emb = g["apply_rotary_pos_emb"]
        self._eager_attention_forward = g["eager_attention_forward"]
        self._all_attention_functions = g["ALL_ATTENTION_FUNCTIONS"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        q_proj_out = self.q_proj(hidden_states)
        if not isinstance(q_proj_out, tuple) or len(q_proj_out) < 2:
            raise RuntimeError(
                "te.LayerNormLinear(return_layernorm_output=True) must return "
                "(linear_output, layernorm_output)"
            )
        q, normed_hidden = q_proj_out[0], q_proj_out[1]

        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(normed_hidden)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(normed_hidden)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        attn_fn = self._eager_attention_forward
        if (
            getattr(self.config, "_attn_implementation", None) is not None
            and self.config._attn_implementation != "eager"
        ):
            attn_fn = self._all_attention_functions[self.config._attn_implementation]

        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def _fuse_qwen3_decoder_mlp_blocks(
    model: nn.Module,
    *,
    te_fp8_params: Optional[bool] = None,
) -> int:
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
        fused = _build_layernorm_mlp_from(post_norm, mlp, te_fp8_params=te_fp8_params)
        # Replace mlp first (so post_norm reference is still valid for build above);
        # then replace post_attention_layernorm with Identity.
        module.mlp = fused
        module.post_attention_layernorm = nn.Identity()
        n += 1
    return n


def _fuse_qwen3_attention_qkv(
    model: nn.Module,
    *,
    te_fp8_params: Optional[bool] = None,
) -> int:
    """Fuse the DFlash attention block's input RMSNorm + q_proj path.

    DFlash reuses the same k/v projection weights for both verifier-hidden and
    noise-hidden activations, so a full qkv merge would duplicate parameters and
    change training semantics. The safe extension is to fuse the q path and keep
    shared k/v projections as TE linears.
    """
    if not TE_AVAILABLE:
        return 0
    n = 0
    for module in model.modules():
        input_norm = getattr(module, "input_layernorm", None)
        attn = getattr(module, "self_attn", None)
        if input_norm is None or attn is None:
            continue
        if not _is_rmsnorm(input_norm):
            continue
        q_proj = getattr(attn, "q_proj", None)
        k_proj = getattr(attn, "k_proj", None)
        v_proj = getattr(attn, "v_proj", None)
        o_proj = getattr(attn, "o_proj", None)
        if not all(isinstance(m, nn.Linear) for m in (q_proj, k_proj, v_proj, o_proj)):
            continue
        module.self_attn = _Qwen3DFlashFusedQProjAttention(
            input_norm,
            attn,
            te_fp8_params=te_fp8_params,
        )
        module.input_layernorm = nn.Identity()
        n += 1
    return n


def _fuse_final_norm_lm_head(
    model: nn.Module,
    *,
    te_fp8_params: Optional[bool] = None,
) -> int:
    """Fuse the top-level RMSNorm + lm_head into te.LayerNormLinear."""
    if not TE_AVAILABLE:
        return 0
    norm = getattr(model, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if not _is_rmsnorm(norm) or not isinstance(lm_head, nn.Linear):
        return 0
    model.lm_head = _build_layernorm_linear_from(
        norm,
        lm_head,
        te_fp8_params=te_fp8_params,
    )
    model.norm = nn.Identity()
    return 1


# ---- nn.Linear -> te.Linear monkey-patch -------------------------------------
def _replace_linear(
    module: nn.Module,
    parent_name: str = "",
    *,
    te_fp8_params: Optional[bool] = None,
) -> int:
    """Recursively replace nn.Linear children with te.Linear, copying weights."""
    if not TE_AVAILABLE:
        return 0
    n_replaced = 0
    for name, child in list(module.named_children()):
        full = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, nn.Linear) and not isinstance(child, te.Linear):
            in_f, out_f = child.in_features, child.out_features
            bias = child.bias is not None
            with _te_param_init_context(te_fp8_params):
                new = te.Linear(
                    in_f,
                    out_f,
                    bias=bias,
                    params_dtype=child.weight.dtype,
                    device=child.weight.device,
                )
            with torch.no_grad():
                new.weight.copy_(child.weight.detach())
                if bias:
                    new.bias.copy_(child.bias.detach())
            setattr(module, name, new)
            n_replaced += 1
        else:
            n_replaced += _replace_linear(child, full, te_fp8_params=te_fp8_params)
    return n_replaced


def wrap_with_te(
    model: nn.Module,
    fp8: bool = True,
    *,
    te_fp8_params: Optional[bool] = None,
) -> nn.Module:
    """In-place wrap: replace every nn.Linear in `model` with te.Linear.

    If env TE_USE_FUSED=1, also fuses Qwen3 (post_attention_layernorm + mlp)
    into te.LayerNormMLP BEFORE doing the plain nn.Linear -> te.Linear sweep.
    The newer attention/lm_head fusions remain enabled by default and can be
    disabled per-run with ``TE_DISABLE_EXTENDED_FUSION=1`` to reproduce the
    older v12-stable MLP-only fused recipe.

    Returns model unchanged if fp8=False or TE unavailable.
    Idempotent: calling twice is a no-op.
    """
    if not fp8:
        return model
    if not TE_AVAILABLE:
        logger.warning("TE not available; returning model unwrapped")
        return model

    if te_fp8_params is None:
        te_fp8_params = _env_flag("TE_FP8_PARAMS")
    use_fused = os.environ.get("TE_USE_FUSED", "0") == "1"
    use_extended_fusion = os.environ.get("TE_DISABLE_EXTENDED_FUSION", "0") != "1"
    n_fused_mlp = 0
    n_fused_attn = 0
    n_fused_lm_head = 0
    if use_fused:
        n_fused_mlp = _fuse_qwen3_decoder_mlp_blocks(model, te_fp8_params=te_fp8_params)
        if use_extended_fusion:
            n_fused_attn = _fuse_qwen3_attention_qkv(model, te_fp8_params=te_fp8_params)
            n_fused_lm_head = _fuse_final_norm_lm_head(model, te_fp8_params=te_fp8_params)
        logger.info(f"[te_wrap] fused {n_fused_mlp} Qwen3 (post_norm+mlp) -> te.LayerNormMLP")
        if use_extended_fusion:
            logger.info(
                "[te_wrap] fused %s DFlash attention blocks (input_norm+q_proj) -> "
                "te.LayerNormLinear and %s final norm+lm_head pairs",
                n_fused_attn,
                n_fused_lm_head,
            )
        else:
            logger.info("[te_wrap] extended TE fusion disabled by TE_DISABLE_EXTENDED_FUSION=1")
        print(f"[te_wrap] fused {n_fused_mlp} Qwen3 (post_norm+mlp) -> te.LayerNormMLP", flush=True)
        if use_extended_fusion:
            print(
                "[te_wrap] fused "
                f"{n_fused_attn} DFlash attention blocks (input_norm+q_proj) -> "
                f"te.LayerNormLinear and {n_fused_lm_head} final norm+lm_head pairs",
                flush=True,
            )
        else:
            print("[te_wrap] extended TE fusion disabled by TE_DISABLE_EXTENDED_FUSION=1", flush=True)

    n = _replace_linear(model, te_fp8_params=te_fp8_params)
    logger.info(f"[te_wrap] replaced {n} nn.Linear with te.Linear")
    if te_fp8_params:
        logger.info("[te_wrap] enabled TE fp8_model_init parameter storage")
    return model


def fusion_coverage(model: nn.Module) -> dict:
    """Classify module coverage across TE/Liger/vanilla buckets."""
    coverage = {
        "te_layernorm_mlp": [],
        "te_layernorm_linear": [],
        "te_linear": [],
        "liger_fused_lce": [],
        "liger_rope": [],
        "nn_linear": [],
        "nn_embedding": [],
        "unfused": [],
    }
    te_linear_cls = getattr(te, "Linear", tuple()) if TE_AVAILABLE else tuple()
    te_ln_linear_cls = getattr(te, "LayerNormLinear", tuple()) if TE_AVAILABLE else tuple()
    te_ln_mlp_cls = getattr(te, "LayerNormMLP", tuple()) if TE_AVAILABLE else tuple()
    if not isinstance(te_linear_cls, type):
        te_linear_cls = tuple()
    if not isinstance(te_ln_linear_cls, type):
        te_ln_linear_cls = tuple()
    if not isinstance(te_ln_mlp_cls, type):
        te_ln_mlp_cls = tuple()
    for name, module in model.named_modules():
        if not name:
            continue
        if te_ln_mlp_cls and isinstance(module, te_ln_mlp_cls):
            coverage["te_layernorm_mlp"].append(name)
        elif te_ln_linear_cls and isinstance(module, te_ln_linear_cls):
            coverage["te_layernorm_linear"].append(name)
        elif te_linear_cls and isinstance(module, te_linear_cls):
            coverage["te_linear"].append(name)
        elif isinstance(module, nn.Linear):
            coverage["nn_linear"].append(name)
            coverage["unfused"].append(name)
        elif isinstance(module, nn.Embedding):
            coverage["nn_embedding"].append(name)
    if getattr(model, "_liger_fused_linear_ce", False):
        coverage["liger_fused_lce"].append("model.forward")
    rope_patches = int(getattr(model, "_liger_rope_patches", 0) or 0)
    for idx in range(rope_patches):
        coverage["liger_rope"].append(f"rope_patch_{idx}")
    return {
        "coverage": coverage,
        "summary": {k: len(v) for k, v in coverage.items()},
    }


def count_linears(model: nn.Module):
    """Backward-compatible summary view over fusion_coverage()."""
    summary = fusion_coverage(model)["summary"]
    return {
        "nn_linear": summary["nn_linear"],
        "te_linear": summary["te_linear"],
        "te_layernorm_linear": summary["te_layernorm_linear"],
        "te_layernorm_mlp": summary["te_layernorm_mlp"],
    }


def unfused_to_fused_state_dict(state_dict: dict, model_arch: str = "qwen3") -> dict:
    """Translate pre-fusion DFlash Qwen3 state dict keys into the fused layout."""
    if model_arch != "qwen3":
        raise ValueError(f"unsupported model_arch: {model_arch}")
    out = dict(state_dict)
    rename_pairs = []
    for key in list(state_dict):
        if ".input_layernorm.weight" in key:
            rename_pairs.append((key, key.replace(".input_layernorm.weight", ".self_attn.q_proj.layer_norm_weight")))
        elif key == "norm.weight":
            rename_pairs.append((key, "lm_head.layer_norm_weight"))
    for src, dst in rename_pairs:
        out[dst] = out.pop(src)
    mlp_prefixes = sorted(
        {
            key[: -len("gate_proj.weight")]
            for key in state_dict
            if key.endswith("gate_proj.weight")
        }
    )
    for prefix in mlp_prefixes:
        gate_w = out.pop(prefix + "gate_proj.weight")
        up_w = out.pop(prefix + "up_proj.weight")
        down_w = out.pop(prefix + "down_proj.weight")
        out[prefix + "layer_norm_weight"] = out.pop(prefix.replace("mlp.", "post_attention_layernorm.") + "weight")
        out[prefix + "fc1_weight"] = torch.cat([gate_w, up_w], dim=0)
        out[prefix + "fc2_weight"] = down_w
        gate_b_key = prefix + "gate_proj.bias"
        up_b_key = prefix + "up_proj.bias"
        down_b_key = prefix + "down_proj.bias"
        if gate_b_key in out and up_b_key in out:
            out[prefix + "fc1_bias"] = torch.cat([out.pop(gate_b_key), out.pop(up_b_key)], dim=0)
        if down_b_key in out:
            out[prefix + "fc2_bias"] = out.pop(down_b_key)
    return out


def fused_to_unfused_state_dict(state_dict: dict, model_arch: str = "qwen3") -> dict:
    """Translate fused-layout DFlash Qwen3 state dict keys back to the pre-fusion layout."""
    if model_arch != "qwen3":
        raise ValueError(f"unsupported model_arch: {model_arch}")
    out = dict(state_dict)
    rename_pairs = []
    for key in list(state_dict):
        if ".self_attn.q_proj.layer_norm_weight" in key:
            rename_pairs.append((key, key.replace(".self_attn.q_proj.layer_norm_weight", ".input_layernorm.weight")))
        elif key == "lm_head.layer_norm_weight":
            rename_pairs.append((key, "norm.weight"))
    for src, dst in rename_pairs:
        out[dst] = out.pop(src)
    mlp_prefixes = sorted(
        {
            key[: -len("layer_norm_weight")]
            for key in state_dict
            if key.endswith("mlp.layer_norm_weight")
        }
    )
    for prefix in mlp_prefixes:
        layer_norm_weight = out.pop(prefix + "layer_norm_weight")
        fc1_weight = out.pop(prefix + "fc1_weight")
        fc2_weight = out.pop(prefix + "fc2_weight")
        split = fc1_weight.shape[0] // 2
        out[prefix.replace("mlp.", "post_attention_layernorm.") + "weight"] = layer_norm_weight
        out[prefix + "gate_proj.weight"] = fc1_weight[:split]
        out[prefix + "up_proj.weight"] = fc1_weight[split:]
        out[prefix + "down_proj.weight"] = fc2_weight
        fc1_bias_key = prefix + "fc1_bias"
        fc2_bias_key = prefix + "fc2_bias"
        if fc1_bias_key in out:
            fc1_bias = out.pop(fc1_bias_key)
            split_bias = fc1_bias.shape[0] // 2
            out[prefix + "gate_proj.bias"] = fc1_bias[:split_bias]
            out[prefix + "up_proj.bias"] = fc1_bias[split_bias:]
        if fc2_bias_key in out:
            out[prefix + "down_proj.bias"] = out.pop(fc2_bias_key)
    return out
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
    "fusion_coverage",
    "count_linears",
    "unfused_to_fused_state_dict",
    "fused_to_unfused_state_dict",
]
