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
from types import MethodType
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention

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
def _te_param_init_context():
    """Optionally allocate TE modules with fp8 parameter storage."""
    if not TE_AVAILABLE or not _env_flag("TE_FP8_PARAMS"):
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


def _dense_attention_mask(
    attention_mask: Any,
    *,
    batch_size: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Normalize HF/flex-attention masks to the dense bool layout TE DPA accepts."""
    if attention_mask is None:
        return None
    if hasattr(attention_mask, "mask_mod") and hasattr(attention_mask, "seq_lengths"):
        batch_idx = torch.zeros((), device=device, dtype=torch.long)
        head_idx = torch.zeros((), device=device, dtype=torch.long)
        q_idx = torch.arange(q_len, device=device, dtype=torch.long).view(q_len, 1)
        kv_idx = torch.arange(kv_len, device=device, dtype=torch.long).view(1, kv_len)
        mask = attention_mask.mask_mod(batch_idx, head_idx, q_idx, kv_idx)
        return mask.to(torch.bool).unsqueeze(0).unsqueeze(0).contiguous()
    if hasattr(attention_mask, "to_dense"):
        attention_mask = attention_mask.to_dense()
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError(
            "TE DPA requires a tensor-like attention_mask; "
            f"got {type(attention_mask)!r}"
        )
    mask = attention_mask.to(device=device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError(
            "TE DPA requires a 2D/3D/4D attention mask after densification; "
            f"got shape {tuple(mask.shape)}"
        )
    if mask.shape[-2:] != (q_len, kv_len):
        raise ValueError(
            "TE DPA attention mask has unexpected trailing dims: "
            f"{tuple(mask.shape)} vs expected (*, *, {q_len}, {kv_len})"
        )
    if mask.shape[0] not in (1, batch_size):
        raise ValueError(
            "TE DPA attention mask batch dim must broadcast to the query batch: "
            f"{tuple(mask.shape)} for batch_size={batch_size}"
        )
    return mask.to(torch.bool).contiguous()


def _in_block_weights(
    length: int,
    *,
    block_size: int = 8,
    gamma: float = 4.0,
    device=None,
    dtype=None,
) -> torch.Tensor:
    idx = torch.arange(length, device=device)
    in_block_idx = idx % block_size
    weights = torch.exp(-((in_block_idx - 1).clamp(min=0)).to(dtype or torch.float32) / gamma)
    return weights * (in_block_idx != 0).to(dtype or torch.float32)


def _compute_accuracy_from_predictions(
    predicted_tokens: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    block_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    correct = (predicted_tokens == targets) & loss_mask.to(torch.bool)
    correct = correct.reshape(1, -1, block_size)
    mask = loss_mask.to(torch.bool).reshape(1, -1, block_size)
    per_block_idx_sum = correct.float().sum(dim=1)
    per_block_idx_denom = mask.float().sum(dim=1)
    total_sum = per_block_idx_sum.sum()
    total_denom = per_block_idx_denom.sum()
    return total_sum / (total_denom + 1e-5), (per_block_idx_sum / (per_block_idx_denom + 1e-5)).reshape(-1)


def dflash_weighted_ce_reference(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    block_size: int = 8,
    gamma: float = 4.0,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    logits = hidden_states @ lm_head_weight.t()
    if bias is not None and bias.numel() > 0:
        logits = logits + bias
    batch, length, vocab = logits.shape
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(batch * length, vocab),
        target_ids.reshape(batch * length),
        reduction="none",
        ignore_index=-100,
    ).view(batch, length)
    weights = _in_block_weights(length, block_size=block_size, gamma=gamma, device=logits.device, dtype=logits.dtype)
    weights = weights.view(1, length)
    mask = loss_mask.to(logits.dtype).view(batch, length)
    weighted_ce = ce * weights * mask
    denom = (weights * mask).sum(dim=1) + 1e-5
    loss = (weighted_ce.sum(dim=1) / denom).mean()
    predicted_tokens = torch.argmax(logits, dim=-1)
    full_acc, per_position_acc = _compute_accuracy_from_predictions(
        predicted_tokens,
        target_ids,
        loss_mask,
        block_size=block_size,
    )
    metrics: dict[str, torch.Tensor] = {
        "loss": loss.detach().clone(),
        "full_acc": full_acc,
    }
    for pos in range(1, len(per_position_acc)):
        metrics[f"position {pos} acc"] = per_position_acc[pos]
    return loss, metrics, predicted_tokens


def dflash_weighted_ce_chunked(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    chunk_size: int,
    block_size: int = 8,
    gamma: float = 4.0,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    if chunk_size <= 0:
        return dflash_weighted_ce_reference(
            hidden_states,
            lm_head_weight,
            target_ids,
            loss_mask,
            block_size=block_size,
            gamma=gamma,
            bias=bias,
        )
    batch, length, _hidden = hidden_states.shape
    weights = _in_block_weights(
        length,
        block_size=block_size,
        gamma=gamma,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    ).view(1, length)
    mask = loss_mask.to(hidden_states.dtype).view(batch, length)
    weighted_ce_sum = hidden_states.new_zeros(batch)
    denom = (weights * mask).sum(dim=1) + 1e-5
    predicted_chunks: list[torch.Tensor] = []
    vocab = lm_head_weight.shape[0]
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        chunk_hidden = hidden_states[:, start:end, :]
        chunk_logits = chunk_hidden @ lm_head_weight.t()
        if bias is not None and bias.numel() > 0:
            chunk_logits = chunk_logits + bias
        chunk_targets = target_ids[:, start:end]
        chunk_ce = torch.nn.functional.cross_entropy(
            chunk_logits.reshape(batch * (end - start), vocab),
            chunk_targets.reshape(batch * (end - start)),
            reduction="none",
            ignore_index=-100,
        ).view(batch, end - start)
        weighted_ce_sum = weighted_ce_sum + (
            chunk_ce
            * weights[:, start:end]
            * mask[:, start:end]
        ).sum(dim=1)
        predicted_chunks.append(torch.argmax(chunk_logits, dim=-1))
    loss = (weighted_ce_sum / denom).mean()
    predicted_tokens = torch.cat(predicted_chunks, dim=1)
    full_acc, per_position_acc = _compute_accuracy_from_predictions(
        predicted_tokens,
        target_ids,
        loss_mask,
        block_size=block_size,
    )
    metrics: dict[str, torch.Tensor] = {
        "loss": loss.detach().clone(),
        "full_acc": full_acc,
    }
    for pos in range(1, len(per_position_acc)):
        metrics[f"position {pos} acc"] = per_position_acc[pos]
    return loss, metrics, predicted_tokens


def _apply_chunked_ce(model: nn.Module, *, chunk_size: int) -> None:
    if chunk_size <= 0 or getattr(model, "_dflash_chunked_ce_size", 0) == chunk_size:
        return

    def _chunked_forward(
        self,
        hidden_states,
        input_ids,
        loss_mask,
        verifier_last_hidden_states,
        lengths=None,
        position_ids=None,
        **kwargs,
    ):
        from speculators.models.dflash.core import (
            create_anchor_block_mask_mod,
            create_block_mask,
            get_base_indices_for_anchored_blocks,
            select_anchors,
        )

        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        num_anchors = self.config.max_anchors
        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = 1 + torch.arange(total_seq_len, dtype=torch.long, device=device).unsqueeze(0)

        anchor_positions, anchor_valid = select_anchors(loss_mask, num_anchors, self.block_size)
        anchor_positions = anchor_positions[anchor_valid]
        if anchor_positions.numel() == 0:
            raise ValueError("No valid anchors were selected for this batch")

        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            lengths=lengths.to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
        )
        attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device)

        mask_tokens_size = anchor_positions.numel() * self.block_size
        mask_token_ids = torch.full((1, mask_tokens_size), self.mask_token_id, dtype=torch.long, device=device)
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)

        fc_output = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output)
        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[:, anchor_positions], self.block_size, input_ids.numel()
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size, input_ids.numel()
        )
        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(self.verifier_norm(verifier_last_hidden_states))
        verifier_preds = torch.argmax(verifier_logits, dim=-1)
        verifier_preds = torch.cat([verifier_preds.new_zeros(1, 1), verifier_preds[:, :-1]], dim=1)
        targets = verifier_preds[:, anchored_block_indices]

        for layer in self.layers:
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        final_hidden = self.norm(noise_embedding)
        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        aligned_loss_mask[:, :: self.block_size] = 0
        loss, metrics, predicted_tokens = dflash_weighted_ce_chunked(
            final_hidden,
            self.lm_head.weight,
            targets,
            aligned_loss_mask,
            chunk_size=chunk_size,
            block_size=self.block_size,
            bias=getattr(self.lm_head, "bias", None),
        )
        return predicted_tokens, loss, metrics

    model.forward = MethodType(_chunked_forward, model)
    model._dflash_chunked_ce_size = chunk_size


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

    with _te_param_init_context():
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
) -> "te.LayerNormLinear":
    """Construct a te.LayerNormLinear mirroring (RMSNorm + Linear)."""
    assert TE_AVAILABLE
    has_bias = getattr(linear, "bias", None) is not None
    eps = _rmsnorm_eps(norm)
    dtype = linear.weight.dtype
    device = linear.weight.device
    with _te_param_init_context():
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

    def __init__(self, input_layernorm: nn.Module, attn: nn.Module):
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
        self._use_te_dpa = bool(
            TE_AVAILABLE and hasattr(te, "DotProductAttention") and _env_flag("TE_USE_DPA")
        )
        self._compile_flex = bool(hasattr(torch, "compile") and _env_flag("DFLASH_COMPILE_FLEX"))
        self._te_dpa = None
        self._compiled_flex_attention = None
        if self._use_te_dpa:
            self._te_dpa = te.DotProductAttention(
                num_attention_heads=attn.q_proj.out_features // self.head_dim,
                kv_channels=self.head_dim,
                num_gqa_groups=attn.k_proj.out_features // self.head_dim,
                attention_dropout=0.0,
                qkv_format="bshd",
                attn_mask_type="arbitrary",
                softmax_scale=self.scaling,
            )
        elif self._compile_flex:
            def _flex_compiled(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                block_mask,
                scale: float,
            ):
                return torch_flex_attention(
                    query,
                    key,
                    value,
                    score_mod=None,
                    block_mask=block_mask,
                    enable_gqa=query.shape[1] != key.shape[1],
                    scale=scale,
                )

            self._compiled_flex_attention = torch.compile(
                _flex_compiled,
                fullgraph=False,
                dynamic=False,
            )

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
        q = self.q_norm(q)

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(normed_hidden)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(normed_hidden)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v_h = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v_h = past_key_values.update(k, v_h, self.layer_idx, cache_kwargs)

        if self._te_dpa is not None:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v_h.transpose(1, 2)
            dense_mask = _dense_attention_mask(
                attention_mask,
                batch_size=bsz,
                q_len=q_len,
                kv_len=ctx_len + q_len,
                device=q.device,
            )
            attn_output = self._te_dpa(
                q,
                k,
                v,
                attention_mask=dense_mask,
                attn_mask_type="arbitrary",
            )
            attn_weights = None
        else:
            if self._compiled_flex_attention is not None and attention_mask is not None:
                flex_out = self._compiled_flex_attention(
                    q.contiguous(),
                    k.contiguous(),
                    v_h.contiguous(),
                    attention_mask,
                    self.scaling,
                )
                attn_output = flex_out.transpose(1, 2).contiguous()
                attn_weights = None
            else:
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
                    v_h,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,
                    **kwargs,
                )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


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


def _fuse_qwen3_attention_qkv(model: nn.Module) -> int:
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
        module.self_attn = _Qwen3DFlashFusedQProjAttention(input_norm, attn)
        module.input_layernorm = nn.Identity()
        n += 1
    return n


def _fuse_final_norm_lm_head(model: nn.Module) -> int:
    """Fuse the top-level RMSNorm + lm_head into te.LayerNormLinear."""
    if not TE_AVAILABLE:
        return 0
    norm = getattr(model, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if not _is_rmsnorm(norm) or not isinstance(lm_head, nn.Linear):
        return 0
    model.lm_head = _build_layernorm_linear_from(norm, lm_head)
    model.norm = nn.Identity()
    return 1


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
            with _te_param_init_context():
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

    use_fused = os.environ.get("TE_USE_FUSED", "0") == "1"
    use_extended_fusion = os.environ.get("TE_DISABLE_EXTENDED_FUSION", "0") != "1"
    chunked_ce_size = int(os.environ.get("DFLASH_FUSED_CE_CHUNK", "0") or "0")
    n_fused_mlp = 0
    n_fused_attn = 0
    n_fused_lm_head = 0
    if use_fused:
        n_fused_mlp = _fuse_qwen3_decoder_mlp_blocks(model)
        if use_extended_fusion:
            n_fused_attn = _fuse_qwen3_attention_qkv(model)
            n_fused_lm_head = _fuse_final_norm_lm_head(model)
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

    n = _replace_linear(model)
    logger.info(f"[te_wrap] replaced {n} nn.Linear with te.Linear")
    if _env_flag("TE_FP8_PARAMS"):
        logger.info("[te_wrap] enabled TE fp8_model_init parameter storage")
    if chunked_ce_size > 0:
        _apply_chunked_ce(model, chunk_size=chunked_ce_size)
        logger.info("[te_wrap] enabled DFlash chunked CE with chunk_size=%s", chunked_ce_size)
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
    "dflash_weighted_ce_reference",
    "dflash_weighted_ce_chunked",
    "fusion_coverage",
    "count_linears",
    "unfused_to_fused_state_dict",
    "fused_to_unfused_state_dict",
]
