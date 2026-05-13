"""Import-safe Liger integration for DFlash training.

The target model here is the speculators DFlash drafter, not a stock HF
causal-LM. We therefore avoid broad global monkey-patching and instead expose
small, explicit hooks:

- ``apply_liger(model, fused_linear_ce=True, rope=True, rms_norm=False)``
- ``dflash_weighted_ce_reference(...)`` for exact-reference checks
- ``dflash_weighted_ce_liger(...)`` for the fused loss path

When Liger is unavailable, ``apply_liger`` becomes a no-op and the fused-loss
helpers fall back to the exact torch reference implementation.
"""
from __future__ import annotations

import logging
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import liger_kernel
    from liger_kernel.transformers import (
        LigerFusedLinearCrossEntropyLoss,
        LigerRMSNorm,
        liger_rotary_pos_emb,
    )
    LIGER_AVAILABLE = True
    LIGER_VERSION = getattr(liger_kernel, "__version__", "?")
except ImportError as e:
    LIGER_AVAILABLE = False
    LIGER_VERSION = None
    logger.warning(f"liger-kernel not importable: {e}")


def _in_block_weights(length: int, *, block_size: int = 8, gamma: float = 4.0, device=None, dtype=None):
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


def _metrics_from_predictions(
    predicted_tokens: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    block_size: int = 8,
) -> dict[str, Any]:
    full_acc, per_position_acc = _compute_accuracy_from_predictions(
        predicted_tokens,
        targets,
        loss_mask,
        block_size=block_size,
    )
    metrics: dict[str, Any] = {"full_acc": full_acc}
    for pos in range(1, len(per_position_acc)):
        metrics[f"position {pos} acc"] = per_position_acc[pos]
    return metrics


def dflash_weighted_ce_reference(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    block_size: int = 8,
    gamma: float = 4.0,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    """Exact torch reference for the DFlash weighted CE path."""
    logits = hidden_states @ lm_head_weight.t()
    if bias is not None:
        logits = logits + bias
    batch, length, vocab = logits.shape
    ce = F.cross_entropy(
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
    metrics = _metrics_from_predictions(
        predicted_tokens,
        target_ids,
        loss_mask,
        block_size=block_size,
    )
    metrics["loss"] = loss.detach().clone()
    return loss, metrics, predicted_tokens


def dflash_weighted_ce_liger(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    block_size: int = 8,
    gamma: float = 4.0,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    """Fused DFlash weighted CE via Liger when available, else torch reference."""
    if not LIGER_AVAILABLE:
        return dflash_weighted_ce_reference(
            hidden_states,
            lm_head_weight,
            target_ids,
            loss_mask,
            block_size=block_size,
            gamma=gamma,
            bias=bias,
        )

    batch, length, hidden = hidden_states.shape
    fused = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100,
        reduction="none",
        return_predicted_tokens=True,
    )
    result = fused(
        lm_head_weight,
        hidden_states.reshape(batch * length, hidden),
        target_ids.reshape(batch * length),
        bias=bias,
    )
    token_loss = result.loss.view(batch, length)
    predicted_tokens = result.predicted_tokens.view(batch, length)
    weights = _in_block_weights(length, block_size=block_size, gamma=gamma, device=hidden_states.device, dtype=hidden_states.dtype)
    weights = weights.view(1, length)
    mask = loss_mask.to(hidden_states.dtype).view(batch, length)
    weighted_ce = token_loss.to(hidden_states.dtype) * weights * mask
    denom = (weights * mask).sum(dim=1) + 1e-5
    loss = (weighted_ce.sum(dim=1) / denom).mean()
    metrics = _metrics_from_predictions(
        predicted_tokens,
        target_ids,
        loss_mask,
        block_size=block_size,
    )
    metrics["loss"] = loss.detach().clone()
    return loss, metrics, predicted_tokens


def _patch_liger_rms_norm(module):
    if not LIGER_AVAILABLE:
        return module
    liger = LigerRMSNorm(
        module.weight.shape[0],
        eps=getattr(module, "variance_epsilon", None) or getattr(module, "eps", 1e-6),
    )
    with torch.no_grad():
        liger.weight.copy_(module.weight.detach().to(liger.weight.dtype))
    liger = liger.to(device=module.weight.device, dtype=module.weight.dtype)
    return liger


def _apply_liger_rope(model):
    patched = 0
    for module in model.modules():
        if hasattr(module, "_apply_rotary_pos_emb"):
            module._apply_rotary_pos_emb = liger_rotary_pos_emb
            patched += 1
    try:
        import speculators.models.dflash.model_definitions as defs

        defs.apply_rotary_pos_emb = liger_rotary_pos_emb
        patched += 1
    except ImportError:
        pass
    return patched


def _apply_liger_rms_norm(model):
    patched = 0
    for name, child in list(model.named_children()):
        if child.__class__.__name__.lower().endswith("rmsnorm") and hasattr(child, "weight"):
            setattr(model, name, _patch_liger_rms_norm(child))
            patched += 1
        else:
            patched += _apply_liger_rms_norm(child)
    return patched


def _liger_forward(self, hidden_states, input_ids, loss_mask, verifier_last_hidden_states, lengths=None, position_ids=None, **kwargs):
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
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        lengths=lengths.to(device),
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=self.block_size,
    )
    attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device)

    mask_tokens_size = num_anchors * self.block_size
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
    aligned_loss_mask = aligned_loss_mask * (
        anchor_valid.repeat_interleave(self.block_size).unsqueeze(0).to(aligned_loss_mask.dtype)
    )
    aligned_loss_mask[:, :: self.block_size] = 0

    loss, metrics, predicted_tokens = dflash_weighted_ce_liger(
        final_hidden,
        self.lm_head.weight,
        targets,
        aligned_loss_mask,
        block_size=self.block_size,
    )
    return predicted_tokens, loss, metrics


def apply_liger(model, *, fused_linear_ce: bool = True, rope: bool = True, rms_norm: bool = False):
    """Apply Liger integrations in-place when available.

    Intended apply order: TransformerEngine first, then Liger.
    """
    if not LIGER_AVAILABLE:
        logger.warning("Liger not available; returning model unmodified")
        return model

    patched = {"fused_linear_ce": False, "rope": 0, "rms_norm": 0}
    if rope:
        patched["rope"] = _apply_liger_rope(model)
    if rms_norm:
        patched["rms_norm"] = _apply_liger_rms_norm(model)
    if fused_linear_ce and hasattr(model, "lm_head") and hasattr(model, "layers") and hasattr(model, "norm"):
        model.forward = MethodType(_liger_forward, model)
        patched["fused_linear_ce"] = True

    logger.info(
        "[liger_wrap] LIGER_VERSION=%s fused_linear_ce=%s rope_patches=%s rms_norm_patches=%s",
        LIGER_VERSION,
        patched["fused_linear_ce"],
        patched["rope"],
        patched["rms_norm"],
    )
    return model


__all__ = [
    "LIGER_AVAILABLE",
    "LIGER_VERSION",
    "apply_liger",
    "dflash_weighted_ce_reference",
    "dflash_weighted_ce_liger",
]
