"""MiniMax-M2.7 family verifier configs."""
from __future__ import annotations
from typing import Optional, Sequence
from .base import BaseVerifier

# Factory-level defaults — override via kwargs.
_M27_HIDDEN_SIZE = 3072
_M27_VOCAB_SIZE = 200064
_M27_MASK_TOKEN_ID = 200054
_M27_NUM_HIDDEN_LAYERS = 62
_M27_LAYER_IDS = (2, 16, 30, 45, 59, 61)


def minimax_m27(
    *,
    name: str = "minimax-m2.7",
    gguf_path: Optional[str] = None,
    hf_path: Optional[str] = None,
    hidden_size: Optional[int] = None,
    num_hidden_layers: Optional[int] = None,
    vocab_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    layer_ids: Optional[Sequence[int]] = None,
    block_size: int = 8,
    drafter_arch: str = "qwen3",
    drafter_hidden_act: str = "silu",
) -> BaseVerifier:
    """MiniMax-M2.7 verifier (62-layer MoE, 200064 vocab).

    DFlash layer taps default to ``(2, 16, 30, 45, 59, 61)`` which matches the
    canonical IQ4 traces emitted by the v2 generator. **Pass ``layer_ids=...``
    to use a different set of taps** (e.g. for a new training schedule).
    Hidden size / vocab / num layers are likewise overridable for users on a
    forked variant of the family.
    """
    return BaseVerifier(
        name=name,
        family="minimax_m2",
        hidden_size=_M27_HIDDEN_SIZE if hidden_size is None else int(hidden_size),
        vocab_size=_M27_VOCAB_SIZE if vocab_size is None else int(vocab_size),
        mask_token_id=_M27_MASK_TOKEN_ID if mask_token_id is None else int(mask_token_id),
        num_hidden_layers=(_M27_NUM_HIDDEN_LAYERS if num_hidden_layers is None
                           else int(num_hidden_layers)),
        layer_ids=tuple(_M27_LAYER_IDS if layer_ids is None else layer_ids),
        drafter_arch=drafter_arch,
        drafter_hidden_act=drafter_hidden_act,
        block_size=block_size,
        gguf_path=gguf_path,
        hf_path=hf_path,
    )


def minimax_m27_iq4_xs(
    *,
    gguf_path: Optional[str] = None,
    hf_path: Optional[str] = None,
    layer_ids: Optional[Sequence[int]] = None,
    **kw,
) -> BaseVerifier:
    """MiniMax-M2.7-UD-IQ4_XS quantized variant. Same shape as the bf16 base."""
    return minimax_m27(
        name="minimax-m2.7-iq4-xs",
        gguf_path=gguf_path,
        hf_path=hf_path,
        layer_ids=layer_ids,
        **kw,
    )
