"""Kimi-K2.5 family verifier config (DFlash drafter reference from z-lab)."""
from __future__ import annotations
from typing import Optional, Sequence
from ..base import BaseVerifier

_K25_HIDDEN_SIZE = 7168
_K25_VOCAB_SIZE = 163840
_K25_MASK_TOKEN_ID = 163838
_K25_NUM_HIDDEN_LAYERS = 61
_K25_LAYER_IDS = (1, 12, 24, 35, 47, 58)


def kimi_k25(
    *,
    name: str = "kimi-k2.5",
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
    """Kimi-K2.5 verifier (61-layer, 163840 vocab).

    Layer taps default to ``(1, 12, 24, 35, 47, 58)`` (z-lab/Kimi-K2.5-DFlash
    reference). Pass ``layer_ids=...`` to override.
    """
    return BaseVerifier(
        name=name,
        family="kimi_k25",
        hidden_size=_K25_HIDDEN_SIZE if hidden_size is None else int(hidden_size),
        vocab_size=_K25_VOCAB_SIZE if vocab_size is None else int(vocab_size),
        mask_token_id=_K25_MASK_TOKEN_ID if mask_token_id is None else int(mask_token_id),
        num_hidden_layers=(_K25_NUM_HIDDEN_LAYERS if num_hidden_layers is None
                           else int(num_hidden_layers)),
        layer_ids=tuple(_K25_LAYER_IDS if layer_ids is None else layer_ids),
        drafter_arch=drafter_arch,
        drafter_hidden_act=drafter_hidden_act,
        block_size=block_size,
        gguf_path=gguf_path,
        hf_path=hf_path,
    )
