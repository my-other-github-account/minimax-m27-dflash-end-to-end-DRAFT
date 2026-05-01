"""Kimi-K2.5 family verifier config (DFlash drafter reference from z-lab)."""
from __future__ import annotations
from typing import Optional
from .base import BaseVerifier


def kimi_k25(
    *,
    name: str = "kimi-k2.5",
    gguf_path: Optional[str] = None,
    hf_path: Optional[str] = None,
) -> BaseVerifier:
    """Kimi-K2.5 verifier (61-layer, 163840 vocab).

    Layer taps are the canonical [1, 12, 24, 35, 47, 58] reported in the
    z-lab/Kimi-K2.5-DFlash reference config.
    """
    return BaseVerifier(
        name=name,
        family="kimi_k25",
        hidden_size=7168,
        vocab_size=163840,
        mask_token_id=163838,
        num_hidden_layers=61,
        layer_ids=(1, 12, 24, 35, 47, 58),
        drafter_arch="qwen3",
        drafter_hidden_act="silu",
        block_size=8,
        gguf_path=gguf_path,
        hf_path=hf_path,
    )
