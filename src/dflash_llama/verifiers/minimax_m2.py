"""MiniMax-M2.7 family verifier configs."""
from __future__ import annotations
from typing import Optional
from .base import BaseVerifier


def minimax_m27(
    *,
    name: str = "minimax-m2.7",
    gguf_path: Optional[str] = None,
    hf_path: Optional[str] = None,
) -> BaseVerifier:
    """MiniMax-M2.7 verifier (62-layer MoE, 200064 vocab).

    DFlash layer taps default to [2, 16, 30, 45, 59, 61] which matches the
    canonical IQ4 traces emitted by the v2 generator.
    """
    return BaseVerifier(
        name=name,
        family="minimax_m2",
        hidden_size=3072,
        vocab_size=200064,
        mask_token_id=200054,
        num_hidden_layers=62,
        layer_ids=(2, 16, 30, 45, 59, 61),
        drafter_arch="qwen3",
        drafter_hidden_act="silu",
        block_size=8,
        gguf_path=gguf_path,
        hf_path=hf_path,
    )


def minimax_m27_iq4_xs(
    *, gguf_path: Optional[str] = None, hf_path: Optional[str] = None
) -> BaseVerifier:
    """MiniMax-M2.7-UD-IQ4_XS quantized variant. Same shape as the bf16 base."""
    return minimax_m27(
        name="minimax-m2.7-iq4-xs",
        gguf_path=gguf_path,
        hf_path=hf_path,
    )
