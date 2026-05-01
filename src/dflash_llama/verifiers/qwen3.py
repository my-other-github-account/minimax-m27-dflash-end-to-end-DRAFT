"""Generic Qwen3 family verifier config.

Qwen3 has many sizes; the most common training targets are 4B and 14B.
Callers should pass explicit hidden_size / num_hidden_layers / layer_ids
unless one of the named factories matches their setup.
"""
from __future__ import annotations
from typing import Optional, Sequence
from .base import BaseVerifier


def qwen3(
    *,
    name: str = "qwen3-generic",
    hidden_size: int,
    num_hidden_layers: int,
    vocab_size: int = 151936,
    mask_token_id: int = 151643,
    layer_ids: Optional[Sequence[int]] = None,
    gguf_path: Optional[str] = None,
    hf_path: Optional[str] = None,
) -> BaseVerifier:
    if layer_ids is None:
        # Default: 6 evenly spaced taps + final layer
        n = num_hidden_layers
        layer_ids = tuple(sorted({
            max(1, n // 12),
            max(2, n // 4),
            max(3, n // 2),
            max(4, 3 * n // 4),
            max(5, 11 * n // 12),
            n - 1,
        }))
    return BaseVerifier(
        name=name,
        family="qwen3",
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        num_hidden_layers=num_hidden_layers,
        layer_ids=tuple(layer_ids),
        drafter_arch="qwen3",
        drafter_hidden_act="silu",
        block_size=8,
        gguf_path=gguf_path,
        hf_path=hf_path,
    )


def qwen3_4b(*, gguf_path: Optional[str] = None, hf_path: Optional[str] = None) -> BaseVerifier:
    return qwen3(
        name="qwen3-4b",
        hidden_size=2560,
        num_hidden_layers=36,
        layer_ids=(2, 8, 16, 24, 32, 35),
        gguf_path=gguf_path,
        hf_path=hf_path,
    )


def qwen3_14b(*, gguf_path: Optional[str] = None, hf_path: Optional[str] = None) -> BaseVerifier:
    return qwen3(
        name="qwen3-14b",
        hidden_size=5120,
        num_hidden_layers=48,
        layer_ids=(2, 12, 22, 32, 42, 47),
        gguf_path=gguf_path,
        hf_path=hf_path,
    )
