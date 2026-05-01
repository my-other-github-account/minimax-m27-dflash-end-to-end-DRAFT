"""DeepSeek-V4 family verifier configs.

Config taken from ``deepseek-ai/DeepSeek-V4-Flash`` (model_type=deepseek_v4,
arch=DeepseekV4ForCausalLM): 43 layers, hidden=4096, vocab=129280, MoE with
256 routed experts and 6 per-token. The ``mask_token_id`` defaults to the
EOS id (1); override via kwargs if your training run uses a different mask.

Layer taps default to ``(2, 11, 21, 32, 41, 42)`` — early, ~25%, ~50%, ~75%,
~95%, final-residual. This is a starting point; tune via ``layer_ids=`` if
your training experiments call for a different schedule.
"""
from __future__ import annotations
from typing import Optional, Sequence
from .base import BaseVerifier

_DSV4F_HIDDEN_SIZE = 4096
_DSV4F_VOCAB_SIZE = 129280
_DSV4F_MASK_TOKEN_ID = 1  # EOS
_DSV4F_NUM_HIDDEN_LAYERS = 43
_DSV4F_LAYER_IDS = (2, 11, 21, 32, 41, 42)


def deepseek_v4_flash(
    *,
    name: str = "deepseek-v4-flash",
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
    """DeepSeek-V4-Flash verifier (43-layer MoE, 129280 vocab, hidden=4096).

    Pass ``layer_ids=...`` to use a different tap schedule. Hidden size /
    vocab / num layers are also overridable for any forked variant.
    """
    return BaseVerifier(
        name=name,
        family="deepseek_v4",
        hidden_size=_DSV4F_HIDDEN_SIZE if hidden_size is None else int(hidden_size),
        vocab_size=_DSV4F_VOCAB_SIZE if vocab_size is None else int(vocab_size),
        mask_token_id=_DSV4F_MASK_TOKEN_ID if mask_token_id is None else int(mask_token_id),
        num_hidden_layers=(_DSV4F_NUM_HIDDEN_LAYERS if num_hidden_layers is None
                           else int(num_hidden_layers)),
        layer_ids=tuple(_DSV4F_LAYER_IDS if layer_ids is None else layer_ids),
        drafter_arch=drafter_arch,
        drafter_hidden_act=drafter_hidden_act,
        block_size=block_size,
        gguf_path=gguf_path,
        hf_path=hf_path,
    )


def deepseek_v4_pro(
    *,
    name: str = "deepseek-v4-pro",
    gguf_path: Optional[str] = None,
    hf_path: Optional[str] = None,
    layer_ids: Optional[Sequence[int]] = None,
    **kw,
) -> BaseVerifier:
    """DeepSeek-V4-Pro verifier — same factory as V4-Flash with a different name.

    DeepSeek published Pro and Flash with the same hidden_size/vocab/layer
    geometry; only the routing/expert config differs (which doesn't affect
    DFlash's hidden-state taps). If the Pro config diverges in a future
    release, override the relevant kwargs explicitly.
    """
    return deepseek_v4_flash(
        name=name,
        gguf_path=gguf_path,
        hf_path=hf_path,
        layer_ids=layer_ids,
        **kw,
    )
