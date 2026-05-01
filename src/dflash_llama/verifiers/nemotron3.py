"""NVIDIA Nemotron-3 family verifier configs.

Config taken from ``nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16``
(model_type=nemotron_h, arch=NemotronHForCausalLM): 88 layers, hidden=4096,
vocab=131072, ``pad_token_id=0``. The ``mask_token_id`` defaults to the
PAD id (0); override via kwargs if your training run uses a different mask.

Layer taps default to ``(3, 22, 44, 66, 86, 87)`` — early, ~25%, ~50%, ~75%,
~98%, final-residual. Override via ``layer_ids=`` for a different schedule.

⚠️ **Hybrid architecture caveat.** Nemotron-3 is hybrid Mamba+MLP+Attention
(see ``hybrid_override_pattern`` in the HF config — a per-layer string of
``M`` / ``E`` / ``*`` markers). The 88 "layers" are not all transformer
blocks. DFlash speculative decoding has historically been validated on
pure-transformer verifiers, so taps onto Mamba states may behave differently
from the MiniMax/Kimi/DSV4 case. Treat this factory as a starting point and
sanity-check your loss curve before assuming the canonical schedule works.

Note also that the underlying ``mlp_hidden_act`` for Nemotron-3 is ``relu2``,
not ``silu``. The ``drafter_hidden_act`` here refers to the drafter (Qwen3),
not the verifier — leave it ``silu`` unless you know what you're doing.

A second factory ``nemotron3_nano_30b_a3b`` is provided for the smaller
30B-A3B variant (52 layers, hidden=2688, same vocab).
"""
from __future__ import annotations
from typing import Optional, Sequence
from .base import BaseVerifier

# --- Nemotron-3 Super 120B-A12B (BF16 reference) ---
_N3S_HIDDEN_SIZE = 4096
_N3S_VOCAB_SIZE = 131072
_N3S_MASK_TOKEN_ID = 0  # PAD
_N3S_NUM_HIDDEN_LAYERS = 88
_N3S_LAYER_IDS = (3, 22, 44, 66, 86, 87)


def nemotron3_super_120b(
    *,
    name: str = "nemotron3-super-120b",
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
    """Nemotron-3-Super-120B-A12B verifier (88-layer, 131072 vocab, hidden=4096).

    Pass ``layer_ids=...`` to use a different tap schedule.
    """
    return BaseVerifier(
        name=name,
        family="nemotron_h",
        hidden_size=_N3S_HIDDEN_SIZE if hidden_size is None else int(hidden_size),
        vocab_size=_N3S_VOCAB_SIZE if vocab_size is None else int(vocab_size),
        mask_token_id=_N3S_MASK_TOKEN_ID if mask_token_id is None else int(mask_token_id),
        num_hidden_layers=(_N3S_NUM_HIDDEN_LAYERS if num_hidden_layers is None
                           else int(num_hidden_layers)),
        layer_ids=tuple(_N3S_LAYER_IDS if layer_ids is None else layer_ids),
        drafter_arch=drafter_arch,
        drafter_hidden_act=drafter_hidden_act,
        block_size=block_size,
        gguf_path=gguf_path,
        hf_path=hf_path,
    )


# --- Nemotron-3 Nano 30B-A3B (smaller MoE sibling) ---
# Config from nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
_N3N30_HIDDEN_SIZE = 2688
_N3N30_VOCAB_SIZE = 131072
_N3N30_MASK_TOKEN_ID = 0
_N3N30_NUM_HIDDEN_LAYERS = 52
# Spread 6 taps across 52 layers: ~5%, ~25%, ~50%, ~75%, ~95%, final
_N3N30_LAYER_IDS = (2, 13, 26, 39, 50, 51)


def nemotron3_nano_30b_a3b(
    *,
    name: str = "nemotron3-nano-30b-a3b",
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
    """Nemotron-3-Nano-30B-A3B verifier (62-layer MoE, 131072 vocab, hidden=3072)."""
    return BaseVerifier(
        name=name,
        family="nemotron_h",
        hidden_size=_N3N30_HIDDEN_SIZE if hidden_size is None else int(hidden_size),
        vocab_size=_N3N30_VOCAB_SIZE if vocab_size is None else int(vocab_size),
        mask_token_id=_N3N30_MASK_TOKEN_ID if mask_token_id is None else int(mask_token_id),
        num_hidden_layers=(_N3N30_NUM_HIDDEN_LAYERS if num_hidden_layers is None
                           else int(num_hidden_layers)),
        layer_ids=tuple(_N3N30_LAYER_IDS if layer_ids is None else layer_ids),
        drafter_arch=drafter_arch,
        drafter_hidden_act=drafter_hidden_act,
        block_size=block_size,
        gguf_path=gguf_path,
        hf_path=hf_path,
    )
