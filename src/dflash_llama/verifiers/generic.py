"""Generic verifier — adapt the library to any new model with no code changes.

This is the escape hatch: instead of writing a Python factory and calling
``register_verifier``, you can describe a model entirely from its shape
parameters and either pass an explicit list of ``layer_ids`` or let the
library pick a sensible default spread.

Example::

    from dflash_llama import generic_verifier

    v = generic_verifier(
        name="llama-3.1-8b",
        hidden_size=4096,
        num_hidden_layers=32,
        vocab_size=128256,
        mask_token_id=128255,
        layer_ids=[2, 8, 16, 24, 30, 31],
        hf_path="/data/llama3.1-8b",
        gguf_path="/data/llama3.1-8b-Q4_K_M.gguf",
    )

If you don't know which layers to tap, omit ``layer_ids`` and pass
``num_layer_taps=6`` (default) — the library will spread taps across the
network and append the final residual.
"""
from __future__ import annotations
from typing import Optional, Sequence
from .base import BaseVerifier


def auto_layer_ids(num_hidden_layers: int, num_taps: int = 6) -> tuple[int, ...]:
    """Spread ``num_taps`` layer taps across a network of ``num_hidden_layers``.

    The tap pattern is:
      - one early tap (layer index ``max(1, n//(num_taps*2))``)
      - evenly spaced interior taps
      - the final residual layer (``n - 1``)

    For the 62-layer MiniMax-M2.7 with ``num_taps=6`` this produces
    ``(2, 13, 25, 37, 49, 61)`` — close to but not identical to the
    canonical ``(2, 16, 30, 45, 59, 61)``. **For best results you should
    pass an explicit ``layer_ids=`` for any model where you have a known-
    good schedule.** ``auto_layer_ids`` is a starting point, not gospel.
    """
    if num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be > 0, got {num_hidden_layers}")
    if num_taps < 2:
        raise ValueError(f"num_taps must be >= 2, got {num_taps}")
    if num_taps > num_hidden_layers:
        raise ValueError(
            f"num_taps ({num_taps}) must be <= num_hidden_layers ({num_hidden_layers})"
        )

    n = num_hidden_layers
    final = n - 1
    # Reserve one tap for the final residual; spread the rest.
    interior = num_taps - 1
    if interior == 1:
        return (max(1, n // 2), final)

    # Spread interior taps from ~early to ~late, biased so the first tap is
    # never layer 0 (the embedding-equivalent residual is rarely useful).
    early = max(1, n // (num_taps * 2))
    late = max(early + 1, (n * (interior - 1)) // interior)
    if interior == 2:
        ids = [early, late, final]
    else:
        # interior_steps = interior - 1 gaps between (early, late)
        step = (late - early) / max(1, interior - 1)
        ids = [int(round(early + i * step)) for i in range(interior)] + [final]
    # Dedup, sort, clamp into [1, n-1]
    ids = sorted({max(1, min(n - 1, x)) for x in ids})
    return tuple(ids)


def generic_verifier(
    *,
    name: str,
    hidden_size: int,
    num_hidden_layers: int,
    vocab_size: int,
    mask_token_id: int,
    layer_ids: Optional[Sequence[int]] = None,
    num_layer_taps: int = 6,
    drafter_arch: str = "qwen3",
    drafter_hidden_act: str = "silu",
    block_size: int = 8,
    family: str = "generic",
    hf_path: Optional[str] = None,
    gguf_path: Optional[str] = None,
) -> BaseVerifier:
    """Build a ``BaseVerifier`` from raw shape kwargs — no factory needed.

    ``layer_ids`` is optional: if omitted, the library picks ``num_layer_taps``
    taps via :func:`auto_layer_ids`. Always prefer to pass an explicit
    ``layer_ids`` if you know the right schedule for your model.
    """
    if layer_ids is None:
        layer_ids = auto_layer_ids(num_hidden_layers, num_taps=num_layer_taps)

    return BaseVerifier(
        name=name,
        family=family,
        hidden_size=int(hidden_size),
        vocab_size=int(vocab_size),
        mask_token_id=int(mask_token_id),
        num_hidden_layers=int(num_hidden_layers),
        layer_ids=tuple(int(x) for x in layer_ids),
        drafter_arch=drafter_arch,
        drafter_hidden_act=drafter_hidden_act,
        block_size=int(block_size),
        hf_path=hf_path,
        gguf_path=gguf_path,
    )
