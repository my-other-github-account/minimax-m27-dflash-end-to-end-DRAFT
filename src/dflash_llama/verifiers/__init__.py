"""Verifier configs and the ``load_verifier`` registry."""
from __future__ import annotations
from typing import Optional

from .base import BaseVerifier
from .minimax_m2 import minimax_m27, minimax_m27_iq4_xs
from .kimi_k25 import kimi_k25
from .qwen3 import qwen3, qwen3_4b, qwen3_14b
from .auto import autodetect_verifier

# Named-string registry. ``load_verifier("minimax-m2.7-iq4-xs", gguf_path=...)``.
_REGISTRY = {
    "minimax-m2.7": minimax_m27,
    "minimax-m2.7-iq4-xs": minimax_m27_iq4_xs,
    "kimi-k2.5": kimi_k25,
    "qwen3-4b": qwen3_4b,
    "qwen3-14b": qwen3_14b,
}


def list_verifiers() -> list[str]:
    """Return the list of registered verifier names."""
    return sorted(_REGISTRY.keys())


def register_verifier(name: str, factory) -> None:
    """Register a custom verifier factory under ``name``.

    The factory is any callable matching the signature of the built-in
    factories (e.g. ``minimax_m27``): it takes ``hf_path``, ``gguf_path``,
    and arbitrary kwargs, and returns a ``BaseVerifier`` instance.

    Use this from a downstream package or notebook to register a new
    verifier without modifying the library:

        from dflash_llama import register_verifier, BaseVerifier

        def my_model_8b(*, hf_path=None, gguf_path=None, **kw):
            return BaseVerifier(
                name="my-model-8b",
                hidden_size=4096, num_hidden_layers=32,
                vocab_size=131072, mask_token_id=131071,
                layer_ids=[2, 8, 16, 24, 30, 31],
                hf_path=hf_path, gguf_path=gguf_path, **kw,
            )
        register_verifier("my-model-8b", my_model_8b)
    """
    _REGISTRY[name.lower()] = factory


def load_verifier(
    name: Optional[str] = None,
    *,
    hf_path: Optional[str] = None,
    gguf_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    gguf_repo: Optional[str] = None,
    gguf_quant: Optional[str] = None,
    revision: Optional[str] = None,
    **overrides,
) -> BaseVerifier:
    """Load a verifier by name, with optional Hub-slug auto-resolution.

    Three ways to point at the underlying model:

    1. **Local paths** — pass ``hf_path=/path/to/dir`` and/or
       ``gguf_path=/path/to/file.gguf``. Useful when you already have files
       on disk (e.g. on a Spark cluster).

    2. **Hub slugs** — pass ``hf_repo="MiniMaxAI/MiniMax-M2"`` and/or
       ``gguf_repo="unsloth/MiniMax-M2-GGUF"`` (with optional ``gguf_quant``
       to pick the quant subdir). The library downloads to its cache and
       returns the local path. Re-runs are no-ops.

    3. **Auto-detect** — pass only ``name=None`` plus ``hf_path=`` (or
       ``hf_repo=``) and we'll inspect ``config.json`` to pick a registry
       entry.

    Examples::

        # Local files (e.g. on a cluster where weights are pre-staged)
        v = load_verifier(
            "minimax-m2.7-iq4-xs",
            gguf_path="/data/models/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf",
            hf_path="/data/models/MiniMax-M2.7-FP8",
        )

        # Hub slugs — auto-download
        v = load_verifier(
            "minimax-m2.7-iq4-xs",
            hf_repo="MiniMaxAI/MiniMax-M2",
            gguf_repo="unsloth/MiniMax-M2-GGUF",
            gguf_quant="UD-IQ4_XS",
        )

        # Hub slug for the small files only — bring your own GGUF
        v = load_verifier(
            "minimax-m2.7-iq4-xs",
            hf_repo="MiniMaxAI/MiniMax-M2",
            gguf_path="./models/UD-IQ4_XS/shard-00001.gguf",
        )
    """
    # Resolve Hub slugs to local paths (cached). Only download what's needed.
    if hf_repo and not hf_path:
        from ..hub import resolve_hf_repo
        hf_path = resolve_hf_repo(hf_repo, revision=revision)
    if gguf_repo and not gguf_path:
        from ..hub import resolve_gguf_repo
        gguf_path = resolve_gguf_repo(gguf_repo, quant=gguf_quant, revision=revision)

    if name is None:
        return autodetect_verifier(hf_path=hf_path, gguf_path=gguf_path)
    name = name.lower()
    if name not in _REGISTRY:
        # Allow generic qwen3 with explicit shape overrides
        if name == "qwen3":
            return qwen3(hf_path=hf_path, gguf_path=gguf_path, **overrides)
        raise KeyError(
            f"unknown verifier {name!r}; known = {list_verifiers()} "
            f"(or use name='qwen3' with explicit hidden_size/num_hidden_layers)"
        )
    return _REGISTRY[name](hf_path=hf_path, gguf_path=gguf_path, **overrides)


__all__ = [
    "BaseVerifier",
    "load_verifier",
    "list_verifiers",
    "register_verifier",
    "autodetect_verifier",
    "minimax_m27",
    "minimax_m27_iq4_xs",
    "kimi_k25",
    "qwen3",
    "qwen3_4b",
    "qwen3_14b",
]
