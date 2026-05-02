"""Verifier configs and the ``load_verifier`` registry.

Validated factories (end-to-end through the library's ingest path):

- ``minimax-m2.7-iq4-xs`` — MiniMax-M2.7 + Unsloth UD-IQ4_XS GGUF.
  The reference path. Validated 2026-04-30 / 2026-05-02 on spark-1.
- ``minimax-m2.7`` — same family, FP8 quant. Kept for documentation.

Plus the ``generic`` adapter (no Python factory needed — pass shape kwargs
at the call site) and ``autodetect_verifier`` (read ``config.json`` and
pick a registry entry, falling back to generic).

**Experimental** factories live in :mod:`dflash_llama.verifiers.experimental`.
Import them explicitly (``from dflash_llama.verifiers.experimental import
kimi_k25``) — they are NOT end-to-end-validated by this library.
"""
from __future__ import annotations
from typing import Optional, Sequence

from .base import BaseVerifier
from .minimax_m2 import minimax_m27, minimax_m27_iq4_xs
from .generic import generic_verifier, auto_layer_ids
from .auto import autodetect_verifier

# Named-string registry — VALIDATED factories only.
# ``load_verifier("minimax-m2.7-iq4-xs", gguf_path=...)``.
#
# Experimental factories live in :mod:`.experimental` and must be opted
# into explicitly via ``register_verifier`` if you want them by name.
_REGISTRY = {
    "minimax-m2.7": minimax_m27,
    "minimax-m2.7-iq4-xs": minimax_m27_iq4_xs,
}


def list_verifiers() -> list[str]:
    """Return the list of registered, validated verifier names.

    Use ``dflash_llama.verifiers.experimental`` to access factories that
    have working shape metadata but have NOT been end-to-end-validated
    through the library's ingest path.
    """
    return sorted(_REGISTRY.keys())


def list_experimental_verifiers() -> list[str]:
    """Return the names of factories available under
    :mod:`dflash_llama.verifiers.experimental`. Importing them does NOT
    register them — call ``register_verifier(name, factory)`` if you want
    name-based lookup.
    """
    from . import experimental
    return sorted(name for name in dir(experimental)
                  if not name.startswith("_") and callable(getattr(experimental, name)))


def register_verifier(name: str, factory) -> None:
    """Register a custom verifier factory under ``name``.

    The factory is any callable matching the signature of the built-in
    factories (e.g. ``minimax_m27``): it takes ``hf_path``, ``gguf_path``,
    and arbitrary kwargs, and returns a :class:`BaseVerifier` instance.

    Use this from a downstream package or notebook to register a new
    verifier without modifying the library, OR to opt-in to one of the
    experimental factories by name::

        from dflash_llama import register_verifier
        from dflash_llama.verifiers.experimental import kimi_k25
        register_verifier("kimi-k2.5", kimi_k25)
        v = load_verifier("kimi-k2.5", gguf_path=...)
    """
    _REGISTRY[name.lower()] = factory


# Shape kwargs that the generic / family factories accept and that
# ``load_verifier`` will forward unmodified.
_SHAPE_KWARGS = (
    "hidden_size",
    "num_hidden_layers",
    "vocab_size",
    "mask_token_id",
    "layer_ids",
    "num_layer_taps",
    "block_size",
    "drafter_arch",
    "drafter_hidden_act",
    "family",
)


def load_verifier(
    name: Optional[str] = None,
    *,
    hf_path: Optional[str] = None,
    gguf_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    gguf_repo: Optional[str] = None,
    gguf_quant: Optional[str] = None,
    revision: Optional[str] = None,
    layer_ids: Optional[Sequence[int]] = None,
    hidden_size: Optional[int] = None,
    num_hidden_layers: Optional[int] = None,
    vocab_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    num_layer_taps: Optional[int] = None,
    block_size: Optional[int] = None,
    drafter_arch: Optional[str] = None,
    drafter_hidden_act: Optional[str] = None,
    family: Optional[str] = None,
    **extra_overrides,
) -> BaseVerifier:
    """Load a verifier by name, with optional Hub-slug auto-resolution.

    Three ways to point at the underlying model:

    1. **Local paths** — pass ``hf_path=/path/to/dir`` and/or
       ``gguf_path=/path/to/file.gguf``. Useful when files are already on
       disk (e.g. on a Spark cluster).

    2. **Hub slugs** — pass ``hf_repo="MiniMaxAI/MiniMax-M2"`` and/or
       ``gguf_repo="unsloth/MiniMax-M2-GGUF"`` (with optional ``gguf_quant``
       to pick the quant subdir). The library downloads to its cache and
       returns the local path. Re-runs are no-ops.

    3. **Auto-detect** — pass only ``name=None`` plus ``hf_path=`` (or
       ``hf_repo=``) and we'll inspect ``config.json`` to pick a registry
       entry, falling back to the ``generic`` adapter.

    **Override knobs** — every shape parameter is overridable at the call
    site. Pass any of ``layer_ids``, ``hidden_size``, ``num_hidden_layers``,
    ``vocab_size``, ``mask_token_id``, ``num_layer_taps``, ``block_size``,
    ``drafter_arch``, ``drafter_hidden_act``, ``family`` to customize the
    verifier without writing a factory. ``layer_ids`` is the most common
    knob — use it to change which layers DFlash taps for hidden states.

    Use ``name="generic"`` (or omit ``name`` and pass shape kwargs
    explicitly) to build a verifier for an unfamiliar model::

        v = load_verifier(
            "generic",
            name_override="llama-3.1-8b",
            hf_path="...",
            gguf_path="...",
            hidden_size=4096,
            num_hidden_layers=32,
            vocab_size=128256,
            mask_token_id=128255,
            layer_ids=[2, 8, 16, 24, 30, 31],
        )

    Only ``minimax-m2.7-iq4-xs`` and ``minimax-m2.7`` ship registered in
    the validated namespace. To use an experimental factory by name, opt
    in explicitly::

        from dflash_llama import register_verifier
        from dflash_llama.verifiers.experimental import qwen3_4b
        register_verifier("qwen3-4b", qwen3_4b)
        v = load_verifier("qwen3-4b", hf_path="...", gguf_path="...")
    """
    # Resolve Hub slugs to local paths (cached). Only download what's needed.
    if hf_repo and not hf_path:
        from ..hub import resolve_hf_repo
        hf_path = resolve_hf_repo(hf_repo, revision=revision)
    if gguf_repo and not gguf_path:
        from ..hub import resolve_gguf_repo
        gguf_path = resolve_gguf_repo(gguf_repo, quant=gguf_quant, revision=revision)

    # Collect all shape overrides into a single dict so factories see them
    # uniformly. Drop ``None`` so factory defaults still win.
    overrides = {
        "layer_ids": layer_ids,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "vocab_size": vocab_size,
        "mask_token_id": mask_token_id,
        "num_layer_taps": num_layer_taps,
        "block_size": block_size,
        "drafter_arch": drafter_arch,
        "drafter_hidden_act": drafter_hidden_act,
        "family": family,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    overrides.update(extra_overrides)

    if name is None:
        # Autodetect, then apply shape overrides AFTER we get the base
        # verifier so users can refine an autodetected family.
        v = autodetect_verifier(hf_path=hf_path, gguf_path=gguf_path)
        return _apply_overrides(v, overrides)

    name = name.lower()

    if name == "generic":
        required = ("hidden_size", "num_hidden_layers", "vocab_size", "mask_token_id")
        missing = [k for k in required if overrides.get(k) is None]
        if missing:
            raise ValueError(
                f"load_verifier(name='generic') needs these shape kwargs: "
                f"{missing}. Use a registered name (one of {list_verifiers()}) "
                f"for known model families, or pass these kwargs to describe "
                f"your custom model."
            )
        return generic_verifier(
            name=overrides.pop("name_override", "generic-model"),
            hf_path=hf_path,
            gguf_path=gguf_path,
            **{k: v for k, v in overrides.items()
               if k in _SHAPE_KWARGS},
        )

    if name not in _REGISTRY:
        # Helpful error: list both registered AND experimental names so
        # the user knows the experimental opt-in path exists.
        try:
            exp = list_experimental_verifiers()
        except Exception:
            exp = []
        msg = (
            f"unknown verifier {name!r}; registered = {list_verifiers()}. "
            f"Use name='generic' with hidden_size/num_hidden_layers/"
            f"vocab_size/mask_token_id/layer_ids to describe a custom "
            f"model."
        )
        if exp:
            msg += (
                f"\n\nExperimental factories (NOT end-to-end-validated): "
                f"{exp}. To use one by name, opt in:\n"
                f"  from dflash_llama.verifiers.experimental import <factory>\n"
                f"  register_verifier(<name>, <factory>)"
            )
        raise KeyError(msg)

    factory = _REGISTRY[name]
    import inspect
    sig = inspect.signature(factory)
    accepted = {k for k in sig.parameters}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        kwargs = dict(overrides)
    else:
        kwargs = {k: v for k, v in overrides.items() if k in accepted}
        dropped = [k for k in overrides if k not in accepted]
        if dropped:
            import warnings
            warnings.warn(
                f"load_verifier({name!r}) factory does not accept "
                f"{dropped}; these overrides were ignored. Use name='generic' "
                f"to build a fully-custom verifier, or update the factory to "
                f"accept these kwargs.",
                RuntimeWarning,
                stacklevel=2,
            )
    return factory(hf_path=hf_path, gguf_path=gguf_path, **kwargs)


def _apply_overrides(v: BaseVerifier, overrides: dict) -> BaseVerifier:
    """Return a copy of ``v`` with shape overrides applied.

    Used after autodetect so users can refine the autodetected family.
    """
    if not overrides:
        return v
    from dataclasses import replace
    apply = {}
    for k in ("hidden_size", "num_hidden_layers", "vocab_size",
              "mask_token_id", "layer_ids", "block_size",
              "drafter_arch", "drafter_hidden_act", "family"):
        if k in overrides:
            apply[k] = (tuple(overrides[k]) if k == "layer_ids" else overrides[k])
    if "num_layer_taps" in overrides and "layer_ids" not in overrides:
        apply["layer_ids"] = auto_layer_ids(
            v.num_hidden_layers, num_taps=overrides["num_layer_taps"]
        )
    if not apply:
        return v
    return replace(v, **apply)


__all__ = [
    "BaseVerifier",
    "load_verifier",
    "list_verifiers",
    "list_experimental_verifiers",
    "register_verifier",
    "autodetect_verifier",
    "generic_verifier",
    "auto_layer_ids",
    "minimax_m27",
    "minimax_m27_iq4_xs",
]
