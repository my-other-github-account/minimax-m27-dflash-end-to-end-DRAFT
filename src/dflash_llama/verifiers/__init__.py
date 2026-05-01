"""Verifier configs and the ``load_verifier`` registry."""
from __future__ import annotations
from typing import Optional, Sequence

from .base import BaseVerifier
from .minimax_m2 import minimax_m27, minimax_m27_iq4_xs
from .kimi_k25 import kimi_k25
from .qwen3 import qwen3, qwen3_4b, qwen3_14b
from .deepseek_v4 import deepseek_v4_flash, deepseek_v4_pro
from .nemotron3 import nemotron3_super_120b, nemotron3_nano_30b_a3b
from .generic import generic_verifier, auto_layer_ids
from .auto import autodetect_verifier

# Named-string registry. ``load_verifier("minimax-m2.7-iq4-xs", gguf_path=...)``.
_REGISTRY = {
    "minimax-m2.7": minimax_m27,
    "minimax-m2.7-iq4-xs": minimax_m27_iq4_xs,
    "kimi-k2.5": kimi_k25,
    "deepseek-v4-flash": deepseek_v4_flash,
    "deepseek-v4-pro": deepseek_v4_pro,
    "nemotron3-super-120b": nemotron3_super_120b,
    "nemotron3-nano-30b-a3b": nemotron3_nano_30b_a3b,
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
       ``gguf_path=/path/to/file.gguf``. Useful when you already have files
       on disk (e.g. on a Spark cluster).

    2. **Hub slugs** — pass ``hf_repo="MiniMaxAI/MiniMax-M2"`` and/or
       ``gguf_repo="unsloth/MiniMax-M2-GGUF"`` (with optional ``gguf_quant``
       to pick the quant subdir). The library downloads to its cache and
       returns the local path. Re-runs are no-ops.

    3. **Auto-detect** — pass only ``name=None`` plus ``hf_path=`` (or
       ``hf_repo=``) and we'll inspect ``config.json`` to pick a registry
       entry, falling back to the ``generic`` adapter.

    **Override knobs** — every shape parameter is overridable from the call
    site. Pass any of ``layer_ids``, ``hidden_size``, ``num_hidden_layers``,
    ``vocab_size``, ``mask_token_id``, ``num_layer_taps``, ``block_size``,
    ``drafter_arch``, ``drafter_hidden_act``, ``family`` to customize the
    verifier without writing a factory. ``layer_ids`` is the most common
    knob — use it to change which layers DFlash taps for hidden states.

    Use ``name="generic"`` (or omit ``name`` and pass shape kwargs explicitly)
    to build a verifier for an unfamiliar model::

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
        # Autodetect, but apply layer_ids/shape overrides AFTER we get the
        # base verifier so users can refine an autodetected family.
        v = autodetect_verifier(hf_path=hf_path, gguf_path=gguf_path)
        return _apply_overrides(v, overrides)

    name = name.lower()

    if name == "generic":
        # Required shape kwargs for the generic path
        required = ("hidden_size", "num_hidden_layers", "vocab_size", "mask_token_id")
        missing = [k for k in required if overrides.get(k) is None]
        if missing:
            raise ValueError(
                f"load_verifier(name='generic') needs these shape kwargs: {missing}. "
                f"Use a registered name (one of {list_verifiers()}) for known model "
                f"families, or pass these kwargs to describe your custom model."
            )
        return generic_verifier(
            name=overrides.pop("name_override", "generic-model"),
            hf_path=hf_path,
            gguf_path=gguf_path,
            **{k: v for k, v in overrides.items()
               if k in ("hidden_size", "num_hidden_layers", "vocab_size",
                        "mask_token_id", "layer_ids", "num_layer_taps",
                        "block_size", "drafter_arch", "drafter_hidden_act", "family")},
        )

    if name not in _REGISTRY:
        if name == "qwen3":
            # qwen3 wants explicit shape overrides positionally
            return qwen3(
                hf_path=hf_path,
                gguf_path=gguf_path,
                **{k: v for k, v in overrides.items()
                   if k in ("hidden_size", "num_hidden_layers", "vocab_size",
                            "mask_token_id", "layer_ids", "block_size",
                            "drafter_arch", "drafter_hidden_act")},
            )
        raise KeyError(
            f"unknown verifier {name!r}; known = {list_verifiers()} "
            f"(or use name='generic' with hidden_size/num_hidden_layers/"
            f"vocab_size/mask_token_id/layer_ids to describe a custom model)"
        )

    factory = _REGISTRY[name]
    # Forward only the kwargs the factory will accept. We rely on
    # ``inspect.signature`` to filter rather than risk a TypeError on factories
    # that don't take ``**kwargs``.
    import inspect
    sig = inspect.signature(factory)
    accepted = {k for k in sig.parameters}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        # Factory accepts **kwargs — pass everything
        kwargs = dict(overrides)
    else:
        kwargs = {k: v for k, v in overrides.items() if k in accepted}
        # Warn the user if they passed something the factory will silently drop
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
        # User asked us to re-spread taps after autodetect picked a family.
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
    "register_verifier",
    "autodetect_verifier",
    "generic_verifier",
    "auto_layer_ids",
    "minimax_m27",
    "minimax_m27_iq4_xs",
    "kimi_k25",
    "deepseek_v4_flash",
    "deepseek_v4_pro",
    "nemotron3_super_120b",
    "nemotron3_nano_30b_a3b",
    "qwen3",
    "qwen3_4b",
    "qwen3_14b",
]
