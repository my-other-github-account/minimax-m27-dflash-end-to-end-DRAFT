"""Auto-detect a verifier from an HF config.json or GGUF file's metadata.

Only validated families autodetect to their named factory. Unknown
``model_type`` values fall back to the ``generic`` adapter with shape
parameters read from ``config.json`` — this is best-effort: callers
should verify that ``layer_ids`` (auto-spread by the generic factory) is
sensible for their model and pass an override otherwise.

Experimental factories (``kimi_k25``, ``qwen3``, ``deepseek_v4_*``,
``nemotron3_*``) are NOT autodetected. To use them by name, opt in
explicitly via :func:`dflash_llama.register_verifier`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .base import BaseVerifier
from .minimax_m2 import minimax_m27, minimax_m27_iq4_xs


def _read_hf_config(hf_path: str) -> Optional[dict]:
    p = Path(hf_path) / "config.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def autodetect_verifier(
    *,
    hf_path: Optional[str] = None,
    gguf_path: Optional[str] = None,
) -> BaseVerifier:
    """Detect the verifier family from on-disk metadata.

    Currently only the HF ``config.json`` path is supported; GGUF metadata
    detection requires ``gguf_reader`` and is left as a TODO.

    Validated families autodetect to their named factory:

    - ``minimax_m2`` → :func:`minimax_m27`
    - ``minimax_m2`` w/ IQ4_XS GGUF → :func:`minimax_m27_iq4_xs`

    Anything else falls back to :func:`generic_verifier` with shape
    parameters from ``config.json``.
    """
    cfg = _read_hf_config(hf_path) if hf_path else None
    if cfg is None and gguf_path is None:
        raise ValueError("autodetect_verifier needs hf_path or gguf_path")
    if cfg is not None:
        mt = (cfg.get("model_type") or "").lower()
        if mt == "minimax_m2":
            # Pick IQ4_XS variant if the GGUF path mentions it; otherwise
            # the FP8/general factory.
            if gguf_path and "iq4" in gguf_path.lower():
                return minimax_m27_iq4_xs(hf_path=hf_path, gguf_path=gguf_path)
            return minimax_m27(hf_path=hf_path, gguf_path=gguf_path)

        # Unknown / experimental family — fall back to generic adapter
        # using shape from config.json. This is best-effort: callers
        # should verify ``layer_ids`` (auto-spread by the generic factory)
        # is sensible for their model.
        from .generic import generic_verifier
        try:
            hidden_size = int(cfg["hidden_size"])
            num_hidden_layers = int(cfg["num_hidden_layers"])
            vocab_size = int(cfg.get("vocab_size", 0))
            mask_token_id = int(
                cfg.get("mask_token_id")
                or cfg.get("pad_token_id")
                or cfg.get("eos_token_id")
                or 0
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"autodetect_verifier: model_type={mt!r} is not a "
                f"validated DFlash family and config.json is missing the "
                f"shape fields needed to build a generic verifier. Pass "
                f"an explicit load_verifier(name='generic', ...) call "
                f"instead, or opt into an experimental factory: see "
                f"dflash_llama.verifiers.experimental."
            )
        return generic_verifier(
            name=f"autodetected-{mt or 'unknown'}",
            family=mt or "unknown",
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            hf_path=hf_path,
            gguf_path=gguf_path,
        )
    raise ValueError(
        f"could not autodetect verifier from hf_path={hf_path} "
        f"gguf_path={gguf_path}; use load_verifier(name=...) with an "
        "explicit family name, or load_verifier(name='generic', "
        "hidden_size=..., num_hidden_layers=...)"
    )
