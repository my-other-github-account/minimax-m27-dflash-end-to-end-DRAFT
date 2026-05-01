"""Auto-detect a verifier from an HF config.json or a GGUF file's metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .base import BaseVerifier
from .minimax_m2 import minimax_m27, minimax_m27_iq4_xs
from .kimi_k25 import kimi_k25
from .qwen3 import qwen3


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
    """Try to detect the verifier family from on-disk metadata.

    Currently only the HF config.json path is supported; GGUF metadata
    detection requires gguf_reader and is left as a TODO.
    """
    cfg = _read_hf_config(hf_path) if hf_path else None
    if cfg is None and gguf_path is None:
        raise ValueError("autodetect_verifier needs hf_path or gguf_path")
    if cfg is not None:
        mt = (cfg.get("model_type") or "").lower()
        if mt == "minimax_m2":
            return minimax_m27(hf_path=hf_path, gguf_path=gguf_path)
        if "kimi" in mt or mt == "deepseek_v3":
            # K2.5 uses the deepseek_v3 model_type in HF config
            return kimi_k25(hf_path=hf_path, gguf_path=gguf_path)
        if "qwen3" in mt:
            return qwen3(
                hidden_size=int(cfg["hidden_size"]),
                num_hidden_layers=int(cfg["num_hidden_layers"]),
                vocab_size=int(cfg.get("vocab_size", 151936)),
                hf_path=hf_path,
                gguf_path=gguf_path,
            )
        # Unknown family — fall back to the generic adapter if we can read
        # enough shape from the config. This is best-effort: callers should
        # double-check ``layer_ids`` and pass an explicit override if the
        # auto-spread isn't right for their model.
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
                f"autodetect_verifier: model_type={mt!r} is not a known DFlash "
                f"family and config.json is missing the shape fields needed to "
                f"build a generic verifier. Pass an explicit "
                f"load_verifier(name='generic', ...) call instead."
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
        f"could not autodetect verifier from hf_path={hf_path} gguf_path={gguf_path}; "
        "use load_verifier(name=...) with an explicit family name, or "
        "load_verifier(name='generic', hidden_size=..., num_hidden_layers=...)"
    )
