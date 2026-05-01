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
    raise ValueError(
        f"could not autodetect verifier from hf_path={hf_path} gguf_path={gguf_path}; "
        "use load_verifier(name=...) with an explicit family name"
    )
