"""HF Hub integration — resolve repo slugs into local paths automatically.

Users should not have to think about disk layout. They tell us a Hub slug
(e.g. ``"unsloth/MiniMax-M2-GGUF"``); we cache it locally and return the path.

We deliberately do NOT take a torch / transformers dependency just for this —
``huggingface_hub`` is already a transitive dep (datasets needs it) and ships
with its own ``hf_hub_download`` and ``snapshot_download`` helpers that are
exactly what we want.

Cache root resolution order:
  1. ``$DFLASH_LLAMA_HOME``                       (explicit override)
  2. ``$XDG_CACHE_HOME/dflash-llama``             (XDG default)
  3. ``~/.cache/dflash-llama``                    (final fallback)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def cache_root() -> Path:
    """Return the directory where downloaded model assets live."""
    explicit = os.environ.get("DFLASH_LLAMA_HOME")
    if explicit:
        return Path(explicit).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve() / "dflash-llama"
    return Path("~/.cache/dflash-llama").expanduser().resolve()


def resolve_hf_repo(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    allow_patterns: Optional[Iterable[str]] = None,
    ignore_patterns: Optional[Iterable[str]] = None,
) -> str:
    """Download (or reuse cached) snapshot of an HF model repo. Returns a path.

    By default we DO NOT download model weights (``*.safetensors``,
    ``*.bin``, ``*.gguf``) — only the small config / tokenizer files needed by
    the trainer. Override ``allow_patterns`` / ``ignore_patterns`` to change.

    Example::

        path = resolve_hf_repo("MiniMaxAI/MiniMax-M2", revision="main")
        # -> /home/user/.cache/dflash-llama/MiniMaxAI__MiniMax-M2/main/
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub not installed. `pip install huggingface_hub`. "
            "(It is already a transitive dependency of `datasets`.)"
        ) from e

    if ignore_patterns is None:
        # Default: skip large weight shards. Trainer only needs config /
        # tokenizer / chat_template / index.json. Users who want the full
        # model should override.
        ignore_patterns = [
            "*.safetensors", "*.bin", "*.pth", "*.pt",
            "*.gguf", "*.onnx", "*.tflite",
            "consolidated.*", "*.msgpack", "*.h5",
        ]

    target_dir = cache_root() / repo_id.replace("/", "__") / (revision or "main")
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
    )
    return str(target_dir)


def resolve_gguf_repo(
    repo_id: str,
    *,
    quant: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Download all GGUF shards of a quantization. Returns the path of the FIRST shard.

    GGUFs are typically distributed in repos like ``unsloth/MiniMax-M2-GGUF``
    with multiple quants (``UD-IQ4_XS``, ``Q4_K_M``, etc.) inside. Pass
    ``quant=`` to pick one. We download the entire shard set and return the
    path of the lowest-numbered shard (which is what llama.cpp expects).

    Example::

        gguf_path = resolve_gguf_repo("unsloth/MiniMax-M2-GGUF", quant="UD-IQ4_XS")
        # -> /home/user/.cache/dflash-llama/unsloth__MiniMax-M2-GGUF/main/UD-IQ4_XS/MiniMax-M2-UD-IQ4_XS-00001-of-00004.gguf
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub not installed. `pip install huggingface_hub`."
        ) from e

    target_dir = cache_root() / repo_id.replace("/", "__") / (revision or "main")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Always restrict to the requested quant subdir (saves bandwidth)
    if quant:
        allow = [f"{quant}/*", f"*{quant}*.gguf"]
    else:
        allow = ["*.gguf"]

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        allow_patterns=allow,
    )

    # Find the first shard (alphabetic sort puts -00001-of-NNNNN first)
    candidates = sorted(target_dir.rglob("*.gguf"))
    if not candidates:
        raise FileNotFoundError(
            f"no .gguf files matched in {repo_id} (quant={quant!r}). "
            f"Try a different quant or check the repo layout."
        )
    if quant:
        candidates = [c for c in candidates if quant in str(c)] or candidates
    return str(candidates[0])


__all__ = ["cache_root", "resolve_hf_repo", "resolve_gguf_repo"]
