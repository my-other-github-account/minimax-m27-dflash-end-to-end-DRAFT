"""On-disk binary format helpers for the llama-dump-hiddens-worker protocol.

The worker reads a ``tokens.bin`` file (uint32 count + int32 tokens) and writes
a hidden-states blob to a ``out_bin`` path with the layout::

    int32 n_layers
    int32 n_tokens
    int32 n_embd
    int32[n_layers]   capture_layer_ids
    int32 n_tokens_in
    int32[n_tokens_in] token_ids
    float32[n_layers, n_tokens, n_embd]   (row-major)

These helpers used to live on ``LlamaCppGGUFBackend`` as static methods. The
legacy spawn-per-prompt backend is gone; the helpers move here so the
persistent-server client/server keep working without depending on the deleted
backend module.
"""
from __future__ import annotations

import os
import struct
from typing import Sequence

import numpy as np


def write_tokens_bin(tokens: Sequence[int], path: str) -> None:
    """Write a ``tokens.bin`` file consumable by ``llama-dump-hiddens-worker``."""
    n = len(tokens)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(np.asarray(tokens, dtype=np.int32).tobytes())
        f.flush()
        os.fsync(f.fileno())


def parse_hidden_bin(path: str) -> tuple[np.ndarray, list[int], list[int]]:
    """Parse an ``out_bin`` written by ``llama-dump-hiddens-worker``.

    Returns ``(hidden_states, token_ids, capture_layer_ids)`` where
    ``hidden_states`` is shaped ``(n_tokens, n_layers, n_embd)`` in float32.
    """
    raw = open(path, "rb").read()
    off = 0
    n_layers, n_tokens, n_embd = struct.unpack_from("<iii", raw, off)
    off += 12
    capture_layers = list(struct.unpack_from(f"<{n_layers}i", raw, off))
    off += 4 * n_layers
    n_toks_in = struct.unpack_from("<i", raw, off)[0]
    off += 4
    token_ids = list(struct.unpack_from(f"<{n_toks_in}i", raw, off))
    off += 4 * n_toks_in
    body = raw[off:]
    expected = n_layers * n_tokens * n_embd * 4
    if len(body) != expected:
        raise ValueError(
            f"body size {len(body)} != expected {expected} "
            f"(n_layers={n_layers} n_tokens={n_tokens} n_embd={n_embd})"
        )
    arr = np.frombuffer(body, dtype=np.float32).reshape(n_layers, n_tokens, n_embd)
    # → (n_tokens, n_layers, n_embd)
    arr = np.transpose(arr, (1, 0, 2)).copy()
    return arr, token_ids, capture_layers


__all__ = ["write_tokens_bin", "parse_hidden_bin"]
