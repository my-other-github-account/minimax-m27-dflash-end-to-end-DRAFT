"""llama.cpp GGUF backend — wraps the ``llama-dump-hiddens`` binary.

The binary reads a tokens.bin file (uint32 count + int32 tokens) from
``$TOKENS_BIN``, runs the model, and writes a hidden-states blob to
``$OUT_BIN`` with this layout::

    int32 n_layers
    int32 n_tokens
    int32 n_embd
    int32[n_layers]  capture_layer_ids
    int32 n_tokens_in
    int32[n_tokens_in] token_ids
    float32[n_layers, n_tokens, n_embd]   (row-major)

We never trust ``$OUT_BIN`` to round-trip ``input_ids`` byte-for-byte —
the binary may auto-add a BOS token. We compare and warn but defer the
final pairing decision to the caller.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .base import BaseBackend


class LlamaCppGGUFBackend(BaseBackend):
    """Subprocess backend that wraps llama-dump-hiddens."""

    name = "llamacpp_gguf"

    def __init__(
        self,
        *,
        gguf_path: str,
        binary: str = "llama-dump-hiddens",
        ctx: int = 4096,
        timeout: int = 600,
        ngl: int = 99,
        extra_args: Sequence[str] = ("-ot", "exps=CPU"),
    ):
        self.gguf_path = str(gguf_path)
        self.binary = str(binary)
        self.ctx = int(ctx)
        self.timeout = int(timeout)
        self.ngl = int(ngl)
        self.extra_args = tuple(extra_args)

    # ----- helpers -----
    @staticmethod
    def _write_tokens_bin(tokens: Sequence[int], path: str) -> None:
        n = len(tokens)
        with open(path, "wb") as f:
            f.write(struct.pack("<I", n))
            f.write(np.asarray(tokens, dtype=np.int32).tobytes())
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _parse_hidden_bin(path: str) -> tuple[np.ndarray, list[int], list[int]]:
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

    # ----- backend API -----
    def run_one(
        self,
        input_ids: Sequence[int],
        *,
        layer_ids: Sequence[int],
        max_seq_len: int,
    ) -> tuple[torch.Tensor, list[int]]:
        if len(input_ids) > max_seq_len:
            raise ValueError(
                f"input_ids length {len(input_ids)} > max_seq_len {max_seq_len}"
            )
        if len(input_ids) > self.ctx - 16:
            raise ValueError(
                f"input_ids length {len(input_ids)} > ctx-16 ({self.ctx - 16})"
            )

        td = Path(tempfile.mkdtemp(prefix="dflash_llama_gguf_"))
        toks_bin = str(td / "tokens.bin")
        out_bin = str(td / "hidden.bin")
        self._write_tokens_bin(input_ids, toks_bin)

        env = os.environ.copy()
        env["TOKENS_BIN"] = toks_bin
        env["OUT_BIN"] = out_bin
        env["CAPTURE_LAYERS"] = ",".join(str(L) for L in layer_ids)
        env.setdefault("LLAMA_LOG_LEVEL", "2")

        cmd = [
            self.binary, "-m", self.gguf_path,
            "-ngl", str(self.ngl),
            *self.extra_args,
            "-c", str(self.ctx),
            "-p", "x",
        ]
        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, timeout=self.timeout
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"llama-dump-hiddens timed out after {self.timeout}s") from e
        finally:
            pass  # don't wipe yet — parser may still need out_bin

        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", errors="replace")[-1500:]
            self._cleanup(td)
            raise RuntimeError(
                f"llama-dump-hiddens rc={proc.returncode}: {tail}"
            )
        if not os.path.exists(out_bin):
            self._cleanup(td)
            raise RuntimeError("llama-dump-hiddens produced no output file")

        try:
            hs_f32, tok_out, cap_out = self._parse_hidden_bin(out_bin)
        finally:
            self._cleanup(td)

        if list(cap_out) != list(layer_ids):
            raise RuntimeError(
                f"layer-id mismatch from binary: requested {list(layer_ids)} got {cap_out}"
            )
        if list(tok_out) != list(input_ids):
            print(
                f"[llamacpp_gguf] WARN token_ids round-trip mismatch "
                f"(in={len(input_ids)} out={len(tok_out)}); using binary output",
                file=sys.stderr,
            )

        return torch.from_numpy(hs_f32), list(tok_out)

    @staticmethod
    def _cleanup(td: Path) -> None:
        for p in td.iterdir():
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        try:
            td.rmdir()
        except OSError:
            pass
