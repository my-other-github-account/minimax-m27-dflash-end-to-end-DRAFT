"""High-level TraceGenerator API.

The library has a single execution path: ``TracegenClientBackend`` ↔
``TraceServer`` ↔ ``llama-dump-hiddens-worker``. The persistent worker
amortizes model load and supports batched decode of multiple same-length
prompts in one ``llama_decode`` call — that's where the throughput win lives.

To get the documented ~2.5× / 60+ traces/min headline rate you MUST batch.
``TraceGenerator.generate(...)`` and ``TraceGenerator.generate_many(...)``
batch automatically. ``generate_one(...)`` is preserved for tests and tiny
demos but routes through the batched path with a single-element group, so
even it benefits from the persistent server (model is loaded once across all
calls on the same ``TraceGenerator`` instance).
"""
from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import torch

from ..verifiers.base import BaseVerifier
from .backends.base import BaseBackend
from .backends.tracegen_client import TracegenClientBackend
from .format import save_trace, VALID_STORAGE


# Default batched-window size. ``llama-dump-hiddens-worker`` is built with
# ``n_seq_max=8`` (verified by the bench: greater values overflow the
# per-seq KV budget at ctx=16384, max_seq_len=2048). Production workers
# may lower this if memory pressure rises; 8 is the sweet spot today.
DEFAULT_BATCH_WIDTH = 8


class _State:
    """Tiny atomic state.json wrapper for resumability."""

    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
            self._write({
                "version": 1,
                "started_at": time.time(),
                "completed_count": 0,
                "skipped_count": 0,
                "failed_count": 0,
                "last_completed_idx": None,
                "last_failed_idx": None,
            })
        self.data = json.loads(path.read_bytes())

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.tmp_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(json.dumps(data, indent=2, sort_keys=True).encode())
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    def update(self, mutator) -> None:
        cur = json.loads(self.path.read_bytes())
        new = mutator(cur)
        self._write(new)
        self.data = new


def _make_backend(name: str, *, verifier: BaseVerifier, **kwargs) -> BaseBackend:
    name = name.lower()
    if name in ("tracegen_client", "trace_server", "server"):
        if not verifier.gguf_path:
            raise ValueError("backend=tracegen_client requires verifier.gguf_path to be set")
        return TracegenClientBackend(
            gguf_path=verifier.gguf_path,
            layer_ids=verifier.layer_ids,
            **kwargs,
        )
    raise ValueError(
        f"unknown backend {name!r}; supported: 'tracegen_client'. "
        "The legacy 'llamacpp_gguf' spawn-per-prompt backend was removed in "
        "favor of the persistent server."
    )


def _align_input_ids(input_ids_list, token_ids_list):
    """Pad/truncate ``input_ids`` to match ``token_ids`` length.

    The worker may emit ``token_ids`` that differ from the caller's
    ``input_ids`` (e.g. it auto-prepends a BOS, or the verifier's tokenizer
    canonicalizes differently). We trust the worker's output as ground
    truth and align ``input_ids`` to it for ``save_trace``.
    """
    input_ids_t = torch.tensor(list(input_ids_list), dtype=torch.int64)
    token_ids_t = torch.tensor(token_ids_list, dtype=torch.int64)
    seq_len = token_ids_t.shape[0]
    if input_ids_t.shape[0] == seq_len:
        return input_ids_t, token_ids_t
    if input_ids_t.shape[0] < seq_len:
        pad_len = seq_len - input_ids_t.shape[0]
        input_ids_t = torch.cat([token_ids_t[:pad_len].clone(), input_ids_t], dim=0)
    else:
        input_ids_t = input_ids_t[-seq_len:].clone()
    return input_ids_t, token_ids_t


def _align_loss_mask(loss_mask, seq_len: int):
    if loss_mask is None:
        return torch.ones(seq_len, dtype=torch.bool)
    loss_mask_t = torch.as_tensor(loss_mask, dtype=torch.bool)
    if loss_mask_t.shape[0] == seq_len:
        return loss_mask_t
    if loss_mask_t.shape[0] > seq_len:
        return loss_mask_t[:seq_len]
    pad = torch.zeros(seq_len - loss_mask_t.shape[0], dtype=torch.bool)
    return torch.cat([loss_mask_t, pad], dim=0)


class TraceGenerator:
    """High-level API for generating self-describing traces.

    Example::

        gen = TraceGenerator(
            verifier=load_verifier("minimax-m2.7-iq4-xs", gguf_path=GGUF),
            storage="fp8_per_tensor_scale",
            backend_kwargs={
                "binary": "/path/to/llama-dump-hiddens-worker",
                "auto_start": True,
                "ctx": 16384,
                "ngl": 99,
                "override_tensor": "exps=CPU",
            },
        )
        gen.generate(
            prompts="/path/to/prompts_arrow",
            output_dir="/path/to/out",
            rows=range(0, 1000),
        )

    By default ``generate(...)`` walks rows in order and batches up to
    ``batch_width=8`` same-length prompts per ``run_many`` call against the
    persistent trace-server. Lower ``batch_width`` if you OOM, raise it only
    after rebuilding ``llama-dump-hiddens-worker`` with a larger
    ``n_seq_max``.
    """

    def __init__(
        self,
        *,
        verifier: BaseVerifier,
        storage: str = "fp8_per_tensor_scale",
        backend: Union[str, BaseBackend] = "tracegen_client",
        backend_kwargs: Optional[dict] = None,
    ):
        if storage not in VALID_STORAGE:
            raise ValueError(f"storage must be one of {VALID_STORAGE}")
        self.verifier = verifier
        self.storage = storage
        if isinstance(backend, BaseBackend):
            self.backend = backend
        else:
            self.backend = _make_backend(
                str(backend), verifier=verifier, **(backend_kwargs or {})
            )

    # ------------------------------------------------------------------
    # Single-row API (thin wrapper around the batched fast path)
    # ------------------------------------------------------------------
    def generate_one(
        self,
        *,
        input_ids,
        output_path: Union[str, Path],
        source_name: str,
        source_row_idx: int,
        max_seq_len: int = 2048,
        loss_mask=None,
        extra_metadata: Optional[dict] = None,
    ) -> dict:
        """Run the backend on one prompt and atomically save a trace file.

        Implemented as a single-prompt call to ``generate_many``. Prefer
        ``generate_many`` (or ``generate``) for production workloads — a
        single prompt cannot exploit the batched-decode fast path.
        """
        results = self.generate_many(
            batch_inputs=[list(input_ids)],
            output_paths=[output_path],
            source_names=[source_name],
            source_row_ids=[int(source_row_idx)],
            max_seq_len=max_seq_len,
            loss_masks=[loss_mask],
            extra_metadatas=[extra_metadata],
        )
        return results[0]

    # ------------------------------------------------------------------
    # Batched API (the fast path)
    # ------------------------------------------------------------------
    def generate_many(
        self,
        *,
        batch_inputs: Sequence[Sequence[int]],
        output_paths: Sequence[Union[str, Path]],
        source_names: Sequence[str],
        source_row_ids: Sequence[int],
        max_seq_len: int = 2048,
        loss_masks: Optional[Sequence] = None,
        extra_metadatas: Optional[Sequence[Optional[dict]]] = None,
    ) -> list[dict]:
        """Run the backend on a batch of prompts and save one trace per prompt.

        All prompts in ``batch_inputs`` MUST share the same length — the
        underlying ``run_many`` requires uniform sequence width. Callers
        that don't have same-length groups should use ``generate(...)``,
        which buckets by length internally.

        Returns a list of metadata dicts (one per saved trace) in the same
        order as ``batch_inputs``.
        """
        n = len(batch_inputs)
        if n == 0:
            return []
        if not (len(output_paths) == len(source_names) == len(source_row_ids) == n):
            raise ValueError(
                "generate_many: output_paths, source_names, source_row_ids must "
                f"have the same length as batch_inputs ({n})"
            )
        if loss_masks is not None and len(loss_masks) != n:
            raise ValueError("generate_many: loss_masks length mismatch")
        if extra_metadatas is not None and len(extra_metadatas) != n:
            raise ValueError("generate_many: extra_metadatas length mismatch")

        lengths = {len(p) for p in batch_inputs}
        if len(lengths) != 1:
            raise ValueError(
                "generate_many requires all prompts to share the same length; "
                f"got {sorted(lengths)}. Use generate(...) for mixed-length input."
            )

        # Single-prompt fast path: don't pay the batched-protocol overhead.
        if n == 1:
            hs, token_ids = self.backend.run_one(
                batch_inputs[0],
                layer_ids=self.verifier.layer_ids,
                max_seq_len=max_seq_len,
            )
            return [self._save_one(
                hs=hs, token_ids=token_ids,
                input_ids_in=batch_inputs[0],
                output_path=output_paths[0],
                source_name=source_names[0],
                source_row_idx=int(source_row_ids[0]),
                loss_mask=(loss_masks[0] if loss_masks else None),
                extra_metadata=(extra_metadatas[0] if extra_metadatas else None),
            )]

        results = self.backend.run_many(
            list(batch_inputs),
            layer_ids=self.verifier.layer_ids,
            max_seq_len=max_seq_len,
        )
        out_meta: list[dict] = []
        for i, (hs, token_ids) in enumerate(results):
            out_meta.append(self._save_one(
                hs=hs, token_ids=token_ids,
                input_ids_in=batch_inputs[i],
                output_path=output_paths[i],
                source_name=source_names[i],
                source_row_idx=int(source_row_ids[i]),
                loss_mask=(loss_masks[i] if loss_masks else None),
                extra_metadata=(extra_metadatas[i] if extra_metadatas else None),
            ))
        return out_meta

    def _save_one(
        self,
        *,
        hs: torch.Tensor,
        token_ids: list[int],
        input_ids_in,
        output_path: Union[str, Path],
        source_name: str,
        source_row_idx: int,
        loss_mask,
        extra_metadata: Optional[dict],
    ) -> dict:
        input_ids_t, token_ids_t = _align_input_ids(input_ids_in, token_ids)
        loss_mask_t = _align_loss_mask(loss_mask, token_ids_t.shape[0])
        return save_trace(
            output_path,
            hidden_states=hs,
            token_ids=token_ids_t,
            input_ids=input_ids_t,
            loss_mask=loss_mask_t,
            source_name=source_name,
            source_row_idx=source_row_idx,
            storage=self.storage,
            layer_ids=list(self.verifier.layer_ids),
            extra_metadata=extra_metadata,
        )

    # ------------------------------------------------------------------
    # Dataset walker (the main production entry point)
    # ------------------------------------------------------------------
    def generate(
        self,
        *,
        prompts: Union[str, Path],
        output_dir: Union[str, Path],
        rows: Optional[Iterable[int]] = None,
        state_path: Optional[Union[str, Path]] = None,
        max_seq_len: int = 2048,
        source_name: Optional[str] = None,
        skip_existing: bool = True,
        log_every: int = 10,
        batch_width: int = DEFAULT_BATCH_WIDTH,
        flush_after_rows: int = 256,
    ) -> dict:
        """Walk an HF prompts dataset and write one trace per row, batched.

        Each output file is ``output_dir / hs_<row>.safetensors``.
        Resumable: re-running skips rows whose output file already exists.

        Rows are read in order and accumulated into per-length buckets.
        When a bucket reaches ``batch_width`` it's fired as a single
        ``run_many`` call (the fast path). Buckets are also flushed after
        every ``flush_after_rows`` rows examined, and once at the end of
        the walk, so partial batches still get written.
        """
        from datasets import load_from_disk

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        sp = Path(state_path) if state_path else (out.parent / f"{out.name}_state.json")
        state = _State(sp)
        ds = load_from_disk(str(prompts))
        total = len(ds)
        if rows is None:
            rows = range(0, total)

        src_name = source_name or Path(prompts).name

        stop = {"flag": False}

        def _sigterm(signum, frame):
            print(f"[gen] received signal {signum}; finishing current batch", flush=True)
            stop["flag"] = True

        old_term = signal.signal(signal.SIGTERM, _sigterm)
        old_int = signal.signal(signal.SIGINT, _sigterm)

        completed = skipped = failed = 0
        # Buckets keyed by seq_len -> list of (row_idx, input_ids, loss_mask)
        buckets: dict[int, list] = {}
        rows_examined = 0

        def _flush_bucket(seq_len: int) -> None:
            nonlocal completed, failed
            group = buckets.pop(seq_len, [])
            if not group:
                return
            try:
                self.generate_many(
                    batch_inputs=[g[1] for g in group],
                    output_paths=[out / f"hs_{g[0]}.safetensors" for g in group],
                    source_names=[src_name] * len(group),
                    source_row_ids=[g[0] for g in group],
                    max_seq_len=max_seq_len,
                    loss_masks=[g[2] for g in group],
                )
            except Exception as e:
                print(f"[gen] batch (seq_len={seq_len}, n={len(group)}) failed: {e}",
                      file=sys.stderr, flush=True)
                failed += len(group)
                last_idx = group[-1][0]
                state.update(lambda d: {
                    **d,
                    "failed_count": d.get("failed_count", 0) + len(group),
                    "last_failed_idx": int(last_idx),
                })
                return
            for g in group:
                completed += 1
            last_idx = group[-1][0]
            state.update(lambda d: {
                **d,
                "completed_count": d.get("completed_count", 0) + len(group),
                "last_completed_idx": int(last_idx),
                "last_completed_at": time.time(),
            })
            if completed % log_every < len(group):
                print(f"[gen] completed={completed} skipped={skipped} failed={failed} "
                      f"(batch seq_len={seq_len} n={len(group)})", flush=True)

        def _flush_full_buckets() -> None:
            for k in [k for k, v in buckets.items() if len(v) >= batch_width]:
                _flush_bucket(k)

        def _flush_all_buckets() -> None:
            for k in list(buckets.keys()):
                _flush_bucket(k)

        try:
            for i in rows:
                if stop["flag"]:
                    break
                if i >= total:
                    break
                out_path = out / f"hs_{i}.safetensors"
                if skip_existing and out_path.exists():
                    skipped += 1
                    continue
                row = ds[int(i)]
                input_ids = list(row["input_ids"])
                if len(input_ids) > max_seq_len:
                    failed += 1
                    state.update(lambda d: {
                        **d,
                        "failed_count": d.get("failed_count", 0) + 1,
                        "last_failed_idx": int(i),
                    })
                    continue
                loss_mask = row.get("loss_mask", None)
                buckets.setdefault(len(input_ids), []).append((int(i), input_ids, loss_mask))
                rows_examined += 1
                _flush_full_buckets()
                if rows_examined % flush_after_rows == 0:
                    _flush_all_buckets()
            # Final flush
            _flush_all_buckets()
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)

        summary = {
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "output_dir": str(out),
            "state_path": str(sp),
            "batch_width": batch_width,
        }
        print(f"[gen] done: {summary}", flush=True)
        return summary


__all__ = ["TraceGenerator", "DEFAULT_BATCH_WIDTH"]
