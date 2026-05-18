"""High-level TraceGenerator API."""
from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional, Union

from ..verifiers.base import BaseVerifier
from .backends.base import BaseBackend
from .backends.llamacpp_gguf import LlamaCppGGUFBackend
from .backends.tracegen_client import TracegenClientBackend
from .format import save_trace, VALID_STORAGE


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
    if name in ("llamacpp_gguf", "gguf"):
        if not verifier.gguf_path:
            raise ValueError("backend=llamacpp_gguf requires verifier.gguf_path to be set")
        return LlamaCppGGUFBackend(gguf_path=verifier.gguf_path, **kwargs)
    if name in ("tracegen_client", "trace_client", "trace_server"):
        return TracegenClientBackend(
            gguf_path=verifier.gguf_path,
            layer_ids=verifier.layer_ids,
            **kwargs,
        )
    raise ValueError(
        f"unknown backend {name!r}; supported: 'llamacpp_gguf', 'tracegen_client'"
    )


class TraceGenerator:
    """High-level API for generating self-describing traces.

    Example::

        gen = TraceGenerator(
            verifier=load_verifier("minimax-m2.7-iq4-xs", gguf_path=GGUF),
            storage="fp8_per_tensor_scale",
            backend="llamacpp_gguf",
        )
        gen.generate(
            prompts="/path/to/prompts_arrow",
            output_dir="/path/to/out",
            rows=range(0, 1000),
        )
    """

    def __init__(
        self,
        *,
        verifier: BaseVerifier,
        storage: str = "fp8_per_tensor_scale",
        backend: Union[str, BaseBackend] = "llamacpp_gguf",
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

    # ----- single-row API (handy for tests / library users) -----
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
        """Run the backend on one prompt and atomically save a trace file."""
        import torch

        hs, token_ids = self.backend.run_one(
            input_ids,
            layer_ids=self.verifier.layer_ids,
            max_seq_len=max_seq_len,
        )
        seq_len = len(token_ids)
        token_ids_t = torch.tensor(token_ids, dtype=torch.int64)
        input_ids_t = torch.tensor(list(input_ids), dtype=torch.int64)
        # Pad input_ids to seq_len if the backend appended a BOS or similar
        if input_ids_t.shape[0] != seq_len:
            if input_ids_t.shape[0] < seq_len:
                pad_len = seq_len - input_ids_t.shape[0]
                input_ids_t = torch.cat(
                    [token_ids_t[:pad_len].clone(), input_ids_t], dim=0
                )
            else:
                input_ids_t = input_ids_t[-seq_len:].clone()
        if loss_mask is None:
            # Default: train on every position (the trainer narrows this to anchors)
            loss_mask_t = torch.ones(seq_len, dtype=torch.bool)
        else:
            loss_mask_t = torch.as_tensor(loss_mask, dtype=torch.bool)
            if loss_mask_t.shape[0] != seq_len:
                # truncate or pad with False to match
                if loss_mask_t.shape[0] > seq_len:
                    loss_mask_t = loss_mask_t[:seq_len]
                else:
                    pad = torch.zeros(seq_len - loss_mask_t.shape[0], dtype=torch.bool)
                    loss_mask_t = torch.cat([loss_mask_t, pad], dim=0)

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

    # ----- batched / resumable -----
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
    ) -> dict:
        """Walk an HF prompts dataset and write one trace per row.

        Each output file is ``output_dir / hs_<row>.safetensors`` and is
        self-describing (see ``format.save_trace``). Resumable: re-running
        skips rows whose output file already exists.
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

        # SIGTERM/SIGINT: flush state and exit on the next loop boundary.
        stop = {"flag": False}

        def _sigterm(signum, frame):
            print(f"[gen] received signal {signum}; finishing current row", flush=True)
            stop["flag"] = True

        old_term = signal.signal(signal.SIGTERM, _sigterm)
        old_int = signal.signal(signal.SIGINT, _sigterm)

        completed = skipped = failed = 0
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
                    state.update(lambda d: {**d, "failed_count": d.get("failed_count", 0) + 1, "last_failed_idx": int(i)})
                    continue
                loss_mask = row.get("loss_mask", None)
                try:
                    self.generate_one(
                        input_ids=input_ids,
                        output_path=out_path,
                        source_name=src_name,
                        source_row_idx=int(i),
                        max_seq_len=max_seq_len,
                        loss_mask=loss_mask,
                    )
                except Exception as e:
                    print(f"[gen] row {i} failed: {e}", file=sys.stderr, flush=True)
                    failed += 1
                    state.update(lambda d: {**d, "failed_count": d.get("failed_count", 0) + 1, "last_failed_idx": int(i)})
                    continue
                completed += 1
                state.update(lambda d: {
                    **d,
                    "completed_count": d.get("completed_count", 0) + 1,
                    "last_completed_idx": int(i),
                    "last_completed_at": time.time(),
                })
                if completed % log_every == 0:
                    print(
                        f"[gen] completed={completed} skipped={skipped} failed={failed}",
                        flush=True,
                    )
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)

        summary = {
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "output_dir": str(out),
            "state_path": str(sp),
        }
        print(f"[gen] done: {summary}", flush=True)
        return summary


__all__ = ["TraceGenerator"]
