"""SelfDescribingTraceDataset — torch Dataset over a self-describing trace dir.

This is a thin wrapper around the trace files emitted by
``dflash_llama.generation.TraceGenerator``. It returns a dict per row with
the same keys the speculators ``ArrowDataset`` would expose, but reads the
hidden states (and prompt fields) directly off the safetensor — there is
no separate prompts-arrow / hidden-states-dir split.

The library still emits a prompts arrow (via ``assemble_prompts_arrow``)
so the speculators trainer can drive its own dataloader; this dataset is
for in-process use (eval, debugging, vocab-map building).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset

from ..generation.format import load_trace


class SelfDescribingTraceDataset(Dataset):
    """Iterable Dataset over ``hs_*.safetensors`` files in a directory."""

    def __init__(
        self,
        traces_dir: str,
        *,
        glob: str = "hs_*.safetensors",
        files: Optional[Sequence[str]] = None,
    ):
        self.traces_dir = Path(traces_dir)
        if files is not None:
            self.files = [Path(f) for f in files]
        else:
            self.files = sorted(
                self.traces_dir.glob(glob),
                key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
            )
        if not self.files:
            raise FileNotFoundError(
                f"no traces matched {glob!r} under {self.traces_dir}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        d = load_trace(path)
        out = {
            "hidden_states": d["hidden_states"],          # bf16 (seq, n_layers, hidden)
            "token_ids": d["token_ids"],                  # int64 (seq,)
            "input_ids": d["input_ids"] if d["input_ids"] is not None else d["token_ids"],
            "loss_mask": d["loss_mask"] if d["loss_mask"] is not None
                            else torch.ones_like(d["token_ids"], dtype=torch.bool),
            "metadata": d["metadata"],
            "_path": str(path),
        }
        return out


__all__ = ["SelfDescribingTraceDataset"]
