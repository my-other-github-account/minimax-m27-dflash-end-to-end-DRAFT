"""Base backend interface.

A backend turns ``input_ids`` into a ``(seq, n_layers, hidden)`` tensor of
hidden states by running the verifier model. Each backend hides its own
runtime details (subprocess vs. in-process, GGUF vs. HF, etc.).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch


class BaseBackend(ABC):
    """Abstract verifier-execution backend."""

    name: str = "base"

    @abstractmethod
    def run_one(
        self,
        input_ids: Sequence[int],
        *,
        layer_ids: Sequence[int],
        max_seq_len: int,
    ) -> tuple[torch.Tensor, list[int]]:
        """Run the verifier on a single prompt.

        Returns ``(hidden_states, token_ids)`` where ``hidden_states`` has
        shape ``(seq, n_layers, hidden)`` (any float dtype — the format
        layer handles the cast) and ``token_ids`` is the verifier-emitted
        token id sequence (length == seq).
        """
        raise NotImplementedError
