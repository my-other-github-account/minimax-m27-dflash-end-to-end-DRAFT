"""TraceGenerator backend that talks to the persistent TraceServer."""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from ...tracegen import TraceClient
from .base import BaseBackend


class TracegenClientBackend(BaseBackend):
    """Backend that delegates hidden-state extraction to ``TraceServer``."""

    name = "tracegen_client"

    def __init__(
        self,
        *,
        socket_path: str = "unix:///tmp/dflash_tracegen.sock",
        request_timeout: float = 120.0,
        connect_timeout: float = 5.0,
        startup_timeout: float = 900.0,
        auto_start: bool = False,
        gguf_path: Optional[str] = None,
        layer_ids: Optional[Sequence[int]] = None,
        binary: str = "llama-dump-hiddens-worker",
        ctx: int = 4096,
        ngl: int = 99,
        override_tensor: Optional[str] = "exps=CPU",
        worker_args: Optional[Sequence[str]] = None,
        server_log_path: Optional[str] = None,
        restart_retries: int = 1,
    ):
        self.client = TraceClient(
            socket_path=socket_path,
            request_timeout=request_timeout,
            connect_timeout=connect_timeout,
            startup_timeout=startup_timeout,
            auto_start=auto_start,
            gguf_path=gguf_path,
            layer_ids=layer_ids,
            binary=binary,
            ctx=ctx,
            ngl=ngl,
            override_tensor=override_tensor,
            worker_args=worker_args,
            server_log_path=server_log_path,
            restart_retries=restart_retries,
        )

    def run_one(
        self,
        input_ids: Sequence[int],
        *,
        layer_ids: Sequence[int],
        max_seq_len: int,
    ) -> tuple[torch.Tensor, list[int]]:
        result = self.client.dump_hiddens(
            input_ids=input_ids,
            layer_ids=layer_ids,
            max_seq_len=max_seq_len,
        )
        return result["hidden_states"], list(result["token_ids"])

    def run_many(
        self,
        batch_inputs: Sequence[Sequence[int]],
        *,
        layer_ids: Sequence[int],
        max_seq_len: int,
    ) -> list[tuple[torch.Tensor, list[int]]]:
        results = self.client.dump_hiddens_many(
            batch_inputs=batch_inputs,
            layer_ids=layer_ids,
            max_seq_len=max_seq_len,
        )
        return [
            (result["hidden_states"], list(result["token_ids"]))
            for result in results
        ]
