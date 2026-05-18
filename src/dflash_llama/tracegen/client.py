"""Unix-socket client for the persistent trace-generation server."""
from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch

from ..generation.backends.llamacpp_gguf import LlamaCppGGUFBackend
from ..generation.format import VALID_STORAGE, save_trace
from .server import SOCKET_PREFIX, _normalize_socket_path


class TraceClient:
    """Client that talks to ``TraceServer`` and mirrors ``generate_one``."""

    def __init__(
        self,
        *,
        socket: Optional[str] = None,
        socket_path: str = "unix:///tmp/dflash_tracegen.sock",
        request_timeout: float = 120.0,
        connect_timeout: float = 5.0,
        startup_timeout: float = 900.0,
        auto_start: bool = False,
        gguf_path: Optional[str] = None,
        layer_ids: Optional[Iterable[int]] = None,
        binary: str = "llama-dump-hiddens-worker",
        ctx: int = 4096,
        ngl: int = 99,
        override_tensor: Optional[str] = "exps=CPU",
        server_log_path: Optional[str | Path] = None,
        storage: str = "fp8_per_tensor_scale",
        restart_retries: int = 1,
    ):
        if storage not in VALID_STORAGE:
            raise ValueError(f"storage must be one of {VALID_STORAGE}")
        self.socket_path = _normalize_socket_path(socket or socket_path)
        self.request_timeout = float(request_timeout)
        self.connect_timeout = float(connect_timeout)
        self.startup_timeout = float(startup_timeout)
        self.auto_start = bool(auto_start)
        self.gguf_path = gguf_path
        self.layer_ids = [int(layer) for layer in layer_ids] if layer_ids is not None else None
        self.binary = str(binary)
        self.ctx = int(ctx)
        self.ngl = int(ngl)
        self.override_tensor = override_tensor
        self.server_log_path = Path(server_log_path) if server_log_path else None
        self.storage = storage
        self.restart_retries = int(restart_retries)
        self._server_proc: Optional[subprocess.Popen] = None
        self._server_log_fh = None

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout)
        sock.connect(self.socket_path)
        sock.settimeout(self.request_timeout)
        return sock

    def _request(self, payload: dict) -> dict:
        with self._connect() as conn:
            conn.sendall(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")
            data = bytearray()
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data.extend(chunk)
                if b"\n" in chunk:
                    break
            if not data:
                raise RuntimeError("server closed connection without responding")
            response = json.loads(data.split(b"\n", 1)[0].decode("utf-8"))
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "trace server request failed"))
        return response

    def _ping(self) -> bool:
        try:
            response = self._request({"op": "ping"})
        except OSError:
            return False
        except RuntimeError:
            return False
        return response.get("status") == "ready"

    def _build_server_cmd(self) -> list[str]:
        if not self.gguf_path:
            raise ValueError("auto_start=True requires gguf_path")
        cmd = [
            sys.executable,
            "-m",
            "dflash_llama.cli",
            "trace-server",
            "--gguf-path",
            self.gguf_path,
            "--socket",
            f"{SOCKET_PREFIX}{self.socket_path}",
            "--ctx",
            str(self.ctx),
            "--ngl",
            str(self.ngl),
            "--binary",
            self.binary,
        ]
        if self.layer_ids is not None:
            cmd += ["--layer-ids", ",".join(str(layer) for layer in self.layer_ids)]
        if self.override_tensor:
            cmd += ["--override-tensor", self.override_tensor]
        if self.server_log_path:
            cmd += ["--log", str(self.server_log_path)]
        return cmd

    def _spawn_server(self) -> None:
        if self._server_proc is not None and self._server_proc.poll() is None:
            return
        cmd = self._build_server_cmd()
        env = os.environ.copy()
        pythonpath = [p for p in sys.path if p]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
        if self.server_log_path:
            self.server_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._server_log_fh = open(self.server_log_path, "a", buffering=1)
            stdout = self._server_log_fh
            stderr = subprocess.STDOUT
        else:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        self._server_proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._server_proc.poll() is not None:
                raise RuntimeError(f"trace server exited during startup with rc={self._server_proc.returncode}")
            if self._ping():
                return
            time.sleep(0.5)
        raise TimeoutError(
            f"trace server did not become ready within {self.startup_timeout:.1f}s"
        )

    def _ensure_server(self) -> None:
        if self._ping():
            return
        if not self.auto_start:
            raise RuntimeError(
                f"trace server is not reachable at {self.socket_path}; set auto_start=True or start it manually"
            )
        self._spawn_server()

    def _restart_server(self) -> None:
        self.close()
        self._spawn_server()

    def close(self) -> None:
        proc = self._server_proc
        self._server_proc = None
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if self._server_log_fh:
            self._server_log_fh.close()
            self._server_log_fh = None

    def __enter__(self) -> "TraceClient":
        self._ensure_server()
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def dump_hiddens(
        self,
        *,
        input_ids: Sequence[int],
        max_seq_len: int = 2048,
        layer_ids: Optional[Sequence[int]] = None,
    ) -> dict:
        requested_layers = [int(layer) for layer in (layer_ids or self.layer_ids or [])]
        if not requested_layers:
            raise ValueError("layer_ids must be provided either on the client or per request")
        if len(input_ids) > max_seq_len:
            raise ValueError(
                f"input_ids length {len(input_ids)} > max_seq_len {max_seq_len}"
            )

        attempts = max(self.restart_retries, 0) + 1
        last_exc = None
        for attempt in range(attempts):
            td = Path(tempfile.mkdtemp(prefix="dflash_trace_client_"))
            out_bin = td / "hidden.bin"
            try:
                self._ensure_server()
                payload = {
                    "op": "dump_hiddens",
                    "request_id": f"{os.getpid()}-{time.time_ns()}",
                    "input_ids": list(int(tok) for tok in input_ids),
                    "layer_ids": requested_layers,
                    "max_seq_len": int(max_seq_len),
                    "out_bin": str(out_bin),
                }
                response = self._request(payload)
                hs_f32, tok_out, cap_out = LlamaCppGGUFBackend._parse_hidden_bin(str(out_bin))
                if list(cap_out) != list(requested_layers):
                    raise RuntimeError(
                        f"layer-id mismatch from server: requested {requested_layers} got {cap_out}"
                    )
                return {
                    "hidden_states": torch.from_numpy(hs_f32),
                    "token_ids": list(tok_out),
                    "capture_layers": list(cap_out),
                    "server_meta": response,
                }
            except (OSError, RuntimeError, TimeoutError) as exc:
                last_exc = exc
                if attempt + 1 >= attempts:
                    break
                if not (self.auto_start or self._server_proc is not None):
                    break
                self._restart_server()
                continue
            finally:
                try:
                    out_bin.unlink()
                except FileNotFoundError:
                    pass
                try:
                    td.rmdir()
                except OSError:
                    pass
        assert last_exc is not None
        raise last_exc

    def generate_one(
        self,
        *,
        input_ids,
        output_path: str | Path,
        source_name: str,
        source_row_idx: int,
        max_seq_len: int = 2048,
        loss_mask=None,
        extra_metadata: Optional[dict] = None,
    ) -> dict:
        result = self.dump_hiddens(
            input_ids=input_ids,
            max_seq_len=max_seq_len,
            layer_ids=self.layer_ids,
        )
        hs = result["hidden_states"]
        token_ids = result["token_ids"]
        seq_len = len(token_ids)
        token_ids_t = torch.tensor(token_ids, dtype=torch.int64)
        input_ids_t = torch.tensor(list(input_ids), dtype=torch.int64)
        if input_ids_t.shape[0] != seq_len:
            if input_ids_t.shape[0] < seq_len:
                pad_len = seq_len - input_ids_t.shape[0]
                input_ids_t = torch.cat(
                    [token_ids_t[:pad_len].clone(), input_ids_t], dim=0
                )
            else:
                input_ids_t = input_ids_t[-seq_len:].clone()
        if loss_mask is None:
            loss_mask_t = torch.ones(seq_len, dtype=torch.bool)
        else:
            loss_mask_t = torch.as_tensor(loss_mask, dtype=torch.bool)
            if loss_mask_t.shape[0] != seq_len:
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
            layer_ids=list(self.layer_ids or []),
            extra_metadata=extra_metadata,
        )


__all__ = ["TraceClient"]
