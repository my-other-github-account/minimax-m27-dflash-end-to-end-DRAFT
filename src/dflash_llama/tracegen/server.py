"""Unix-socket trace-generation server backed by a persistent llama worker."""
from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..generation.backends.llamacpp_gguf import LlamaCppGGUFBackend
from ._proc import parent_deathsig_preexec

SOCKET_PREFIX = "unix://"


def _normalize_socket_path(bind: str) -> str:
    if bind.startswith(SOCKET_PREFIX):
        return bind[len(SOCKET_PREFIX):]
    return bind


def _recv_json_line(conn: socket.socket, max_bytes: int = 8 * 1024 * 1024) -> dict:
    chunks = bytearray()
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        chunks.extend(chunk)
        if len(chunks) > max_bytes:
            raise ValueError("request exceeded max size")
        if b"\n" in chunk:
            break
    if not chunks:
        raise EOFError("peer closed without sending a request")
    line = chunks.split(b"\n", 1)[0]
    return json.loads(line.decode("utf-8"))


def _send_json_line(conn: socket.socket, payload: dict) -> None:
    conn.sendall(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")


class _PersistentWorker:
    """Owns the long-lived llama-dump-hiddens worker subprocess."""

    def __init__(
        self,
        *,
        gguf_path: str,
        binary: str,
        ctx: int,
        ngl: int,
        override_tensor: Optional[str],
        worker_args: Optional[Sequence[str]],
        startup_timeout: float,
        request_timeout: float,
        log_path: Optional[str | Path],
    ):
        self.gguf_path = str(gguf_path)
        self.binary = str(binary)
        self.ctx = int(ctx)
        self.ngl = int(ngl)
        self.override_tensor = override_tensor
        self.worker_args = [str(arg) for arg in (worker_args or [])]
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        self.log_path = Path(log_path) if log_path else None
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None
        self._lock = threading.Lock()

    def _build_cmd(self) -> list[str]:
        cmd = [
            self.binary,
            "-m",
            self.gguf_path,
            "-ngl",
            str(self.ngl),
            "-c",
            str(self.ctx),
            "-p",
            "x",
        ]
        if self.override_tensor:
            cmd += ["-ot", self.override_tensor]
        cmd += self.worker_args
        return cmd

    @staticmethod
    def _readline_with_timeout(stream, timeout: float) -> str:
        fd = stream.fileno()
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            raise TimeoutError(f"worker timed out after {timeout:.1f}s")
        line = stream.readline()
        if not line:
            raise RuntimeError("worker exited before producing protocol output")
        return line.rstrip("\n")

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        cmd = self._build_cmd()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "a", buffering=1)
            stderr = self._log_fh
        else:
            stderr = None
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
            preexec_fn=parent_deathsig_preexec(),
        )
        ready = self._readline_with_timeout(self._proc.stdout, self.startup_timeout)
        if not ready.startswith("READY\t"):
            self.stop()
            raise RuntimeError(f"unexpected worker banner: {ready}")

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            if self._log_fh:
                self._log_fh.close()
                self._log_fh = None
            return
        try:
            if proc.stdin:
                try:
                    proc.stdin.write("QUIT\n")
                    proc.stdin.flush()
                except BrokenPipeError:
                    pass
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        finally:
            if self._log_fh:
                self._log_fh.close()
                self._log_fh = None

    def run_job(
        self,
        *,
        request_id: str,
        tokens_bin: str,
        out_bin: str,
        capture_layers: Iterable[int],
    ) -> dict:
        capture_csv = ",".join(str(int(layer)) for layer in capture_layers)
        with self._lock:
            self.start()
            if self._proc is None or self._proc.poll() is not None:
                raise RuntimeError("worker is not running")
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None
            try:
                self._proc.stdin.write(
                    f"{request_id}\t{tokens_bin}\t{out_bin}\t{capture_csv}\n"
                )
                self._proc.stdin.flush()
            except BrokenPipeError as exc:
                self.stop()
                raise RuntimeError("worker pipe broke while sending request") from exc

            line = self._readline_with_timeout(self._proc.stdout, self.request_timeout)
            fields = line.split("\t", 4)
            if len(fields) < 2:
                raise RuntimeError(f"malformed worker response: {line}")
            if fields[0] == "OK":
                if len(fields) != 5 or fields[1] != request_id:
                    raise RuntimeError(f"mismatched worker response: {line}")
                return {
                    "request_id": request_id,
                    "n_layers": int(fields[2]),
                    "n_tokens": int(fields[3]),
                    "n_embd": int(fields[4]),
                }
            if fields[0] == "ERR":
                if len(fields) < 3:
                    raise RuntimeError(f"malformed worker error: {line}")
                if fields[1] != request_id:
                    raise RuntimeError(f"mismatched worker error: {line}")
                raise RuntimeError(fields[2])
            raise RuntimeError(f"unknown worker response: {line}")


class TraceServer:
    """Persistent hidden-state extraction service on a Unix socket."""

    def __init__(
        self,
        *,
        gguf_path: str,
        layer_ids: Iterable[int],
        bind: str = "unix:///tmp/dflash_tracegen.sock",
        n_ctx: int = 4096,
        n_gpu_layers: int = 99,
        override_tensor: Optional[str] = "exps=CPU",
        binary: str = "llama-dump-hiddens-worker",
        worker_args: Optional[Sequence[str]] = None,
        temp_root: Optional[str | Path] = None,
        startup_timeout: float = 900.0,
        request_timeout: float = 900.0,
        worker_log_path: Optional[str | Path] = None,
    ):
        self.gguf_path = str(gguf_path)
        self.layer_ids = [int(layer) for layer in layer_ids]
        self.bind = bind
        self.socket_path = _normalize_socket_path(bind)
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.override_tensor = override_tensor
        self.binary = str(binary)
        self.worker_args = [str(arg) for arg in (worker_args or [])]
        self.temp_root = Path(temp_root) if temp_root else None
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        self.worker_log_path = Path(worker_log_path) if worker_log_path else None
        self._worker = _PersistentWorker(
            gguf_path=self.gguf_path,
            binary=self.binary,
            ctx=self.n_ctx,
            ngl=self.n_gpu_layers,
            override_tensor=self.override_tensor,
            worker_args=self.worker_args,
            startup_timeout=self.startup_timeout,
            request_timeout=self.request_timeout,
            log_path=self.worker_log_path,
        )
        self._listener: Optional[socket.socket] = None
        self._stop = threading.Event()

    def start(self) -> "TraceServer":
        if self._listener is not None:
            return self
        self._stop.clear()
        if self.temp_root:
            self.temp_root.mkdir(parents=True, exist_ok=True)
        self._worker.start()
        sock_path = Path(self.socket_path)
        sock_path.parent.mkdir(parents=True, exist_ok=True)
        if sock_path.exists():
            sock_path.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(self.socket_path)
        listener.listen(128)
        listener.settimeout(1.0)
        self._listener = listener
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            finally:
                self._listener = None
        try:
            Path(self.socket_path).unlink()
        except FileNotFoundError:
            pass
        self._worker.stop()

    def __enter__(self) -> "TraceServer":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

    def _handle_dump_hiddens(self, request: dict) -> dict:
        input_ids = list(request["input_ids"])
        max_seq_len = int(request["max_seq_len"])
        if len(input_ids) > max_seq_len:
            raise ValueError(
                f"input_ids length {len(input_ids)} exceeds max_seq_len {max_seq_len}"
            )
        if len(input_ids) > self.n_ctx - 16:
            raise ValueError(
                f"input_ids length {len(input_ids)} exceeds ctx-16 ({self.n_ctx - 16})"
            )
        out_bin = str(request["out_bin"])
        capture_layers = [int(x) for x in request.get("layer_ids") or self.layer_ids]
        if not capture_layers:
            raise ValueError("request omitted layer_ids and server has no default layer_ids")
        req_id = str(request.get("request_id", "req"))
        td = Path(
            tempfile.mkdtemp(
                prefix="dflash_tracegen_req_",
                dir=str(self.temp_root) if self.temp_root else None,
            )
        )
        tokens_bin = td / "tokens.bin"
        try:
            LlamaCppGGUFBackend._write_tokens_bin(input_ids, str(tokens_bin))
            worker_meta = self._worker.run_job(
                request_id=req_id,
                tokens_bin=str(tokens_bin),
                out_bin=out_bin,
                capture_layers=capture_layers,
            )
        finally:
            try:
                tokens_bin.unlink()
            except FileNotFoundError:
                pass
            try:
                td.rmdir()
            except OSError:
                pass
        return {
            "ok": True,
            "request_id": req_id,
            "capture_layers": capture_layers,
            **worker_meta,
        }

    def _handle_request(self, request: dict) -> dict:
        op = request.get("op", "dump_hiddens")
        if op == "ping":
            return {"ok": True, "status": "ready"}
        if op == "dump_hiddens":
            return self._handle_dump_hiddens(request)
        raise ValueError(f"unsupported op {op!r}")

    def serve_forever(self) -> None:
        self.start()
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                raise
            with conn:
                conn.settimeout(self.request_timeout)
                try:
                    request = _recv_json_line(conn)
                    response = self._handle_request(request)
                except Exception as exc:  # noqa: BLE001
                    response = {"ok": False, "error": str(exc)}
                _send_json_line(conn, response)


__all__ = ["TraceServer", "SOCKET_PREFIX"]
