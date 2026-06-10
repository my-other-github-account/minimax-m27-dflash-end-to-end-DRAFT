"""OpenAI-compatible llama.cpp server wrapped as a Python context manager.

Public surface::

    LlamaServer(verifier_gguf, drafter_gguf=None, ...) — context manager

Usage::

    with LlamaServer(verifier_gguf=..., drafter_gguf=..., port=8080) as srv:
        print(srv.url)  # → "http://localhost:8080/v1"
        # any OpenAI client points at srv.url
"""
from __future__ import annotations

import contextlib
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

DEFAULT_BINARY = "llama-server"
DEFAULT_LUCEBOX_DFLASH_BINARY = "dflash_server"


def _resolve_binary(binary: Optional[str | Path]) -> str:
    if binary is not None:
        return str(binary)
    if Path(DEFAULT_BINARY).exists():
        return DEFAULT_BINARY
    found = shutil.which("llama-server")
    if found:
        return found
    raise FileNotFoundError(
        f"Could not locate llama-server. Pass binary= or put llama-server on PATH. "
        f"Tried command: {DEFAULT_BINARY}"
    )



def _resolve_lucebox_binary(binary: Optional[str | Path]) -> str:
    if binary is not None:
        return str(binary)
    if Path(DEFAULT_LUCEBOX_DFLASH_BINARY).exists():
        return str(DEFAULT_LUCEBOX_DFLASH_BINARY)
    found = shutil.which(DEFAULT_LUCEBOX_DFLASH_BINARY)
    if found:
        return found
    raise FileNotFoundError(
        "Could not locate Lucebox dflash_server. Pass binary= or put "
        "dflash_server on PATH."
    )

def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


class LlamaServer:
    """Context manager that runs llama.cpp's OpenAI-compatible server.

    The server exposes ``http://<host>:<port>/v1/chat/completions`` and
    ``/v1/completions``. Plug any OpenAI client at ``srv.url``.

    Parameters
    ----------
    verifier_gguf : path
        Target model GGUF.
    drafter_gguf : path, optional
        DFlash drafter GGUF. If provided, speculative decoding is enabled
        with ``--spec-type dflash``.
    spec_type : {"dflash", "draft", None}
        Speculation mode. Default "dflash" when drafter is provided.
    draft_max : int
        Speculative chain length when drafter is provided.
    host : str
        Bind address. Default "0.0.0.0" (all interfaces).
    port : int
        Listen port. Default 8080.
    ctx : int
        Verifier context length.
    n_gpu_layers / n_gpu_layers_draft : int
        --ngl / --ngld. Default 99 (full GPU offload).
    override_tensor : str, optional
        --override-tensor / -ot. Default ``exps=CPU`` to keep MoE experts off-GPU.
    draft_device : str, optional
        --device-draft. Default ``CUDA0``.
    binary : path, optional
        llama-server binary. Default: PATH lookup.
    parallel : int
        --parallel slots (concurrent requests). Default 1.
    extra_args : list[str], optional
        Additional flags passed through.
    startup_timeout : float
        Seconds to wait for the port to come up before raising.
    log_path : path, optional
        If set, server stdout+stderr is appended to this file. Otherwise
        inherits the parent's stdout/stderr.
    """

    def __init__(
        self,
        verifier_gguf: str | Path,
        drafter_gguf: Optional[str | Path] = None,
        *,
        spec_type: Optional[str] = "dflash",
        draft_max: int = 7,
        host: str = "0.0.0.0",
        port: int = 8080,
        ctx: int = 8192,
        n_gpu_layers: int = 99,
        n_gpu_layers_draft: int = 99,
        override_tensor: Optional[str] = "exps=CPU",
        draft_device: Optional[str] = "CUDA0",
        binary: Optional[str | Path] = None,
        parallel: int = 1,
        extra_args: Optional[list[str]] = None,
        startup_timeout: float = 600.0,
        log_path: Optional[str | Path] = None,
    ):
        self.verifier_gguf = str(verifier_gguf)
        self.drafter_gguf = str(drafter_gguf) if drafter_gguf else None
        self.spec_type = spec_type if drafter_gguf else None
        self.draft_max = draft_max
        self.host = host
        self.port = port
        self.ctx = ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_gpu_layers_draft = n_gpu_layers_draft
        self.override_tensor = override_tensor
        self.draft_device = draft_device
        self.binary = _resolve_binary(binary)
        self.parallel = parallel
        self.extra_args = list(extra_args or [])
        self.startup_timeout = startup_timeout
        self.log_path = Path(log_path) if log_path else None
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}/v1"

    @property
    def base_url(self) -> str:
        """Same as ``url`` but without the ``/v1`` suffix."""
        return self.url[:-3]

    def _build_cmd(self) -> list[str]:
        cmd = [
            self.binary,
            "-m", self.verifier_gguf,
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.ctx),
            "-ngl", str(self.n_gpu_layers),
            "--parallel", str(self.parallel),
        ]
        if self.override_tensor:
            cmd += ["-ot", self.override_tensor]
        if self.drafter_gguf:
            cmd += [
                "-md", self.drafter_gguf,
                "--draft-max", str(self.draft_max),
                "-ngld", str(self.n_gpu_layers_draft),
            ]
            if self.spec_type == "dflash":
                cmd += ["--spec-type", "dflash"]
            elif self.spec_type:
                cmd += ["--spec-type", self.spec_type]
            if self.draft_device:
                cmd += ["-devd", self.draft_device]
        cmd += self.extra_args
        return cmd

    def start(self):
        if self._proc is not None:
            raise RuntimeError("Server already running")
        cmd = self._build_cmd()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "a")
            self._log_fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} START\n")
            self._log_fh.write(f"=== cmd: {' '.join(cmd)}\n")
            self._log_fh.flush()
            stdout, stderr = self._log_fh, subprocess.STDOUT
        else:
            stdout, stderr = None, None
        self._proc = subprocess.Popen(
            cmd,
            stdout=stdout, stderr=stderr,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
        if not _wait_for_port(self.host if self.host != "0.0.0.0" else "127.0.0.1",
                               self.port, self.startup_timeout):
            self.stop()
            raise TimeoutError(
                f"llama-server did not bind {self.host}:{self.port} in "
                f"{self.startup_timeout:.0f}s. Check log: {self.log_path}"
            )
        return self

    def stop(self):
        if self._proc is None:
            return
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            else:
                self._proc.terminate()
            self._proc.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                else:
                    self._proc.kill()
            except ProcessLookupError:
                pass
        finally:
            self._proc = None
            if self._log_fh:
                self._log_fh.close()
                self._log_fh = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


class LuceboxDFlashServer:
    """Context manager for Lucebox's native OpenAI-compatible DFlash server.

    Lucebox's patched ``dflash_server`` takes the target GGUF as a positional
    argument and the drafter GGUF via ``--draft``. Typical/top-k accept is
    controlled by ``DFLASH_ACCEPT_MODE`` plus ``DFLASH_ACCEPT_ETA`` or
    ``DFLASH_ACCEPT_TOPK``; the verifier remains authoritative for every
    committed token.
    """

    VALID_ACCEPT_MODES = {"strict", "eta", "topk"}

    def __init__(
        self,
        target_gguf: str | Path,
        draft_gguf: str | Path,
        *,
        binary: Optional[str | Path] = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        max_ctx: int = 2304,
        default_max_tokens: int = 256,
        think_max_tokens: Optional[int] = None,
        hard_limit_reply_budget: Optional[int] = None,
        model_name: Optional[str] = None,
        prefix_cache_slots: Optional[int] = None,
        accept_mode: str = "strict",
        accept_eta: Optional[float] = None,
        accept_topk: Optional[int] = None,
        pass_accept_flags: bool = False,
        extra_args: Optional[list[str]] = None,
        startup_timeout: float = 600.0,
        log_path: Optional[str | Path] = None,
    ):
        accept_mode = accept_mode.lower()
        if accept_mode not in self.VALID_ACCEPT_MODES:
            raise ValueError(
                f"accept_mode must be one of {sorted(self.VALID_ACCEPT_MODES)}, "
                f"got {accept_mode!r}"
            )
        if accept_mode == "eta" and accept_eta is None:
            raise ValueError("accept_eta is required when accept_mode='eta'")
        if accept_mode == "topk" and accept_topk is None:
            raise ValueError("accept_topk is required when accept_mode='topk'")

        self.target_gguf = str(target_gguf)
        self.draft_gguf = str(draft_gguf)
        self.binary = _resolve_lucebox_binary(binary)
        self.host = host
        self.port = port
        self.max_ctx = max_ctx
        self.default_max_tokens = default_max_tokens
        self.think_max_tokens = think_max_tokens
        self.hard_limit_reply_budget = hard_limit_reply_budget
        self.model_name = model_name
        self.prefix_cache_slots = prefix_cache_slots
        self.accept_mode = accept_mode
        self.accept_eta = accept_eta
        self.accept_topk = accept_topk
        self.pass_accept_flags = pass_accept_flags
        self.extra_args = list(extra_args or [])
        self.startup_timeout = startup_timeout
        self.log_path = Path(log_path) if log_path else None
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}/v1"

    @property
    def base_url(self) -> str:
        """Same as ``url`` but without the ``/v1`` suffix."""
        return self.url[:-3]

    def _build_cmd(self) -> list[str]:
        cmd = [
            self.binary,
            self.target_gguf,
            "--draft", self.draft_gguf,
            "--host", self.host,
            "--port", str(self.port),
            "--max-ctx", str(self.max_ctx),
            "--default-max-tokens", str(self.default_max_tokens),
        ]
        if self.think_max_tokens is not None:
            cmd += ["--think-max-tokens", str(self.think_max_tokens)]
        if self.hard_limit_reply_budget is not None:
            cmd += ["--hard-limit-reply-budget", str(self.hard_limit_reply_budget)]
        if self.model_name:
            cmd += ["--model-name", self.model_name]
        if self.prefix_cache_slots is not None:
            cmd += ["--prefix-cache-slots", str(self.prefix_cache_slots)]
        if self.pass_accept_flags and self.accept_mode != "strict":
            cmd += ["--accept-mode", self.accept_mode]
            if self.accept_mode == "eta":
                cmd += ["--accept-eta", str(self.accept_eta)]
            elif self.accept_mode == "topk":
                cmd += ["--accept-topk", str(self.accept_topk)]
        cmd += self.extra_args
        return cmd

    def _build_env(self, base_env: Optional[dict[str, str]] = None) -> dict[str, str]:
        env = dict(base_env or {})
        for key in ("DFLASH_ACCEPT_MODE", "DFLASH_ACCEPT_ETA", "DFLASH_ACCEPT_TOPK"):
            env.pop(key, None)
        if self.accept_mode == "strict":
            return env
        env["DFLASH_ACCEPT_MODE"] = self.accept_mode
        if self.accept_mode == "eta":
            env["DFLASH_ACCEPT_ETA"] = str(self.accept_eta)
        elif self.accept_mode == "topk":
            env["DFLASH_ACCEPT_TOPK"] = str(self.accept_topk)
        return env

    def start(self):
        if self._proc is not None:
            raise RuntimeError("Server already running")
        cmd = self._build_cmd()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "a")
            self._log_fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} START\n")
            self._log_fh.write(f"=== cmd: {' '.join(cmd)}\n")
            self._log_fh.flush()
            stdout, stderr = self._log_fh, subprocess.STDOUT
        else:
            stdout, stderr = None, None
        self._proc = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            env=self._build_env(os.environ),
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
        if not _wait_for_port(self.host if self.host != "0.0.0.0" else "127.0.0.1",
                              self.port, self.startup_timeout):
            self.stop()
            raise TimeoutError(
                f"Lucebox dflash_server did not bind {self.host}:{self.port} in "
                f"{self.startup_timeout:.0f}s. Check log: {self.log_path}"
            )
        return self

    def stop(self):
        if self._proc is None:
            return
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            else:
                self._proc.terminate()
            self._proc.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                else:
                    self._proc.kill()
            except ProcessLookupError:
                pass
        finally:
            self._proc = None
            if self._log_fh:
                self._log_fh.close()
                self._log_fh = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


__all__ = ["LlamaServer", "LuceboxDFlashServer", "DEFAULT_BINARY", "DEFAULT_LUCEBOX_DFLASH_BINARY"]
