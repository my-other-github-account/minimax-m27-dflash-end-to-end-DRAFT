from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import torch
from safetensors import safe_open


def _write_fake_worker(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import os
import struct
import sys

def parse_csv(s):
    return [int(x) for x in s.split(",") if x]

print("READY\\t4096", flush=True)
for line in sys.stdin:
    line = line.rstrip("\\n")
    if line == "QUIT":
        break
    req_id, tokens_bin, out_bin, layer_csv = line.split("\\t")
    layers = parse_csv(layer_csv)
    with open(tokens_bin, "rb") as f:
        raw = f.read()
    n_tokens = struct.unpack_from("<I", raw, 0)[0]
    token_ids = list(struct.unpack_from(f"<{n_tokens}i", raw, 4))
    n_embd = 4
    body = []
    for layer_idx in range(len(layers)):
        for token_idx in range(n_tokens):
            for dim in range(n_embd):
                body.append(float(layer_idx * 1000 + token_idx * 10 + dim))
    with open(out_bin, "wb") as out:
        out.write(struct.pack("<iii", len(layers), n_tokens, n_embd))
        out.write(struct.pack(f"<{len(layers)}i", *layers))
        out.write(struct.pack("<i", n_tokens))
        out.write(struct.pack(f"<{n_tokens}i", *token_ids))
        out.write(struct.pack(f"<{len(body)}f", *body))
    print(f"OK\\t{req_id}\\t{len(layers)}\\t{n_tokens}\\t{n_embd}", flush=True)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _start_server(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline:
        if Path(server.socket_path).exists():
            return thread
        time.sleep(0.05)
    raise TimeoutError("server socket did not appear in time")


def _short_socket_path(name: str) -> str:
    return f"/tmp/{name}_{os.getpid()}.sock"


def test_trace_server_client_roundtrip(tmp_path):
    from dflash_llama.tracegen import TraceClient, TraceServer

    worker = tmp_path / "fake_worker.py"
    _write_fake_worker(worker)
    socket_path = _short_socket_path("tracegen_roundtrip")
    server = TraceServer(
        gguf_path="/tmp/fake.gguf",
        layer_ids=[2, 16],
        bind=f"unix://{socket_path}",
        binary=str(worker),
        request_timeout=5,
        startup_timeout=5,
    )
    thread = _start_server(server)
    try:
        client = TraceClient(
            socket_path=f"unix://{socket_path}",
            layer_ids=[2, 16],
            request_timeout=5,
        )
        result = client.dump_hiddens(input_ids=[7, 8, 9], max_seq_len=16)
        hs = result["hidden_states"]
        assert tuple(hs.shape) == (3, 2, 4)
        assert result["token_ids"] == [7, 8, 9]
        assert torch.equal(hs[1, 1], torch.tensor([1010.0, 1011.0, 1012.0, 1013.0]))
    finally:
        server.stop()
        thread.join(timeout=5)


def test_trace_client_generate_one_writes_fp8_trace(tmp_path):
    from dflash_llama.generation.format import load_trace
    from dflash_llama.tracegen import TraceClient, TraceServer

    worker = tmp_path / "fake_worker.py"
    _write_fake_worker(worker)
    socket_path = _short_socket_path("tracegen_generate_one")
    server = TraceServer(
        gguf_path="/tmp/fake.gguf",
        layer_ids=[2, 16],
        bind=f"unix://{socket_path}",
        binary=str(worker),
        request_timeout=5,
        startup_timeout=5,
    )
    thread = _start_server(server)
    try:
        client = TraceClient(
            socket_path=f"unix://{socket_path}",
            layer_ids=[2, 16],
            request_timeout=5,
        )
        out_path = tmp_path / "hs_0.safetensors"
        meta = client.generate_one(
            input_ids=[3, 4, 5],
            output_path=out_path,
            source_name="unit",
            source_row_idx=0,
        )
        assert meta["storage"] == "fp8_per_tensor_scale"
        trace = load_trace(out_path)
        assert tuple(trace["hidden_states"].shape) == (3, 2, 4)
        with safe_open(str(out_path), framework="pt") as f:
            assert "hidden_states_scale" in f.keys()
            assert f.metadata()["storage"] == "fp8_per_tensor_scale"
    finally:
        server.stop()
        thread.join(timeout=5)


def test_trace_generator_tracegen_client_backend(tmp_path):
    from dflash_llama import TraceGenerator
    from dflash_llama.generation.format import load_trace
    from dflash_llama.verifiers import generic_verifier
    from dflash_llama.tracegen import TraceServer

    worker = tmp_path / "fake_worker.py"
    _write_fake_worker(worker)
    socket_path = _short_socket_path("tracegen_backend")
    server = TraceServer(
        gguf_path="/tmp/fake.gguf",
        layer_ids=[2, 16],
        bind=f"unix://{socket_path}",
        binary=str(worker),
        request_timeout=5,
        startup_timeout=5,
    )
    thread = _start_server(server)
    try:
        verifier = generic_verifier(
            name="fake-generic",
            hidden_size=4,
            num_hidden_layers=32,
            vocab_size=1024,
            mask_token_id=1023,
            layer_ids=[2, 16],
            gguf_path="/tmp/fake.gguf",
        )
        gen = TraceGenerator(
            verifier=verifier,
            backend="tracegen_client",
            backend_kwargs={
                "socket_path": f"unix://{socket_path}",
                "request_timeout": 5,
            },
        )
        out_path = tmp_path / "hs_1.safetensors"
        gen.generate_one(
            input_ids=[1, 2],
            output_path=out_path,
            source_name="unit",
            source_row_idx=1,
        )
        trace = load_trace(out_path)
        assert tuple(trace["hidden_states"].shape) == (2, 2, 4)
    finally:
        server.stop()
        thread.join(timeout=5)


def test_trace_client_build_server_cmd_includes_worker_args(tmp_path):
    from dflash_llama.tracegen import TraceClient

    log_path = tmp_path / "tracegen.log"
    client = TraceClient(
        socket_path=f"unix://{_short_socket_path('tracegen_autostart')}",
        auto_start=True,
        gguf_path="/tmp/fake.gguf",
        layer_ids=[2, 16],
        binary="/tmp/fake-worker",
        override_tensor="exps=CPU",
        worker_args=["--no-mmap", "--mlock"],
        server_log_path=log_path,
    )
    cmd = client._build_server_cmd()
    assert "--worker-arg" in cmd
    assert cmd.count("--worker-arg") == 2
    assert cmd[cmd.index("--worker-arg") + 1] == "--no-mmap"
    assert cmd[cmd.index("--worker-arg", cmd.index("--worker-arg") + 1) + 1] == "--mlock"
