"""Tests for the persistent batched-decode trace-server / trace-client wiring.

These tests do NOT exercise the actual llama-dump-hiddens-worker binary or
load any model — they only check that the Python plumbing (imports, class
construction, CLI parser, backend dispatch) is correctly wired. The
end-to-end smoke against a real worker is documented in
``repro/08-tracegen-server.md`` and requires CUDA + a GGUF on disk.
"""
from __future__ import annotations

import pytest


def test_tracegen_module_imports() -> None:
    """Top-level package exports the persistent-server API."""
    import dflash_llama

    assert dflash_llama.TraceServer is not None
    assert dflash_llama.TraceClient is not None


def test_tracegen_client_class_is_constructible_without_model() -> None:
    """TraceClient init must NOT spawn a worker or open a socket — that
    happens lazily on first request. The class needs to be importable and
    constructible in CI without CUDA."""
    from dflash_llama import TraceClient

    client = TraceClient(
        socket_path="unix:///tmp/dflash_test_should_not_exist.sock",
        auto_start=False,  # do not spawn a worker in CI
        gguf_path="/tmp/does_not_exist.gguf",
        layer_ids=[2, 16, 30, 45, 59, 61],
        binary="llama-dump-hiddens-worker",
    )
    assert client.layer_ids == [2, 16, 30, 45, 59, 61]
    assert client.binary == "llama-dump-hiddens-worker"
    assert client.auto_start is False
    # We never connected, so there should be no server process.
    assert client._server_proc is None


def test_tracegen_client_normalizes_legacy_binary_name() -> None:
    """A user that says binary='llama-dump-hiddens' on the persistent-server
    code path should silently be upgraded to the '-worker' binary, since the
    one-shot binary cannot talk JSONL on stdin/stdout."""
    from dflash_llama import TraceClient

    client = TraceClient(
        socket_path="unix:///tmp/dflash_test_should_not_exist.sock",
        auto_start=False,
        gguf_path="/tmp/does_not_exist.gguf",
        layer_ids=[2],
        binary="llama-dump-hiddens",  # the legacy one-shot name
    )
    assert client.binary == "llama-dump-hiddens-worker"


def test_tracegen_client_backend_is_registered() -> None:
    """The generation backend registry should accept 'tracegen_client'."""
    from dflash_llama.generation import TracegenClientBackend
    from dflash_llama.generation.backends import LlamaCppGGUFBackend

    assert TracegenClientBackend is not None
    # Both backends share the same base interface.
    assert hasattr(TracegenClientBackend, "run_one")
    assert hasattr(LlamaCppGGUFBackend, "run_one")


def test_cli_exposes_trace_server_subcommand() -> None:
    """`dflash-llama trace-server --help` should be a real, documented entry
    point — not just an internal helper."""
    from dflash_llama.cli import build_parser

    parser = build_parser()
    subactions = parser._subparsers._group_actions[0].choices
    assert "trace-server" in subactions, sorted(subactions)

    # Required args are present
    trace_server_parser = subactions["trace-server"]
    arg_names = {a.dest for a in trace_server_parser._actions}
    for required in ("gguf_path", "layer_ids", "socket", "ctx", "ngl", "binary"):
        assert required in arg_names, f"trace-server parser missing --{required}"


def test_cli_generate_supports_tracegen_client_backend() -> None:
    """`dflash-llama generate --backend tracegen_client` should be valid."""
    from dflash_llama.cli import build_parser

    parser = build_parser()
    # Parse a minimal valid command that uses the new backend; should not raise.
    args = parser.parse_args([
        "generate",
        "--verifier", "minimax-m2.7-iq4-xs",
        "--gguf-path", "/tmp/does_not_exist.gguf",
        "--prompts", "/tmp/does_not_exist_prompts",
        "--out", "/tmp/does_not_exist_out",
        "--backend", "tracegen_client",
    ])
    assert args.backend == "tracegen_client"
    assert args.socket.startswith("unix://") or args.socket.startswith("tcp://")


def test_make_backend_rejects_unknown_names() -> None:
    """Defensive: unknown backend names produce a helpful error that lists
    the actually-supported ones."""
    from dflash_llama.generation.trace_generator import _make_backend
    from dflash_llama.verifiers import minimax_m27_iq4_xs

    verifier = minimax_m27_iq4_xs(gguf_path="/tmp/does_not_exist.gguf")
    with pytest.raises(ValueError, match="tracegen_client"):
        _make_backend("nonsense_backend", verifier=verifier)
