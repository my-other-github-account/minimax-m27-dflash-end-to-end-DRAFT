"""Tests for the dflash_llama.hub slug resolver.

We don't actually hit the network here — that would make tests flaky and slow.
Instead we monkeypatch huggingface_hub.snapshot_download and verify the
calling convention + cache layout are correct.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import dflash_llama
from dflash_llama import hub


def test_top_level_exports():
    """Hub helpers exported from the top-level package."""
    assert dflash_llama.cache_root is hub.cache_root
    assert dflash_llama.resolve_hf_repo is hub.resolve_hf_repo
    assert dflash_llama.resolve_gguf_repo is hub.resolve_gguf_repo


def test_cache_root_explicit_override(tmp_path, monkeypatch):
    """DFLASH_LLAMA_HOME wins over XDG_CACHE_HOME and the home fallback."""
    monkeypatch.setenv("DFLASH_LLAMA_HOME", str(tmp_path / "explicit"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert hub.cache_root() == (tmp_path / "explicit").resolve()


def test_cache_root_xdg_fallback(tmp_path, monkeypatch):
    """XDG_CACHE_HOME is honored when the explicit override is unset."""
    monkeypatch.delenv("DFLASH_LLAMA_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert hub.cache_root() == (tmp_path / "xdg" / "dflash-llama").resolve()


def test_cache_root_home_fallback(monkeypatch):
    """Final fallback is ~/.cache/dflash-llama."""
    monkeypatch.delenv("DFLASH_LLAMA_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert hub.cache_root() == Path("~/.cache/dflash-llama").expanduser().resolve()


def test_resolve_hf_repo_skips_weights_by_default(tmp_path, monkeypatch):
    """resolve_hf_repo passes ignore_patterns to skip large weights."""
    monkeypatch.setenv("DFLASH_LLAMA_HOME", str(tmp_path))
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        # Materialise a config.json so the cache dir exists
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(kwargs["local_dir"]) / "config.json").write_text("{}")

    with patch.dict(sys.modules, {"huggingface_hub": type(sys)("huggingface_hub")}):
        sys.modules["huggingface_hub"].snapshot_download = fake_snapshot_download
        path = hub.resolve_hf_repo("MiniMaxAI/MiniMax-M2")

    assert captured["repo_id"] == "MiniMaxAI/MiniMax-M2"
    # Default behaviour: skip weights
    assert any("safetensors" in p for p in captured["ignore_patterns"])
    assert any("gguf" in p for p in captured["ignore_patterns"])
    assert "MiniMaxAI__MiniMax-M2" in path  # Cache layout uses __ separator


def test_resolve_gguf_repo_filters_by_quant(tmp_path, monkeypatch):
    """resolve_gguf_repo restricts allow_patterns to the requested quant."""
    monkeypatch.setenv("DFLASH_LLAMA_HOME", str(tmp_path))
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        # Simulate two GGUF shards being downloaded
        target = Path(kwargs["local_dir"]) / "UD-IQ4_XS"
        target.mkdir(parents=True, exist_ok=True)
        (target / "MiniMax-M2-UD-IQ4_XS-00001-of-00004.gguf").write_bytes(b"GGUF")
        (target / "MiniMax-M2-UD-IQ4_XS-00002-of-00004.gguf").write_bytes(b"GGUF")

    with patch.dict(sys.modules, {"huggingface_hub": type(sys)("huggingface_hub")}):
        sys.modules["huggingface_hub"].snapshot_download = fake_snapshot_download
        path = hub.resolve_gguf_repo("unsloth/MiniMax-M2-GGUF", quant="UD-IQ4_XS")

    # Allow patterns are quant-scoped
    assert any("UD-IQ4_XS" in p for p in captured["allow_patterns"])
    # Returns the FIRST shard (lowest-numbered)
    assert path.endswith("00001-of-00004.gguf")


def test_resolve_gguf_repo_no_files_raises(tmp_path, monkeypatch):
    """If snapshot_download produced no .gguf files, we raise a clear error."""
    monkeypatch.setenv("DFLASH_LLAMA_HOME", str(tmp_path))

    def fake_snapshot_download(**kwargs):
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
        # No GGUFs created on purpose

    with patch.dict(sys.modules, {"huggingface_hub": type(sys)("huggingface_hub")}):
        sys.modules["huggingface_hub"].snapshot_download = fake_snapshot_download
        with pytest.raises(FileNotFoundError, match="no .gguf files matched"):
            hub.resolve_gguf_repo("unsloth/MiniMax-M2-GGUF", quant="UD-IQ4_XS")


def test_load_verifier_with_hf_repo_dispatches_to_resolver(tmp_path, monkeypatch):
    """load_verifier(name, hf_repo=...) calls resolve_hf_repo and uses the result."""
    monkeypatch.setenv("DFLASH_LLAMA_HOME", str(tmp_path))
    fake_local = tmp_path / "fake_hf_dir"
    fake_local.mkdir()

    with patch("dflash_llama.hub.resolve_hf_repo", return_value=str(fake_local)) as mock:
        v = dflash_llama.load_verifier(
            "minimax-m2.7-iq4-xs",
            hf_repo="MiniMaxAI/MiniMax-M2",
            revision="main",
        )
    mock.assert_called_once_with("MiniMaxAI/MiniMax-M2", revision="main")
    assert v.hf_path == str(fake_local)


def test_load_verifier_local_path_skips_resolver(tmp_path):
    """When hf_path is given directly, no Hub call is made."""
    with patch("dflash_llama.hub.resolve_hf_repo") as mock:
        v = dflash_llama.load_verifier(
            "minimax-m2.7-iq4-xs",
            hf_path=str(tmp_path),
        )
    mock.assert_not_called()
    assert v.hf_path == str(tmp_path)
