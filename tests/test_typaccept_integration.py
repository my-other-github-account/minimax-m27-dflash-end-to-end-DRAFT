"""No-GPU tests for the MiniMax typical/top-k accept reproduction plumbing."""
from __future__ import annotations

import math
from pathlib import Path


def test_lucebox_server_cmd_builder_and_topk_accept_env():
    from dflash_llama.inference.server import LuceboxDFlashServer

    srv = LuceboxDFlashServer(
        target_gguf="/models/target.gguf",
        draft_gguf="/models/draft.gguf",
        binary="/usr/bin/true",
        host="127.0.0.1",
        port=9101,
        max_ctx=2304,
        default_max_tokens=256,
        think_max_tokens=128,
        hard_limit_reply_budget=64,
        model_name="minimax-m27-typaccept",
        prefix_cache_slots=4,
        accept_mode="topk",
        accept_topk=3,
    )

    cmd = srv._build_cmd()
    assert cmd[:2] == ["/usr/bin/true", "/models/target.gguf"]
    assert "--draft" in cmd and "/models/draft.gguf" in cmd
    assert "--max-ctx" in cmd and "2304" in cmd
    assert "--default-max-tokens" in cmd and "256" in cmd
    assert "--think-max-tokens" in cmd and "128" in cmd
    assert "--hard-limit-reply-budget" in cmd and "64" in cmd
    assert "--model-name" in cmd and "minimax-m27-typaccept" in cmd
    assert "--prefix-cache-slots" in cmd and "4" in cmd
    # Native accept flags are compatibility-only; default plumbing is via env.
    assert "--accept-mode" not in cmd
    assert "--accept-topk" not in cmd

    flagged = LuceboxDFlashServer(
        target_gguf="/models/target.gguf",
        draft_gguf="/models/draft.gguf",
        binary="/usr/bin/true",
        accept_mode="topk",
        accept_topk=3,
        pass_accept_flags=True,
    )
    flagged_cmd = flagged._build_cmd()
    assert "--accept-mode" in flagged_cmd and "topk" in flagged_cmd
    assert "--accept-topk" in flagged_cmd and "3" in flagged_cmd

    env = srv._build_env({})
    assert env["DFLASH_ACCEPT_MODE"] == "topk"
    assert env["DFLASH_ACCEPT_TOPK"] == "3"
    assert "DFLASH_ACCEPT_ETA" not in env
    assert srv.url == "http://127.0.0.1:9101/v1"


def test_lucebox_server_eta_and_strict_accept_env():
    from dflash_llama.inference.server import LuceboxDFlashServer

    eta_srv = LuceboxDFlashServer(
        target_gguf="target.gguf",
        draft_gguf="draft.gguf",
        binary="/usr/bin/true",
        accept_mode="eta",
        accept_eta=0.93,
    )
    eta_env = eta_srv._build_env({})
    assert eta_env["DFLASH_ACCEPT_MODE"] == "eta"
    assert eta_env["DFLASH_ACCEPT_ETA"] == "0.93"
    assert "DFLASH_ACCEPT_TOPK" not in eta_env

    strict_srv = LuceboxDFlashServer(
        target_gguf="target.gguf",
        draft_gguf="draft.gguf",
        binary="/usr/bin/true",
        accept_mode="strict",
    )
    strict_env = strict_srv._build_env({
        "KEEP": "1",
        "DFLASH_ACCEPT_MODE": "topk",
        "DFLASH_ACCEPT_TOPK": "3",
        "DFLASH_ACCEPT_ETA": "0.5",
    })
    assert strict_env == {"KEEP": "1"}


def test_export_lucebox_parser_and_checkpoint_file_normalization(tmp_path):
    from dflash_llama.cli import build_parser
    from dflash_llama.inference.gguf_export import normalize_lucebox_checkpoint

    ckpt_dir = tmp_path / "step_00020000"
    ckpt_dir.mkdir()
    safetensors = ckpt_dir / "model_lucebox_layout.safetensors"
    safetensors.write_bytes(b"stub")

    assert normalize_lucebox_checkpoint(ckpt_dir) == ckpt_dir
    assert normalize_lucebox_checkpoint(safetensors) == ckpt_dir

    parser = build_parser()
    args = parser.parse_args([
        "export-lucebox",
        "--checkpoint", str(safetensors),
        "--out", str(tmp_path / "draft.gguf"),
    ])
    assert args.checkpoint == str(safetensors)
    assert args.output == str(tmp_path / "draft.gguf")
    assert args.force_block_size == 8


def test_typaccept_benchmark_metrics_and_quality_gate():
    from dflash_llama.inference.benchmark import (
        al_true,
        has_verbatim_repeat_loop,
        quality_gate_passes,
        summarize_ar_vs_spec_50,
    )

    assert al_true(100, 50) == 2.0
    assert math.isinf(al_true(100, 100))
    assert al_true(0, 0) is None

    assert quality_gate_passes({"content": "answer", "reasoning_tokens": 1}) is True
    assert quality_gate_passes({"content": "", "reasoning_tokens": 1}) is False
    assert quality_gate_passes({"content": "answer", "reasoning_tokens": 0}) is False

    repeated_9gram = "a b c d e f g h i a b c d e f g h i"
    assert has_verbatim_repeat_loop(repeated_9gram, min_ngram=9) is True
    assert quality_gate_passes({"content": repeated_9gram, "reasoning_tokens": 1}) is False

    rows = [
        {"name": "p1", "ar_wall_sec": 10.0, "spec_wall_sec": 8.0, "n_pred": 100, "n_acc": 50,
         "content": "ok one", "reasoning_tokens": 3},
        {"name": "p2", "ar_wall_sec": 20.0, "spec_wall_sec": 12.0, "n_pred": 100, "n_acc": 20,
         "content": "ok two", "reasoning_tokens": 2},
    ]
    summary = summarize_ar_vs_spec_50(rows)
    assert summary["prompt_count"] == 2
    assert summary["quality_passed"] == 2
    assert summary["full_wall_speedup"] == 30.0 / 20.0
    assert summary["al_true"] == 200.0 / (200.0 - 70.0)
