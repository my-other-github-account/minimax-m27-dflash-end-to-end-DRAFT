"""Unit tests for the inference module — no GPU, no buun build, no network.

Validates:
  - log parser handles real llama-speculative-simple output
  - chain prediction math is correct
  - z-score sign + magnitude
  - SpeculativeReport round-trips JSON
  - GGUF-export module imports cleanly even when buun is absent
  - Server config builder produces a sensible argv
"""
from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Real llama-speculative-simple output fragment from the May-2 bench.
# Trimmed but structurally identical to what the binary emits.
SAMPLE_LOG = textwrap.dedent("""\
    === DMAX=2 === 2026-05-02T08:14:00-07:00
    main: target_dft_loaded
    ... lots of llama logs ...
    n_drafted = 634
    n_accept  = 68
    accept    = 10.726%

    rejection histogram (position → count [%]):
      pos  0:  254 ( 80.1%)
      pos  1:   58 ( 18.3%)
      all ok:    5 (  1.6%)

    draft:

    statistics copyspec: #calls(b,g,a) = 1 317 0, #gen drafts = 0, #acc drafts = 0
    statistics dflash: #calls(b,g,a) = 1 317 0, #gen drafts = 317, #acc drafts = 0

    target:

    common_perf_print:    sampling time =      92.30 ms
    common_perf_print:    samplers time =      37.24 ms /   385 tokens
    common_perf_print:        load time =   19394.87 ms
    common_perf_print: prompt eval time =  161644.16 ms /   980 tokens (  164.94 ms per token,     6.06 tokens per second)
    common_perf_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
    common_perf_print:       total time =  173929.46 ms /   981 tokens
""")


def test_parse_speculative_log(tmp_path):
    from dflash_llama.inference import parse_speculative_log

    log = tmp_path / "dmax2.log"
    log.write_text(SAMPLE_LOG)
    parsed = parse_speculative_log(log)

    assert parsed["n_iter"] == 254 + 58 + 5  # 317
    assert parsed["rej"] == [254, 58]
    assert parsed["all_ok"] == 5
    assert parsed["n_drafted"] == 634
    assert parsed["n_accept"] == 68
    assert parsed["dflash_path_fired"] is True
    assert parsed["dflash_n_calls"] == 317
    assert parsed["prompt_tps"] == pytest.approx(6.06, abs=0.01)


def test_chain_pred_from_val(tmp_path):
    from dflash_llama.inference import chain_pred_from_val

    val = tmp_path / "val_metrics.json"
    # Real numbers from iq4_full_5L_bf16only_20260501_081933 ckpt_best
    val.write_text(json.dumps({
        "loss_epoch": 5.936,
        "full_acc_epoch": 0.1165,
        "position 1 acc_epoch": 0.20528,
        "position 2 acc_epoch": 0.14110,
        "position 3 acc_epoch": 0.11163,
    }))
    p, chained, loss = chain_pred_from_val(val)
    assert len(p) == 3
    assert p[0] == pytest.approx(0.20528)
    # chain[0] = p[0]; chain[1] = p[0]*p[1]; chain[2] = p[0]*p[1]*p[2]
    assert chained[0] == pytest.approx(0.20528)
    assert chained[1] == pytest.approx(0.20528 * 0.14110)
    assert chained[2] == pytest.approx(0.20528 * 0.14110 * 0.11163)
    assert loss == pytest.approx(5.936)


def test_z_score():
    from dflash_llama.inference import z_score

    # Measured exactly equals predicted ⇒ z=0
    assert z_score(0.20, 0.20, 100) == pytest.approx(0.0)
    # Measured > predicted ⇒ z > 0
    assert z_score(0.30, 0.20, 100) > 0
    # Measured < predicted ⇒ z < 0
    assert z_score(0.10, 0.20, 100) < 0
    # Edge cases
    assert math.isnan(z_score(0.5, 0.0, 100))
    assert math.isnan(z_score(0.5, 1.0, 100))
    assert math.isnan(z_score(0.5, 0.5, 0))


def test_chain_measured_per_position():
    from dflash_llama.inference import chain_measured, per_position_conditional

    parsed = {"n_iter": 317, "rej": [254, 58], "all_ok": 5,
              "n_drafted": 634, "n_accept": 68}

    cm = chain_measured(parsed)
    # chain-pos-1 = (317 - 254) / 317 = 0.1987
    assert cm[0] == pytest.approx(63 / 317)
    # chain-pos-2 = (317 - 254 - 58) / 317 = 0.0158
    assert cm[1] == pytest.approx(5 / 317)

    pp = per_position_conditional(parsed)
    assert pp[0]["position"] == 1
    assert pp[0]["n_reached"] == 317
    assert pp[0]["n_accepted"] == 63
    assert pp[0]["p_k"] == pytest.approx(63 / 317)
    # pos 2 is conditional on having reached it (= n_reached - rej[0])
    assert pp[1]["n_reached"] == 63
    assert pp[1]["n_accepted"] == 63 - 58  # 5
    assert pp[1]["p_k"] == pytest.approx(5 / 63)


def test_speculative_report_roundtrip(tmp_path):
    from dflash_llama.inference import (
        SpeculativeReport, parse_speculative_log,
    )

    log = tmp_path / "dmax2.log"
    log.write_text(SAMPLE_LOG)
    parsed = parse_speculative_log(log)

    rpt = SpeculativeReport(
        drafter_label="test-drafter",
        val_loss=5.936,
        training_per_pos=[0.20528, 0.14110, 0.11163],
        training_chained=[0.20528, 0.20528*0.14110, 0.20528*0.14110*0.11163],
    )
    rpt.add_run(2, parsed)

    # Markdown smoke
    md = rpt.markdown()
    assert "DFlash speculative-decode report" in md
    assert "test-drafter" in md
    assert "chain-pos-1" in md
    # Has z-score notation
    assert "z=" in md

    # JSON round-trip
    out = tmp_path / "report.json"
    rpt.to_json(out)
    loaded = json.loads(out.read_text())
    assert loaded["drafter_label"] == "test-drafter"
    assert loaded["runs"]["2"]["n_iter"] == 317  # JSON keys become strings
    # z-scores present on per-position rows
    assert loaded["runs"]["2"]["per_position"][0]["training_p_k"] == pytest.approx(0.20528)
    assert "z" in loaded["runs"]["2"]["per_position"][0]


def test_gguf_export_imports_without_buun():
    """Module must import even when buun-llama-cpp isn't installed."""
    from dflash_llama.inference import (
        export_to_gguf, prep_for_buun_converter,
        register_minimax_fp8_tokenizer_hash, FP8_TOKENIZER_HASH,
    )
    assert callable(export_to_gguf)
    assert callable(prep_for_buun_converter)
    # Hash is the FP8-quant tokenizer hash, not the upstream MiniMax-M2 one
    assert FP8_TOKENIZER_HASH.startswith("a77756c3")


def test_llama_server_cmd_builder():
    """Server config builder produces a sensible argv without launching."""
    from dflash_llama.inference import LlamaServer

    srv = LlamaServer(
        verifier_gguf="/path/to/v.gguf",
        drafter_gguf="/path/to/d.gguf",
        port=9001,
        host="127.0.0.1",
        binary="/usr/bin/true",  # any existing path so resolver doesn't error
    )
    cmd = srv._build_cmd()
    # core flags
    assert "-m" in cmd and "/path/to/v.gguf" in cmd
    assert "-md" in cmd and "/path/to/d.gguf" in cmd
    assert "--port" in cmd and "9001" in cmd
    assert "--spec-type" in cmd and "dflash" in cmd
    assert "--draft-max" in cmd
    # OAI-compat URL
    assert srv.url == "http://127.0.0.1:9001/v1"


def test_llama_server_no_drafter_disables_spec():
    """No drafter ⇒ no --spec-type / -md / etc — pure verifier server."""
    from dflash_llama.inference import LlamaServer

    srv = LlamaServer(
        verifier_gguf="/path/to/v.gguf",
        drafter_gguf=None,
        binary="/usr/bin/true",
    )
    cmd = srv._build_cmd()
    assert "-md" not in cmd
    assert "--spec-type" not in cmd
    assert "--draft-max" not in cmd


def test_load_checkpoint_state_dict_unfuses_te_keys(tmp_path):
    from dflash_llama.training.eval import _load_checkpoint_state_dict

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    save_file(
        {
            "layers.0.self_attn.q_proj.layer_norm_weight": torch.ones(4, dtype=torch.bfloat16),
            "layers.0.mlp.layer_norm_weight": torch.full((4,), 2, dtype=torch.bfloat16),
            "layers.0.mlp.fc1_weight": torch.arange(0, 16, dtype=torch.bfloat16).reshape(4, 4),
            "layers.0.mlp.fc2_weight": torch.arange(0, 8, dtype=torch.bfloat16).reshape(2, 4),
            "layers.0.mlp._extra_state": torch.tensor([1], dtype=torch.uint8),
            "lm_head.layer_norm_weight": torch.full((4,), 3, dtype=torch.bfloat16),
        },
        str(ckpt / "model.safetensors"),
    )

    state_dict = _load_checkpoint_state_dict(ckpt)

    assert "layers.0.input_layernorm.weight" in state_dict
    assert "layers.0.post_attention_layernorm.weight" in state_dict
    assert "layers.0.mlp.gate_proj.weight" in state_dict
    assert "layers.0.mlp.up_proj.weight" in state_dict
    assert "layers.0.mlp.down_proj.weight" in state_dict
    assert "norm.weight" in state_dict
    assert not any(key.endswith("._extra_state") for key in state_dict)


def test_prep_for_buun_converter_normalizes_te_checkpoint_and_expands_vocab(tmp_path):
    from dflash_llama.inference import prep_for_buun_converter

    src = tmp_path / "src"
    out = tmp_path / "out"
    verifier = tmp_path / "verifier"
    src.mkdir()
    verifier.mkdir()

    (verifier / "tokenizer.json").write_text("{}")

    cfg = {
        "draft_vocab_size": 4,
        "block_size": 8,
        "mask_token_id": 200054,
        "aux_hidden_state_layer_ids": [2, 16, 30, 45, 59],
        "transformer_layer_config": {
            "vocab_size": 6,
            "hidden_size": 3,
            "model_type": "qwen3",
        },
    }
    (src / "config.json").write_text(json.dumps(cfg))

    save_file(
        {
            "d2t": torch.tensor([0, 1, 4, 4], dtype=torch.int64),
            "t2d": torch.tensor([1, 1, 1, 1], dtype=torch.bool),
            "lm_head.weight": torch.arange(0, 12, dtype=torch.bfloat16).reshape(4, 3),
            "layers.0.mlp.layer_norm_weight": torch.ones(3, dtype=torch.bfloat16),
            "layers.0.mlp.fc1_weight": torch.arange(0, 12, dtype=torch.bfloat16).reshape(4, 3),
            "layers.0.mlp.fc2_weight": torch.arange(0, 6, dtype=torch.bfloat16).reshape(2, 3),
            "layers.0.mlp._extra_state": torch.tensor([1], dtype=torch.uint8),
        },
        str(src / "model.safetensors"),
    )

    prep_for_buun_converter(src, out, verifier_meta_dir=verifier, verbose=False)

    out_cfg = json.loads((out / "config.json").read_text())
    assert out_cfg["vocab_size"] == 8
    assert out_cfg["draft_vocab_size"] == 8

    with safe_open(str(out / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        assert "layers.0.post_attention_layernorm.weight" in keys
        assert "layers.0.mlp.gate_proj.weight" in keys
        assert "layers.0.mlp.up_proj.weight" in keys
        assert "layers.0.mlp.down_proj.weight" in keys
        assert not any(key.endswith("._extra_state") for key in keys)
        assert list(f.get_tensor("lm_head.weight").shape) == [8, 3]
