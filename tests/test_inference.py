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
