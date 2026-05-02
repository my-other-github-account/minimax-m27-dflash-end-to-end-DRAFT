"""Inference utilities — GGUF export, llama.cpp OpenAI-compat server,
speculative-decode benchmarking with per-position + chain-cumulative
acceptance reporting.

Public surface
--------------

- :func:`export_to_gguf` — convert a speculators-format DFlash drafter
  checkpoint into a buun-loadable GGUF (rebakes lm_head, registers
  tokenizer hash, runs buun's converter).
- :class:`LlamaServer` — context manager that runs llama-server with
  optional DFlash speculative decoding; exposes an OpenAI-compatible
  endpoint at ``http://<host>:<port>/v1``.
- :func:`benchmark` — sweep ``llama-speculative-simple`` across draft-max
  values; returns a :class:`SpeculativeReport` with per-position p_k and
  chain-cumulative ∏p_i with z-scores against training prediction.
- :func:`parse_speculative_log` — low-level log parser.
- :func:`chain_pred_from_val` — read training ``val_metrics.json`` →
  per-position + chained predictions.
"""
from __future__ import annotations

from .gguf_export import (
    export_to_gguf,
    prep_for_buun_converter,
    register_minimax_fp8_tokenizer_hash,
    verify_gguf_metadata,
    FP8_TOKENIZER_HASH,
)
from .server import LlamaServer
from .benchmark import benchmark, DEFAULT_PROMPT
from .analyze import (
    SpeculativeReport,
    parse_speculative_log,
    chain_pred_from_val,
    chain_measured,
    per_position_conditional,
    z_score,
)

__all__ = [
    # GGUF export
    "export_to_gguf",
    "prep_for_buun_converter",
    "register_minimax_fp8_tokenizer_hash",
    "verify_gguf_metadata",
    "FP8_TOKENIZER_HASH",
    # Server
    "LlamaServer",
    # Benchmark + report
    "benchmark",
    "DEFAULT_PROMPT",
    "SpeculativeReport",
    "parse_speculative_log",
    "chain_pred_from_val",
    "chain_measured",
    "per_position_conditional",
    "z_score",
]
