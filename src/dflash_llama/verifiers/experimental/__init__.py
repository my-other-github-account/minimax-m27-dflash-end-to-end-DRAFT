"""EXPERIMENTAL verifier factories — registered but NOT end-to-end verified.

Everything in this submodule has working shape metadata, but **none of these
have been validated through this library's full ingest path** (trace-gen
→ DFlash training → GGUF export → ``llama-speculative-simple`` benchmark
with measured non-zero pos-2 acceptance against training prediction).

Use at your own risk. If you successfully end-to-end-validate one of these
families, please open a PR moving the factory back to the main
:mod:`dflash_llama.verifiers` package along with a short report (val_loss,
chain-pos-1/-2 measured vs predicted, drafter GGUF SHA).

Validated factories (in main :mod:`dflash_llama.verifiers`):

- ``minimax-m2.7-iq4-xs`` — MiniMax-M2.7 + Unsloth UD-IQ4_XS GGUF + DFlash
  drafter (5L, draft_vocab=32768). Validated 2026-04-30 / 2026-05-02 on
  spark-1, chain-pos-1 measured 19.9–24.0% (predicted 20.5%) with z within
  ±2σ across dmax=2/4/7. The reference path.

- ``minimax-m2.7`` — same family, FP8 quant. Trace-gen retired (FP8
  storage NaN-corrupts hidden states with ``|x|>448``); kept for
  documentation purposes only.

The shape metadata of every factory in this submodule was sourced from
public model cards. The drafter architecture choice (qwen3) was inherited
from the speculators reference impl. Anything beyond shape correctness
(e.g. is the d2t rebake right? does the chosen tap schedule produce
trainable hiddens? does the resulting GGUF round-trip through
``llama-speculative-simple``?) is unproven for these families.
"""
from __future__ import annotations

from .kimi_k25 import kimi_k25  # noqa: F401
from .qwen3 import qwen3, qwen3_4b, qwen3_14b  # noqa: F401
from .deepseek_v4 import deepseek_v4_flash, deepseek_v4_pro  # noqa: F401
from .nemotron3 import nemotron3_super_120b, nemotron3_nano_30b_a3b  # noqa: F401

__all__ = [
    "kimi_k25",
    "qwen3",
    "qwen3_4b",
    "qwen3_14b",
    "deepseek_v4_flash",
    "deepseek_v4_pro",
    "nemotron3_super_120b",
    "nemotron3_nano_30b_a3b",
]
