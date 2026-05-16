"""Helpers for building subprocess launch environments for training flows."""
from __future__ import annotations

import os
from typing import Mapping


def build_training_env(
    *,
    te_fp8_params: bool = False,
    compile_flex_attention: bool = False,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the subprocess env for DFlash training and smoke launches."""
    env = os.environ.copy()
    if compile_flex_attention:
        env.pop("TORCHDYNAMO_DISABLE", None)
        env.pop("TORCH_COMPILE_DISABLE", None)
        env["DFLASH_COMPILE_FLEX"] = "1"
    else:
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["TORCH_COMPILE_DISABLE"] = "1"
        env.pop("DFLASH_COMPILE_FLEX", None)

    if te_fp8_params:
        env["TE_FP8_PARAMS"] = "1"
    else:
        env.pop("TE_FP8_PARAMS", None)

    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    return env


__all__ = ["build_training_env"]
