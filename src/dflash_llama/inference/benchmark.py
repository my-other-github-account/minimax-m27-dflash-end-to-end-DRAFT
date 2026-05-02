"""Run a DFlash speculative-decode benchmark sweep and produce a
SpeculativeReport with per-position + chain-cumulative accept rates.

Public surface::

    benchmark(verifier_gguf, drafter_gguf, *, ...) -> SpeculativeReport
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Iterable

from .analyze import (
    SpeculativeReport,
    chain_pred_from_val,
    parse_speculative_log,
)

DEFAULT_PROMPT = (
    "Write a Python function that computes the nth Fibonacci number iteratively. "
    "Then explain step by step what your code does, and discuss its time and "
    "space complexity."
)

DEFAULT_BINARY = "/home/user/dflash_clean_repro/build_clean/bin/llama-speculative-simple"


def _resolve_binary(binary: Optional[str | Path]) -> str:
    if binary is not None:
        return str(binary)
    if Path(DEFAULT_BINARY).exists():
        return DEFAULT_BINARY
    found = shutil.which("llama-speculative-simple")
    if found:
        return found
    raise FileNotFoundError(
        f"Could not locate llama-speculative-simple. Pass binary= or build buun-llama-cpp. "
        f"Tried: {DEFAULT_BINARY}"
    )


def _resolve_drafter_label(drafter_gguf: str | Path,
                           explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return Path(drafter_gguf).stem


def benchmark(
    verifier_gguf: str | Path,
    drafter_gguf: str | Path,
    *,
    val_metrics: Optional[str | Path] = None,
    prompt: str = DEFAULT_PROMPT,
    dmax_sweep: Iterable[int] = (2, 4, 7),
    n_tokens: int = 384,
    ctx: int = 8192,
    temperature: float = 0.0,
    n_gpu_layers: int = 99,
    n_gpu_layers_draft: int = 99,
    override_tensor: Optional[str] = "exps=CPU",
    draft_device: Optional[str] = "CUDA0",
    binary: Optional[str | Path] = None,
    log_dir: str | Path = "/tmp/dflash_bench",
    drafter_label: Optional[str] = None,
    progress: bool = True,
    extra_args: Optional[list[str]] = None,
) -> SpeculativeReport:
    """Run llama-speculative-simple over a dmax sweep and return a report.

    Parameters
    ----------
    verifier_gguf : path
        Target model GGUF (e.g. one shard of MiniMax-M2.7-UD-IQ4_XS).
    drafter_gguf : path
        DFlash drafter GGUF produced by ``export_to_gguf``.
    val_metrics : path, optional
        Path to ``val_metrics.json`` from training. If provided, training
        per-position p_i and chained ∏p_i become the prediction baseline
        for z-scores. If omitted, the report omits z-scores.
    prompt : str
        Prompt to bench against (default: a Fibonacci spec — high local
        redundancy, well-characterized prediction baseline).
    dmax_sweep : iterable[int]
        --draft-max values to sweep. Default (2, 4, 7).
    n_tokens : int
        --n (generated tokens per run).
    ctx : int
        --ctx-size for the verifier.
    temperature : float
        --temp (0.0 = greedy).
    n_gpu_layers / n_gpu_layers_draft : int
        --ngl / --ngld.
    override_tensor : str, optional
        Pass-through to llama-speculative-simple --override-tensor (-ot).
        Default ``exps=CPU`` keeps MoE experts off-GPU for IQ4_XS targets.
    draft_device : str, optional
        Pass-through to ``--device-draft``. Default ``CUDA0``.
    binary : path, optional
        llama-speculative-simple binary. Defaults to spark-1's clean build.
    log_dir : path
        Where to write per-dmax logs. Created if missing.
    drafter_label : str, optional
        Label for the report. Default: drafter_gguf stem.
    progress : bool
        Show a tqdm progress bar across the sweep. Falls back to print()
        if tqdm is not installed.
    extra_args : list[str], optional
        Additional flags appended to every llama-speculative-simple call.

    Returns
    -------
    SpeculativeReport with per-dmax metrics and (if val_metrics provided)
    z-scores against training predictions.
    """
    binary = _resolve_binary(binary)
    verifier_gguf = str(verifier_gguf)
    drafter_gguf = str(drafter_gguf)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    label = _resolve_drafter_label(drafter_gguf, drafter_label)

    # Prediction baseline from training
    if val_metrics:
        per_pos, chained, val_loss = chain_pred_from_val(val_metrics)
    else:
        per_pos, chained, val_loss = [], [], None

    report = SpeculativeReport(
        drafter_label=label,
        val_loss=val_loss,
        training_per_pos=per_pos,
        training_chained=chained,
    )

    dmax_list = list(dmax_sweep)

    # Progress bar
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(dmax_list, desc=f"benchmark[{label}]", unit="dmax")
        except ImportError:
            iterator = dmax_list

            def _print_step(d):
                print(f"  -> dmax={d} starting...", flush=True)
            for d in dmax_list:
                _print_step(d)  # we'll print inline below if no tqdm
            iterator = dmax_list
    else:
        iterator = dmax_list

    for dmax in iterator:
        log_path = log_dir / f"{label}_dmax{dmax}.log"
        cmd = [
            binary,
            "-m", verifier_gguf,
            "-md", drafter_gguf,
            "--spec-type", "dflash",
            "--draft-max", str(dmax),
            "-p", prompt,
            "-n", str(n_tokens),
            "-ngl", str(n_gpu_layers),
            "-ngld", str(n_gpu_layers_draft),
            "-c", str(ctx),
            "--temp", str(temperature),
        ]
        if override_tensor:
            cmd += ["-ot", override_tensor]
        if draft_device:
            cmd += ["-devd", draft_device]
        if extra_args:
            cmd += list(extra_args)

        t0 = time.time()
        with open(log_path, "w") as f:
            f.write(f"=== cmd: {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        dt = time.time() - t0

        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-speculative-simple exited rc={proc.returncode} for dmax={dmax}. "
                f"See log: {log_path}"
            )

        parsed = parse_speculative_log(log_path)
        if "_error" in parsed:
            raise RuntimeError(
                f"Could not parse log {log_path} ({parsed['_error']}). "
                f"Check that the binary really fired the DFlash code path."
            )
        parsed["wall_clock_sec"] = dt
        parsed["log_path"] = str(log_path)
        report.add_run(dmax, parsed)

        if not progress:
            print(f"  dmax={dmax} done: n_iter={parsed['n_iter']}, "
                  f"n_accept={parsed['n_accept']} ({dt:.1f}s)", flush=True)

    return report


__all__ = ["benchmark", "DEFAULT_PROMPT", "DEFAULT_BINARY"]
