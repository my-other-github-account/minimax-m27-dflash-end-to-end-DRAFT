"""Run a DFlash speculative-decode benchmark sweep and produce a
SpeculativeReport with per-position + chain-cumulative accept rates.

Public surface::

    benchmark(verifier_gguf, drafter_gguf, *, ...) -> SpeculativeReport
    benchmark_ar_vs_dflash(verifier_gguf, drafter_gguf, *, prompts=...) -> list[dict]
"""
from __future__ import annotations

import json
import os
import re
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

DEFAULT_BINARY = "llama-speculative-simple"


def _resolve_binary(binary: Optional[str | Path]) -> str:
    if binary is not None:
        return str(binary)
    if Path(DEFAULT_BINARY).exists():
        return DEFAULT_BINARY
    found = shutil.which("llama-speculative-simple")
    if found:
        return found
    raise FileNotFoundError(
        f"Could not locate llama-speculative-simple. Pass binary= or put it on PATH. "
        f"Tried command: {DEFAULT_BINARY}"
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
        llama-speculative-simple binary. Defaults to PATH lookup.
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
            # PR #22105's llama.cpp-dflash uses bare --dflash; --spec-type is for
            # upstream's draft-mode merge that has different semantics. If your
            # binary is from a different lineage, override via extra_args.
            "--dflash",
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


# ---------------------------------------------------------------------------
# AR-vs-DFlash apples-to-apples benchmark (multi-prompt suite)
# ---------------------------------------------------------------------------

# 8-prompt entropy-spectrum suite (validated on Kimi-K2.5 and MiniMax-M2.7).
# Stable order; spans low→high output entropy.
DEFAULT_PROMPT_SUITE = {
    "pythag":    "Explain the Pythagorean theorem in one paragraph.",
    "code_fib":  "Write a Python function to compute the nth Fibonacci number, memoized.",
    "code_sort": "Implement quicksort in Rust with generics.",
    "chain":     ("A farmer has 12 chickens and 8 cows. Each chicken lays 2 eggs per day. "
                  "How many eggs in 30 days? Show work."),
    "summary":   "Summarize the plot of Shakespeare's Hamlet in exactly 5 bullet points.",
    "creative":  "Write the opening 150 words of a cyberpunk noir story set on Titan.",
    "factual":   "List 10 notable open-weight LLMs released in 2024, with one sentence each.",
    "translate": ("Translate this to formal French: 'Speculative decoding is a technique "
                  "for accelerating inference...'"),
}


def _resolve_completion_binary(binary: Optional[str | Path], dflash_binary: str) -> str:
    """Find llama-completion sibling of the dflash speculative binary."""
    if binary:
        return str(binary)
    sibling = Path(dflash_binary).parent / "llama-completion"
    if sibling.exists():
        return str(sibling)
    cli = shutil.which("llama-completion")
    if cli:
        return cli
    raise FileNotFoundError(
        "llama-completion not found. Pass ar_binary= explicitly. "
        "(Newer llama.cpp builds rename llama-cli for non-conversation use.)"
    )


def benchmark_ar_vs_dflash(
    verifier_gguf: str | Path,
    drafter_gguf: str | Path,
    *,
    prompts: Optional[dict[str, str]] = None,
    draft_max: int = 16,
    n_tokens: int = 256,
    ctx: int = 4096,
    temperature: float = 0.0,
    n_gpu_layers: int = 99,
    n_gpu_layers_draft: int = 99,
    override_tensor: Optional[str] = "exps=CPU",
    draft_device: Optional[str] = "CUDA0",
    flash_attn: bool = True,
    seed: int = 42,
    dflash_binary: Optional[str | Path] = None,
    ar_binary: Optional[str | Path] = None,
    log_dir: str | Path = "/tmp/dflash_ar_vs_dflash",
    warmup: bool = True,
    progress: bool = True,
) -> list[dict]:
    """Run AR baseline vs DFlash across a prompt suite and return per-prompt results.

    AR baseline uses ``llama-completion`` (not ``llama-cli`` — that's deprecated for
    one-shot in newer builds; see speculative-decode-benchmark-sweep skill Trap 3).
    DFlash uses ``llama-speculative-simple --dflash``.

    Both runs share verifier GGUF, ngl, ctx, n_tokens, temperature, top_k=1, seed —
    the cleanest apples-to-apples comparison the skill recommends.

    Parameters
    ----------
    verifier_gguf, drafter_gguf : path
        Target + draft GGUFs.
    prompts : dict[str, str], optional
        Prompt name → prompt text. Default: ``DEFAULT_PROMPT_SUITE`` (8 prompts
        spanning the entropy spectrum).
    draft_max : int
        DFlash --draft-max.
    n_tokens : int
        --n (tokens to generate per prompt per mode).
    ctx, temperature, n_gpu_layers, n_gpu_layers_draft, override_tensor,
    draft_device, flash_attn, seed : llama.cpp args (see binary --help).
    dflash_binary : path
        ``llama-speculative-simple`` binary. Default: PATH lookup.
    ar_binary : path
        ``llama-completion`` binary. Default: same dir as dflash_binary,
        else PATH.
    log_dir : path
        Per-prompt logs land here.
    warmup : bool
        Run a small AR pass first to page-cache the verifier (avoids paying
        cold mmap cost on the first measured prompt).
    progress : bool
        Show a tqdm bar across prompts.

    Returns
    -------
    list of dicts, one per prompt::

        {
          "name": "pythag",
          "prompt": "...",
          "ar": {"decode_tps": 4.32, "wall_clock_sec": 65.1, "log_path": "..."},
          "dflash": {"decode_tps": 4.81, "n_drafted": 469, "n_accept": 192,
                     "accept_pct": 40.94, "wall_clock_sec": 98.3,
                     "rej_per_pos": [...], "all_ok": 5, "log_path": "..."},
          "speedup": 1.113,            # dflash / ar
        }
    """
    dflash_binary = _resolve_binary(dflash_binary)
    ar_binary = _resolve_completion_binary(ar_binary, dflash_binary)
    verifier_gguf = str(verifier_gguf)
    drafter_gguf = str(drafter_gguf)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if prompts is None:
        prompts = DEFAULT_PROMPT_SUITE

    common = [
        "-ngl", str(n_gpu_layers),
        "-c", str(ctx),
        "-n", str(n_tokens),
        "--temp", str(temperature),
        "--top-k", "1",
        "--seed", str(seed),
    ]
    if flash_attn:
        common += ["-fa", "on"]
    if override_tensor:
        common += ["-ot", override_tensor]

    def _run(cmd: list[str], log_path: Path) -> float:
        t0 = time.time()
        with open(log_path, "w") as f:
            f.write(f"=== cmd: {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
            )
        dt = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(
                f"Run failed rc={proc.returncode}; see {log_path}"
            )
        return dt

    # Warmup (page-cache the verifier shards)
    if warmup:
        warmup_prompt = log_dir / "warmup_prompt.txt"
        warmup_prompt.write_text("warmup probe")
        _run(
            [ar_binary, "-m", verifier_gguf, *common, "-no-cnv",
             "-f", str(warmup_prompt)],
            log_dir / "warmup.log",
        )

    # Iterate prompts
    items = list(prompts.items())
    if progress:
        try:
            from tqdm.auto import tqdm
            items = tqdm(items, desc="ar_vs_dflash", unit="prompt")
        except ImportError:
            pass

    _AR_DECODE_RE = re.compile(
        r"\beval time =\s+([\d.]+)\s+ms /\s+(\d+)\s+runs"
    )

    def _parse_ar_decode_tps(log_path: Path) -> Optional[float]:
        """Parse decode t/s from llama-completion log. Returns None if not found."""
        try:
            text = log_path.read_text()
        except Exception:
            return None
        # Prefer common_perf_print line (matches what llama-completion emits)
        for m in _AR_DECODE_RE.finditer(text):
            ms = float(m.group(1))
            n = int(m.group(2))
            if n > 1 and ms > 0:
                return n / (ms / 1000.0)
        return None

    results: list[dict] = []
    for name, prompt in items:
        prompt_path = log_dir / f"{name}_prompt.txt"
        prompt_path.write_text(prompt)

        # AR (llama-completion)
        ar_log = log_dir / f"{name}_ar.log"
        ar_dt = _run(
            [ar_binary, "-m", verifier_gguf, *common, "-no-cnv",
             "-f", str(prompt_path)],
            ar_log,
        )
        ar_decode = _parse_ar_decode_tps(ar_log)

        # DFlash (llama-speculative-simple --dflash)
        dflash_cmd = [
            dflash_binary,
            "-m", verifier_gguf,
            "-md", drafter_gguf,
            "--dflash", "--draft-max", str(draft_max),
            "-ngld", str(n_gpu_layers_draft),
            *common,
            "-f", str(prompt_path),
        ]
        if draft_device:
            dflash_cmd += ["-devd", draft_device]
        dflash_log = log_dir / f"{name}_dflash.log"
        df_dt = _run(dflash_cmd, dflash_log)

        df_parsed = parse_speculative_log(dflash_log)
        df_decode = None
        # parse_speculative_log might give us throughput; if not, parse from same regex
        df_decode = _parse_ar_decode_tps(dflash_log) or df_parsed.get("decode_tps")

        speedup = (df_decode / ar_decode) if (ar_decode and df_decode) else None
        results.append({
            "name": name,
            "prompt": prompt,
            "ar": {
                "decode_tps": ar_decode,
                "wall_clock_sec": ar_dt,
                "log_path": str(ar_log),
            },
            "dflash": {
                "decode_tps": df_decode,
                "n_drafted": df_parsed.get("n_drafted"),
                "n_accept": df_parsed.get("n_accept"),
                "accept_pct": (
                    100.0 * df_parsed["n_accept"] / df_parsed["n_drafted"]
                    if df_parsed.get("n_drafted") else None
                ),
                "wall_clock_sec": df_dt,
                "rej_per_pos": df_parsed.get("rej"),
                "all_ok": df_parsed.get("all_ok"),
                "log_path": str(dflash_log),
            },
            "speedup": speedup,
        })

    return results


# ---------------------------------------------------------------------------
# Typical/top-k accept 50-prompt reporting helpers
# ---------------------------------------------------------------------------

def al_true(n_pred: int | float, n_acc: int | float) -> Optional[float]:
    """Return verifier-authoritative accepted length: n_pred/(n_pred-n_acc)."""
    if n_pred == 0:
        return None
    denom = n_pred - n_acc
    if denom == 0:
        return float("inf")
    return n_pred / denom


def has_verbatim_repeat_loop(text: str, *, min_ngram: int = 9) -> bool:
    """Detect a repeated contiguous n-gram loop in whitespace-tokenized text."""
    words = text.split()
    if min_ngram <= 0 or len(words) < 2 * min_ngram:
        return False
    seen: set[tuple[str, ...]] = set()
    for i in range(0, len(words) - min_ngram + 1):
        gram = tuple(words[i:i + min_ngram])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def quality_gate_passes(row: dict, *, min_ngram: int = 9) -> bool:
    """Quality gate for the typaccept protocol.

    Required: nonempty response content, positive reasoning token count, and no
    repeated >8-gram verbatim loop.
    """
    content = str(row.get("content") or "").strip()
    reasoning_tokens = row.get("reasoning_tokens") or 0
    if not content:
        return False
    if reasoning_tokens <= 0:
        return False
    if has_verbatim_repeat_loop(content, min_ngram=min_ngram):
        return False
    return True


def summarize_ar_vs_spec_50(rows: Iterable[dict]) -> dict:
    """Aggregate the public 50-prompt AR-vs-spec protocol.

    Each row may use flat keys (``ar_wall_sec``, ``spec_wall_sec``, ``n_pred``,
    ``n_acc``) or nested benchmark output (``ar.wall_clock_sec`` and
    ``dflash.wall_clock_sec`` / ``dflash.n_drafted`` / ``dflash.n_accept``).
    """
    rows = list(rows)

    def get(row: dict, flat: str, nested_obj: str, nested_key: str):
        if flat in row:
            return row.get(flat)
        nested = row.get(nested_obj) or {}
        return nested.get(nested_key)

    total_ar = 0.0
    total_spec = 0.0
    total_pred = 0.0
    total_acc = 0.0
    quality_passed = 0
    for row in rows:
        ar_wall = get(row, "ar_wall_sec", "ar", "wall_clock_sec") or 0.0
        spec_wall = get(row, "spec_wall_sec", "dflash", "wall_clock_sec") or 0.0
        n_pred = get(row, "n_pred", "dflash", "n_drafted") or 0.0
        n_acc = get(row, "n_acc", "dflash", "n_accept") or 0.0
        total_ar += float(ar_wall)
        total_spec += float(spec_wall)
        total_pred += float(n_pred)
        total_acc += float(n_acc)
        if quality_gate_passes(row):
            quality_passed += 1
    return {
        "prompt_count": len(rows),
        "quality_passed": quality_passed,
        "quality_failed": len(rows) - quality_passed,
        "full_wall_speedup": (total_ar / total_spec) if total_spec else None,
        "al_true": al_true(total_pred, total_acc),
        "n_pred": total_pred,
        "n_acc": total_acc,
        "ar_wall_sec": total_ar,
        "spec_wall_sec": total_spec,
    }


__all__ = [
    "benchmark",
    "benchmark_ar_vs_dflash",
    "al_true",
    "has_verbatim_repeat_loop",
    "quality_gate_passes",
    "summarize_ar_vs_spec_50",
    "DEFAULT_PROMPT",
    "DEFAULT_PROMPT_SUITE",
    "DEFAULT_BINARY",
]
