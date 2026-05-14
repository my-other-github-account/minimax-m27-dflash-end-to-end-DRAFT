#!/usr/bin/env python3
"""Run the Phase 2 perf sweep on a single Spark host.

This harness is meant to run *on* ``spark-2`` after the patched speculators
checkout and the TE/Liger runtime are in place. It launches one cell at a time,
waits until step 300's in-epoch validation lands, then terminates the training
process and records throughput / memory / stability guardrails.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from datasets import concatenate_datasets, load_from_disk
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

BASE_TOTAL_SEQ_LEN = 2048
MAX_ANCHORS = 1024
NUM_LAYERS = 6
LR = 3e-4
WARMUP_STEPS = 100
LOG_FREQ = 5
TARGET_STEP_DEFAULT = 300
VAL_MAX_BATCHES = 80
MICRO_BS_DEFAULT = (1, 2, 3, 4, 6, 8)
OOM_MARKERS = (
    "out of memory",
    "CUDA error: out of memory",
    "CUDA out of memory",
)
DEFAULT_VERIFIER_PATH = Path("/home/user/iq4_full_run/verifier_meta_v11")
DEFAULT_VOCAB_DATA_PATH = Path("/home/user/iq4_full_run/iq4_v10/prompts")
DEFAULT_HIDDEN_STATES_PATH = Path("/home/user/iq4_full_run/iq4_v10/hidden_states")
DEFAULT_SPECULATORS_REPO = Path("/home/user/speculators-phase2")
DEFAULT_TRAIN_SCRIPT = Path("/home/user/speculators-phase2/scripts/train.py")
DEFAULT_PYTHON_BIN = Path("/home/user/venvs/vllm/bin/python")
DEFAULT_TORCHRUN_BIN = Path("/home/user/venvs/vllm/bin/torchrun")


@dataclass(frozen=True)
class SweepConfig:
    config_id: str
    description: str
    fp8_recipe_kind: str
    te_use_fused: bool
    disable_extended_te: bool
    liger_fused_linear_ce: bool
    liger_rope: bool
    liger_rms_norm: bool = False
    te_use_dpa: bool = False
    dflash_fused_ce_chunk: int = 0
    te_fp8_params: bool = False
    nvte_fused_attn: bool = False
    compile_flex_attention: bool = False


CONFIGS: dict[str, SweepConfig] = {
    "C1": SweepConfig(
        config_id="C1",
        description="bf16 baseline (v11)",
        fp8_recipe_kind="",
        te_use_fused=False,
        disable_extended_te=False,
        liger_fused_linear_ce=False,
        liger_rope=False,
    ),
    "C3": SweepConfig(
        config_id="C3",
        description="FP8 current v12-stable (TE MLP fused)",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=True,
        liger_fused_linear_ce=False,
        liger_rope=False,
    ),
    "C4": SweepConfig(
        config_id="C4",
        description="FP8 full TE fusion",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=False,
        liger_rope=False,
    ),
    "C5": SweepConfig(
        config_id="C5",
        description="FP8 full TE + Liger fused linear CE",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=True,
        liger_rope=False,
    ),
    "C6": SweepConfig(
        config_id="C6",
        description="FP8 full TE + Liger fused linear CE + RoPE",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=True,
        liger_rope=True,
    ),
    "C7": SweepConfig(
        config_id="C7",
        description="FP8 full TE fusion + TE DPA",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=False,
        liger_rope=False,
        te_use_dpa=True,
        nvte_fused_attn=True,
    ),
    "C8": SweepConfig(
        config_id="C8",
        description="FP8 full TE fusion + chunked DFlash CE (128)",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=False,
        liger_rope=False,
        dflash_fused_ce_chunk=128,
    ),
    "C9": SweepConfig(
        config_id="C9",
        description="FP8 full TE fusion + fp8_model_init params",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=False,
        liger_rope=False,
        te_fp8_params=True,
    ),
    "C10": SweepConfig(
        config_id="C10",
        description="FP8 full TE fusion + DPA + chunked CE + fp8 params",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=False,
        liger_rope=False,
        te_use_dpa=True,
        dflash_fused_ce_chunk=128,
        te_fp8_params=True,
        nvte_fused_attn=True,
    ),
    "C11": SweepConfig(
        config_id="C11",
        description="FP8 full TE fusion + chunked CE + fp8 params (no DPA)",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=False,
        liger_rope=False,
        te_use_dpa=False,
        dflash_fused_ce_chunk=128,
        te_fp8_params=True,
    ),
    "C12": SweepConfig(
        config_id="C12",
        description="FP8 full TE fusion + chunked CE + fp8 params bs=6 (no DPA)",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=False,
        liger_rope=False,
        te_use_dpa=False,
        dflash_fused_ce_chunk=128,
        te_fp8_params=True,
    ),
    "C13": SweepConfig(
        config_id="C13",
        description="FP8 full TE fusion + TE DPA with NVTE fused attention enabled",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=False,
        liger_rope=False,
        te_use_dpa=True,
        nvte_fused_attn=True,
    ),
    "C14": SweepConfig(
        config_id="C14",
        description="FP8 full TE fusion + DPA + chunked CE + fp8 params + NVTE fused attention",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=False,
        liger_rope=False,
        te_use_dpa=True,
        dflash_fused_ce_chunk=128,
        te_fp8_params=True,
        nvte_fused_attn=True,
    ),
    "C15": SweepConfig(
        config_id="C15",
        description="FP8 full TE fusion + fp8 params + compiled flex attention",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=False,
        liger_rope=False,
        te_fp8_params=True,
        compile_flex_attention=True,
    ),
    "C16": SweepConfig(
        config_id="C16",
        description="FP8 full TE fusion + chunked CE + fp8 params + compiled flex attention",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=False,
        liger_rope=False,
        dflash_fused_ce_chunk=128,
        te_fp8_params=True,
        compile_flex_attention=True,
    ),
    "C17": SweepConfig(
        config_id="C17",
        description="NVFP4 safe recipe on full TE fusion",
        fp8_recipe_kind="nvfp4_safe",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=False,
        liger_rope=False,
    ),
    "C18": SweepConfig(
        config_id="C18",
        description="FP8 full TE fusion + fp8 params + low-level Liger CE",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=True,
        liger_rope=False,
        te_fp8_params=True,
    ),
    "C19": SweepConfig(
        config_id="C19",
        description="FP8 full TE fusion + fp8 params + Liger CE + compiled flex attention",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=True,
        liger_rope=False,
        te_fp8_params=True,
        compile_flex_attention=True,
    ),
    "C20": SweepConfig(
        config_id="C20",
        description="FP8 full TE fusion + fp8 params + Liger CE + Liger RoPE",
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        disable_extended_te=False,
        liger_fused_linear_ce=True,
        liger_rope=True,
        te_fp8_params=True,
    ),
}


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _hhmmss_to_seconds(value: str) -> int:
    hh, mm, ss = value.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def _tau_from_val_metrics(metrics: dict[str, float]) -> float | None:
    cumulative = 1.0
    running = 1.0
    found = False
    for pos in range(1, 8):
        key = f"position {pos} acc_epoch"
        if key not in metrics:
            break
        found = True
        running *= float(metrics[key])
        cumulative += running
    return cumulative if found else None


def _parse_train_points(log_text: str) -> list[dict[str, float | int | str]]:
    points: list[dict[str, float | int | str]] = []
    current_loss: float | None = None
    current_ts: str | None = None
    current_full_acc: float | None = None
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        ts_match = re.match(r"^\[(\d\d:\d\d:\d\d)\]", line)
        if ts_match:
            current_ts = ts_match.group(1)
        loss_match = re.search(r"train/loss=([0-9.eE+-]+|nan)", line)
        if loss_match:
            try:
                current_loss = float(loss_match.group(1))
            except ValueError:
                current_loss = float("nan")
        acc_match = re.search(r"train/full_acc=([0-9.eE+-]+)", line)
        if acc_match:
            current_full_acc = float(acc_match.group(1))
        step_match = re.search(r"global_step=(\d+)", line)
        if step_match and current_ts is not None:
            points.append(
                {
                    "timestamp": current_ts,
                    "step": int(step_match.group(1)),
                    "loss": current_loss,
                    "full_acc": current_full_acc,
                }
            )
    return points


def _seconds_between(start_hms: str, end_hms: str) -> int:
    start = _hhmmss_to_seconds(start_hms)
    end = _hhmmss_to_seconds(end_hms)
    if end < start:
        end += 24 * 3600
    return end - start


def _extract_event_time(log_text: str, pattern: str) -> str | None:
    match = re.search(pattern, log_text)
    if match:
        return match.group(1)
    return None


def _parse_fp8_receipt(log_text: str) -> dict[str, bool]:
    return {
        "has_fp8_line": "[FP8]" in log_text,
        "split_accumulator_ok": log_text.count("use_split_accumulator=True") >= 3,
        "has_te_mlp": "te_layernorm_mlp" in log_text,
        "has_te_ln_linear": "te_layernorm_linear" in log_text,
        "has_liger": "LIGER_VERSION=" in log_text,
    }


def _status_from_log(log_text: str) -> str:
    lowered = log_text.lower()
    if any(marker.lower() in lowered for marker in OOM_MARKERS):
        return "OOM"
    if "train/loss=nan" in lowered or " nan" in lowered or "NaN-SKIP" in log_text:
        return "NaN"
    return "other"


def _build_default_data_sources_json(path: Path) -> Path:
    payload = [
        {
            "name": "iq4_v10",
            "datapath": "/home/user/iq4_full_run/iq4_v10/prompts",
            "hidden_states_path": "/home/user/iq4_full_run/iq4_v10/hidden_states",
        },
        {
            "name": "train_paired_v3",
            "datapath": "/home/user/iq4_full_run/QUARANTINED_FP8_20260502_155004/train_paired_v3/prompts",
            "hidden_states_path": "/home/user/iq4_full_run/QUARANTINED_FP8_20260502_155004/train_paired_v3/hidden_states",
        },
        {
            "name": "v4_pool",
            "datapath": "/home/user/iq4_full_run/v4_pool/prompts_renum",
            "hidden_states_path": "/home/user/iq4_full_run/v4_pool/hidden_states_renum",
        },
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _materialize_combined_pool(
    *,
    data_sources_path: Path,
    dest_root: Path,
) -> tuple[Path, Path]:
    prompts_out = dest_root / "combined_prompts"
    hidden_states_out = dest_root / "combined_hidden_states"
    if prompts_out.exists() and hidden_states_out.exists():
        return prompts_out, hidden_states_out

    sources = json.loads(data_sources_path.read_text())
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"invalid data sources file: {data_sources_path}")

    def _keep_indices_for_anchorable_rows(ds, *, block_size: int = 8) -> list[int]:
        keep_indices: list[int] = []
        for idx, row in enumerate(ds):
            if any(row["loss_mask"][:-block_size]):
                keep_indices.append(idx)
        return keep_indices

    def _filter_indices_with_finite_hidden_states(
        indices: list[int],
        *,
        hidden_states_root: Path,
    ) -> list[int]:
        kept: list[int] = []
        bad = 0
        for idx in indices:
            hs_path = hidden_states_root / f"hs_{idx}.safetensors"
            hs = load_file(str(hs_path))["hidden_states"].float()
            if hs.isfinite().all():
                kept.append(idx)
            else:
                bad += 1
        if bad:
            print(
                f"[perf_sweep] filtering {bad} non-finite hidden-state rows "
                f"from {hidden_states_root}"
            )
        return kept

    common_cols = ["input_ids", "loss_mask", "seq_len"]
    datasets = []
    source_keep_indices: list[list[int]] = []
    for src in sources:
        ds = load_from_disk(src["datapath"])
        keep_indices = _keep_indices_for_anchorable_rows(ds)
        if len(keep_indices) != len(ds):
            print(
                f"[perf_sweep] filtering {len(ds) - len(keep_indices)} zero-anchor rows "
                f"from {src['datapath']}"
            )
        if "train_paired_v3" in str(src.get("name", "")) or "train_paired_v3" in str(src["hidden_states_path"]):
            keep_indices = _filter_indices_with_finite_hidden_states(
                keep_indices,
                hidden_states_root=Path(src["hidden_states_path"]),
            )
        if len(keep_indices) != len(ds):
            ds = ds.select(keep_indices)
        source_keep_indices.append(keep_indices)
        keep = [col for col in common_cols if col in ds.column_names]
        datasets.append(ds.select_columns(keep))
    combined = concatenate_datasets(datasets)
    combined.set_format(type="torch", columns=common_cols, output_all_columns=True)
    prompts_out.parent.mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(str(prompts_out))

    first_prompts = Path(sources[0]["datapath"])
    for name in ("d2t.npy", "t2d.npy", "token_freq.pt", "dataset_info.json", "state.json"):
        src = first_prompts / name
        dst = prompts_out / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    hidden_states_out.mkdir(parents=True, exist_ok=True)
    out_idx = 0
    for src, keep_indices in zip(sources, source_keep_indices, strict=True):
        hs_root = Path(src["hidden_states_path"])
        hs_files = {
            int(p.stem.split("_")[1]): p
            for p in hs_root.glob("hs_*.safetensors")
        }
        for row_idx in keep_indices:
            hs_file = hs_files.get(row_idx)
            if hs_file is None:
                raise FileNotFoundError(f"missing hidden-state shard hs_{row_idx}.safetensors under {hs_root}")
            dst = hidden_states_out / f"hs_{out_idx}.safetensors"
            if not dst.exists():
                dst.symlink_to(hs_file)
            out_idx += 1
    return prompts_out, hidden_states_out


def _build_env(
    *,
    speculators_repo: Path,
    disable_extended_te: bool,
    te_use_dpa: bool,
    dflash_fused_ce_chunk: int,
    te_fp8_params: bool,
    nvte_fused_attn: bool,
    compile_flex_attention: bool,
) -> dict[str, str]:
    env = os.environ.copy()
    py_parts = [str(SRC_ROOT), str(speculators_repo / "src")]
    if env.get("PYTHONPATH"):
        py_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_parts)
    venv_site = Path("/home/user/venvs/vllm/lib/python3.12/site-packages/nvidia")
    nccl_lib = str(venv_site / "nccl" / "lib")
    cudnn_lib = str(venv_site / "cudnn" / "lib")
    nccl_inc = str(venv_site / "nccl" / "include")
    cudnn_inc = str(venv_site / "cudnn" / "include")
    cuda_inc = str(venv_site / "cu13" / "include")
    env["LD_LIBRARY_PATH"] = ":".join(
        part for part in (nccl_lib, cudnn_lib, env.get("LD_LIBRARY_PATH", "")) if part
    )
    env["CPATH"] = ":".join(
        part for part in (nccl_inc, cudnn_inc, cuda_inc, env.get("CPATH", "")) if part
    )
    if compile_flex_attention:
        env.pop("TORCHDYNAMO_DISABLE", None)
        env.pop("TORCH_COMPILE_DISABLE", None)
    else:
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["TORCH_COMPILE_DISABLE"] = "1"
    env["NVTE_FUSED_ATTN"] = "1" if nvte_fused_attn else "0"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if te_use_dpa:
        env["TE_USE_DPA"] = "1"
    else:
        env.pop("TE_USE_DPA", None)
    if dflash_fused_ce_chunk > 0:
        env["DFLASH_FUSED_CE_CHUNK"] = str(dflash_fused_ce_chunk)
    else:
        env.pop("DFLASH_FUSED_CE_CHUNK", None)
    if te_fp8_params:
        env["TE_FP8_PARAMS"] = "1"
    else:
        env.pop("TE_FP8_PARAMS", None)
    if compile_flex_attention:
        env["DFLASH_COMPILE_FLEX"] = "1"
    else:
        env.pop("DFLASH_COMPILE_FLEX", None)
    if disable_extended_te:
        env["TE_DISABLE_EXTENDED_FUSION"] = "1"
    else:
        env.pop("TE_DISABLE_EXTENDED_FUSION", None)
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
    ):
        env.pop(key, None)
    return env


def _build_train_cmd(
    *,
    python_bin: Path,
    torchrun_bin: Path,
    train_script: Path,
    verifier_path: Path,
    vocab_data_path: Path,
    default_hidden_states_path: Path,
    data_sources_path: Path | None,
    save_path: Path,
    config: SweepConfig,
    micro_bs: int,
    master_port: int,
    max_anchors: int,
    val_every_steps: int,
    val_in_epoch_max_batches: int,
    seq_len_per_micro: int,
    num_workers: int,
    prefetch_factor: int,
) -> list[str]:
    use_torchrun = config.fp8_recipe_kind in ("", "bf16", "none")
    cmd: list[str]
    if use_torchrun:
        cmd = [
            str(torchrun_bin),
            f"--master_port={master_port}",
            "--nproc-per-node=1",
            str(train_script),
        ]
    else:
        cmd = [str(python_bin), str(train_script)]
    cmd += [
        "--speculator-type",
        "dflash",
        "--verifier-name-or-path",
        str(verifier_path),
        "--data-path",
        str(vocab_data_path),
        "--hidden-states-path",
        str(default_hidden_states_path),
        "--save-path",
        str(save_path),
        "--epochs",
        "1",
        "--total-seq-len",
        str(seq_len_per_micro * micro_bs),
        "--max-anchors",
        str(max_anchors),
        "--num-workers",
        str(num_workers),
        "--on-missing",
        "skip",
        "--target-layer-ids",
        "2",
        "16",
        "30",
        "45",
        "59",
        "--draft-arch",
        "qwen3",
        "--draft-hidden-act",
        "silu",
        "--mask-token-id",
        "200054",
        "--block-size",
        "8",
        "--hidden-states-dtype",
        "bfloat16",
        "--num-layers",
        str(NUM_LAYERS),
        "--draft-vocab-size",
        "32768",
        "--lr",
        str(LR),
        "--scheduler-warmup-steps",
        str(WARMUP_STEPS),
        "--noise-std",
        "0.05",
        "--log-freq",
        str(LOG_FREQ),
    ]
    if num_workers > 0:
        cmd += ["--prefetch-factor", str(prefetch_factor)]
    if data_sources_path is not None:
        cmd += ["--data-sources", str(data_sources_path)]
    if val_every_steps > 0:
        cmd += ["--val-every-steps", str(val_every_steps)]
    if val_in_epoch_max_batches > 0:
        cmd += ["--val-in-epoch-max-batches", str(val_in_epoch_max_batches)]
    if config.fp8_recipe_kind:
        cmd += ["--fp8-recipe-kind", config.fp8_recipe_kind]
    if config.te_use_fused:
        cmd.append("--te-use-fused")
    if config.liger_fused_linear_ce:
        cmd.append("--liger-fused-linear-ce")
    if config.liger_rope:
        cmd.append("--liger-rope")
    if config.liger_rms_norm:
        cmd.append("--liger-rms-norm")
    return cmd


class MemoryPoller(threading.Thread):
    def __init__(self, interval_sec: float = 5.0):
        super().__init__(daemon=True)
        self.interval_sec = interval_sec
        self.max_mem_gb = 0.0
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
                first = out.splitlines()[0].strip()
                mem_gb = float(first) / 1024.0
                self.max_mem_gb = max(self.max_mem_gb, mem_gb)
            except Exception:
                pass
            self._stop_event.wait(self.interval_sec)

    def stop(self) -> None:
        self._stop_event.set()


@dataclass
class CellResult:
    config: str
    description: str
    micro_bs: int
    throughput_tok_s: float | None
    peak_gpu_mem_gb: float | None
    step_time_ms: float | None
    loss_0: float | None
    loss_final: float | None
    loss_descending: bool
    nan_skips: int
    tau: float | None
    delta_vs_bf16_bs1: float | None
    status: str
    smoke_ok: bool
    fp8_receipt_ok: bool
    split_accumulator_ok: bool
    fail_reason: str | None
    log_path: str
    val_json: str | None
    wall_time_sec: float | None


def _wait_for_step_target(
    proc: subprocess.Popen[bytes],
    *,
    log_path: Path,
    target_step: int,
    val_json_path: Path | None,
    timeout_sec: int,
) -> tuple[bool, str]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            if val_json_path is not None and val_json_path.exists():
                return True, text
            if f"global_step={target_step}" in text and val_json_path is None:
                return True, text
        if proc.poll() is not None:
            break
        time.sleep(2)
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    return False, text


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_smoke(
    *,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    timeout_sec: int,
) -> tuple[bool, str]:
    full_cmd = ["timeout", str(timeout_sec), *cmd]
    with log_path.open("wb") as logf:
        proc = subprocess.run(full_cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    text = log_path.read_text(errors="replace")
    timed_out = proc.returncode == 124
    saw_step = "global_step=" in text
    clean = "NaN-SKIP" not in text and "train/loss=nan" not in text.lower()
    return timed_out and saw_step and clean, text


def _run_cell(
    *,
    config: SweepConfig,
    micro_bs: int,
    args: argparse.Namespace,
    baseline: CellResult | None,
) -> CellResult:
    run_root = args.run_root / f"{config.config_id}_bs{micro_bs}"
    run_root.mkdir(parents=True, exist_ok=True)
    smoke_log = run_root / "smoke.log"
    train_log = run_root / "train.log"
    save_path = run_root / "ckpt"
    save_path.mkdir(parents=True, exist_ok=True)
    val_json = None
    if args.val_every_steps > 0 and args.val_in_epoch_max_batches > 0:
        val_json = save_path / "val_in_epoch" / f"step_{args.target_step:08d}.json"
    env = _build_env(
        speculators_repo=args.speculators_repo,
        disable_extended_te=config.disable_extended_te,
        te_use_dpa=config.te_use_dpa,
        dflash_fused_ce_chunk=config.dflash_fused_ce_chunk,
        te_fp8_params=config.te_fp8_params,
        nvte_fused_attn=config.nvte_fused_attn,
        compile_flex_attention=config.compile_flex_attention,
    )
    smoke_ok = True
    smoke_text = ""
    if not args.skip_smoke:
        smoke_cmd = _build_train_cmd(
            python_bin=args.python_bin,
            torchrun_bin=args.torchrun_bin,
            train_script=args.train_script,
            verifier_path=args.verifier_path,
            vocab_data_path=args.vocab_data_path,
            default_hidden_states_path=args.default_hidden_states_path,
            data_sources_path=args.data_sources_path,
            save_path=save_path / "smoke",
            config=config,
            micro_bs=micro_bs,
            master_port=args.base_port + micro_bs,
        max_anchors=64,
        val_every_steps=0,
        val_in_epoch_max_batches=0,
        seq_len_per_micro=args.seq_len_per_micro,
    )
        smoke_ok, smoke_text = _run_smoke(
            cmd=smoke_cmd,
            env=env,
            log_path=smoke_log,
            timeout_sec=args.smoke_timeout_sec,
        )
    fp8_receipt_ok = True
    split_acc_ok = True
    fp8_receipt = _parse_fp8_receipt(smoke_text)
    if not args.skip_smoke and config.fp8_recipe_kind:
        fp8_receipt_ok = fp8_receipt["has_fp8_line"]
        split_acc_ok = fp8_receipt["split_accumulator_ok"]
        if config.te_use_fused and not fp8_receipt["has_te_mlp"]:
            fp8_receipt_ok = False
        if not config.disable_extended_te and not fp8_receipt["has_te_ln_linear"]:
            fp8_receipt_ok = False
        if (config.liger_fused_linear_ce or config.liger_rope) and not fp8_receipt["has_liger"]:
            fp8_receipt_ok = False
    if not smoke_ok or not fp8_receipt_ok or not split_acc_ok:
        reason = "smoke_failed"
        if smoke_ok and not fp8_receipt_ok:
            reason = "fp8_receipt_missing"
        if smoke_ok and fp8_receipt_ok and not split_acc_ok:
            reason = "split_accumulator_missing"
        return CellResult(
            config=config.config_id,
            description=config.description,
            micro_bs=micro_bs,
            throughput_tok_s=None,
            peak_gpu_mem_gb=None,
            step_time_ms=None,
            loss_0=None,
            loss_final=None,
            loss_descending=False,
            nan_skips=smoke_text.count("NaN-SKIP"),
            tau=None,
            delta_vs_bf16_bs1=None,
            status="other",
            smoke_ok=smoke_ok,
            fp8_receipt_ok=fp8_receipt_ok,
            split_accumulator_ok=split_acc_ok,
            fail_reason=reason,
            log_path=str(smoke_log),
            val_json=None,
            wall_time_sec=None,
        )

    train_cmd = _build_train_cmd(
        python_bin=args.python_bin,
        torchrun_bin=args.torchrun_bin,
        train_script=args.train_script,
        verifier_path=args.verifier_path,
        vocab_data_path=args.vocab_data_path,
        default_hidden_states_path=args.default_hidden_states_path,
        data_sources_path=args.data_sources_path,
        save_path=save_path,
        config=config,
        micro_bs=micro_bs,
        master_port=args.base_port + 100 + micro_bs,
        max_anchors=args.max_anchors,
        val_every_steps=args.val_every_steps,
        val_in_epoch_max_batches=args.val_in_epoch_max_batches,
        seq_len_per_micro=args.seq_len_per_micro,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    poller = MemoryPoller(interval_sec=5.0)
    start_time = time.time()
    with train_log.open("wb") as logf:
        proc = subprocess.Popen(
            train_cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        poller.start()
        reached, log_text = _wait_for_step_target(
            proc,
            log_path=train_log,
            target_step=args.target_step,
            val_json_path=val_json,
            timeout_sec=args.cell_timeout_sec,
        )
        end_time = time.time()
        _terminate_process_group(proc)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc)
    poller.stop()
    log_text = train_log.read_text(errors="replace") if train_log.exists() else log_text
    if args.skip_smoke and config.fp8_recipe_kind:
        fp8_receipt = _parse_fp8_receipt(log_text)
        fp8_receipt_ok = fp8_receipt["has_fp8_line"]
        split_acc_ok = fp8_receipt["split_accumulator_ok"]
        if config.te_use_fused and not fp8_receipt["has_te_mlp"]:
            fp8_receipt_ok = False
        if not config.disable_extended_te and not fp8_receipt["has_te_ln_linear"]:
            fp8_receipt_ok = False
        if (config.liger_fused_linear_ce or config.liger_rope) and not fp8_receipt["has_liger"]:
            fp8_receipt_ok = False
    status = "OK" if reached else _status_from_log(log_text)
    points = _parse_train_points(log_text)
    loss_0 = None
    loss_final = None
    step_time_ms = None
    throughput = None
    tau = None
    fail_reason = None
    if points:
        loss_0 = next((p["loss"] for p in points if p["step"] == 0), points[0]["loss"])
        loss_final = points[-1]["loss"]
        usable = [p for p in points if isinstance(p["step"], int) and int(p["step"]) > 0]
        if len(usable) >= 2:
            deltas = []
            for prev, cur in zip(usable, usable[1:]):
                dt = _seconds_between(str(prev["timestamp"]), str(cur["timestamp"]))
                ds = int(cur["step"]) - int(prev["step"])
                if dt > 0 and ds > 0:
                    deltas.append((dt / ds) * 1000.0)
            if deltas:
                step_time_ms = sum(deltas) / len(deltas)
        start_point = next((p for p in points if p["step"] == 0), points[0])
        target_point = next(
            (p for p in points if p["step"] == args.target_step),
            None,
        )
        trigger_time = (
            str(target_point["timestamp"])
            if target_point is not None
            else _extract_event_time(
                log_text,
                rf"\[(\d\d:\d\d:\d\d)\][^\n]*In-epoch validation triggered at[^\n]*global_step={args.target_step}",
            ) or _extract_event_time(
                log_text,
                rf"\[(\d\d:\d\d:\d\d)\][^\n]*In-epoch val_metrics saved",
            )
        )
        if trigger_time:
            elapsed = _seconds_between(str(start_point["timestamp"]), trigger_time)
            if elapsed > 0:
                throughput = (
                    args.seq_len_per_micro * args.max_anchors * micro_bs * args.target_step
                ) / elapsed
    nan_skips = log_text.count("NaN-SKIP")
    if val_json is not None and val_json.exists():
        val_metrics = json.loads(val_json.read_text())
        tau = _tau_from_val_metrics(val_metrics)
    loss_descending = (
        loss_0 is not None
        and loss_final is not None
        and loss_final < loss_0
    )
    if not fp8_receipt_ok:
        status = "other"
        fail_reason = "fp8_receipt_missing"
    elif not split_acc_ok:
        status = "other"
        fail_reason = "split_accumulator_missing"
    elif not reached:
        fail_reason = "target_step_not_reached"
    elif nan_skips > 5:
        status = "NaN"
        fail_reason = "nan_skip_limit_exceeded"
    elif not loss_descending:
        status = "other"
        fail_reason = "loss_not_descending"
    elif config.fp8_recipe_kind and not fp8_receipt["split_accumulator_ok"]:
        status = "other"
        fail_reason = "split_accumulator_missing"
    baseline_tok_s = None
    if baseline is not None:
        baseline_tok_s = baseline.throughput_tok_s
    elif args.baseline_throughput_tok_s:
        baseline_tok_s = args.baseline_throughput_tok_s
    if baseline_tok_s and throughput:
        delta_vs_bf16_bs1 = throughput / baseline_tok_s
    else:
        delta_vs_bf16_bs1 = None
    if baseline is not None and baseline.tau is not None and tau is not None and config.config_id != "C1":
        if tau < baseline.tau - 0.10:
            status = "other"
            fail_reason = "tau_direction_fail"
    return CellResult(
        config=config.config_id,
        description=config.description,
        micro_bs=micro_bs,
        throughput_tok_s=throughput,
        peak_gpu_mem_gb=round(poller.max_mem_gb, 3) if poller.max_mem_gb else None,
        step_time_ms=round(step_time_ms, 3) if step_time_ms is not None else None,
        loss_0=float(loss_0) if loss_0 is not None else None,
        loss_final=float(loss_final) if loss_final is not None else None,
        loss_descending=bool(loss_descending),
        nan_skips=nan_skips,
        tau=round(tau, 6) if tau is not None else None,
        delta_vs_bf16_bs1=round(delta_vs_bf16_bs1, 6) if delta_vs_bf16_bs1 is not None else None,
        status=status,
        smoke_ok=smoke_ok,
        fp8_receipt_ok=fp8_receipt_ok,
        split_accumulator_ok=split_acc_ok,
        fail_reason=fail_reason,
        log_path=str(train_log),
        val_json=str(val_json) if val_json is not None and val_json.exists() else None,
        wall_time_sec=round(end_time - start_time, 3),
    )


def _render_markdown(results: Iterable[CellResult]) -> str:
    lines = [
        "| config | bs | tok/s | peak_mem_GB | step_ms | loss_300 | nan_skips | tau | status | delta_vs_bf16_bs1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for result in results:
        lines.append(
            "| {config} | {bs} | {tok} | {mem} | {step_ms} | {loss} | {nan} | {tau} | {status} | {delta} |".format(
                config=result.config,
                bs=result.micro_bs,
                tok="--" if result.throughput_tok_s is None else f"{result.throughput_tok_s:.3f}",
                mem="--" if result.peak_gpu_mem_gb is None else f"{result.peak_gpu_mem_gb:.3f}",
                step_ms="--" if result.step_time_ms is None else f"{result.step_time_ms:.3f}",
                loss="--" if result.loss_final is None else f"{result.loss_final:.6f}",
                nan=result.nan_skips,
                tau="--" if result.tau is None else f"{result.tau:.6f}",
                status=result.status,
                delta="--" if result.delta_vs_bf16_bs1 is None else f"{result.delta_vs_bf16_bs1:.3f}x",
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", action="append", choices=sorted(CONFIGS), help="Config(s) to run; default = full sweep order")
    parser.add_argument("--micro-bs", nargs="+", type=int, default=list(MICRO_BS_DEFAULT))
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "repro" / "artifacts" / f"perf_sweep_{_timestamp_utc()}")
    parser.add_argument("--results-prefix", type=str, default=f"perf_sweep_{_timestamp_utc()}")
    parser.add_argument("--speculators-repo", type=Path, default=DEFAULT_SPECULATORS_REPO)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--torchrun-bin", type=Path, default=DEFAULT_TORCHRUN_BIN)
    parser.add_argument("--verifier-path", type=Path, default=DEFAULT_VERIFIER_PATH)
    parser.add_argument("--vocab-data-path", type=Path, default=DEFAULT_VOCAB_DATA_PATH)
    parser.add_argument("--default-hidden-states-path", type=Path, default=DEFAULT_HIDDEN_STATES_PATH)
    parser.add_argument("--data-sources-path", type=Path, default=None)
    parser.add_argument("--base-port", type=int, default=29540)
    parser.add_argument("--max-anchors", type=int, default=MAX_ANCHORS)
    parser.add_argument("--seq-len-per-micro", type=int, default=BASE_TOTAL_SEQ_LEN)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--val-in-epoch-max-batches", type=int, default=VAL_MAX_BATCHES)
    parser.add_argument("--target-step", type=int, default=TARGET_STEP_DEFAULT)
    parser.add_argument("--val-every-steps", type=int, default=0, help="Validation cadence for train runs; 0 disables validation")
    parser.add_argument(
        "--baseline-throughput-tok-s",
        type=float,
        default=None,
        help="Optional external bf16 bs=1 baseline throughput for delta computation",
    )
    parser.add_argument("--smoke-timeout-sec", type=int, default=60)
    parser.add_argument("--cell-timeout-sec", type=int, default=3600)
    parser.add_argument("--skip-smoke", action="store_true", help="Run a single train process per cell and validate FP8 receipt from train.log")
    parser.add_argument(
        "--reuse-pool-from",
        type=Path,
        default=None,
        help="Existing combined-pool root containing combined_prompts/ and combined_hidden_states/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.reuse_pool_from is not None:
        args.vocab_data_path = args.reuse_pool_from / "combined_prompts"
        args.default_hidden_states_path = args.reuse_pool_from / "combined_hidden_states"
        args.data_sources_path = None
    else:
        should_materialize_default_pool = (
            args.data_sources_path is None
            and args.vocab_data_path == DEFAULT_VOCAB_DATA_PATH
            and args.default_hidden_states_path == DEFAULT_HIDDEN_STATES_PATH
        )
        if should_materialize_default_pool:
            args.data_sources_path = _build_default_data_sources_json(args.run_root / "full_55k_sources.json")
    if args.data_sources_path is not None:
        prompts_path, hidden_states_path = _materialize_combined_pool(
            data_sources_path=args.data_sources_path,
            dest_root=args.run_root,
        )
        args.vocab_data_path = prompts_path
        args.default_hidden_states_path = hidden_states_path
        args.data_sources_path = None
    config_order = args.config_id or ["C1", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20"]
    jsonl_path = args.run_root / f"{args.results_prefix}.jsonl"
    md_path = args.run_root / f"{args.results_prefix}.md"
    results: list[CellResult] = []
    baseline: CellResult | None = None
    for config_id in config_order:
        config = CONFIGS[config_id]
        for micro_bs in args.micro_bs:
            result = _run_cell(
                config=config,
                micro_bs=micro_bs,
                args=args,
                baseline=baseline,
            )
            results.append(result)
            with jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(result), sort_keys=True) + "\n")
            md_path.write_text(_render_markdown(results))
            if config.config_id == "C1" and micro_bs == 1 and result.status == "OK":
                baseline = result
            if result.status == "OOM":
                break
    md_path.write_text(_render_markdown(results))
    print(json.dumps({"jsonl": str(jsonl_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
