"""Smoke-test wrapper around the speculators trainer.

Mirrors the v2 ``smoke_train.sh`` semantics: launches torchrun with the
production hyperparameters but with ``epochs=1`` and ``max-anchors=64``,
runs for ``timeout_sec`` seconds, then declares pass if:

  - process was killed by the timeout (rc 124, the canonical "ran the
    full duration without crashing" signal), AND
  - the log shows at least one ``global_step=`` line, AND
  - the log does NOT contain any of the canonical failure markers.

We never *require* the trainer to finish — this is a smoke test. A real
training run uses ``DFlashTrainer.train``.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..verifiers.base import BaseVerifier
from .launch import build_training_env


CANONICAL_FAILURE_MARKERS = (
    "R54: hs prompt prefix mismatch",
    "anchor_positions include padding",
    "don't match input ids",
    "t2d has",
    "d2t has",
)


@dataclass
class SmokeResult:
    rc: int
    timed_out: bool
    log_path: str
    env: dict | None = None
    saw_global_step: bool = False
    failure_markers_hit: list = field(default_factory=list)
    passed: bool = False
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "rc": self.rc,
            "timed_out": self.timed_out,
            "log_path": self.log_path,
            "env": self.env,
            "saw_global_step": self.saw_global_step,
            "failure_markers_hit": list(self.failure_markers_hit),
            "passed": self.passed,
            "message": self.message,
        }



def _resolve_torchrun_smoke() -> str:
    """Locate torchrun next to sys.executable, then PATH, then bare name."""
    import shutil
    import sys
    cand = Path(sys.executable).parent / "torchrun"
    if cand.exists():
        return str(cand)
    found = shutil.which("torchrun")
    if found:
        return found
    return "torchrun"


def run_smoke_test(
    *,
    paired_dir: str,
    verifier: BaseVerifier,
    save_path: str = "/tmp/dflash-smoke",
    log_path: str = "/tmp/dflash-smoke.log",
    timeout_sec: int = 90,
    port: int = 29501,
    speculators_train_script: Optional[str] = None,
    extra_env: Optional[dict] = None,
    dry_run: bool = False,
    # === DFlash FP8 / TransformerEngine wrap (optional) ===
    fp8_recipe_kind: str = "",
    te_use_fused: bool = False,
    te_fp8_params: bool = False,
    compile_flex_attention: bool = False,
    liger_fused_linear_ce: bool = False,
    liger_rope: bool = False,
    liger_rms_norm: bool = False,
    use_torchrun: Optional[bool] = None,
) -> SmokeResult:
    """Run a 90-second smoke training run against a paired-trace directory.

    ``paired_dir`` must contain ``prompts/`` (HF Dataset + d2t.npy + t2d.npy
    + token_freq.pt) and ``hidden_states/`` (the trace files).

    When ``fp8_recipe_kind`` is set (e.g. ``"current_fp8"``), the smoke launcher
    automatically switches from torchrun to direct ``python`` invocation — torchrun
    sets ``RANK``/``WORLD_SIZE`` which routes the trainer through FSDP and silently
    skips the FP8/TE wrap. See ``repro/06-fp8-training.md`` for details. Pass
    ``use_torchrun=True/False`` to override the default.
    """
    paired_dir = Path(paired_dir)
    train_script = speculators_train_script or os.environ.get(
        "SPECULATORS_TRAIN_SCRIPT",
        os.path.expanduser("~/repos/speculators/scripts/train.py"),
    )
    save_path_p = Path(save_path)
    save_path_p.mkdir(parents=True, exist_ok=True)

    target_layer_ids = verifier.trainer_target_layer_ids()

    # Launch-mode selection: same policy as DFlashTrainer.train (FP8 -> direct python).
    if use_torchrun is None:
        use_torchrun = (fp8_recipe_kind in ("", "bf16", "none"))

    if use_torchrun:
        launcher = [
            _resolve_torchrun_smoke(),
            f"--master_port={port}",
            "--nproc-per-node=1",
            train_script,
        ]
    else:
        launcher = [sys.executable, train_script]

    cmd = [
        *launcher,
        "--speculator-type", "dflash",
        "--verifier-name-or-path", str(verifier.hf_path or verifier.gguf_path or ""),
        "--data-path", str(paired_dir / "prompts"),
        "--hidden-states-path", str(paired_dir / "hidden_states"),
        "--save-path", str(save_path_p),
        "--epochs", "1",
        "--total-seq-len", "2048",
        "--max-anchors", "64",
        "--num-workers", "1", "--prefetch-factor", "2",
        "--on-missing", "skip",
        "--target-layer-ids", *(str(L) for L in target_layer_ids),
        "--draft-arch", verifier.drafter_arch,
        "--draft-hidden-act", verifier.drafter_hidden_act,
        "--mask-token-id", str(verifier.mask_token_id),
        "--block-size", str(verifier.block_size),
        "--hidden-states-dtype", "bfloat16",
        "--num-layers", "5",
        "--draft-vocab-size", "32768",
        "--log-freq", "5",
    ]
    if fp8_recipe_kind:
        cmd += ["--fp8-recipe-kind", fp8_recipe_kind]
    if te_use_fused:
        cmd.append("--te-use-fused")
    if liger_fused_linear_ce:
        cmd.append("--liger-fused-linear-ce")
    if liger_rope:
        cmd.append("--liger-rope")
    if liger_rms_norm:
        cmd.append("--liger-rms-norm")
    cmd_with_timeout = ["timeout", str(timeout_sec), *cmd]
    env = build_training_env(
        te_fp8_params=te_fp8_params,
        compile_flex_attention=compile_flex_attention,
        extra_env=extra_env,
    )

    if dry_run:
        return SmokeResult(
            rc=0,
            timed_out=False,
            log_path=log_path,
            env=env,
            passed=True,
            message="dry_run=True; would run: " + " ".join(shlex.quote(c) for c in cmd_with_timeout),
        )

    print(f"[smoke] cmd: {' '.join(shlex.quote(c) for c in cmd_with_timeout)}", flush=True)
    with open(log_path, "wb") as logf:
        proc = subprocess.run(
            cmd_with_timeout,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
    rc = proc.returncode
    timed_out = rc == 124

    log_text = Path(log_path).read_text(errors="replace") if Path(log_path).exists() else ""
    saw_step = bool(re.search(r"global_step=\d+", log_text))
    hits = [m for m in CANONICAL_FAILURE_MARKERS if m in log_text]

    passed = timed_out and saw_step and not hits
    msg = (
        "PASS" if passed
        else f"FAIL: rc={rc} timed_out={timed_out} saw_global_step={saw_step} "
             f"failure_markers={hits}"
    )
    print(f"[smoke] {msg}", flush=True)
    return SmokeResult(
        rc=rc,
        timed_out=timed_out,
        log_path=log_path,
        env=env,
        saw_global_step=saw_step,
        failure_markers_hit=hits,
        passed=passed,
        message=msg,
    )


__all__ = ["run_smoke_test", "SmokeResult", "CANONICAL_FAILURE_MARKERS"]
