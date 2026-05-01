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
    saw_global_step: bool = False
    failure_markers_hit: list = field(default_factory=list)
    passed: bool = False
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "rc": self.rc,
            "timed_out": self.timed_out,
            "log_path": self.log_path,
            "saw_global_step": self.saw_global_step,
            "failure_markers_hit": list(self.failure_markers_hit),
            "passed": self.passed,
            "message": self.message,
        }


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
) -> SmokeResult:
    """Run a 90-second torchrun smoke against a paired-trace directory.

    ``paired_dir`` must contain ``prompts/`` (HF Dataset + d2t.npy + t2d.npy
    + token_freq.pt) and ``hidden_states/`` (the trace files).
    """
    paired_dir = Path(paired_dir)
    train_script = speculators_train_script or os.environ.get(
        "SPECULATORS_TRAIN_SCRIPT",
        os.path.expanduser("~/repos/speculators/scripts/train.py"),
    )
    save_path_p = Path(save_path)
    save_path_p.mkdir(parents=True, exist_ok=True)

    target_layer_ids = verifier.trainer_target_layer_ids()
    cmd = [
        "torchrun",
        f"--master_port={port}",
        "--nproc-per-node=1",
        train_script,
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
    cmd_with_timeout = ["timeout", str(timeout_sec), *cmd]

    if dry_run:
        return SmokeResult(
            rc=0,
            timed_out=False,
            log_path=log_path,
            passed=True,
            message="dry_run=True; would run: " + " ".join(shlex.quote(c) for c in cmd_with_timeout),
        )

    env = os.environ.copy()
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})

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
        saw_global_step=saw_step,
        failure_markers_hit=hits,
        passed=passed,
        message=msg,
    )


__all__ = ["run_smoke_test", "SmokeResult", "CANONICAL_FAILURE_MARKERS"]
