"""DFlashTrainer — high-level wrapper around the speculators training pipeline.

Pragmatic choice: we shell out to torchrun rather than driving speculators
in-process. The speculators training entry point's argparse interface is
much more stable than its programmatic API. When the latter stabilises we
can swap to it without users having to change anything.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from ..verifiers.base import BaseVerifier
from .prompts import assemble_prompts_arrow
from .vocab_maps import build_vocab_maps
from .smoke import run_smoke_test, SmokeResult


class DFlashTrainer:
    """End-to-end DFlash training driver.

    Workflow::

        trainer = DFlashTrainer(
            traces_dir=...,
            verifier=load_verifier("minimax-m2.7-iq4-xs", hf_path=HF),
            num_layers=5, draft_vocab_size=32768,
        )
        trainer.prepare()                 # assemble_prompts_arrow + vocab maps
        trainer.smoke(timeout_sec=90)     # 90s torchrun smoke
        trainer.train(epochs=17, save_to=CKPT)
        trainer.offline_eval(checkpoint=CKPT / "checkpoint_best")
    """

    def __init__(
        self,
        *,
        traces_dir: str,
        verifier: BaseVerifier,
        drafter_arch: Optional[str] = None,
        num_layers: int = 5,
        draft_vocab_size: int = 32768,
        paired_dir: Optional[str] = None,
    ):
        self.traces_dir = Path(traces_dir)
        self.verifier = verifier
        self.drafter_arch = drafter_arch or verifier.drafter_arch
        self.num_layers = int(num_layers)
        self.draft_vocab_size = int(draft_vocab_size)
        self.paired_dir = Path(paired_dir) if paired_dir else self.traces_dir.parent / "paired"
        self._prepared = False

    # -----------------------------------------------------------------
    def prepare(self, *, force: bool = False) -> dict:
        """Build the paired prompts arrow + vocab maps."""
        prompts_dir = self.paired_dir / "prompts"
        report = {}
        if force or not prompts_dir.exists():
            report["assemble"] = assemble_prompts_arrow(
                self.traces_dir, output_dir=self.paired_dir
            )
        else:
            report["assemble"] = {"skipped": True, "reason": f"{prompts_dir} already exists"}

        if force or not (prompts_dir / "t2d.npy").exists():
            report["vocab_maps"] = build_vocab_maps(
                prompts_dir,
                verifier_vocab_size=self.verifier.vocab_size,
                draft_vocab_size=self.draft_vocab_size,
            )
        else:
            report["vocab_maps"] = {"skipped": True, "reason": "t2d.npy already exists"}
        self._prepared = True
        return report

    # -----------------------------------------------------------------
    def _build_train_cmd(
        self,
        *,
        save_to: str,
        epochs: int,
        lr: float,
        max_anchors: int,
        log_freq: int,
        scheduler_warmup_steps: int,
        save_best: bool,
        port: int,
        speculators_train_script: Optional[str],
        total_seq_len: int,
    ) -> list[str]:
        train_script = speculators_train_script or os.environ.get(
            "SPECULATORS_TRAIN_SCRIPT",
            os.path.expanduser("~/repos/speculators/scripts/train.py"),
        )
        target_layer_ids = self.verifier.trainer_target_layer_ids()
        cmd = [
            "torchrun",
            f"--master_port={port}",
            "--nproc-per-node=1",
            train_script,
            "--speculator-type", "dflash",
            "--verifier-name-or-path", str(self.verifier.hf_path or self.verifier.gguf_path or ""),
            "--data-path", str(self.paired_dir / "prompts"),
            "--hidden-states-path", str(self.paired_dir / "hidden_states"),
            "--save-path", str(save_to),
            "--epochs", str(epochs),
            "--total-seq-len", str(total_seq_len),
            "--max-anchors", str(max_anchors),
            "--num-workers", "1", "--prefetch-factor", "2",
            "--on-missing", "skip",
            "--target-layer-ids", *(str(L) for L in target_layer_ids),
            "--draft-arch", self.drafter_arch,
            "--draft-hidden-act", self.verifier.drafter_hidden_act,
            "--mask-token-id", str(self.verifier.mask_token_id),
            "--block-size", str(self.verifier.block_size),
            "--hidden-states-dtype", "bfloat16",
            "--num-layers", str(self.num_layers),
            "--draft-vocab-size", str(self.draft_vocab_size),
            "--lr", str(lr),
            "--scheduler-warmup-steps", str(scheduler_warmup_steps),
            "--log-freq", str(log_freq),
        ]
        if save_best:
            cmd.append("--save-best")
        return cmd

    # -----------------------------------------------------------------
    def train(
        self,
        *,
        save_to: str,
        epochs: int = 17,
        lr: float = 3e-5,
        max_anchors: int = 512,
        total_seq_len: int = 2048,
        log_freq: int = 5,
        scheduler_warmup_steps: int = 100,
        save_best: bool = True,
        port: int = 29502,
        speculators_train_script: Optional[str] = None,
        log_path: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict:
        """Run a full training job.

        Returns ``{"rc": int, "log_path": str, "cmd": [...]}``. ``dry_run=True``
        returns the exact command that would be invoked without executing it.
        """
        if not self._prepared and not (self.paired_dir / "prompts" / "t2d.npy").exists():
            raise RuntimeError(
                "trainer.prepare() must be called first (or pass an already-prepared paired_dir)"
            )
        save_to_p = Path(save_to)
        save_to_p.mkdir(parents=True, exist_ok=True)
        if log_path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_path = str(save_to_p / f"train_{ts}.log")

        cmd = self._build_train_cmd(
            save_to=str(save_to_p),
            epochs=epochs, lr=lr, max_anchors=max_anchors,
            log_freq=log_freq,
            scheduler_warmup_steps=scheduler_warmup_steps,
            save_best=save_best, port=port,
            speculators_train_script=speculators_train_script,
            total_seq_len=total_seq_len,
        )

        if dry_run:
            return {"rc": 0, "log_path": log_path, "cmd": cmd, "dry_run": True}

        print(f"[train] cmd: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        with open(log_path, "wb") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        rc = proc.returncode
        return {"rc": rc, "log_path": log_path, "cmd": cmd, "save_path": str(save_to_p)}

    # -----------------------------------------------------------------
    def smoke(
        self,
        *,
        timeout_sec: int = 90,
        save_path: str = "/tmp/dflash-smoke",
        log_path: str = "/tmp/dflash-smoke.log",
        port: int = 29501,
        speculators_train_script: Optional[str] = None,
        dry_run: bool = False,
    ) -> SmokeResult:
        """Run the 90-second smoke against ``self.paired_dir``."""
        return run_smoke_test(
            paired_dir=str(self.paired_dir),
            verifier=self.verifier,
            save_path=save_path,
            log_path=log_path,
            timeout_sec=timeout_sec,
            port=port,
            speculators_train_script=speculators_train_script,
            dry_run=dry_run,
        )

    # -----------------------------------------------------------------
    def offline_eval(
        self,
        *,
        checkpoint: str,
        max_batches: int = 60,
        total_seq_len: int = 2048,
    ) -> dict:
        from .eval import offline_eval

        return offline_eval(
            checkpoint=checkpoint,
            paired_dir=str(self.paired_dir),
            verifier_path=str(self.verifier.hf_path or self.verifier.gguf_path or ""),
            max_batches=max_batches,
            total_seq_len=total_seq_len,
        )


__all__ = ["DFlashTrainer"]
