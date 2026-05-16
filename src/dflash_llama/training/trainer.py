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
from .launch import build_training_env
from .prompts import assemble_prompts_arrow
from .vocab_maps import build_vocab_maps
from .smoke import run_smoke_test, SmokeResult


def _resolve_torchrun() -> str:
    """Locate torchrun next to sys.executable, then PATH, then bare name.

    subprocess.run() with a bare "torchrun" only works when torchrun is on the
    caller's PATH, which is unreliable when DFlashTrainer is invoked from a
    script that didn't source the venv. We look in the venv's bin/ first,
    then shutil.which, then fall back to the bare name so the user gets a
    clean FileNotFoundError if nothing exists.
    """
    import shutil
    import sys
    cand = Path(sys.executable).parent / "torchrun"
    if cand.exists():
        return str(cand)
    found = shutil.which("torchrun")
    if found:
        return found
    return "torchrun"



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
        """Build the paired prompts arrow + vocab maps + hidden_states symlink farm.

        Speculators' ArrowDataset expects hidden states named
        ``hs_<dataset_position>.safetensors`` aligned with arrow row order, but
        v3 traces are keyed by prompt-row index. We build a symlink farm in
        ``paired_dir/hidden_states/`` mapping each dataset position to the
        underlying source trace file.
        """
        prompts_dir = self.paired_dir / "prompts"
        self.paired_dir.mkdir(parents=True, exist_ok=True)
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
        # Build hs_<dataset_position>.safetensors symlink farm aligned with the arrow.
        report["hs_symlinks"] = self._build_hs_symlink_farm()
        self._prepared = True
        return report


    # -----------------------------------------------------------------
    def _build_hs_symlink_farm(self) -> dict:
        """Materialise paired_dir/hidden_states/hs_<i>.safetensors -> traces_dir/<source>.

        Reads the freshly built arrow's source_row_idx column to build a
        deterministic mapping aligned with dataset row order.
        """
        from datasets import load_from_disk
        prompts_dir = self.paired_dir / "prompts"
        hs_dir = self.paired_dir / "hidden_states"
        if hs_dir.exists() and hs_dir.is_symlink():
            hs_dir.unlink()
        hs_dir.mkdir(parents=True, exist_ok=True)

        ds = load_from_disk(str(prompts_dir))
        if "source_row_idx" not in ds.column_names:
            return {"skipped": True, "reason": "arrow missing source_row_idx column"}

        n_linked = 0
        n_missing = 0
        missing_examples: list[str] = []
        for dataset_position, source_row_idx in enumerate(ds["source_row_idx"]):
            src_file = self.traces_dir / f"hs_{int(source_row_idx)}.safetensors"
            link = hs_dir / f"hs_{dataset_position}.safetensors"
            if link.exists() or link.is_symlink():
                link.unlink()
            if not src_file.exists():
                n_missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(str(src_file))
                continue
            link.symlink_to(src_file.resolve())
            n_linked += 1

        return {
            "hidden_states_dir": str(hs_dir),
            "n_linked": n_linked,
            "n_missing": n_missing,
            "missing_first_5": missing_examples,
            "n_rows": len(ds),
        }

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
        # === DFlash in-epoch val + step-tagged ckpt cadence ===
        val_every_steps: int = 0,
        val_in_epoch_max_batches: int = 0,
        save_every_n_vals: int = 0,
        # === DFlash FP8 / TransformerEngine wrap ===
        fp8_recipe_kind: str = "",
        te_use_fused: bool = False,
        te_fp8_params: bool = False,
        compile_flex_attention: bool = False,
        liger_fused_linear_ce: bool = False,
        liger_rope: bool = False,
        liger_rms_norm: bool = False,
        # === Launch mode ===
        use_torchrun: Optional[bool] = None,
    ) -> list[str]:
        train_script = speculators_train_script or os.environ.get(
            "SPECULATORS_TRAIN_SCRIPT",
            os.path.expanduser("~/repos/speculators/scripts/train.py"),
        )
        target_layer_ids = self.verifier.trainer_target_layer_ids()

        # === Launch mode selection: torchrun vs direct python ===
        # FP8 single-GPU training MUST bypass torchrun. The speculators trainer's
        # setup_model() routes to the FSDP branch when any of RANK/WORLD_SIZE/etc
        # are set in the environment — which torchrun --nproc-per-node=1 always sets,
        # even on a single GPU. The FP8/TE wrap fires only in the single-GPU branch,
        # so torchrun would silently run plain bf16 with no warning. See
        # repro/06-fp8-training.md for the full failure-mode analysis.
        #
        # Default policy: drop torchrun whenever fp8 is requested. Caller can force
        # either mode via use_torchrun=True/False.
        if use_torchrun is None:
            use_torchrun = (fp8_recipe_kind in ("", "bf16", "none"))

        cmd: list[str] = []
        if use_torchrun:
            torchrun_bin = _resolve_torchrun()
            cmd = [
                torchrun_bin,
                f"--master_port={port}",
                "--nproc-per-node=1",
                train_script,
            ]
        else:
            # Direct python launch. Use sys.executable so the venv is preserved.
            cmd = [sys.executable, train_script]

        cmd += [
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
        # In-epoch val + step-tagged ckpt flags (passed as 0 = disabled by default)
        if val_every_steps > 0:
            cmd += ["--val-every-steps", str(val_every_steps)]
        if val_in_epoch_max_batches > 0:
            cmd += ["--val-in-epoch-max-batches", str(val_in_epoch_max_batches)]
        if save_every_n_vals > 0:
            cmd += ["--save-every-n-vals", str(save_every_n_vals)]
        # FP8 / TE wrap flags
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
        # === DFlash in-epoch val + step-tagged ckpt cadence ===
        val_every_steps: int = 0,
        val_in_epoch_max_batches: int = 0,
        save_every_n_vals: int = 0,
        # === DFlash FP8 / TransformerEngine wrap ===
        fp8_recipe_kind: str = "",
        te_use_fused: bool = False,
        te_fp8_params: bool = False,
        compile_flex_attention: bool = False,
        liger_fused_linear_ce: bool = False,
        liger_rope: bool = False,
        liger_rms_norm: bool = False,
        # === Launch mode override ===
        use_torchrun: Optional[bool] = None,
    ) -> dict:
        """Run a full training job.

        Returns ``{"rc": int, "log_path": str, "cmd": [...], "env": {...}}``.
        ``dry_run=True`` returns the exact command/env that would be invoked
        without executing it.

        FP8 training (DGX Spark sm_121a verified-stable):
            fp8_recipe_kind="current_fp8", te_use_fused=True,
            te_fp8_params=True, compile_flex_attention=True

        Liger add-ons (apply after TE wrap):
            liger_fused_linear_ce=True, liger_rope=True

        When ``fp8_recipe_kind`` is set, the launcher automatically switches from
        torchrun to direct ``python`` invocation — torchrun on a single GPU sets
        ``RANK``/``WORLD_SIZE`` env vars, which routes the trainer through the
        FSDP branch and silently skips the TE wrap. See ``repro/06-fp8-training.md``.

        In-epoch validation (off by default, opt-in for long runs):
            val_every_steps=145, val_in_epoch_max_batches=80, save_every_n_vals=1
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
            val_every_steps=val_every_steps,
            val_in_epoch_max_batches=val_in_epoch_max_batches,
            save_every_n_vals=save_every_n_vals,
            fp8_recipe_kind=fp8_recipe_kind,
            te_use_fused=te_use_fused,
            te_fp8_params=te_fp8_params,
            compile_flex_attention=compile_flex_attention,
            liger_fused_linear_ce=liger_fused_linear_ce,
            liger_rope=liger_rope,
            liger_rms_norm=liger_rms_norm,
            use_torchrun=use_torchrun,
        )
        env = build_training_env(
            te_fp8_params=te_fp8_params,
            compile_flex_attention=compile_flex_attention,
        )

        if dry_run:
            return {
                "rc": 0,
                "log_path": log_path,
                "cmd": cmd,
                "env": env,
                "dry_run": True,
            }

        print(f"[train] cmd: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        with open(log_path, "wb") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        rc = proc.returncode
        return {
            "rc": rc,
            "log_path": log_path,
            "cmd": cmd,
            "env": env,
            "save_path": str(save_to_p),
        }

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
        fp8_recipe_kind: str = "",
        te_use_fused: bool = False,
        te_fp8_params: bool = False,
        compile_flex_attention: bool = False,
        liger_fused_linear_ce: bool = False,
        liger_rope: bool = False,
        liger_rms_norm: bool = False,
        use_torchrun: Optional[bool] = None,
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
            fp8_recipe_kind=fp8_recipe_kind,
            te_use_fused=te_use_fused,
            te_fp8_params=te_fp8_params,
            compile_flex_attention=compile_flex_attention,
            liger_fused_linear_ce=liger_fused_linear_ce,
            liger_rope=liger_rope,
            liger_rms_norm=liger_rms_norm,
            use_torchrun=use_torchrun,
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
