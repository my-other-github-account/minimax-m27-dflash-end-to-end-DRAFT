"""End-to-end smoke test on a synthetic trace directory.

Verifies the whole pipeline up to the torchrun shell-out:

  - assemble_prompts_arrow walks the trace dir and emits a valid HF Dataset
  - build_vocab_maps generates t2d/d2t/token_freq with correct shapes/dtypes
  - DFlashTrainer.smoke(dry_run=True) builds the right torchrun command
  - DFlashTrainer.train(dry_run=True) produces the production-style command

We never actually call torchrun (no GPU on macmini, no speculators dep
in tests). dry_run=True is the contract for "show me what you'd run".
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from datasets import load_from_disk

from dflash_llama.training import (
    DFlashTrainer,
    assemble_prompts_arrow,
    build_vocab_maps,
)
from dflash_llama.verifiers import load_verifier


def test_assemble_prompts_emits_valid_arrow(synthetic_trace_dir, tmp_path):
    out_dir = tmp_path / "paired"
    report = assemble_prompts_arrow(synthetic_trace_dir, output_dir=out_dir)
    assert report["n_traces"] == 5
    assert report["n_rows"] == 5
    assert report["n_skipped"] == 0
    assert report["by_source"] == {"synthetic": 5}

    prompts_dir = out_dir / "prompts"
    assert prompts_dir.exists()
    ds = load_from_disk(str(prompts_dir))
    assert len(ds) == 5
    assert "input_ids" in ds.column_names
    assert "loss_mask" in ds.column_names
    assert "source_name" in ds.column_names
    assert "source_row_idx" in ds.column_names

    # hidden_states symlink dir is populated
    hs_dir = out_dir / "hidden_states"
    assert hs_dir.exists()
    assert len(list(hs_dir.glob("hs_*.safetensors"))) == 5


def test_full_prepare_then_dryrun_train(synthetic_trace_dir, tmp_path):
    verifier = load_verifier(
        "generic", name_override="synthetic-qwen3-tiny",
        hidden_size=64, num_hidden_layers=6,
        vocab_size=2048, mask_token_id=0,
        layer_ids=(0, 1, 2, 3, 4, 5),
        hf_path="/dummy/hf",
    )
    paired = tmp_path / "paired"
    trainer = DFlashTrainer(
        traces_dir=str(synthetic_trace_dir),
        verifier=verifier,
        num_layers=2,
        draft_vocab_size=128,
        paired_dir=str(paired),
    )
    report = trainer.prepare()
    assert "assemble" in report
    assert "vocab_maps" in report

    # vocab maps were written
    prompts = paired / "prompts"
    assert (prompts / "t2d.npy").exists()
    assert (prompts / "d2t.npy").exists()
    t2d = np.load(prompts / "t2d.npy")
    assert t2d.dtype == np.bool_
    assert int(t2d.sum()) == 128

    # dry-run train command — must include all required flags
    out = tmp_path / "ckpt"
    res = trainer.train(save_to=str(out), dry_run=True, epochs=3, max_anchors=64, lr=5e-5)
    assert res["dry_run"] is True
    cmd = res["cmd"]
    assert "torchrun" in cmd[0]
    cmd_str = " ".join(cmd)
    assert "--speculator-type dflash" in cmd_str
    assert "--data-path" in cmd_str
    assert "--hidden-states-path" in cmd_str
    assert "--num-layers 2" in cmd_str
    assert "--draft-vocab-size 128" in cmd_str
    assert "--mask-token-id 0" in cmd_str
    assert "--epochs 3" in cmd_str
    assert "--lr 5e-05" in cmd_str
    # target-layer-ids drops the final tap
    assert "--target-layer-ids 0 1 2 3 4 " in cmd_str + " "  # 0..4, not 5


def test_smoke_dryrun(synthetic_trace_dir, tmp_path):
    verifier = load_verifier(
        "generic", name_override="synthetic-qwen3-tiny",
        hidden_size=64, num_hidden_layers=6,
        vocab_size=2048, mask_token_id=0,
        layer_ids=(0, 1, 2, 3, 4, 5),
        hf_path="/dummy/hf",
    )
    paired = tmp_path / "paired"
    trainer = DFlashTrainer(
        traces_dir=str(synthetic_trace_dir),
        verifier=verifier,
        num_layers=2, draft_vocab_size=128,
        paired_dir=str(paired),
    )
    trainer.prepare()
    res = trainer.smoke(dry_run=True, timeout_sec=15)
    assert res.passed
    # smoke command should set max-anchors=64 and epochs=1
    assert "--max-anchors 64" in res.message
    assert "--epochs 1" in res.message


def test_assemble_skips_legacy_traces(tmp_path):
    """Traces without our v3 schema should be skipped, not crash."""
    from safetensors.torch import save_file
    import torch

    d = tmp_path / "mixed"
    d.mkdir()
    # 2 valid v3 traces
    from dflash_llama.generation.format import save_trace
    for i in range(2):
        save_trace(
            d / f"hs_{i}.safetensors",
            hidden_states=torch.randn(8, 4, 16),
            token_ids=torch.arange(8, dtype=torch.int64),
            input_ids=torch.arange(8, dtype=torch.int64),
            loss_mask=torch.ones(8, dtype=torch.bool),
            source_name="ok", source_row_idx=i,
        )
    # 1 legacy-style trace (no metadata)
    save_file(
        {"hidden_states": torch.zeros(4, 4, 16, dtype=torch.bfloat16),
         "token_ids": torch.zeros(4, dtype=torch.int64)},
        str(d / "hs_99.safetensors"),
        metadata=None,
    )
    out = tmp_path / "out"
    report = assemble_prompts_arrow(d, output_dir=out)
    assert report["n_rows"] == 2  # only the valid ones
    assert report["n_skipped"] == 1
