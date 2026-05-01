"""Tiny end-to-end demo on synthetic data — runs in ~5 seconds on CPU.

This script exercises the entire library pipeline (sans the torchrun
shell-out) on synthetic random data. Useful as a smoke test for an
install or as a starting point for adapting to your own corpus.
"""
from pathlib import Path
import tempfile

import torch

from dflash_llama import (
    DFlashTrainer,
    SelfDescribingTraceDataset,
    TraceGenerator,
    load_verifier,
)
from dflash_llama.generation.format import save_trace


def main():
    workdir = Path(tempfile.mkdtemp(prefix="dflash_llama_tiny_"))
    print(f"workdir: {workdir}")

    # 1) Synthesise 5 trace files (skip the real GGUF backend).
    traces = workdir / "traces"
    traces.mkdir()
    torch.manual_seed(0)
    for i in range(5):
        seq = 24 + i * 2
        save_trace(
            traces / f"hs_{i}.safetensors",
            hidden_states=torch.randn(seq, 6, 64) * 1500.0,  # exercise saturating fp8
            token_ids=torch.randint(0, 1024, (seq,), dtype=torch.int64),
            input_ids=torch.randint(0, 1024, (seq,), dtype=torch.int64),
            loss_mask=torch.ones(seq, dtype=torch.bool),
            source_name="tiny_smoke", source_row_idx=i,
            storage="fp8_per_tensor_scale",
            layer_ids=[0, 1, 2, 3, 4, 5],
        )

    # 2) Load them via the dataset to confirm format integrity.
    ds = SelfDescribingTraceDataset(str(traces))
    print(f"loaded {len(ds)} traces, first row hs shape={tuple(ds[0]['hidden_states'].shape)}")

    # 3) prepare() — assemble prompts arrow + vocab maps.
    verifier = load_verifier(
        "qwen3", hidden_size=64, num_hidden_layers=6,
        vocab_size=1024, mask_token_id=0,
        layer_ids=(0, 1, 2, 3, 4, 5),
        hf_path="/dummy",
    )
    trainer = DFlashTrainer(
        traces_dir=str(traces), verifier=verifier,
        num_layers=2, draft_vocab_size=64,
        paired_dir=str(workdir / "paired"),
    )
    report = trainer.prepare()
    print(f"prepare report: {report}")

    # 4) Show the torchrun command we *would* invoke.
    res = trainer.train(save_to=str(workdir / "ckpt"), dry_run=True, epochs=1)
    print(f"would-run cmd: {' '.join(res['cmd'])}")
    print("OK")


if __name__ == "__main__":
    main()
