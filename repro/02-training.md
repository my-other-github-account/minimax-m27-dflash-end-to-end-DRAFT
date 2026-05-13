# §2 — Training the DFlash Drafter

End-to-end recipe to train a 5-layer DFlash drafter from a directory of self-describing traces (§1) using `dflash-llama train` (or the `DFlashTrainer` Python API).

> **Looking for FP8 training?** This page covers the bf16 path (the v11 baseline). For Float8CurrentScaling(HYBRID) + fused TE LayerNormMLP on DGX Spark sm_121a (+18% throughput, verified step-for-step ≥ bf16 through step 2,610), see [§6 — FP8 training](06-fp8-training.md).

> **No pairing step.** v2 required a brittle `build_paired_dataset.py` that sha256-matched hidden-state files against a separate prompts dataset. The new self-describing trace format makes this a 30-second enumeration: `assemble_prompts_arrow` walks the directory and reads the `input_ids` / `loss_mask` / `source_row_idx` directly off each safetensor.

## Pipeline at a glance

```
┌────────────┐   prepare()   ┌─────────────────┐   train()   ┌──────────────┐
│  traces/   │ ────────────▶ │  paired/         │ ──────────▶ │  checkpoint  │
│  hs_*.st   │               │   prompts/       │             │              │
│  (§1)      │               │   hidden_states/ │             │              │
└────────────┘               │   t2d.npy        │             └──────────────┘
                             │   d2t.npy        │
                             │   token_freq.pt  │
                             └─────────────────┘
```

`prepare()` does two things:

1. **`assemble_prompts_arrow`** — reads every trace, emits an HF Dataset of `{input_ids, loss_mask, source_name, source_row_idx}` rows, and creates a `hidden_states/` directory of symlinks pointing back at the original safetensors. The trainer's `--data-path` is `paired/prompts`; its `--hidden-states-path` is `paired/hidden_states`.
2. **`build_vocab_maps`** — counts loss-mask token frequencies, picks the top-K, and writes the canonical `t2d.npy` (bool mask, `sum() == draft_vocab_size`), `d2t.npy` (int64 offset table: `verifier_token = draft_id + d2t[draft_id]`), and `token_freq.pt`.

## CLI quickstart

```bash
# Smoke first (mandatory — 90s torchrun, exit 124 = pass)
dflash-llama smoke \
    --verifier minimax-m2.7-iq4-xs \
    --hf-path /path/to/MiniMax-M2.7-FP8 \
    --traces /path/to/traces \
    --timeout 90

# Full run (17 epochs, max_anchors=512, lr=3e-5)
dflash-llama train \
    --verifier minimax-m2.7-iq4-xs \
    --hf-path /path/to/MiniMax-M2.7-FP8 \
    --traces /path/to/traces \
    --output /path/to/checkpoint \
    --epochs 17 \
    --lr 3e-5 \
    --max-anchors 512
```

## Python API quickstart

```python
from dflash_llama import DFlashTrainer, load_verifier

verifier = load_verifier(
    "minimax-m2.7-iq4-xs",
    hf_path="/path/to/MiniMax-M2.7-FP8",
)
trainer = DFlashTrainer(
    traces_dir="/path/to/traces",
    verifier=verifier,
    drafter_arch="qwen3",
    num_layers=5,
    draft_vocab_size=32768,
)

trainer.prepare()                    # assemble_prompts_arrow + build_vocab_maps
result = trainer.smoke(timeout_sec=90)
assert result.passed, result.message

trainer.train(
    save_to="/path/to/checkpoint",
    epochs=17,
    lr=3e-5,
    max_anchors=512,
)

trainer.offline_eval(
    checkpoint="/path/to/checkpoint/checkpoint_best",
    max_batches=60,
)
```

## Hyperparameters (production defaults)

| flag | default | notes |
|---|---|---|
| `epochs` | 17 | matches v2 production run |
| `total_seq_len` | 2048 | trainer pads to this |
| `max_anchors` | 512 (full) / 64 (smoke) | per-batch anchor budget |
| `lr` | 3e-5 | with `scheduler_warmup_steps=100` |
| `block_size` | 8 | DFlash block size |
| `num_workers` | 1, prefetch=2 | speculators dataloader |
| `on_missing` | skip | tolerate per-row failures |
| `hidden_states_dtype` | bfloat16 | applied after fp8 scale-back |
| `save_best` | True | writes `checkpoint_best/` |

## Implementation note: torchrun shell-out

The trainer **shells out to torchrun** invoking `speculators/scripts/train.py`. We pick this over an in-process call because the speculators argparse interface is much more stable than its programmatic API. To swap to the in-process path later, override `DFlashTrainer.train` — the call shape is unchanged.

You can supply a custom path to the speculators training script via `--train-script` or the `SPECULATORS_TRAIN_SCRIPT` env var. Default: `~/repos/speculators/scripts/train.py`.

## Verifying the smoke

The smoke wrapper enforces the v2 pass criteria:

- **rc == 124** — process was killed by `timeout`, meaning it ran the full 90 seconds without crashing.
- **Log shows `global_step=N`** with `N >= 1`.
- **No canonical failure markers** — `R54: hs prompt prefix mismatch`, `anchor_positions include padding`, `don't match input ids`, `t2d has`, `d2t has`.

Inspect `trainer.smoke(...).failure_markers_hit` if any of these fire.

## Vocab-map dtype contract

`build_vocab_maps` enforces (and re-asserts after every speculators call):

```python
t2d.dtype == np.bool_
t2d.shape == (verifier_vocab_size,)
int(t2d.sum()) == draft_vocab_size

d2t.dtype == np.int64
d2t.shape == (draft_vocab_size,)
verifier_token = draft_id + d2t[draft_id]   # offset semantics
```

If the speculators helper returns torch tensors (it does in some versions), they are coerced to numpy with the canonical dtypes. This was the v2 `build_vocab_maps.py` bug — fixed once and tested in `tests/test_vocab_maps.py`.

## What about layer 61 vs 62?

For MiniMax-M2.7 the tap list `[2, 16, 30, 45, 59, 61]` ends at the final residual stream. The speculators trainer auto-appends "the final layer", which it labels as 62. Semantically the two are the same hidden state — `tap_idx[5]` is the last residual. The library accepts whatever `layer_ids` the verifier config declares and passes `layer_ids[:-1]` to the trainer (so the trainer's auto-append re-creates the 6-tap input the drafter expects).
