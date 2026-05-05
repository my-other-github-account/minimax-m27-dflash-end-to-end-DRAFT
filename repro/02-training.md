# §2 — Training the DFlash Drafter

End-to-end recipe to train a 5-layer DFlash drafter from a directory of self-describing traces (§1) using `dflash-llama train` (or the `DFlashTrainer` Python API).

> **No pairing step.** v2 required a brittle `build_paired_dataset.py` that sha256-matched hidden-state files against a separate prompts dataset. The new self-describing trace format makes this a 30-second enumeration: `assemble_prompts_arrow` walks the directory and reads the `input_ids` / `loss_mask` / `source_row_idx` directly off each safetensor.

> **🔥 FP8 production training is verified end-to-end as of 2026-05-05.** See [§FP8 training (production)](#fp8-training-production) below for the one-call recipe that produced today's stable v12 launch (+42% throughput vs bf16). Always read [§Pitfalls](#pitfalls) first — there are four silent-failure modes to avoid.

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
    --hf-path /home/dnola/models/MiniMax-M2.7-FP8 \
    --traces /path/to/traces \
    --timeout 90

# Full run (17 epochs, max_anchors=512, lr=3e-5)
dflash-llama train \
    --verifier minimax-m2.7-iq4-xs \
    --hf-path /home/dnola/models/MiniMax-M2.7-FP8 \
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
    hf_path="/home/dnola/models/MiniMax-M2.7-FP8",
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

## FP8 training (production)

**Status:** verified end-to-end on the Spark cluster (DGX Spark GB10, sm_121, 119 GB unified) on 2026-05-05 against MiniMax-M2.7-IQ4-XS v12 (35,891 self-describing v3 traces). Microbenched at +42% throughput vs bf16 + intermediate=4096; production survives 305+ steps with zero NaN-skip events.

### One-call Python invocation (the v12 launcher)

```python
from dflash_llama import DFlashTrainer, FP8Recipe, load_verifier

verifier = load_verifier(
    "minimax-m2.7-iq4-xs",
    hf_repo="MiniMaxAI/MiniMax-M2",
    gguf_repo="unsloth/MiniMax-M2-GGUF",
    gguf_quant="UD-IQ4_XS",
)

trainer = DFlashTrainer(
    traces_dir="data/traces_v12",        # 35,891 self-describing v3 traces
    verifier=verifier,
    num_layers=5,
    draft_vocab_size=32768,
    paired_dir="data/paired_v12",
)
trainer.prepare()

trainer.train(
    save_to="data/ckpt_v12",
    epochs=15,
    lr=3e-5,
    max_anchors=512,
    fp8_recipe="current_fp8",            # Float8CurrentScaling HYBRID
    te_use_fused=True,                   # fused te.LayerNormMLP — THE memory lever
    drafter_intermediate_size=6144,      # recipe-faithful for MiniMax-M2.7
    nan_skip=True,                       # defensive optimizer guard
    # use_torchrun=False                 # default — avoids silent-bf16 trap
)
```

That is the entire production recipe. Each kwarg is in the call because something on 2026-05-04 or 2026-05-05 broke without it.

### CLI equivalent

```bash
dflash-llama train \
    --verifier minimax-m2.7-iq4-xs \
    --hf-repo MiniMaxAI/MiniMax-M2 \
    --gguf-repo unsloth/MiniMax-M2-GGUF --gguf-quant UD-IQ4_XS \
    --traces data/traces_v12 \
    --output data/ckpt_v12 \
    --epochs 15 --lr 3e-5 --max-anchors 512 \
    --fp8-recipe current_fp8 \
    --drafter-intermediate-size 6144
```

`--te-use-fused` is on by default; `--nan-skip` is on by default; the launcher is direct python (NOT torchrun) by default. Pass `--no-te-fused` / `--no-nan-skip` / `--use-torchrun` to override (only do this if you've read the Pitfalls below).

### Required runtime env

```bash
# 1. Newer cuBLAS than the system ships with — symbol mismatch otherwise
export LD_PRELOAD=/path/to/te-extras/nvidia/cublas/lib/libcublasLt.so.13

# 2. flex_attention pattern-matcher trips dynamo → disable both
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
```

See [`repro/04-fp8-bringup.md`](04-fp8-bringup.md) for the full TE source-build recipe (NVTE_CUDA_ARCHS=121, sibling te venv with .pth linkage, ninja from vllm venv) — the difference between a 2-minute and a 30-minute build.

### Throughput notes

| recipe | inter | tok/s (200-step microbench, sm_121) | notes |
|---|---|---|---|
| bf16 | 4096 | 1.00× (baseline) | reference |
| current_fp8 + fused te.LayerNormMLP | 4096 | ~1.27× | FP8 GEMM only |
| current_fp8 + fused te.LayerNormMLP | **6144** | **~1.42×** | **production** — fusion eliminates intermediate norm materialization, which is what makes inter=6144 fit |

FP8 weights/GEMM saved roughly **zero** activation memory at v11 sizes — the cliff is in the FFN backward graph (O(intermediate²) without grad checkpointing). Fusing `post_attention_layernorm + mlp` into one `te.LayerNormMLP` is what eliminates the intermediate norm materialization and unlocks inter=6144. **FP8 alone, without fusion, doesn't get you the +42%.**

---

## Pitfalls

Four silent-failure modes that bit production runs in May 2026. The library refuses or warns on each, but understand them before forcing your way past.

### 1. Silent-bf16 trap (torchrun + FP8 + single-GPU)

torchrun --nproc-per-node=1 sets `RANK` / `WORLD_SIZE` even on a single-GPU machine. speculators routes through its FSDP branch the moment those env vars are present. The TE wrap is only patched into the single-GPU branch, so torchrun + FP8 on one GPU silently runs bf16 — no `[FP8]` line in the log, no error, just "weird, this isn't faster than bf16."

**Fix:** `DFlashTrainer.train(use_torchrun=False)` (the new default). The library raises `RuntimeError: Silent-bf16 trap` if you combine `use_torchrun=True` with `fp8_recipe` set on a single-GPU machine.

### 2. The split-accumulator NaN

`Float8CurrentScaling`'s default in stock TransformerEngine has `use_split_accumulator=False` on **fprop** (it's True on dgrad/wgrad). On a noised DFlash drafter, this NaN's at step ~38–40 the moment the LR schedule crests ~1.2e-4. With `True` on **all three** GEMMs (fprop + dgrad + wgrad), the same recipe / data / LR survives 305+ steps with zero NaN-skip events.

**Fix:** `FP8Recipe(use_split_accumulator=True)` is the default in this library. The `make_te_recipe` builder forces this via `MMParams` on `fp8_gemm_fprop` / `fp8_gemm_dgrad` / `fp8_gemm_wgrad`. Don't flip it off.

### 3. Float8BlockScaling is silently NON-CONVERGENT on sm_120/121

TransformerEngine issue #2382. Loss stays finite but never decreases — it just sits at the initial loss value forever. Costs you a 14-epoch run before you notice.

**Fix:** the library refuses `kind="block_fp8"` on sm_120/121 at recipe-construction time (`make_te_recipe` raises `RuntimeError`). Future-us is forbidden from re-discovering this empirically.

### 4. NVFP4 / MXFP8 status on sm_121

- **MXFP8 via TransformerEngine:** BLOCKED on sm_120/121 (cuBLAS lacks the non-TN GEMM layouts; TE issue #2668). Symptom: cublasLt error at first matmul. Workaround if you really need MX: build `torchao.prototype.mx_formats` from source with `TORCH_CUDA_ARCH_LIST="12.1a"` (the trailing `a` is mandatory) — that bypass re-quantizes dim0+dim1 so all GEMMs are TN.
- **NVFP4(disable_rht=True):** silently broken on sm_121. Variance test: 8 trials × different SR seeds → bit-identical losses ⇒ stochastic-rounding not firing. The Harry-Chen polyfill restores SR for small shapes but production-shape FFN backward (`mul_cvt_8x` at `ptx.cuh:935`) hits unfixable arch-specific PTX errors. **NVFP4 is research-only on Spark.**

The library refuses `kind="mxfp8"` on sm_120/121 at recipe-construction time. NVFP4 isn't even an FP8Recipe kind — there's no production path.

---

## Hyperparameters (production defaults)

| flag | bf16 default | FP8 production (v12) | notes |
|---|---|---|---|
| `epochs` | 17 | 15 | matches v2/v12 |
| `total_seq_len` | 2048 | 2048 | trainer pads to this |
| `max_anchors` | 512 | 512 | per-batch anchor budget |
| `lr` | 3e-5 | 3e-5 | with `scheduler_warmup_steps=100` |
| `block_size` | 8 | 8 | DFlash block size |
| `num_workers` | 1, prefetch=2 | 1, prefetch=2 | speculators dataloader |
| `on_missing` | skip | skip | tolerate per-row failures |
| `hidden_states_dtype` | bfloat16 | bfloat16 | applied after fp8 scale-back |
| `save_best` | True | True | writes `checkpoint_best/` |
| `drafter_intermediate_size` | 4096 | **6144** | recipe-faithful for MiniMax-M2.7 |
| `fp8_recipe` | None | `"current_fp8"` | Float8CurrentScaling HYBRID |
| `te_use_fused` | n/a | True | THE memory lever |
| `nan_skip` | True | True | defensive |
| `use_torchrun` | False | False | new default in 0.2.0+ |

---

## Implementation note: torchrun shell-out

The trainer **shells out** to `speculators/scripts/train.py`. We pick this over an in-process call because the speculators argparse interface is much more stable than its programmatic API.

In 0.2.0+ the default launcher is **direct python** (`sys.executable <train_script> ...`), not torchrun. This avoids the silent-bf16 trap on single-GPU FP8 runs. Pass `use_torchrun=True` for genuine multi-GPU runs.

You can supply a custom path to the speculators training script via `--train-script` or the `SPECULATORS_TRAIN_SCRIPT` env var. Default: `~/repos/speculators/scripts/train.py`.

The FP8 flags (`--fp8-recipe-kind`, `--fp8-split-accumulator`, `--te-use-fused`, `--drafter-intermediate-size`, `--nan-skip`) are only emitted when `fp8_recipe` is set, so an unpatched speculators install still consumes the bf16 path. To make speculators **understand** these flags, run:

```bash
python -m dflash_llama.scripts.patch_speculators_for_fp8 ~/repos/speculators
```

(idempotent; see `scripts/patch_speculators_for_fp8.py`).

---

## Verifying the smoke

The smoke wrapper enforces the v2 pass criteria:

- **rc == 124** — process was killed by `timeout`, meaning it ran the full 90 seconds without crashing.
- **Log shows `global_step=N`** with `N >= 1`.
- **No canonical failure markers** — `R54: hs prompt prefix mismatch`, `anchor_positions include padding`, `don't match input ids`, `t2d has`, `d2t has`.

For FP8 runs, **also grep for `[FP8]`** in the log. If there's no `[FP8] enabled: kind=current_fp8 ...` line, you've hit the silent-bf16 trap.

---

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

---

## What about layer 61 vs 62?

For MiniMax-M2.7 the tap list `[2, 16, 30, 45, 59, 61]` ends at the final residual stream. The speculators trainer auto-appends "the final layer", which it labels as 62. Semantically the two are the same hidden state — `tap_idx[5]` is the last residual. The library accepts whatever `layer_ids` the verifier config declares and passes `layer_ids[:-1]` to the trainer (so the trainer's auto-append re-creates the 6-tap input the drafter expects).

---

## The metric that matters: chained per-position acceptance

For speculative decoding, the metric you should track day-to-day is the **chained cumulative acceptance** ∏ p_i (k=1..N), not the per-position teacher-forced conditional. p_1 alone tells you nothing about how a draft of N=7 will perform — what matters is the joint probability that all 7 are accepted. `analyze.chain_pred_from_val()` computes this from `val_metrics.json` (training prediction); `analyze.chain_measured(parsed)` computes it from a `llama-speculative-simple` log (measured ground truth).

The daily-monitoring layout we use during a run:

```
[v12 epoch 4 step 305 | tau (chain ∏ p_i)]
  k=1: 0.812     k=5: 0.341
  k=2: 0.661     k=6: 0.271
  k=3: 0.530     k=7: 0.214
  k=4: 0.428
  → tau@dmax=7 = 0.214 (training-pred 0.219; z=−0.18σ)
```

That `tau@dmax=7` is the headline number — it predicts the speculative-decoding speedup directly. Lead with it in any training report. Per-position teacher-forced conditionals are diagnostic (they tell you which position is collapsing if `tau` is unexpectedly low) but they're not the headline metric.
