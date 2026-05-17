# §6 — FP8 training with TransformerEngine

> ⚠️ **DRAFT.** End-to-end-validated **through 2,655 steps** on DGX Spark sm_121a — zero NaN, slightly *better* per-step τ than the bf16 baseline (see [§6.5 Verified results](#65-verified-results)). Full-convergence FP8 (matching v11's bf16 τ ≈ 2.41 at step ~32k) has not yet been demonstrated end-to-end.

This page documents the production-stable recipe for training a DFlash drafter
with [NVIDIA TransformerEngine](https://github.com/NVIDIA/TransformerEngine) FP8
GEMMs and fused LayerNormMLP on a single DGX Spark (GB10, sm_121a). It is the
sequel to [§2 — Training a DFlash drafter](02-training.md), which covered the
bf16 path.

## Placeholder key

| Placeholder | Meaning | Example |
|---|---|---|
| `${WORKSPACE}` | top-level work dir | `/opt/dflash` |
| `${TE_VENV}` | dedicated venv with TE source-built | `${HOME}/venvs/te` |
| `${VLLM_VENV}` | venv with speculators + datasets + transformers | `${HOME}/venvs/vllm` |
| `${POOL}` | paired-data pool (prompts arrow + symlink farm) | `${WORKSPACE}/pool` |
| `${SAVE_DIR}` | where checkpoints land | `${WORKSPACE}/checkpoints/v12_fp8` |

## 6.1 — The two bugs that bit the first launch (READ THIS FIRST)

The 2026-05-04 v12 production launch silently went NaN at step 38 after appearing
to train healthily for 35 steps. Wasted ~5 hours of compute. The root cause was
**two independent bugs that compound** — fixing only one is not enough.

### Bug A — `torchrun --nproc-per-node=1` silently skips the FP8 wrap

The speculators trainer's `setup_model()` has two branches:
- Single-GPU branch (`is_distributed=False`)
- FSDP branch (`is_distributed=True`)

`is_distributed=True` triggers when env vars `RANK`, `WORLD_SIZE`, etc. are
present — which `torchrun --nproc-per-node=1` **always** sets, even on a single
GPU. The FP8/TE wrap helper `_maybe_wrap_te()` only runs in the single-GPU
branch. With torchrun, you get **silent bf16 with no error or warning**.

**Symptom**: launch is clean, no errors, step-0 loss is in the right range, but
the `[FP8] TE_VERSION=...` log line is missing. The model trains at the bf16
throughput, not the +18% FP8 throughput.

**Fix**: launch with plain `python scripts/train.py`, not `torchrun`. There is
no functional benefit to torchrun on a single GPU; it only exists for multi-GPU
coordination. The `DFlashTrainer.train()` wrapper in this repo automatically
drops torchrun whenever `fp8_recipe_kind` is set.

### Bug B — Bare `Float8CurrentScaling()` overflows fprop at LR ~1.2e-4

TE's default `Float8CurrentScaling()` constructor sets:

```
fp8_gemm_fprop=MMParams(use_split_accumulator=False)   ← THE BUG
fp8_gemm_dgrad=MMParams(use_split_accumulator=True)
fp8_gemm_wgrad=MMParams(use_split_accumulator=True)
```

NVIDIA's throughput-default `use_split_accumulator=False` saves ~3-5% on short
benches, but on a real DFlash drafter at `intermediate=6144` with a linear LR
warmup, the unsplit fprop accumulator silently **overflows to inf around step
38–40** once LR-warmup crosses ~1.0e-4 to 1.2e-4. NaN propagates, `clip_grad_norm_`
is helpless (norm of NaN is NaN), `optimizer.step()` poisons all weights,
checkpoints save the frozen-NaN model.

**The 200-step microbench does not catch this** — it either doesn't reach the
dangerous LR range or uses synthetic inputs with thinner-tailed activations than
real DFlash data.

**Fix**: explicitly construct the recipe with split-accumulator ON for all
three GEMMs:

```python
from transformer_engine.common.recipe import Float8CurrentScaling, Format, MMParams

recipe = Float8CurrentScaling(
    fp8_format=Format.HYBRID,
    fp8_gemm_fprop=MMParams(use_split_accumulator=True),  # MUST OVERRIDE
    fp8_gemm_dgrad=MMParams(use_split_accumulator=True),
    fp8_gemm_wgrad=MMParams(use_split_accumulator=True),
)
```

The `te_wrap.get_recipe("current_fp8")` factory in this repo does this correctly.
**Verified 2026-05-05**: identical run that NaN'd at step 38 with bare default
trained cleanly past step 2,655 with split-acc forced on all 3. Cost: <5% throughput.

### Defense in depth — gradient-level NaN-skip

Even with both fixes above, FP8 has narrower dynamic range than bf16. A single
unlucky batch could still produce a NaN grad. Patch 04 in this repo adds a
**gradient-level** NaN-skip guard between `clip_grad_norm_` and `optimizer.step()`:

```python
has_nan_grad = any(
    p.grad is not None and not torch.isfinite(p.grad).all()
    for p in self.model.parameters()
)
if has_nan_grad:
    root_logger.warning(f"[NaN-SKIP] step={self.global_step} skipping optimizer step")
    self.opt.zero_grad()
    self.global_step += 1
    continue  # bypass opt.step + scheduler.step + logging
```

With Bug B fixed, this guard never fires in production (verified 2,655+ clean
steps). It's insurance for unknown future failure modes.

> **Critical**: do NOT rely on a *loss-level* `torch.isnan(loss)` guard alone. By
> the time loss is NaN, `loss.backward()` has already populated NaN gradients
> across all parameters, and even if you skip `opt.step()`, the next iteration's
> `clip_grad_norm_` may still scale them. The gradient-level guard is correct
> because it runs AFTER `clip_grad_norm_` and only skips the optimizer step,
> leaving the weights intact.

## 6.2 — Environment setup

You need a venv with NVIDIA TransformerEngine **source-built** for sm_121a.
The pre-built wheel does not include the sm_121a kernels on most distributions.

```bash
# Create a dedicated venv (don't mix TE with vllm).
python3 -m venv ${TE_VENV}
source ${TE_VENV}/bin/activate
pip install torch  # match your CUDA version
pip install ninja pybind11

# Clone and build TE
git clone https://github.com/NVIDIA/TransformerEngine.git /tmp/TransformerEngine
cd /tmp/TransformerEngine
git submodule update --init --recursive
export TORCH_CUDA_ARCH_LIST="12.1a"
export NVTE_CUDA_ARCHS=121a
NVTE_FRAMEWORK=pytorch pip install -e .

# Verify
python -c "import transformer_engine.pytorch as te; print(te.__version__)"
```

Then install `dflash-llama` itself into the same venv (the trainer imports
`dflash_llama.training.te_wrap`):

```bash
pip install -e ${WORKSPACE}/dflash-llama-repo
```

Speculators must also be installed and **patches/01-06 applied** (see
[§0 step 5](00-spark-from-scratch.md#step-5-install-speculators)). Patches 04, 05,
and 06 add the in-epoch val, FP8/TE wrap, and CLI flags respectively.

## 6.3 — Launch via DFlashTrainer (Python API)

```python
from dflash_llama.training import DFlashTrainer
from dflash_llama.verifiers import load_verifier

trainer = DFlashTrainer(
    traces_dir="${POOL}/traces",
    verifier=load_verifier("minimax-m2.7-iq4-xs", hf_path="${WORKSPACE}/verifier_meta"),
    num_layers=6,                  # v12 used 6L (v11 was 5L)
    draft_vocab_size=32768,
    paired_dir="${POOL}/paired",
)
trainer.prepare()

# Optional but recommended: 90-second smoke first
smoke = trainer.smoke(
    fp8_recipe_kind="current_fp8",
    te_use_fused=True,
    te_fp8_params=True,
    compile_flex_attention=True,
)
assert smoke.passed, smoke.message

# Full training run
result = trainer.train(
    save_to="${SAVE_DIR}",
    epochs=15,
    lr=3e-4,
    max_anchors=1024,
    total_seq_len=2048,
    log_freq=5,
    scheduler_warmup_steps=100,
    save_best=True,
    # FP8 / TE
    fp8_recipe_kind="current_fp8",
    te_use_fused=True,
    te_fp8_params=True,
    compile_flex_attention=True,
    # In-epoch validation (recommended for long runs)
    val_every_steps=145,
    val_in_epoch_max_batches=80,
    save_every_n_vals=1,
)
print(result)  # {"rc": 0, "log_path": ..., "save_path": ...}
```

When `fp8_recipe_kind` is non-empty, `DFlashTrainer.train()`:
1. Drops torchrun in favor of direct `python scripts/train.py` invocation (Bug A fix).
2. Passes `--fp8-recipe-kind current_fp8 --te-use-fused` to the underlying trainer.
3. `te_fp8_params=True` sets `TE_FP8_PARAMS=1`, and `wrap_with_te(model, te_fp8_params=True)` creates TE layers under `te.fp8_model_init(...)` so weight storage uses TE's FP8-resident container.
4. `compile_flex_attention=True` sets `DFLASH_COMPILE_FLEX=1` and removes `TORCHDYNAMO_DISABLE=1` / `TORCH_COMPILE_DISABLE=1` from the launched subprocess env, which is the switch that lets the pre-existing compiled flex-attn path actually run.
5. The trainer's patched `setup_model()` calls `_maybe_wrap_te()` after `model.to(local_rank)`.
6. `_maybe_wrap_te()` calls `wrap_with_te(model)` and `get_recipe("current_fp8")` from `dflash_llama.training.te_wrap` — both producing the production-stable split-accumulator recipe (Bug B fix).
7. Every train/val forward pass runs under `te.fp8_autocast(enabled=True, fp8_recipe=recipe)`.
8. The gradient-level NaN-skip guard (patch 04) sits between `clip_grad_norm_` and `optimizer.step()`.

Saved C15 checkpoints remain usable by the library inference path:

- The speculators checkpointer materializes TE FP8 tensors to plain CPU tensors
  before `save_pretrained()`.
- `DFlashTrainer.offline_eval()` normalizes TE-fused checkpoint keys back to the
  eager layout before loading.
- `export_to_gguf()` uses the same normalization on read, so GGUF export stays a
  weights-only transform with no dependency on training-time compile state.

## 6.4 — Verification checklist (run AT LAUNCH, do not skip)

After kicking off the run, wait 60 seconds, then:

1. **Confirm the `[FP8]` log line appeared**:
   ```bash
   grep "\[FP8\]" ${LOG} | head -3
   ```
   You should see something like:
   ```
   [FP8] TE_VERSION=2.14.1+... recipe=Float8CurrentScaling(...) fused=True linears_before={...} linears_after={..., te_layernorm_mlp: 6, te_linear: 27, ...}
   ```
   If absent → **silent-bf16 trap (Bug A)**. Kill the run and verify launcher is direct python.

2. **Confirm split-accumulator is True on all 3 GEMMs**:
   ```bash
   grep -oE "fp8_gemm_(fprop|dgrad|wgrad)=MMParams\\(use_split_accumulator=(True|False)" ${LOG} | sort -u
   ```
   You should see exactly three lines, ALL `=True`. If `fprop=False` → **Bug B**. Kill the run and verify `te_wrap.get_recipe("current_fp8")` is being used.

3. **Confirm step-0 loss is in the expected range**: ~10–12 for a fresh DFlash drafter.

4. **Wait 12+ minutes** (≈ to step 50 at 1 step/min) and verify loss is still finite:
   ```bash
   grep -oE "train/loss=[0-9.naN-]+" ${LOG} | tail -10
   ```
   No `nan` anywhere. If you see NaN, you fell into Bug B despite the recipe construction — re-check by grepping the `[FP8]` line for split-accumulator status (step 2 above).

5. **Confirm zero NaN-SKIP events** (insurance fired = something's wrong):
   ```bash
   grep -c "NaN-SKIP" ${LOG}
   ```
   Should be `0`. A non-zero count is not fatal — the guard did its job — but it
   means at least one batch produced NaN gradients, which warrants investigation.

6. **Peak GPU memory** (`nvidia-smi`): should be ~64–65 GB at `intermediate=6144,
   micro-bs=1, FP8CS+fused`. Significantly higher (>72 GB) suggests the fused
   MLP didn't fire (`grep "te_layernorm_mlp"` count = 0 → check `--te-use-fused`).

If any check fails, kill the run and fix. A 12-hour silent-bf16 run wastes a
day of compute that you'll only discover at convergence-comparison time. A
5-hour FP8-default run wastes a half-day because step 0–35 looks healthy and
val at step 145 is the first hard signal — don't wait that long, check at step 50.

## 6.5 — Verified results (DGX Spark sm_121a, 2026-05-05)

| Metric | bf16 baseline | FP8CS(HYBRID) + fused | Δ |
|---|---|---|---|
| Throughput | 114 tok/s | **135 tok/s** | **+18%** |
| Peak GPU | 63.0 GB | 64.4 GB | +1.4 GB (+2.2%) |
| Wall / 200 steps | 3,579s | 3,035s | −15% |
| NaN events | 0 | **0** | ✓ |
| NaN-SKIP guard fires | 0 | **0** | ✓ |
| Steps trained (max) | ~33,000 | **2,655** | partial |

Step-for-step τ comparison (chain-cumulative, the metric that matters; same trace
pool, same seed, same LR schedule, vals every 145 steps):

| step | v11 bf16 τ | v12 FP8 τ | Δτ |
|---|---|---|---|
| 145 | 1.056 | 1.063 | **+0.007** |
| 580 | 1.257 | 1.349 | **+0.092** |
| 725 | 1.308 | 1.425 | **+0.117** ← biggest early gap |
| 1,160 | 1.506 | 1.561 | **+0.055** |
| 1,740 | 1.653 | 1.662 | +0.009 |
| 2,030 | 1.691 | 1.718 | +0.028 |
| 2,320 | 1.758 | 1.772 | +0.013 |
| **2,610** | **1.795** | **1.798** | **+0.003** |

**FP8 was ahead at every single matched step.** The gap was largest in the
early-mid stretch (peaked at +0.117 τ around step 725) and converged toward
parity by step ~2,000. Combined with the +18% throughput, FP8 is **strictly
better** step-for-step *and* walltime-for-walltime through the validated window.

> The comparison only covers the first ~8% of training. The bf16 baseline went
> on to reach τ=2.46 at step ~32,770; FP8 was halted at step 2,610. **Full-convergence
> FP8 has not yet been demonstrated end-to-end.** Whether FP8 holds parity all
> the way to convergence is an open empirical question — but the early-window
> trajectory is encouraging.

## 6.6 — What was tried and ruled out

These alternative precision recipes were tested and **rejected** for production on sm_121a:

- **`Float8CurrentScaling(HYBRID)` with default `use_split_accumulator=False`** — NaNs at step 38 (Bug B above). Fixed by forcing split-acc=True everywhere.
- **`MXFP8BlockScaling`** — not available in the TE 2.14.1 build for sm_121a (only sm_10x).
- **`Float8BlockScaling`** — available but not benchmarked in the v12 spike; current_fp8 was confirmed stable first and the campaign moved on.
- **`NVFP4BlockScaling` (NVFP4 with Harry-Chen polyfill)** — built, but polyfill is incomplete: misses `mul_cvt_bf16_to_fp4_8x_stochastic_rounding` (TE's `ptx.cuh:935`). Production-shape FFN fails with per-thread arch-specific PTX errors. A 64×64 SR-variance microbench appears to pass but doesn't actually exercise the missing instruction — **don't trust the microbench**, the real workload fails.
- **`DelayedScaling` (legacy E4M3)** — recipe is exposed via `get_recipe("delayed_e4m3")` for documentation continuity but has not been benchmarked against current_fp8 on sm_121a.

## 6.7 — Resuming an FP8 run

The trainer's checkpointer saves the model in the wrapped (TE) form. To resume:

```python
trainer.train(
    save_to="${SAVE_DIR}",        # same path as before; checkpointer auto-detects last ckpt
    fp8_recipe_kind="current_fp8",  # MUST match the original run; mixing FP8/bf16 weights is unsafe
    te_use_fused=True,
    # other args same as before
)
```

The trainer's `setup_trainer()` calls `checkpointer.previous_epoch` to detect the
resume point, then `setup_model()` calls `_maybe_wrap_te()` AFTER
`load_model_state_dict`. State-dict keys match because TE-wrapped modules use
the same param names as `nn.Linear` / `nn.RMSNorm` (TE's `Linear.weight` /
`LayerNormMLP.fc1_weight` etc. — verify match via `model.state_dict().keys()` on
the first resume attempt; if you see "unexpected keys" the wrap order is wrong).

## 6.8 — Cross-references

- [§2 — Training a DFlash drafter](02-training.md) — bf16 baseline path
- [`src/dflash_llama/training/te_wrap.py`](../src/dflash_llama/training/te_wrap.py) — the wrap helpers and recipe factory
- [`patches/speculators/04-trainer-nan-guard-and-midepoch-ckpt.patch`](../patches/speculators/04-trainer-nan-guard-and-midepoch-ckpt.patch) — in-epoch val + grad-level NaN-skip
- [`patches/speculators/05-trainer-te-fp8-wrap.patch`](../patches/speculators/05-trainer-te-fp8-wrap.patch) — TE wrap + fp8_autocast forward
- [`patches/speculators/06-train-script-fp8-flags.patch`](../patches/speculators/06-train-script-fp8-flags.patch) — `--fp8-recipe-kind` / `--te-use-fused` CLI flags
- [`patches/speculators/08-checkpointer-te-fp8-save.patch`](../patches/speculators/08-checkpointer-te-fp8-save.patch) — materialize TE FP8 tensors before checkpoint save
- [`repro/scripts/training/launch_full_fp8.sh`](scripts/training/launch_full_fp8.sh) — reference launcher
