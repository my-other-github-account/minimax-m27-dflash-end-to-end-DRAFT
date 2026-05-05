# Changelog

All notable changes to dflash-llama. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project follows semver.

## 0.2.0 (2026-05-05)

FP8 production training is real. Verified end-to-end on the Spark cluster
(DGX Spark GB10, sm_121) against MiniMax-M2.7-IQ4-XS v12 (35,891
self-describing v3 traces) on 2026-05-05.

### Added

- **FP8 production training**: `Float8CurrentScaling` HYBRID + fused
  `te.LayerNormMLP`, **+42% throughput** on sm_120/121. Validated on
  MiniMax-M2.7-IQ4-XS v12 run 2026-05-05; survives 305+ steps with zero
  NaN-skip events.
- `dflash_llama.training.fp8` module with:
  - `FP8Recipe` dataclass (`use_split_accumulator=True` is the default)
  - `make_te_recipe()` factory (forces split-accumulator on all 3 GEMMs)
  - `wrap_with_te()` module wrapper (Linear → te.Linear, mlp+ln → te.LayerNormMLP)
  - `fp8_autocast_ctx()` context-manager helper
  - `current_arch()` capability probe
- `DFlashTrainer.train()` kwargs:
  - `fp8_recipe: FP8Recipe | str | None = None`
  - `te_use_fused: bool = True`
  - `drafter_intermediate_size: int | None = None`
  - `nan_skip: bool = True`
  - `use_torchrun: bool = False` (NEW DEFAULT — see Changed)
- New CLI flags on `dflash-llama train`: `--fp8-recipe`, `--no-te-fused`,
  `--drafter-intermediate-size`, `--no-nan-skip`, `--use-torchrun`.
- New CLI subcommand `dflash-llama check-fp8` — arch capability probe.
- New repro doc [`repro/04-fp8-bringup.md`](repro/04-fp8-bringup.md) — TE
  source-build with all four blockers (cuBLAS LD_PRELOAD trick,
  NVTE_CUDA_ARCHS=121, sibling te venv with .pth linkage, ninja from vllm
  venv).
- New repro section in [`repro/02-training.md`](repro/02-training.md) with
  the v12 launcher + Pitfalls + chained per-position metric layout.
- `scripts/patch_speculators_for_fp8.py` — idempotent patcher applying
  TrainerConfig fields + `_maybe_wrap_te` helper + fp8_autocast forward
  wrapper + NaN-skip optimizer guard to a speculators install.
- 18 new tests in `tests/test_fp8_recipe.py` covering the recipe
  dataclass, arch refusal logic, silent-bf16 trap guard, train-cmd flag
  plumbing.

### Changed

- **`DFlashTrainer.train()` default launcher is now direct python**
  (`use_torchrun=False`), not torchrun. This avoids the silent-bf16 trap
  on single-GPU FP8 runs (torchrun --nproc-per-node=1 sets RANK/WORLD_SIZE
  which routes speculators through its FSDP branch — and the TE wrap only
  lives in the single-GPU branch). Pass `use_torchrun=True` for genuine
  multi-GPU runs.
- The trainer now refuses `use_torchrun=True` combined with `fp8_recipe`
  set on a single-GPU machine (`WORLD_SIZE=1`) — would otherwise silently
  drop to bf16 with no log line indicating the regression.
- README + repro docs lead with the chained per-position acceptance ∏ p_i
  as the headline training metric, not the per-position teacher-forced
  conditional.

### Refused (hard errors at recipe-construction time)

- `Float8BlockScaling` (`kind="block_fp8"`) on sm_120/121: silently
  NON-CONVERGENT (TransformerEngine #2382). Raises with skill-citation.
- `MXFP8` via TE (`kind="mxfp8"`) on sm_120/121: cuBLAS lacks non-TN GEMM
  layouts (TransformerEngine #2668). Raises with workaround pointer
  (torchao.prototype.mx_formats with TORCH_CUDA_ARCH_LIST="12.1a").

### Critical lessons reflected in the code

- **split_accumulator=True is forced on ALL three GEMMs** (fprop + dgrad
  + wgrad). Without this, the recipe NaN's at step ~40 on noised DFlash
  drafters when LR crests ~1.2e-4. With this, identical recipe / data /
  LR survives 305+ steps cleanly. This is the most important hard-won
  default in the library.
- The **fused te.LayerNormMLP**, not FP8 alone, is what makes
  `intermediate=6144` fit. FP8 weights/GEMM saved roughly zero activation
  memory at v11 sizes; the cliff is the FFN backward graph
  (O(intermediate²) without grad checkpointing), and fusion is what
  eliminates intermediate norm materialization.
- **NVFP4 (`disable_rht=True`) is silently broken on sm_121**. Variance
  test: 8 trials × different SR seeds → bit-identical losses ⇒ SR not
  firing. The Harry-Chen polyfill restores SR for small shapes but
  production-shape FFN backward (`mul_cvt_8x` at `ptx.cuh:935`) hits
  unfixable arch-specific PTX errors. NVFP4 is research-only on Spark;
  there is no NVFP4 path in the library.
- v12 pool is +46.5% larger than v11 (35,891 self-describing v3 traces
  vs 24,492 v11 prompts). The library's `assemble_prompts_arrow` exposes
  the assemble-from-trace-dir step.

### Migration from 0.1.0

- Existing bf16 callers: no action required. `fp8_recipe=None` is the
  default, and the dropped-torchrun launcher is functionally equivalent
  for bf16.
- Production users on Spark / Hopper: opt in via
  `trainer.train(fp8_recipe="current_fp8", drafter_intermediate_size=6144)`
  and run `python scripts/patch_speculators_for_fp8.py
  ~/repos/speculators` to teach speculators to consume the new flags.

## 0.1.0 (2026-04-30)

Initial release. Self-describing fp8 trace generation, DFlash drafter
training (bf16, via shell-out to speculators trainer), GGUF export,
OpenAI-compatible serving, per-position + chain-cumulative speculative
decoding benchmark. 79 tests; validated end-to-end on
MiniMax-M2.7-IQ4-XS.
