# Phase 3 Perf Optimization

## Goal

Phase 3 targeted a sustained throughput win over the existing Phase 2 baseline on
the repaired three-source pool while preserving training stability and real
convergence behavior.

Reference baseline:

- Phase 2 `C4` full-pool winner:
  `repro/artifacts/phase2_perf/c4_bs4_full3_finite_probe_fast.jsonl`
  - `throughput_tok_s=413231.921182266`
  - `loss_0=10.688`
  - `loss_final=8.5`
  - `wall_time_sec=290.086`

Reference bf16 anchor baseline for uplift accounting:

- `176231.2605 tok/s`

## What Changed

Three meaningful changes were introduced during Phase 3:

1. Real `TE fp8_model_init` parameter storage in the TE-wrapped DFlash model.
2. Exact Liger fused linear CE for the current weighted hard-CE DFlash loss.
3. Compiled flex-attention on the stable `fp8_params` stack, after fixing a
   harness bug that had been exporting `TORCHDYNAMO_DISABLE=1` and
   `TORCH_COMPILE_DISABLE=1` even for compile-enabled cells.

The winning configuration was:

- `C15`
- FP8 full TE fusion
- `TE_FP8_PARAMS=1`
- compiled flex attention enabled
- no Liger CE in the winning cell

## Winning Result

Short-run probe:

- artifact: `repro/artifacts/phase3_perf/C15_bs5_compile_fp8params.jsonl`
- config: `C15`
- micro batch: `5`
- `throughput_tok_s=9532509.090909092`
- uplift vs bf16 bs=1 baseline: `54.090909x`
- `status=OK`
- `loss_descending=true`
- `nan_skips=0`
- `fp8_receipt_ok=true`
- `split_accumulator_ok=true`

Ratification run:

- artifact: `repro/artifacts/phase3_perf/C15_bs5_compile_fp8params_200.jsonl`
- target step: `200`
- `throughput_tok_s=8738133.333333334`
- uplift vs bf16 bs=1 baseline: `49.583333x`
- `status=OK`
- `loss_0=10.625`
- `loss_final=5.5`
- `loss_descending=true`
- `nan_skips=0`
- `fp8_receipt_ok=true`
- `split_accumulator_ok=true`
- `step_time_ms=1205.128`
- `wall_time_sec=258.093`

## Convergence Check

The ratified winner was required to preserve wall-clock convergence, not just
raw throughput.

Phase 2 reference `C4`:

- loss descent per wall time:
  `(10.688 - 8.5) / 290.086 = 0.00754 loss/sec`

Phase 3 winner `C15` 200-step ratification:

- loss descent per wall time:
  `(10.625 - 5.5) / 258.093 = 0.01986 loss/sec`

So the winning Phase 3 config is materially better than the Phase 2 reference
under the requested wall-clock convergence criterion.

## Rejected Branches

- Exact Liger fused CE (`C18`) was functional and trainable, but did not beat
  the best non-Liger path at `bs=5`.
- Compiled flex + exact Liger (`C19`) removed the eager flex warning and was
  extremely fast on the short probe, but failed the convergence-parity check on
  the sustained run.
- Exact Liger `bs=6` runs remained host-fragile on the current `spark-2`
  environment, including one pre-step wedge and one dataloader-side
  `prefetch_factor` invariant when forcing `num_workers=0`.
- Liger RoPE (`C20`) patched correctly (`liger_rope=7` in coverage) but was not
  competitive enough on wall-clock progress to remain on the critical path.

## Conclusion

Phase 3 succeeded with `C15 bs=5`:

- gate passed on throughput
- sustained `200`-step stability passed
- convergence-per-wallclock check passed

This is the Phase 3 winner on `spark-2`.
