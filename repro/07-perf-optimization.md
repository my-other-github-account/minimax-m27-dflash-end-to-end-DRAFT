# 07 - Perf Optimization

## Goal

Find a stable Phase 2 training configuration on `spark-2` that delivers at least `2.0x` throughput versus the bf16 `C1 bs=1` baseline while preserving:

- no NaN runaway
- descending loss
- acceptable `tau`
- the existing DFlash anti-degenerate guardrails

## Final result

The winning configuration is:

- `C4`
- `micro_bs=4`
- `fp8_recipe_kind=current_fp8`
- full TE fusion enabled
- no Liger fused CE
- no Liger RoPE

Measured on the repaired three-source pool:

- baseline `C1 bs=1`:
  `176231.2605 tok/s`
- winner `C4 bs=4`:
  `413231.9212 tok/s`
- uplift:
  `2.3448x`

The run stayed stable:

- `status=OK`
- `nan_skips=0`
- `loss_0=10.688`
- `loss_final=8.500`
- `tau=1.028925`

## Artifacts

Repo-local copies of the relevant remote artifacts are in:

- `repro/artifacts/phase2_perf/c1_bs1_baseline_final.jsonl`
- `repro/artifacts/phase2_perf/c1_bs1_baseline_final.md`
- `repro/artifacts/phase2_perf/focus_fp8_bs2.jsonl`
- `repro/artifacts/phase2_perf/focus_fp8_bs2.md`
- `repro/artifacts/phase2_perf/c4_bs4_probe.jsonl`
- `repro/artifacts/phase2_perf/c4_bs4_probe.md`
- `repro/artifacts/phase2_perf/c4_bs4_full3_finite_probe_fast.jsonl`
- `repro/artifacts/phase2_perf/c4_bs4_full3_finite_probe_fast.md`

The decisive full-pool winner is:

- `repro/artifacts/phase2_perf/c4_bs4_full3_finite_probe_fast.jsonl`

## What changed

### 1. Expanded TE fusion

The final winning path uses full TE fusion:

- Qwen3 MLP fusion via `te.LayerNormMLP`
- DFlash attention `input_layernorm + q_proj` fusion via `te.LayerNormLinear`
- final `norm + lm_head` fusion

Winning FP8 receipt summary:

- `TE_VERSION=2.16.0.dev0+76c2a9e9`
- all three GEMM paths show `use_split_accumulator=True`
- fusion coverage after wrapping:
  - `te_layernorm_mlp=6`
  - `te_layernorm_linear=7`
  - `te_linear=20`
  - `unfused=0`

### 2. Liger outcome

Liger fused CE and RoPE were integrated and benchmarked, but neither variant was stable enough to win:

- `C5 bs=2`:
  faster than `C4 bs=2`, but `loss_descending=false`, `tau=1.0`
- `C6 bs=2`:
  same issue

So the final winning recipe keeps Liger disabled for the training path.

### 3. Data-pool repair

The original three-source pool was not benchmarkable as-is.

Root causes:

- `train_paired_v3` had a broken staged-pool path
- it also contained many bad rows:
  - `71` zero-anchor prompt rows
  - `12480` rows with non-finite hidden states

Final pool-build remediation in `perf_sweep.py`:

- filter zero-anchor rows
- filter non-finite hidden-state rows for `train_paired_v3`
- preserve prompt/hidden-state index alignment after filtering

Final repaired three-source pool size:

- `41163` rows

That repaired pool is what the winning `C4 bs=4` result used.

## Sweep summary

### Baseline

`C1 bs=1`

- `throughput_tok_s=176231.2605`
- `step_time_ms=11200.0`
- `tau=1.028687`
- `status=OK`

### FP8 candidates

`C4 bs=2`

- `throughput_tok_s=277768.4768`
- `tau=1.034335`
- `status=OK`

`C5 bs=2`

- `throughput_tok_s=281496.9128`
- `loss_descending=false`
- `tau=1.0`
- `status=other`

`C6 bs=2`

- `throughput_tok_s=281496.9128`
- `loss_descending=false`
- `tau=1.000195`
- `status=other`

### Stable-subset scaling check

`C4 bs=4` on the stable subset (`iq4_v10 + v4_renum`)

- `throughput_tok_s=427990.2041`
- `tau=1.029637`
- `status=OK`
- uplift vs baseline:
  `2.43x`

### Final full-pool winner

`C4 bs=4` on the repaired three-source pool

- `throughput_tok_s=413231.9212`
- `step_time_ms=20200.0`
- `tau=1.028925`
- `nan_skips=0`
- `status=OK`
- uplift vs baseline:
  `2.3448x`

## Recommended launch shape

The validated winning shape is:

```bash
python repro/scripts/training/perf_sweep.py \
  --config-id C4 \
  --micro-bs 4 \
  --data-sources-path /tmp/perf_src_full3_repaired.json \
  --target-step 10
```

Config meaning:

- `C4` = FP8 current recipe + full TE fusion
- no Liger fused CE
- no Liger RoPE

## Conclusion

Phase 2 performance optimization succeeded.

The successful path was not Liger fused CE; it was:

- full TE fusion
- FP8 current recipe
- batch-size scaling to `bs=4`
- and filtering invalid `train_paired_v3` rows out of the intended three-source pool
