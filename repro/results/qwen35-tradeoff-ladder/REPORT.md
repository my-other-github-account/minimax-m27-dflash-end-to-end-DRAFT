# TRADEOFF LADDER QWEN35 CANONICAL

Binary md5: 49d9a23d4eb2a655c4f35a18a7a1905a
Prompts: $WORK/t_87c41e13/eval_prompts_50.jsonl
Regime: max_tokens=512, think_max_tokens=256, max_ctx=8192, temperature=0, thinking enabled, prefix_cache_slots=0

| width | AL_true | wall_speedup_vs_w0 | decode_tok_s | reasoning_tokens_total | visible_nonempty | per_pos_len |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | 42.243109 | 0 | 50/50 | 0 |
| 1 | 1.000000 | 0.677309 | 28.224855 | 0 | 50/50 | 0 |
| 2 | 1.847842 | 1.167625 | 49.576470 | 0 | 50/50 | 1 |
| 4 | 3.005400 | 1.657941 | 71.851582 | 0 | 50/50 | 3 |
| 8 | 4.170740 | 1.945501 | 85.425023 | 0 | 50/50 | 7 |
| 12 | 5.344468 | 1.590571 | 68.859038 | 0 | 50/50 | 11 |
| 16 | 6.165703 | 1.776376 | 77.502013 | 0 | 50/50 | 15 |

Trend: PEAK-then-fall at width 8.
Best full-wall speedup: width 8 = 1.945501x.

Note: usage.reasoning_tokens_total was 0 for this server response schema despite thinking enabled; every row had nonempty visible text, satisfying the honest visible-text fallback gate.

JSON: $WORK/t_TRADEOFF_LADDER/ladder_metrics.json
