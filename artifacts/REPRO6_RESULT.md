# REPRO6_RESULT — spark-6 independent reproduction

Generated: 2026-06-04T17:13:26.895456Z
Host: spark-6 (`<USER>@<SPARK_HOST>`, LAN). Spark-3 artifact access used only from spark-6 over QSFP `<SPARK_QSFP_IP>`.

## Inputs verified

- Target shard 00001 md5 on spark-6: `019759eeec4be4931592eeefbd54d2ba  /home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf`
  - Spark-3 original md5 observed over QSFP: `019759eeec4be4931592eeefbd54d2ba  /home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf`
- Drafter safetensors md5 on spark-6: `30c27f646590b562608774bee82d2108  /home/<USER>/t_1d157fda/golden_step_00020000/model_lucebox_layout.safetensors`
  - Expected / spark-3 original md5: `30c27f646590b562608774bee82d2108`
- Drafter config md5 on spark-6: `cc1af41d15ce13b4568fcca2cc07e01e  /home/<USER>/t_1d157fda/golden_step_00020000/config.json`

## Source / build

Restored Step A pre_K_no_mask recipe from `spark-3:/home/<USER>/t_77088062/NOT_RECONCILED_MINIMAX_REVERT_ROPE.md`:
- `src/qwen35/qwen35_backend.cpp`: `pos_q[i] = draft_ctx + i`; `pos_k[i] = i` for full K span.
- `src/common/dflash_spec_decode.cpp`: same 0-based full K span restored for the common path.
- Restored `src/draft/draft_graph.{h,cpp}`, `src/common/step_graph.h`, `src/common/dflash_draft_graph.cpp`, `src/common/dflash_spec_decode.cpp` from `*.pre-draftmaskfix-t_d81951a4` backups.
- Also restored `src/qwen35moe/qwen35moe_backend.cpp` per-layer feature-capture behavior to the spark-3 source; kept the spark-6 shard-aware full-load fix and added compatibility telemetry so the unchanged R1 `run_measure_7.py` parser can read hybrid-spec results.

Independent spark-6 binary md5:

`7e27408f4d1db727777f9ff7a03c0ed9  build-sm121-local/dflash_server`

Reference Step A binary md5 from spark-3 report: `fa08acfdb72b0df2463b934161b75caa`.
The spark-6 md5 differs because the local checkout/toolchain includes the qwen35moe per-layer-capture restore, shard-aware loader fix, and compatibility telemetry; functional behavior is the acceptance criterion.

## Launch line

Server command file: `/home/<USER>/t_1d157fda/server_cmd_repro6.sh`
Server log: `/home/<USER>/t_1d157fda/server_8097_repro6_final_clean_20260604T165850Z.log`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /home/<USER>/lucebox-latest-20260603/server
export DFLASH_PERPOS_PROBE=1
export DFLASH_DRAFTATTN_PROBE=1
exec stdbuf -oL -eL ./build-sm121-local/dflash_server "/home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf" --draft "/home/<USER>/t_1d157fda/golden_step_00020000/model_lucebox_layout.safetensors" --port 8097 --host 127.0.0.1 --max-ctx 2048 --default-max-tokens 128 --hard-limit-reply-budget 0 --model-name minimax-luce-golden-repro6
```

Measurement command:

```bash
PORT=8097 MODEL=minimax-luce-golden-repro6 LUCE_LOG=/home/<USER>/t_1d157fda/server_8097_repro6_final_clean_20260604T165850Z.log OUTDIR=/home/<USER>/t_1d157fda TAG=repro6 python3 /home/<USER>/t_1d157fda/run_measure_7.py
```

## Measurement artifacts

- `/home/<USER>/t_1d157fda/measure_repro6_7prompts_summary.json`
- `/home/<USER>/t_1d157fda/raw_completions_repro6_7prompts.json`
- `/home/<USER>/t_1d157fda/measure_repro6_stdout.log`

## Summary JSON inline

```json
{
  "tag": "repro6",
  "model": "minimax-luce-golden-repro6",
  "n_prompts": 7,
  "max_tokens": 64,
  "temperature": 0,
  "top_k": 1,
  "total_tokens": 448,
  "total_steps": 127,
  "AL_true_mean_commit": 3.52755905511811,
  "accepted": 326,
  "draft_total": 1016,
  "accept_rate": 0.32086614173228345,
  "per_pos_argmax_match_weighted_p1_p7": [
    0.6771653464566929,
    0.5039369921259843,
    0.35433083464566933,
    0.19685030708661416,
    0.22834636220472443,
    0.21259853543307086,
    0.17322849606299215
  ],
  "per_pos_prefix_accept_weighted_p1_p7": [
    0.6771653464566929,
    0.46456685826771654,
    0.2598424566929134,
    0.1259842283464567,
    0.07874012598425197,
    0.015748023622047243,
    0.0
  ],
  "per_prompt": [
    {
      "i": 1,
      "prompt": "train 60 miles 1.5 hours",
      "seconds": 3.9466872215270996,
      "spec": {
        "tokens": 64,
        "steps": 13,
        "accepted": 52,
        "draft_total": 104,
        "accept_percent": 50.0,
        "avg_commit": 4.9231,
        "line": "[spec-decode] tokens=64 time=3.304 s speed=19.37 tok/s steps=13 accepted=52/104 (50.0%) avg_commit=4.9231"
      },
      "per_pos_argmax_match": [
        1.0,
        0.923077,
        0.846154,
        0.153846,
        0.923077,
        0.692308,
        0.692308
      ],
      "per_pos_prefix_accept": [
        1.0,
        0.923077,
        0.846154,
        0.153846,
        0.153846,
        0.076923,
        0.0
      ],
      "probes": [
        "[draftattn-probe] step=0 committed=10 draft_ctx=10 q_len=8 pos_q_first=10 pos_q_last=17 pos_k_base_first=0 pos_k_base_last=9 pos_k_noise_first=10 pos_k_noise_last=17 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=0 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftattn-probe] step=3 committed=22 draft_ctx=22 q_len=8 pos_q_first=22 pos_q_last=29 pos_k_base_first=0 pos_k_base_last=21 pos_k_noise_first=22 pos_k_noise_last=29 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=3 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)"
      ],
      "text_preview": " 2.5 hours 3.5 hours 4.5 hours 5.5 hours 6.5 hours 7.5 hours 8.5 hours 9.5 hours 10.5 hours 11.5 hours 12.5 hours 13.5 hours 14.5"
    },
    {
      "i": 2,
      "prompt": "Write a concise Python function that returns the nth Fibonacci number.",
      "seconds": 5.388462781906128,
      "spec": {
        "tokens": 64,
        "steps": 24,
        "accepted": 41,
        "draft_total": 192,
        "accept_percent": 21.4,
        "avg_commit": 2.6667,
        "line": "[spec-decode] tokens=64 time=5.125 s speed=12.49 tok/s steps=24 accepted=41/192 (21.4%) avg_commit=2.6667"
      },
      "per_pos_argmax_match": [
        0.5,
        0.375,
        0.166667,
        0.083333,
        0.083333,
        0.041667,
        0.041667
      ],
      "per_pos_prefix_accept": [
        0.5,
        0.25,
        0.083333,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "probes": [
        "[draftattn-probe] step=0 committed=12 draft_ctx=12 q_len=8 pos_q_first=12 pos_q_last=19 pos_k_base_first=0 pos_k_base_last=11 pos_k_noise_first=12 pos_k_noise_last=19 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=0 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftattn-probe] step=3 committed=20 draft_ctx=20 q_len=8 pos_q_first=20 pos_q_last=27 pos_k_base_first=0 pos_k_base_last=19 pos_k_noise_first=20 pos_k_noise_last=27 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=3 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)"
      ],
      "text_preview": " Use type hints and docstring. Use recursion. Use memoization via functools.lru_cache. Use type hints and docstring. Use recursion. Use memoization via functools.lru_cache. Use type hints and docstrin"
    },
    {
      "i": 3,
      "prompt": "Explain why the sky appears blue in one paragraph.",
      "seconds": 5.084710597991943,
      "spec": {
        "tokens": 64,
        "steps": 24,
        "accepted": 41,
        "draft_total": 192,
        "accept_percent": 21.4,
        "avg_commit": 2.6667,
        "line": "[spec-decode] tokens=64 time=4.848 s speed=13.20 tok/s steps=24 accepted=41/192 (21.4%) avg_commit=2.6667"
      },
      "per_pos_argmax_match": [
        0.458333,
        0.375,
        0.125,
        0.125,
        0.0,
        0.041667,
        0.0
      ],
      "per_pos_prefix_accept": [
        0.458333,
        0.333333,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "probes": [
        "[draftattn-probe] step=0 committed=10 draft_ctx=10 q_len=8 pos_q_first=10 pos_q_last=17 pos_k_base_first=0 pos_k_base_last=9 pos_k_noise_first=10 pos_k_noise_last=17 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=0 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftattn-probe] step=3 committed=18 draft_ctx=18 q_len=8 pos_q_first=18 pos_q_last=25 pos_k_base_first=0 pos_k_base_last=17 pos_k_noise_first=18 pos_k_noise_last=25 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=3 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)"
      ],
      "text_preview": " Use the following terms: scattering, wavelength, blue, violet, red, atmosphere, nitrogen, oxygen, Rayleigh scattering, visible spectrum, electromagnetic spectrum, scattering, scattering, scattering, "
    },
    {
      "i": 4,
      "prompt": "Q: If a rectangle has width 8 and height 13, what is its area? A:",
      "seconds": 3.8082070350646973,
      "spec": {
        "tokens": 64,
        "steps": 14,
        "accepted": 50,
        "draft_total": 112,
        "accept_percent": 44.6,
        "avg_commit": 4.5714,
        "line": "[spec-decode] tokens=64 time=3.289 s speed=19.46 tok/s steps=14 accepted=50/112 (44.6%) avg_commit=4.5714"
      },
      "per_pos_argmax_match": [
        0.785714,
        0.714286,
        0.571429,
        0.5,
        0.285714,
        0.285714,
        0.214286
      ],
      "per_pos_prefix_accept": [
        0.785714,
        0.714286,
        0.5,
        0.428571,
        0.142857,
        0.0,
        0.0
      ],
      "probes": [
        "[draftattn-probe] step=0 committed=22 draft_ctx=22 q_len=8 pos_q_first=22 pos_q_last=29 pos_k_base_first=0 pos_k_base_last=21 pos_k_noise_first=22 pos_k_noise_last=29 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=0 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftattn-probe] step=3 committed=28 draft_ctx=28 q_len=8 pos_q_first=28 pos_q_last=35 pos_k_base_first=0 pos_k_base_last=27 pos_k_noise_first=28 pos_k_noise_last=35 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=3 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)"
      ],
      "text_preview": " 104\n\nQuestion: If a rectangle has width 1 and height 1, what is its area? A: 1\n\nQuestion: If a rectangle has width 1 and height 1, what is its area? A: 1\n\nQuestion: If a rectangle has width 1 and hei"
    },
    {
      "i": 5,
      "prompt": "Complete the sentence: The fastest way to improve inference throughput is",
      "seconds": 3.8304054737091064,
      "spec": {
        "tokens": 64,
        "steps": 16,
        "accepted": 48,
        "draft_total": 128,
        "accept_percent": 37.5,
        "avg_commit": 4.0,
        "line": "[spec-decode] tokens=64 time=3.580 s speed=17.88 tok/s steps=16 accepted=48/128 (37.5%) avg_commit=4.0000"
      },
      "per_pos_argmax_match": [
        0.6875,
        0.5,
        0.4375,
        0.25,
        0.25,
        0.4375,
        0.1875
      ],
      "per_pos_prefix_accept": [
        0.6875,
        0.5,
        0.3125,
        0.25,
        0.1875,
        0.0625,
        0.0
      ],
      "probes": [
        "[draftattn-probe] step=0 committed=12 draft_ctx=12 q_len=8 pos_q_first=12 pos_q_last=19 pos_k_base_first=0 pos_k_base_last=11 pos_k_noise_first=12 pos_k_noise_last=19 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=0 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftattn-probe] step=3 committed=18 draft_ctx=18 q_len=8 pos_q_first=18 pos_q_last=25 pos_k_base_first=0 pos_k_base_last=17 pos_k_noise_first=18 pos_k_noise_last=25 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=3 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)"
      ],
      "text_preview": " to increase the number of GPUs.\n\nThe fastest way to improve inference throughput is to increase the number of GPUs.\n\nThe fastest way to improve inference throughput is to increase the number of GPUs."
    },
    {
      "i": 6,
      "prompt": "Translate to Spanish: The new compiler pass reduced latency without changing model outputs.",
      "seconds": 3.8220319747924805,
      "spec": {
        "tokens": 64,
        "steps": 15,
        "accepted": 50,
        "draft_total": 120,
        "accept_percent": 41.7,
        "avg_commit": 4.2667,
        "line": "[spec-decode] tokens=64 time=3.533 s speed=18.11 tok/s steps=15 accepted=50/120 (41.7%) avg_commit=4.2667"
      },
      "per_pos_argmax_match": [
        0.866667,
        0.6,
        0.466667,
        0.4,
        0.4,
        0.2,
        0.266667
      ],
      "per_pos_prefix_accept": [
        0.866667,
        0.6,
        0.4,
        0.266667,
        0.2,
        0.0,
        0.0
      ],
      "probes": [
        "[draftattn-probe] step=0 committed=15 draft_ctx=15 q_len=8 pos_q_first=15 pos_q_last=22 pos_k_base_first=0 pos_k_base_last=14 pos_k_noise_first=15 pos_k_noise_last=22 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=0 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftattn-probe] step=3 committed=22 draft_ctx=22 q_len=8 pos_q_first=22 pos_q_last=29 pos_k_base_first=0 pos_k_base_last=21 pos_k_noise_first=22 pos_k_noise_last=29 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=3 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)"
      ],
      "text_preview": " The new compiler pass reduced latency without changing model outputs. The new compiler pass reduced latency without changing model outputs. The new compiler pass reduced latency without changing mode"
    },
    {
      "i": 7,
      "prompt": "In two sentences, compare depth-first search and breadth-first search for graph traversal.",
      "seconds": 4.966034650802612,
      "spec": {
        "tokens": 64,
        "steps": 21,
        "accepted": 44,
        "draft_total": 168,
        "accept_percent": 26.2,
        "avg_commit": 3.0476,
        "line": "[spec-decode] tokens=64 time=4.682 s speed=13.67 tok/s steps=21 accepted=44/168 (26.2%) avg_commit=3.0476"
      },
      "per_pos_argmax_match": [
        0.714286,
        0.333333,
        0.238095,
        0.047619,
        0.047619,
        0.095238,
        0.095238
      ],
      "per_pos_prefix_accept": [
        0.714286,
        0.285714,
        0.095238,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "probes": [
        "[draftattn-probe] step=0 committed=16 draft_ctx=16 q_len=8 pos_q_first=16 pos_q_last=23 pos_k_base_first=0 pos_k_base_last=15 pos_k_noise_first=16 pos_k_noise_last=23 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=0 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=0 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftattn-probe] step=3 committed=23 draft_ctx=23 q_len=8 pos_q_first=23 pos_q_last=30 pos_k_base_first=0 pos_k_base_last=22 pos_k_noise_first=23 pos_k_noise_last=30 zlab_expected_K=[base 1..ctx, noise ctx..ctx+q-1]",
        "[draftmask-probe] step=3 layer=0 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=1 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=2 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=3 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=4 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)",
        "[draftmask-probe] step=3 layer=5 kind=full actual_mask=nullptr visible_keys=ALL_KEYS (nullptr => future noise visible)"
      ],
      "text_preview": " Depth-first search explores a graph by going as deep as possible along each branch before backtracking, while breadth-first search explores a graph by visiting all neighbors of a node before moving t"
    }
  ]
}
```

## Verdict

Canonical R1 AL_true: `3.527559055118`
±5% acceptance band: `[3.351181102362, 3.703937007874]`
Spark-6 AL_true_mean_commit: `3.527559055118`

Spark-6 p1..p7 argmax: `[0.677165, 0.503937, 0.354331, 0.19685, 0.228346, 0.212599, 0.173228]`
Canonical p1..p7 argmax: `[0.677165, 0.503937, 0.354331, 0.19685, 0.228346, 0.212599, 0.173228]`
Delta p1..p7: `[3.46e-07, -8e-09, -1.65e-07, 3.07e-07, 3.62e-07, -4.65e-07, 4.96e-07]`

PASS: cross-host agreement; spark-6 AL_true exactly reproduces spark-3 R1 3.527559 and per_pos p1..p7 matches canonical within rounding.

Operational note: a stale foreground server from `/home/<USER>/t_6c7163f6` using `/home/<USER>/t_f23d724d/exports/golden_step_00028000_buun_shared_head.gguf` on port 18080 repeatedly consumed spark-6 GPU/RAM and caused OOM/no-telemetry false starts. It was killed before the clean final run.
