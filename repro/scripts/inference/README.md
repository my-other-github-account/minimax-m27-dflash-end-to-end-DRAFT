# `repro/scripts/inference/`

Helpers for the inference stages.

## §3 (chain-pos validation, GGUF conversion) — see [`../../03-inference.md`](../../03-inference.md)

| script | what it does |
|---|---|
| `prep_full_for_buun_converter.py` | Rebake the trained drafter (compact 32k draft vocab) into legacy target-lm-head shape (200064) before GGUF conversion. |
| `prep_for_pr22105_v2_with_d2t_rebake.py` | Variant of the above for the PR22105 d2t-aware path. |
| `verify_gguf.py` | Sanity-check the converted GGUF metadata (arch, target_layer_ids, block_size, mask_token_id, n_target_features). |
| `run_dflash_smoke.sh` | Fibonacci+dmax smoke benchmark via `llama-speculative-simple`. |
| `compute_chain_z.py` | z-score chain-pos measurements vs `val_metrics.json` predictions. |

## §4 (empirical tau against llama-benchy) — see [`../../04-empirical-tau-llama-benchy.md`](../../04-empirical-tau-llama-benchy.md)

| script | what it does | called from |
|---|---|---|
| `launch_server_dflash.sh` | Wraps `llama-server` with the three mandatory `DFLASH_*` env vars and the four mandatory CLI flags. | §4.5 step 1 |
| `launch_server_ar.sh` | Same server, no `-md`, no `--draft-max` — autoregressive baseline. | §4.5 step 5 |
| `wait_for_server.sh` | Polls `/v1/models` until the server is ready (60 s timeout). | §4.5 steps 1, 5 |
| `tau_capture_proxy.py` | aiohttp transparent proxy that snapshots `timings.draft_n` / `timings.draft_n_accepted` on every `/v1/chat/completions`. Streaming + non-streaming. | §4.5 step 2 |
| `bench.sh` | Drives `eugr/llama-benchy` against the proxy (default `PORT=8081`). | §4.5 steps 3, 6 |
| `summarize_empirical_tau.py` | Reads the proxy JSONL, prints the §4.6 receipt table. | §4.5 steps 4, 7 |

Everything runs on a single spark host. The §4 scripts assume `llama-server`
from the dflash fork is on `$PATH` (or set `LLAMA_SERVER=/path/to/llama-server`).
