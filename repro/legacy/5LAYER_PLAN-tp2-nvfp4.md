# 5-Layer DFlash Reproduction Plan — MiniMax-M2.7 (2026-04-28)

## Context

Initial 3-target-layer build (`target_layer_ids=[2, 31, 60]`) yielded a drafter with
~73% per-position teacher-forced val acc but **0.71% real accept rate** in vLLM
spec-decode. The training proxy metric was decoupled from deployment performance.
After reading the DFlash paper (arxiv 2602.06036), several gaps with paper
methodology were identified.

## Paper-Required Changes

The DFlash paper (z-lab.ai/projects/dflash) specifies:

1. **5 target hidden features**, uniformly between layer 2 and layer N-3.
   For MiniMax-M2.7 (62 layers, N-3 = 59) → `target_layer_ids=[2, 16, 30, 45, 59]`.
   (Table 6 of paper: 5 features outperforms 3 features on every benchmark.)
2. **Block size 16** (10 for LLaMA-3.1, 8 for some configs). Tables 5 + 7 show
   BS16 trains stronger drafters that also generalize back to BS8 inference.
3. **Loss decay weighting** `w_k = exp(-(k-1)/γ)`, γ=4 for BS=8, γ=7 for BS=16.
   Errors at early positions invalidate later tokens; weight earlier positions.
4. **Target-model-regenerated responses.** Training data uses target's own outputs,
   not the original dataset responses ("better target alignment"). The paper used
   ~800K samples from Nemotron-V2 + CodeAlpaca (with target-regenerated responses).
5. **AdamW lr=6e-4**, cosine schedule, warmup 0.04, **6 epochs**, max_seq_len=3072,
   512 anchors per sequence.

## What We Have vs Paper

| Item | Paper | Ours (3L baseline) | 5L plan |
|---|---|---|---|
| target layers | 5 | 3 | **5** ✓ |
| layer_ids | uniform 2..N-3 | [2, 31, 60] (60>N-3) | **[2, 16, 30, 45, 59]** ✓ |
| block_size | 16 | 8 | 8 → 16 (TBD) |
| loss weighting | exp decay γ=4 (BS8) | uniform | TBD patch |
| training samples | ~800K | 1281–7729 | grow to 800K |
| training data | target-regen | original-dataset | target-regen via UltraChat-200K + Magpie + Nemotron |
| lr | 6e-4 | 2e-5 | 6e-4 |
| epochs | 6 | 30 | 6 |
| hs cache dtype | bf16 (paper) | bf16 | **fp32** (R34 patch — see below) |

## Patches Required

All 4 must be active on every TP rank before launching data-gen:

| Patch | File | Purpose |
|---|---|---|
| R33 (×2 markers) | `vllm/model_executor/models/interfaces.py` | fp32 sum + finite-clamp guard in `_maybe_add_hidden_state` to prevent bf16 overflow at deep layers |
| R33 (×1 marker) | `vllm/v1/spec_decode/extract_hidden_states.py` | re-zero proposer's persistent buffer between requests |
| **R34_UPCAST_HS** (×1 marker, NEW for 5L) | `vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py` | upcast hidden states to fp32 before save, preserves NVFP4 verifier signal |

### R34: fp32 hs cache (new for 5L run)

```python
# At save time, instead of:
"hidden_states": hidden_states.detach().cpu(),
# Use:
"hidden_states": hidden_states.detach().to(torch.float32).cpu(),
```

**Cost:** ~2x disk (300KB vs 150KB per sample). For 800K samples: ~240GB total.
Both node2 (2.5TB free) and node3 (1.4TB free) have ample headroom.

**Benefit:** Preserves NVFP4 verifier's signal at deep layers without bf16 mantissa
truncation. The drafter still trains in bf16 (downcast at load time), but the
cached training-target signal is faithful to what the runtime verifier produces.

**Why fp32 not bf16:** Per R33 history, NVFP4 verifier outputs at layers 60+ saturate
to mantissa values in the 4096-65536 range. bf16 has only 8 bits of mantissa, so
values above ~256 lose >8 bits of precision in the residual sum. fp32 has 23 bits,
fully preserving the dynamic range.

## Launch Sequence

1. `vllm_tp2_5L.sh` — same script on node2 (rank 0) + node3 (rank 1).
   Auto-detects rank by NIC IP. Verifies all 4 patches before launch.
   Writes to `${WORKSPACE}/dflash_minimax/data/preprocessed_5L/hs_staging`.

2. `validator_daemon_5L.py` — long-running process polling staging every 30s,
   moves clean→`hs_clean_pool/`, NaN→`hs_quarantine/`, dedups by hash.

3. `datagen_5L_loop.sh` — endless loop driving prompts at the endpoint.
   Uses Trap 13 wrapper to skip indices already in pool/quarantine.

4. **macmini watchdog cron** — every 15 min: confirms vLLM alive,
   sample count growing, restarts on failure, posts status to Telegram.

## Pool Targets

| Phase | Pool size | What |
|---|---|---|
| Smoketest gate | 200 | Verify R33+R34 holding (no NaN in 200 consecutive) |
| Mid-training viability | 5,000 | Train smoke drafter, expected loss curve like paper |
| Production | 50,000+ | Train production drafter, check real accept rate |
| Paper parity | 800,000 | Full DFlash quality — overnight + days of grind |

## Files

- `repro/vllm_tp2_5L.sh` — TP=2 launcher (5 target layers, R33+R34, no chunked prefill)
- `repro/datagen_5L_loop.sh` — endless data-gen client driver
- `repro/datagen_skip_existing_wrapper.py` — Trap 13 monkey-patch
- `repro/validator_daemon_5L.py` — staging→pool/quarantine validator
- `patches/vllm/r34_upcast_hs.py` — applies R34 patch to vLLM venv

## Status Log

- 2026-04-28 21:33 PDT: vLLM TP=2 endpoint launched with 5 target layers, R34 hs upcast active.
