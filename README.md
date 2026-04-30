# MiniMax-M2.7 DFlash Speculative Decoding — End-to-End (DRAFT)

> # ⚠️ THIS REPO IS A DRAFT — NOT WORKING END-TO-END YET
>
> The Generation section produced a real, validated trace pool that trained a working drafter on the original cluster, but the *reproduction recipe documented here has not been re-walked from scratch on a clean machine*. The Training and Inference sections are stubs reflecting last-known-good config rather than fully harvested evidence.
>
> Treat this as a **work-in-progress reference**, not a paste-and-run pipeline. Specifically:
> - Generation: scripts and recipe are real, but cluster-specific assumptions still live in the placeholders (see §1's "Placeholder key")
> - Training: config is from the run that trained the production drafter, but the surrounding evidence (loss curves, exact data split) hasn't been pulled
> - Inference: build path documented but the smoke-test harness and the chain-gated diff script are still on the original machine
>
> Tracking issue / TODOs are in the per-section "What this section needs" callouts.

---

End-to-end reproduction of DFlash speculative decoding for the MiniMax-M2.7 family on GB10-class hardware, organized into **three independently-runnable sections**:

1. **[Generation](repro/01-generation.md)** — produce a clean FP8 hidden-state trace pool using vLLM TP=4 across 4 GB10 nodes. Output: `~6,500 hs_<N>.safetensors` files of shape `[seq_len, 6, 3072]` in `bfloat16`. Includes the validator daemon, the prompt-source rotation, the wedge-recovery procedure, and the empirical sanity-check evidence that the produced pool's distribution matches the validated reference. **This section is the most fleshed out.**

2. **[Training](repro/02-training.md)** — train a 5-layer DFlash drafter from the trace pool using `speculators.train`. Output: a PyTorch checkpoint convertible to GGUF. *Stub — last-known-good config + metrics, pending a focused reproducibility pass.*

3. **[Inference](repro/03-inference.md)** — run DFlash speculative decoding with `llama-server` / `llama-cli` using PR #22105 + the 4 patches in this repo. Includes the metric-framing finding that resolved the "47× apparent gap" (chain-cumulative vs per-position-conditional). *Stub — last-known-good build path, pending evidence harvest.*

---

## What this repo contains

### Section bundle (`repro/`)

```
repro/
├── 01-generation.md                # Section 1 — fully fleshed out
├── 02-training.md                  # Section 2 — stub
├── 03-inference.md                 # Section 3 — stub
├── scripts/
│   └── generation/
│       ├── vllm_tp4_5L_FP8_CLEAN.sh        # 4-rank vLLM TP=4 launcher (current production)
│       ├── validator_daemon_5L.py          # R62 zero-rate gate + R64 extended validation
│       ├── data_generation_offline_ledgered.py
│       └── multi_dataset_prompt_loader.py  # prompt-source rotation
└── legacy/                         # prior-era artifacts (preserved for historical record)
    ├── REPRODUCE-tp2-nvfp4-nan-bug.md      # the original 2-spark TP=2 NVFP4 NaN-bug investigation
    ├── 5LAYER_PLAN-tp2-nvfp4.md
    ├── vllm_tp2_clean.sh           # earlier TP=2 launcher (NVFP4 era)
    ├── vllm_tp2_5L.sh
    ├── data_gen.sh / datagen_5L_loop.sh / datagen_skip_existing_wrapper.py
    ├── validator_daemon.py         # earlier validator without R62 gate
    ├── deep_audit.py
    ├── prep_for_pr22105_converter.py       # speculators-ckpt → GGUF-converter input
    └── start_dflash_server.sh
```

### Patch bundle (`patches/`)

The patches in `patches/vllm/` and `patches/speculators/` originated from the earlier 2-spark TP=2 NVFP4 NaN-bug investigation. **The vLLM patches were eventually reverted** in the current production recipe — see [Section 1 §1.4](repro/01-generation.md#14-the-we-flailed-for-days-lesson--read-first). They remain in the repo because:
- The speculators patches (R27/R28/R29/R30/R31/R32) are still load-bearing for trainer dtype/NaN issues
- The vLLM patches are kept as historical record (and may be useful for diagnosing similar bugs on different model + quantization combinations)

```
patches/
├── vllm/                                    # all REVERTED in current production — kept for history
│   ├── 01-interfaces-aux-overflow-fix.patch
│   ├── 02-extract-hidden-states-buffer-zero.patch
│   └── r34_upcast_hs.py
├── speculators/                             # still required for training
│   ├── 01-eagle3-core-dtype-fixes.patch
│   ├── 02-train-script-dtype-cast.patch
│   ├── 03-data-empty-sample-dtypes.patch
│   └── 04-trainer-nan-guard-and-midepoch-ckpt.patch
└── llama.cpp/                               # required for inference (Section 3)
    ├── 01-minimax-m2-cb-hooks.patch
    ├── 02-variable-length-target-layer-ids.patch
    ├── 03-converter-drop-drafter-only-tensors.patch
    └── 04-dflash-begin-reset-state.patch
```

`all-vllm.patch` and `all-speculators.patch` are concatenated convenience bundles — also primarily of historical interest now.

---

## Tested environment

### Generation (Section 1) — current production

| Component | Value |
|---|---|
| Hardware | 4 × DGX Spark GB10, QSFP 200 Gbps interconnect, MTU 9000 |
| CUDA | 13.x (Blackwell SM 12.1a) |
| vLLM | `0.20.1rc1.dev23+gde3da0b97` (nightly main, late April 2026) — **vanilla, no patches** |
| Speculators | main @ `67bafe6` ("Dflash verifier targets" PR #477) |
| Verifier | `MiniMax-M2.7-FP8` (62 layers, 200 064 vocab, FP8 quant, 3072 hidden) |
| Topology | TP=4 across 4 nodes, no Ray, plain TCP NCCL over QSFP |
| Output pool | 6515 files / 127 GB at the time of the §1 evidence snapshot |

### Training & Inference (Sections 2 & 3) — last-known-good

| Component | Value |
|---|---|
| Speculators training | same speculators commit, applies patches in `patches/speculators/` |
| Inference base | PR #22105 tip @ `67cb0d507` (`ruixiang63/llama.cpp@dflash`) |
| Inference fork | `my-other-github-account/llama.cpp@dflash-minimax-m2` @ `2c32f36fc` |
| Inference target | `MiniMax-M2.7-GGUF/UD-IQ4_XS` (4 shards) |
| Drafter GGUF | `MiniMax-M2.7-DFlash.gguf` (3.14 GB, MD5 `785c5b5a6bcf8eecb545a1bebb75eb4e`) |

---

## History

This repository started as a patch bundle for two compounding bugs in vLLM that produced silent NaN-poisoning at deep layers (60+) of MiniMax-M2.7 in NVFP4 quantization, plus 4 patches against PR #22105 to enable DFlash inference for MiniMax-M2 family targets. Both ends still work.

In the months since, the trace-generation pipeline migrated from TP=2 / NVFP4 / 4-layer to **TP=4 / FP8 / 6-layer**, and the multi-day patch saga in `vllm/...interfaces.py` and `extract_hidden_states.py` was eventually unwound — the bf16-overflow problem turned out to be better handled by quarantining bad files at the validator (`R62 zero-rate gate`) than by trying to patch the extraction. Files generated by the naive zero-fallback (R33) trained the drafter to ignore deep layers, which then catastrophically failed at inference. The current production recipe (Section 1) is the one that actually trained the working drafter.

The earlier-era artifacts are preserved under `repro/legacy/` for historical record. The current recipe is in `repro/01-generation.md` + `repro/scripts/generation/`.
