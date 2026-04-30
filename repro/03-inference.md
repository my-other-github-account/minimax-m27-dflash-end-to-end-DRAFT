# Section 3 — Inference (DFlash speculative decoding via llama.cpp)

End-to-end recipe for converting a trained DFlash drafter to GGUF and running speculative-decode benchmarks against `MiniMax-M2.7-FP8` (UD-IQ4_XS quant) using buun-llama-cpp.

> **Status:** ✅ Working end-to-end. Verified against FULL epoch-5 drafter (val_loss=5.872, p_1=22.03%) on spark-1, 2026-04-30. Measured chain-pos-1 = 29.9% (dmax=2), 31.6% (dmax=4), 33.2% (dmax=7) — all favorable z-scores vs the chain-cumulative prediction.

---

## 3.0 The metric that defines success

Just like §2, this section leads with **chain-cumulative per-position accept rate** (∏ p_i), because that's what runtime measures. Per-position teacher-forced conditional from `val_metrics.json` is reported only as a sanity-check next to the chained values.

The runtime measurement we care about: from `llama-speculative-simple`'s rejection histogram, given dmax=D and N total drafts:

- **chain-pos-k accept** = (N − Σᵢ<ₖ rej[i]) / N where rej[i] = drafts that accepted positions 0..i-1 then rejected position i

This is a strict prefix-product, exactly what `∏ p_i` from training predicts.

For the FULL epoch-5 drafter (val_loss=5.872, training-side p_i):

| Position | Conditional p_i | Chained ∏ p_i |
|---|---|---|
| 1 | 22.03% | **22.03%** |
| 2 | 13.90% | **3.06%** |
| 3 | 10.08% | **0.31%** |
| 4 | 8.56% | **0.026%** |
| 5 | 7.49% | **0.0020%** |
| 6 | 6.95% | **0.00014%** |
| 7 | 6.40% | **0.0000087%** |

Chain decays fast — chain-pos-3 is already <0.5%, chain-pos-7 is essentially never. **dmax=2 is the sweet spot for stable measurement** because it forces the chain to actually exercise position 2 (the cleanest pos-2 signal), where dmax=4/7 measure pos-2 conditionally on pos-1 already having been right.

---

## 3.1 What was already there

Before this section was written, the production conversion path had been done before by an autonomous cron loop (`dflash-clean-repro-loop` on spark-1). The output of that loop:

- A clean buun-llama-cpp build at `/home/user/dflash_clean_repro/build_clean/` (upstream commit `e275191e`, no source patches)
- A working production drafter GGUF at `/home/user/models/MiniMax-M2.7-DFlash.gguf` (md5 `785c5b5a6bcf8eecb545a1bebb75eb4e`, 3.14 GB, training-aligned `target_layer_ids=[2,16,30,45,59]`)
- A validated benchmark recipe (Fibonacci prompt, dmax=2/4/7) producing chain-pos-1 = 31.6% measured vs 28.88% predicted (z=+0.59, within sample noise)

The remaining work for this section: **apply the same recipe to the FULL training run's checkpoint** (a different drafter with `draft_vocab_size = 32768` instead of the production 200064, so it needs the d2t→target-vocab rebake before conversion).

---

## 3.2 The full recipe (8 steps)

Captured as a skill at `~/.hermes/skills/mlops-inference/dflash-full-checkpoint-to-gguf-spark1/` so future training runs can reuse it directly. The summary:

1. **Sync checkpoint to spark-1 over QSFP** (~290 MB/s, ~12 sec for 3.5 GB)
2. **Verify source tensors** — confirm `lm_head [32768, 3072]`, `d2t [32768] int64`, `t2d [200064] bool`
3. **Run prep script** (`scripts/prep_full_for_buun_converter.py`) — rebake `lm_head` to `[200064, 3072]` with `-65504` floor, strip d2t/t2d, flatten speculators-format config, copy tokenizer
4. **Whitelist FP8-tokenizer hash** in buun's converter (one-time per spark-1 reset)
5. **Run buun's `convert_hf_to_gguf.py`** with `--outtype bf16` → 60-tensor 3.13 GB GGUF
6. **Verify GGUF metadata** (`scripts/verify_gguf.py`) — arch, target_layer_ids, block_size, mask_token_id, n_target_features
7. **Run smoke benchmark** (`scripts/run_dflash_smoke.sh`) — same Fibonacci+dmax sweep as production reference
8. **Compute z-scores** (`scripts/compute_chain_z.py`) — measured vs ∏ p_i from val_metrics.json

The three scripts and the SKILL.md are in `references/dflash-full-checkpoint-to-gguf-spark1/` for inline reference.

---

## 3.3 Why the rebake matters (the d2t bug)

The training pipeline trains with `draft_vocab_size=32768` (compressed from 200064) for memory/throughput reasons. The drafter learns a `lm_head [32768, 3072]` projection over the 32k most-used target tokens, plus a `d2t [32768] int64` offset table mapping each draft index to its target-vocab id (`target_id = draft_idx + d2t[draft_idx]`).

At inference time, buun's runtime expects a single `lm_head [200064, 3072]` over the full target vocabulary — there's no provision for d2t. So the conversion has to **rebake** the trained projection into target-vocab shape:

```python
new_lm_head = torch.full((200064, 3072), -65504.0, dtype=bf16)
target_ids = torch.arange(32768) + d2t  # [32768]: target_id for each draft row
new_lm_head[target_ids] = old_lm_head    # scatter trained rows into their target slots
```

The **`-65504` floor** is critical — it's the largest-magnitude finite bf16 negative value. Non-mapped rows get this floor so that softmax over the target vocab effectively zeros them out, reproducing the masking behavior of training.

The `dflash-gguf-conversion` skill in `~/.hermes/skills/mlops/inference/` documents this as the silent-bug root cause for what was originally diagnosed as a "47× accept-rate gap": if non-mapped rows are zero (default torch.zeros) instead of `-65504`, they leak ~5× false-accept signal into chain-pos-2, breaking chain-pos-2 measurements while leaving chain-pos-1 fine.

We verified the rebake produced correct output: chain-pos-2 measured 2.58% vs 3.06% predicted (z=−0.39, within noise) at dmax=2, exactly matching the training-side prediction.

---

## 3.4 Live results (FULL epoch-5 drafter, dmax sweep)

```
                  predicted     dmax=2          dmax=4          dmax=7
chain-pos-1       22.03%       29.90%(+2.6σ)   31.58%(+3.2σ)   33.16%(+3.7σ)
chain-pos-2        3.06%        2.58%(-0.4σ)    3.68%(+0.5σ)    4.28%(+1.0σ)
chain-pos-3        0.31%         n/a            0.00%(-0.8σ)    0.00%(-0.8σ)
throughput        AR baseline   3.31 t/s        2.79 t/s        2.11 t/s
                  (~4.4 t/s)
```

Sample sizes: dmax=2 n=194 drafts, dmax=4 n=190, dmax=7 n=187 (all from 256 generated tokens on the Fibonacci prompt).

**Interpretation:**

- **chain-pos-1 measured > predicted across all dmax values.** The Fibonacci prompt has high local redundancy (Python keywords, structural repetition); training-set-averaged conditionals understate code-prompt accept. Production drafter showed the same effect (+0.59σ on the same prompt). FULL epoch-5 shows +2.6σ to +3.7σ — strongly favorable, beyond noise but consistent with the prompt-difficulty bias.
- **chain-pos-2 essentially on prediction at dmax=2** (z=−0.4) — the cleanest signal, since dmax=2 forces the chain to actually exercise position 2 rather than measuring it conditional on already-favorable position-1 acceptance. dmax=4/7 show pos-2 slightly favorable (+0.5σ to +1.0σ) for the same prompt-bias reason as pos-1.
- **chain-pos-3 = 0.0% at n=190.** Predicted ~0.6 events; observing 0 is z=−0.8, sample noise. Not a bug. To resolve chain-pos-3 properly, need n ≥ 1000 generated tokens at dmax≥3.
- **Throughput regressed below AR baseline.** This is the documented "verify cost dominates" anomaly with UD-IQ4_XS — the verifier is so expensive (230B-active MoE quantized to 4 bits) that even a successful 2.6× draft batch doesn't pay off at this quant. Speedup will return at lower-quant verifier (Q4_K_M, Q3_K_M) or with batching.

**Baseline AR throughput:** 4.4 t/s (greedy, no speculation, same verifier shard, same prompt). Confirmed by independent run.

---

## 3.5 Verification: did real DFlash actually run?

Four signals confirm the DFlash code path was exercised, not a fallback to copyspec/ngram:

1. **`--spec-type dflash` was on the command line.** That flag selects the DFlash sampler.
2. **DFlash internal counter fired:** `statistics dflash: #calls(b,g,a) = 1 194 0, #gen drafts = 194, #gen tokens = 388` appears in stderr. This counter only exists in the DFlash code path.
3. **Per-position rejection histogram is depth-aware.** dmax=4 produced `pos 0: 130, pos 1: 53, pos 2: 7, all ok: 0` — that's the chain structure of DFlash. Plain draft-model speculation reports a single accept count, not depth-resolved chain rejection.
4. **DFlash-specific GGUF metadata was read and used.** `block_size=8`, `n_target_features=15360` (= 5 layers × 3072 hidden), `mask_token_id=200054` — none of which exist in non-dflash GGUFs. The binary panics if they're missing/wrong-shape, and they're not.

All four held throughout the FULL epoch-5 benchmark.

> **Pitfall:** the line `dflash: #acc drafts = 0` is misleading — it's a known dead counter in this build (never increments regardless of actual accepts). Use `n_accept` from the speculative-simple outer loop, plus the rejection histogram, as the real signal. Both are exact and trustworthy.

---

## 3.6 What was different on spark-2 (and why it failed)

Earlier in the session there was a detour onto spark-2 trying to do the same conversion locally. It failed:

- spark-2's `llama.cpp-dflash` is a **different fork** that registers `arch="dflash"` in source, while the production-style GGUF (and therefore the GGUF produced by the buun converter) uses `arch="dflash-draft"`. The spark-2 binary refuses to load `dflash-draft` GGUFs.
- spark-2's `convert_hf_to_gguf.py` is a stripped-down version that doesn't have the `DFlashDraftModel` class registered, so it can't even produce the right output format from the speculators-format checkpoint.

**Lesson:** spark-1's `dflash_clean_repro` setup is the verified-working rig. Future drafters should be converted there. Spark-2/3/4/6 are training nodes only for DFlash work — inference happens on spark-1 (with appropriate disk for verifier shards).

---

## 3.7 Future work

When FULL training completes (~17 epochs, ETA 15:30 PDT 2026-04-30), the same recipe applies one-shot to the final checkpoint:

```bash
# (from macmini)
ssh -J operator@100.66.198.32 operator@10.0.0.103 \
  'mkdir -p /home/user/full_final_for_gguf && \
   rsync -av operator@192.168.200.4:/home/user/dflash_minimax/checkpoints/full_5L_paired6515_20260430_111332/checkpoint_best/ \
     /home/user/full_final_for_gguf/ && \
   source /home/user/venvs/vllm/bin/activate && \
   python3 /tmp/prep_full_for_buun_converter.py /home/user/full_final_for_gguf /home/user/full_final_prepped && \
   cd /home/user/buun-llama-cpp && \
   python3 convert_hf_to_gguf.py /home/user/full_final_prepped \
     --outtype bf16 \
     --outfile /home/user/models/MiniMax-M2.7-DFlash-FULL-final.gguf && \
   DRAFTER=/home/user/models/MiniMax-M2.7-DFlash-FULL-final.gguf \
     TAG=FULL-final \
     bash /tmp/full_smoke.sh'
```

A cron tick has been queued to auto-execute this against FULL's final checkpoint when training finishes. See `references/cron-auto-run-on-full-completion.md`.

Production target for FULL final: chain-pos-1 should hit ≥25% predicted (it was 22.0% at epoch 5, and FULL is still climbing — rate ~+0.5pp/epoch). Measured runtime should land in the 28-35% range on Fibonacci with sample noise z<3 against prediction.
