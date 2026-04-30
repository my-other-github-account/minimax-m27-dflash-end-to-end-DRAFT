# DFlash Drafter Offline Validation — full reference

This file is a vendored copy of the `dflash-drafter-offline-validation` agent skill, embedded here so the §2 training doc doesn't depend on out-of-tree material. It contains the complete framing for **chain-gated cumulative vs per-position teacher-forced conditional** accuracy, plus the diagnostic playbook for when offline eval and runtime diverge.

## When to use this material

Load the offline-eval harness when:

- A DFlash drafter shows low runtime accept rate in llama.cpp / vLLM speculative-decode and you don't know whether the drafter, the GGUF conversion, or the runtime is at fault.
- You're about to spend hours retraining or patching the runtime — STOP and run this first.
- You need to confirm a freshly converted GGUF is faithful to the source safetensors.
- Position-0 marginal accept matches training-pos-1 acc, but pos-1+ marginals diverge.
- Anyone asks "are we correctly using the adapter?" before they invest in better drafters.

## The core insight

Training stores `val_metrics.json` like this (real example from a 5-layer DFlash on MiniMax-M2.7-FP8):

```json
{
  "loss_epoch": 5.38,
  "full_acc_epoch": 0.1383,
  "position 1 acc_epoch": 0.2888,
  "position 2 acc_epoch": 0.2017,
  "position 3 acc_epoch": 0.1374,
  "position 4 acc_epoch": 0.1056,
  "position 5 acc_epoch": 0.0842,
  "position 6 acc_epoch": 0.0732,
  "position 7 acc_epoch": 0.0720
}
```

These are **teacher-forced parallel-block argmax accuracies** computed by `compute_metrics()` in `speculators/models/dflash/metrics.py`. The drafter is fed `[anchor, MASK*7]` together with verifier hidden states from the configured target layers, and per-position accuracy is measured against the verifier's own greedy predictions (NOT raw ground-truth tokens).

**At inference time, llama.cpp does the same parallel-block forward.** Therefore the **runtime marginal pos-N accept rate** (per-position rejection histogram) should match the training pos-N acc within sample noise. If they don't, something is broken between the safetensors and the runtime.

## Step 1 — Locate the trained checkpoint and val_metrics.json

```bash
ls ${CHECKPOINTS}/<run_dir>/checkpoint_best/
# Expected: config.json, model.safetensors, optimizer_state_dict.pt, val_metrics.json, ...
cat ${CHECKPOINTS}/<run_dir>/checkpoint_best/val_metrics.json
```

Also need:

- The training repo: `${WORKSPACE}/repos/speculators/`
- The training data: `--data-path` and `--hidden-states-path` from the training script
- The verifier model: `--verifier-name-or-path` (e.g. `${MODELS}/MiniMax-M2.7-FP8`)
- A python env with `speculators`, `transformers`, `torch`, `safetensors`

## Step 2 — Run the offline validation harness

The harness (`repro/scripts/training/dflash_offline_eval.py`) loads the checkpoint via `DFlashDraftModel.from_pretrained`, builds the val split with `split_ratio=-0.1` (trainer convention — last 10%), and runs the canonical parallel-block forward with the trainer's collate_fn.

```bash
cd ${WORKSPACE}/repos/speculators/scripts
source ${WORKSPACE}/venvs/vllm/bin/activate
python repro/scripts/training/dflash_offline_eval.py \
    --ckpt ${CHECKPOINTS}/<run>/checkpoint_best \
    --data ${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired/prompts \
    --hs   ${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired/hidden_states \
    --verifier ${MODELS}/MiniMax-M2.7-FP8 \
    --max-batches 60
```

Expected runtime: ~1 min on a Spark (single GPU, batch_size=1).

Expected output format:

```
[eval] full_acc     = 0.1366    (training reported 0.1383)
[eval] position 1 acc = 0.3146  (training reported 0.2888)
[eval] position 2 acc = 0.1843  (training reported 0.2017)
...
```

Sample-noise tolerance: with 30-60 batches you should be within ±3 absolute pp of training values. If you're off by >5 pp on multiple positions → the checkpoint or env is broken.

## ⚠️ Step 2.75 — CRITICAL: distinguish training-conditional from runtime-cumulative metrics

**This is the single most important framing in this whole training section.** Get it wrong and you'll spend days hunting nonexistent bugs.

### The metric trap

Training `val_metrics.json` reports:

```
position 1 acc = 0.2888
position 2 acc = 0.2017
position 3 acc = 0.1374
...
full_acc       = 0.1383
```

These are **per-position CONDITIONAL accuracies under teacher forcing**: `p_k = P(draft[k] == target[k] | all earlier hiddens correct)`. Each position is scored independently against ground-truth target tokens, with the drafter fed correct hiddens at every preceding position.

**`full_acc` = 0.1383 is NOT cumulative chain-accept either.** It's the fraction of teacher-forced rounds where ALL 7 positions were simultaneously correct given oracle hiddens. It's closer to `P(all 7 correct | oracle hiddens)`, which is much higher than the chain-gated runtime expectation.

### What runtime measures

Llama.cpp speculative decoding does **chain-gated** verification:

- chain-pos-1 is tested every round
- chain-pos-2 is only tested if chain-pos-1 was accepted
- chain-pos-k is only tested if positions 1..k-1 were all accepted

So runtime `P(reach AND accept chain-pos-k)` = **prefix product** `∏_{i=1..k} p_i` under the (optimistic) assumption that runtime conditional accept tracks training teacher-forced conditional. This is the **chain-gated cumulative** prediction.

### Numerical example for the MiniMax-M2.7 5-layer DFlash above

| pos k | training p_k (cond) | chain-gated cumulative ∏ p_i | what runtime can possibly measure |
|------:|--------------------:|-----------------------------:|----------------------------------:|
| 1 | 28.88% | **28.88%** | chain-pos-1 accept |
| 2 | 20.17% | **5.82%** | chain-pos-2 accept (full block at dmax=2) |
| 3 | 13.74% | **0.80%** | |
| 4 | 10.56% | **0.085%** | |
| 5 | 8.42% | **0.0071%** | |
| 6 | 7.32% | **0.00052%** | |
| 7 | 7.20% | **0.000037%** | full block at dmax=7 |

So at dmax=7, **0% measured full-block over 295 rounds is consistent with the chain-gated prediction** — expected count = 295 × 0.00000037 ≈ 0. Don't panic. Don't conclude "DFlash is broken." Don't start patching.

### Practical comparison rules

When you read a runtime rejection histogram, convert:

- `chain-pos-1 accept rate` = `1 - hist[pos 0]`
- `chain-pos-2 accept rate` = `1 - (hist[pos 0] + hist[pos 1])` = sum from "all_ok" forward
- `chain-pos-k accept rate` = `1 - cumulative_reject[1..k]`

Compare these to `∏ p_i`, **NOT to the bare training pos-N values**.

### Sample-size sanity

Use a binomial proportion z-score before declaring divergence:

```python
from math import sqrt
def z_score(p_obs, p_pred, n):
    se = sqrt(p_pred * (1 - p_pred) / n)
    return (p_obs - p_pred) / se if se > 0 else float('nan')
```

At `n=50` rounds and `p_pred=0.058` (chain-pos-2 prediction), an observed `1/50 = 0.02` gives `z = -1.15` — **not significant at p=0.05.** This is sample noise around the prediction, not evidence of a bug. Need n>500 to distinguish 2% from 5.8% reliably.

### When the metric trap looks like a 47× gap (real session, do not repeat)

In one session, the author compared a measured 0.29% accept rate at dmax=7 against a "training claim" of 13.83% full-block, called this "47× off and catastrophic," and burned multiple days patching llama.cpp source and GGUF metadata in pursuit of the gap. **The 47× gap was entirely metric mismatch.** Chain-gated cumulative at dmax=7 is 0.000037%, so 0.29% is actually 800× *above* the chain-gated prediction (still mostly noise from rare easy-context rounds).

After realizing this, the path forward was:

1. Revert all patches to clean baseline (preserve diffs in `patches_archive/`)
2. Rewrite the GGUF metadata back to its training-config-aligned values
3. Build chain-gated training-side eval (NOT teacher-forced — feed drafter the chain it would actually generate, not oracle hiddens)
4. Run runtime on the same val samples
5. Diff cumulative chain-pos-k accept rates side-by-side with z-scores

If those match within sample noise, **the drafter is performing as designed**. The training pos-N marginals overstate what runtime can ever achieve.

### Throughput implications you should also flag

Even if accept rates exactly match the chain-gated prediction, throughput may regress vs autoregressive baseline because:

- Verify cost grows roughly linearly with draft chain length
- Acceptance gain is bounded by `E[tokens accepted per round] = Σ ∏ p_i`
- Net speedup factor ≈ `(1 + E[accepted]) / verify_cost_multiplier`

For the example numbers, this gives ~1.36× speedup ceiling under independence. If your verify is more than 1.36× costlier than autoregressive (likely for big targets like IQ4 quants), **speculative will be slower than baseline**, and no amount of accept-rate work fixes that.

## Step 2.5 — Cosine-similarity offset diagnostic (do this FIRST when porting to a new runtime)

Before chasing pos-N marginal mismatches, **prove that your runtime is feeding the drafter the same hidden states it was trained on**. The training data is canonical: it's the captured FP8 traces. Each safetensor file stores `[seq_len, n_aux+1, hidden]` bf16 tensors. Pull sample 0 and compare it byte-by-byte against what your runtime captures on the same input tokens.

### The HF-tuple-vs-engine-layer off-by-one (CRITICAL — usually -1 for llama.cpp ports)

**This is the single most likely silent bug for any DFlash drafter ported to llama.cpp.** It produces ~3-10% accept where you expected 30-80%.

**The semantic mismatch:**

- HuggingFace's `model.forward(output_hidden_states=True).hidden_states` is a tuple where:
  - `hidden_states[0]` = embedding output (PRE-layer-0)
  - `hidden_states[N]` for N≥1 = OUTPUT of layer N-1
- vLLM's `extract_hidden_states` connector and the speculators data-gen pipeline read this HF tuple directly. So config `target_layer_ids = [2, 16, 30, 45, 59]` captures the OUTPUT of layers `[1, 15, 29, 44, 58]` (zero-indexed transformer layers).
- llama.cpp's `cb(cur, "l_out", il)` callback fires AFTER layer `il`. So when llama.cpp is asked to capture `target_layer_ids = [2, 16, 30, 45, 59]`, it captures the output of layers `[2, 16, 30, 45, 59]` — **off by +1 relative to training**.

**Net effect: capture index needs to be `target_layer_ids[i] - 1`** to match training semantics in llama.cpp.

**Verify empirically before patching** — capture llama.cpp's hidden states at multiple candidate layers (e.g. `[N-1, N, N+1]` for each N in the config), compare cosine similarity to the FP8 reference trace, find the offset that gives ≥0.99 cosine.

**Apply the fix by rewriting GGUF metadata** (don't change llama.cpp source — keeps it stock):

```bash
python scripts/patch_target_layer_ids.py /path/to/drafter.gguf --delta -1
```

The patcher writes a `.bak`, only touches the array elements, and preserves type/count headers.

## Step 3 — Compare runtime marginals to offline reference

```
n_drafted = 190  n_accept = 34  accept = 17.9%
rejection histogram (position → count):
  pos 0:  65 ( 68.4%)
  pos 1:  26 ( 27.4%)
  all ok:  4 (  4.2%)
```

Translate to **marginal pos-N acc**:

- `n_rounds = pos0_count + pos1_count + ... + all_ok_count` (=95 above)
- `marginal pos-0 = (rounds where pos-0 was accepted) / n_rounds = (pos1 + pos2 + ... + all_ok) / n_rounds`
- For the example above: marginal pos-0 = 30/95 = **31.6%** ← compare to training pos-1 (28.88%)
- `marginal pos-1 = all_ok / n_rounds = 4/95 = 4.2%` ← compare to training pos-2 (20.17%)

**Important — the off-by-one:** training reports `position N acc` for N in 1..7 (block-pos 0 is the anchor). Llama.cpp reports rejection at "pos 0" meaning the FIRST drafted token, which corresponds to **block-pos 1** in training nomenclature.

| runtime pos | training metric |
|-------------|-----------------|
| pos 0 (1st draft token) | position 1 acc |
| pos 1 (2nd draft token) | position 2 acc |
| pos 2 | position 3 acc |
| ... | ... |

Get a longer sample (≥500 generated tokens) for stable marginal stats — short prompts give noisy pos-N>2 estimates.

## Step 4 — Diagnose based on the comparison

**FIRST — apply the chain-gating sanity check from Step 2.75.** If your "runtime pos-1+ marginal divergence" is actually chain-gated cumulative `∏ p_i` measuring at the predicted level, there is no bug — it's correct behavior. Compute the z-score before assuming GGUF/runtime fault.

| Offline pos-N matches training? | Chain-pos-1 runtime ≈ ∏ p_1? | Chain-pos-k runtime ≈ ∏ p_1..p_k within z<2? | Diagnosis |
|---|---|---|---|
| ✅ | ✅ | ✅ | **Workflow correct AND DFlash is performing as designed.** The drafter is the ceiling. |
| ✅ | ✅ | ❌ (runtime BELOW chain-gated prediction) | **GGUF conversion or runtime bug specific to later positions** — only after confirming `n_rounds ≥ 500`. Most likely: zero-row dilution in d2t-rebaked lm_head, off-by-one in target_layer_ids, per-layer hidden_norm timing, or cross-ring slot indexing. |
| ✅ | ❌ (chain-pos-1 below offline pos-1) | ❌ | **GGUF conversion broke the drafter outright.** Check tensor shapes, fc/hidden_norm wiring, target_layer_ids in GGUF metadata, mask_token_id. Step 2.5 cosine diagnostic is the next move. |
| ❌ | n/a | n/a | **Checkpoint is corrupted or you're running a different one.** Re-download from training output, verify with `val_metrics.json` hash. |

## Common GGUF conversion bugs

### Zero-row dilution in d2t-rebaked lm_head

DFlash drafters typically have `lm_head.weight` of shape `[draft_vocab_size, hidden]` (e.g. `[32768, 3072]`) plus a `d2t` tensor `[draft_vocab_size]` mapping draft indices to target token IDs.

For runtimes with **no native d2t support** (llama.cpp upstream), you must rebake `lm_head` to `[target_vocab_size, hidden]` (e.g. `[200064, 3072]`) so argmax gives target-vocab IDs directly. The naive scatter:

```python
expanded[d2t[i] + i] = original[i]  # for i in 0..draft_vocab_size
```

leaves `target_vocab_size - draft_vocab_size` rows as **zero vectors**. At runtime, `argmax(softmax(logits))` over the 200K vocab can occasionally pick a zero row — its logit is `0 + bias = constant` regardless of hidden state, so when the drafter's confidence is low (later block positions) zero rows can win.

**Fix**: set non-mapped rows to `-inf` instead of zeros so they can never be argmax winners:

```python
expanded = torch.full((target_vocab_size, hidden), float("-inf"))
for i in range(draft_vocab_size):
    expanded[d2t[i] + i] = original[i]
```

GGUF doesn't store `-inf` cleanly in some quants. Use a very negative finite value (e.g. `-1e9`) or store `lm_head` as F16/F32. For BF16 storage, `-65504` (BF16 most-negative) works.

### Wrong target_layer_ids in GGUF metadata

The drafter MUST advertise the same `target_layer_ids` array that training used (e.g. `[2, 16, 30, 45, 59]` for MiniMax-M2.7 5-layer). Mismatch breaks every position because the verifier captures hidden states from the wrong layers.

### Wrong mask_token_id

The training script's `--mask-token-id` (e.g. 200054 for MiniMax) must match `dflash-draft.dflash.mask_token_id` in the GGUF and the runtime's mask token. Mismatch causes every block position 1..7 to attend to the wrong embedding.

## Pitfalls

1. **`@torch.compile` on `DFlashDraftModel.forward`** — bypass with `torch.compiler.set_stance("force_eager")` BEFORE loading the model. Without this, single-batch eval crashes with `RuntimeError: repeats must be 0-dim or 1-dim tensor`.
2. **`load_verifier_weights()` is mandatory** — if you skip it, `verifier_lm_head.weight` stays NaN-initialized and `compute_metrics` returns garbage.
3. **Trainer val convention is `split_ratio=-0.1`** (negative = last 10%), NOT `split_ratio=0.1` (which gives the FIRST 10%).
4. **Single-batch eval forward needs `create_collate_fn`** — bare `DataLoader(batch_size=1)` doesn't work.
5. **Training pos-N indexing is 1-based** — runtime "pos 0" rejection = training "position 1 acc".
6. **Runtime marginal pos-N for N≥2 needs a long generation** — at draft-max=2 you only get pos-0 and pos-1 stats.
7. **`DraftVocabMixin` requires `t2d` and `d2t` to load before `load_verifier_weights()`** — `from_pretrained` handles this.
8. **`--draft-max=1` ALWAYS shows 100% accept** in `llama-speculative-simple` — degenerate. Always benchmark at draft-max ≥ 2.
9. **`n_drafted = K * n_rounds`** — `n_drafted` counts every draft TOKEN proposed, not every draft ROUND.
10. **Don't trust accept-rate aggregates as primary signal** — instrument logits directly for the first ~10 rounds.
11. **Don't borrow conclusions from other systems' notes** — verify on YOUR data with empirical tests.
12. **When patching becomes whack-a-mole, STOP and clean-room reproduce.** Inventory every local edit, save to `patches_archive/`, reset everything, redo the apples-to-apples comparison from scratch with chain-gated math, then bisect one patch at a time.

## Reuse checklist

Before declaring "the drafter works at inference":

- [ ] `val_metrics.json` exists in `<ckpt>/checkpoint_best/`
- [ ] Offline eval reproduces full_acc within ±2 pp and pos-1..7 within ±5 pp of `val_metrics.json`
- [ ] **Computed chain-gated cumulative predictions** ∏ p_i for k=1..7 from training pos-N
- [ ] Runtime chain-pos-1 accept (= 1 - hist[pos 0]) within z<2 of training p_1
- [ ] Runtime chain-pos-k accept within z<2 of ∏ p_1..p_k for k where n_rounds gives statistical power (typically only k=1,2 at n=50; need n>500 for k=3)
- [ ] If divergence: confirmed it's GGUF conversion, NOT chain-gating math, NOT sample noise

Only AFTER all checks pass, move on to "make the drafter better." Confirming repro saves days of training the wrong thing.
