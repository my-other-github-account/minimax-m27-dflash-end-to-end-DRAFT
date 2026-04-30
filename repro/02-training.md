# Section 2 — Training the DFlash Drafter

End-to-end recipe to train a 5-layer DFlash drafter for `MiniMax-M2.7-FP8`, starting from the hidden-state pool produced in §1 and ending with a validated checkpoint ready for GGUF conversion in §3.

> **Pickup point:** §1 produced a pool of `hs_<N>.safetensors` files at `${DATA_ROOT}/preprocessed_5L_FP8/hs_clean_pool/`. This section assumes that's done and the file count + distribution sanity checks from §1.16 passed. Everything below runs on a single Spark — pick the one with the most disk free; the verifier model + paired data + checkpoint dirs together need ~250 GB.

---

## 2.0 The metric you actually care about: **chained per-position accuracy**

Before any training runs, the most important framing in this section:

The trainer's `val_metrics.json` reports **per-position teacher-forced conditional** accuracy:

```json
{"position 1 acc_epoch": 0.205, "position 2 acc_epoch": 0.131, ...}
```

These are `p_k = P(draft[k] == target[k] | all earlier hiddens correct)`. Each position is scored independently with the drafter fed correct hiddens at every preceding position. **This is NOT what the runtime measures.**

At runtime, llama.cpp/vLLM speculative decoding does **chain-gated** verification: position `k` is only tested if positions `1..k-1` were all accepted. So the runtime probability of reaching-and-accepting position `k` is the **chain-cumulative product** `∏_{i=1..k} p_i`.

For a fresh epoch of the production-like 6515-paired-file run with `p = [0.205, 0.131, 0.098, 0.083, 0.075, 0.069, 0.065]`:

| Position | Conditional `p_k` | **Chained `∏ p_i`** |
|---|---|---|
| 1 | 20.5% | **20.5%** |
| 2 | 13.1% | **2.69%** |
| 3 | 9.8% | **0.263%** |
| 4 | 8.3% | **0.0218%** |
| 5 | 7.5% | **0.00164%** |
| 6 | 6.9% | **0.000113%** |
| 7 | 6.5% | **7.35e-08** |

**Always read training results in chained form.** A "21.8% pos-1 acc" sounds great until you realize chain-pos-7 is `1.4e-7` — i.e. functionally never. The chained values match what you'll measure at runtime as `1 - cumulative_reject_at_k`. The full evidence-and-discussion is in `references/dflash-drafter-offline-validation.md` (also embedded at the bottom of this section).

The expected number of tokens accepted per round is `E[acc] = Σ_{k=1..7} ∏ p_i`, which directly bounds speedup ceiling: `1 + E[acc]` — the SMOKE-vs-FULL comparison in §2.7 uses this throughout.

---

## 2.1 Inputs and prerequisites

After §1 you should have on the **single** Spark you're training on:

```
${DATA_ROOT}/preprocessed_5L_FP8/hs_clean_pool/        # ~6,500 files, ~127 GB
${MODELS}/MiniMax-M2.7-FP8/                            # verifier, ~750 GB
${WORKSPACE}/repos/speculators/                        # @ commit 67bafe6
${WORKSPACE}/repos/speculators/patches/                # the 4 R27-R32 NaN patches
```

Plus the **prompt sources** you used during §1's loop. The trace generation rotates across them:

```
${DATA_ROOT}/preprocessed/combined_48k/prompts/        # 48,000 rows
${DATA_ROOT}/preprocessed/bonus_seed4321/prompts/      # 12,000 rows (subset of combined_48k)
${DATA_ROOT}/preprocessed/mixed_<seed>/prompts/        # 6k–13k each, one or more rotations
${DATA_ROOT}/preprocessed_5L_FP8/train1176/prompts/    # the original 1176-curated subset
```

One-time spec check — confirm each prompt source loads as an HF `Dataset` with `input_ids` and `loss_mask`:

```python
from datasets import load_from_disk
ds = load_from_disk("${DATA_ROOT}/preprocessed/combined_48k/prompts")
print(len(ds), ds.column_names)
# Expected: <N> ['input_ids', 'attention_mask', 'loss_mask']
```

If a source is missing or corrupt, regenerate it via §1's `multi_dataset_prompt_loader.py` invocation — don't try to skip it; if its samples ended up in the pool, you need its prompts to pair them.

Verify the verifier model loads (this also pre-fetches the tokenizer):

```bash
source ${WORKSPACE}/venvs/vllm/bin/activate
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('${MODELS}/MiniMax-M2.7-FP8', trust_remote_code=True); print(t.vocab_size)"
# Expected: 200064
```

---

## 2.2 Why your 6,500-file pool needs to be re-paired before training

The trainer requires a **strict 1:1 index pairing**: `hs_<i>.safetensors` MUST be paired with row `i` of the prompts dataset, and the safetensor's saved `token_ids` MUST equal the first `len(token_ids)` tokens of that prompt's `input_ids`.

If you generated the pool with §1's autonomous loop (which rotates prompt sources over time) and the validator daemon (which renumbers files monotonically as `hs_<idx>.safetensors`), then `hs_<N>.safetensors` from your pool **does NOT correspond to row N of any single prompt source** — it's the Nth file the validator promoted, drawn from whichever source was active at that loop tick.

You will see this exact failure cascade if you point the trainer at the pool directly:

```
UserWarning: R54: hs prompt prefix mismatch for index 5959
UserWarning: Loaded token ids tensor([200034, 200019, ...]) for index 5721 don't match input ids tensor([200034, 200019, ...])
ValueError: anchor_positions include padding locations: [0, 0, 0, ...]
```

The padding-anchor `ValueError` is **downstream** of the prefix mismatch — it's not a bug to fix in the anchor logic. Fix the pairing.

### Confirming the pool is in fact complete (one-time sanity check)

Each Spark in your §1 TP=N cluster has its own `hs_staging/` with raw datagen output. You may have heard "every rank wrote its own" and worried that the pool only captured 1/N of the data. **Empirically, the pool IS the complete dataset** — verified by hashing `token_ids` of every staging file on a peer rank vs every pool file:

```
node1 hs_staging/      17,657 files, 6,515 unique by token_ids
node2 hs_clean_pool/   6,515 files,  6,515 unique by token_ids
intersection (sha256 of token_ids):  6,515  (i.e. pool ⊆ staging exactly)
samples in staging but not in pool:  0
```

The 17,657 vs 6,515 ratio (~2.7× duplication) is because the offline datagen retries prompts across loop ticks, writing a new `cmpl-<vllm_request_id>` filename each time but with identical `token_ids`. The validator on the canonical Spark dedupes via first-4KB sha when promoting to pool, collapsing each unique sample to one `hs_<N>.safetensors`. **You are not leaving any data on the floor when you train on the pool.**

If you want to verify on your own cluster:

```bash
# On the canonical pool host:
${WORKSPACE}/repos/speculators-repro/repro/scripts/training/audit_pool_completeness.py \
    --pool ${DATA_ROOT}/preprocessed_5L_FP8/hs_clean_pool \
    --peer-host other-spark-hostname \
    --peer-staging ${DATA_ROOT}/preprocessed_5L_FP8/hs_staging
```

Expected output: `pool ⊆ peer-staging: True, samples in staging not in pool: 0`. If non-zero, your validator missed some — see `references/pool-completeness-debugging.md`.

---

## 2.3 Build the paired dataset (`train_all_paired/`)

Strategy: hash each pool file's `token_ids`, hash every prompt source's `input_ids`, match by sha256. Emit a paired dataset where prompts are taken in pool-index order and hidden-states are symlinked back to the pool (no data duplicated).

```bash
cd ${WORKSPACE}/repos/speculators-repro
python repro/scripts/training/build_paired_dataset.py \
    --pool ${DATA_ROOT}/preprocessed_5L_FP8/hs_clean_pool \
    --prompt-source combined_48k=${DATA_ROOT}/preprocessed/combined_48k/prompts \
    --prompt-source bonus_seed4321=${DATA_ROOT}/preprocessed/bonus_seed4321/prompts \
    --prompt-source mixed_1=${DATA_ROOT}/preprocessed/mixed_<seed1>/prompts \
    --prompt-source mixed_2=${DATA_ROOT}/preprocessed/mixed_<seed2>/prompts \
    --prompt-source train1176_prompts=${DATA_ROOT}/preprocessed_5L_FP8/train1176/prompts \
    --output ${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired
```

Pass every prompt source you used during §1's loop (including `train1176_prompts` if you have any pool files from before the loop started — these will fail to match against any of the rotation sources). The script tries each source in order and uses the first match.

### Reference output

For the production 6515-file pool, all 6515 files match across 4 sources:

```
matched:    6515 / 6515  (100.00%)
by_source:
  combined_48k       — 2,331
  mixed_1777545051   — 2,290
  mixed_1777534621   — 1,237
  train1176_prompts  —   657
```

If you see less than 100%, either you're missing a source (find it and add `--prompt-source name=path`) or some pool files are corrupt (rare; the validator should have caught them in §1). Open `<output>/pairing_report.json` and `match_table.jsonl` to see which sources matched what.

### What the script writes

```
${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired/
├── prompts/
│   ├── data-00000-of-00001.arrow      # paired prompt rows in pool-index order
│   ├── dataset_info.json
│   └── state.json
├── hidden_states/
│   └── hs_<i>.safetensors             # 6,515 symlinks back to hs_clean_pool
├── pairing_report.json                # match counts per source
└── match_table.jsonl                  # per-file source attribution
```

The trainer reads from `prompts/` and `hidden_states/` only.

### Pitfalls

- **Chat-template prefix is identical across all prompts** (`[200034, 200019, 28463, 10, 2985]` for MiniMax). Don't try to match by the first few tokens; use full-sequence sha256.
- **`list == tensor` is always False** in Python. If you write your own pair-by-content tool, convert both sides to lists or both to tensors before comparing.
- **Cross-source duplicate prompts are common** — `bonus_seed4321` is a strict subset of `combined_48k`, for example. The pairer's "first matching source wins" rule handles this correctly: each pool file gets attributed to its actual generation source, but the prompt content is identical across sources, so train-time content is right either way.
- **Pool token_ids ARE the full prompt** (no truncation, no offset). Empirically `len(token_ids) == len(prompt.input_ids) == prompt.seq_len` for pool files generated by §1's pipeline.

---

## 2.4 Generate the vocab maps (`d2t.npy`, `t2d.npy`, `token_freq.pt`)

The DFlash drafter operates on a reduced vocabulary (default 32,768) while the verifier uses the full MiniMax vocab (200,064). Three files in the paired dataset's `prompts/` subdir bridge the two:

| File | Shape | Dtype | Meaning |
|---|---|---|---|
| `token_freq.pt` | `dict[int, int]` | — | Frequency count of each verifier token under the loss mask |
| `t2d.npy` | `(200064,)` | **bool** | `True` where verifier token is in the draft vocab; `sum() == 32768` |
| `d2t.npy` | `(32768,)` | **int64** | **Offset** table: `verifier_token = draft_id + d2t[draft_id]` |

### Critical format notes (where everyone gets bitten)

- **`t2d` is a bool mask, NOT an index map.** Saving it as int with sentinel `-1` for OOV trips a trainer assertion: `t2d has 536687232 non-zero values, expected 32768`.
- **`d2t` stores OFFSETS, NOT target token ids.** The canonical formula is `d2t[i] = selected_token_ids[i] - i` where `selected_token_ids` is the top-K-by-freq token ids sorted ascending. Saving raw target ids will trip a trainer shape check.
- **All three files must live inside the `prompts/` subdir.** The trainer looks there. Writing them next to the dataset directory (one level up) silently falls back to "no token_freq → use full verifier vocab," which then fails with `draft_vocab_size equals verifier vocab_size`.

### Run the generator

```bash
python repro/scripts/training/build_vocab_maps.py \
    --paired-dir ${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired \
    --verifier-vocab-size 200064 \
    --draft-vocab-size 32768
```

The script wraps speculators' canonical `build_vocab_mappings_from_distribution` and validates the output format before writing.

### Reference values

For the 6515-file `train_all_paired` dataset:

- **2.47 M loss-mask tokens, 59,171 unique**
- **Top-32K coverage of loss-mask tokens: 98.39%** ← this is the number to watch
- Below 95% coverage, your dataset is too narrow to train a 32K-vocab drafter — investigate whether your prompt source is single-domain.
- `d2t.unique() == 22,363` (production reference `train1176/prompts/d2t.npy` had 22,083 — very close; the small difference is just data scale).

---

## 2.5 Verify with a 90-second smoke run BEFORE the multi-hour real run

Always do this. The trainer cold-starts the verifier model (3-5 GB to GPU), takes ~60s, and within the first ~30 steps will surface every data-side bug. A 90s smoke is dramatically cheaper than realizing at hour 3 of 17 epochs that your `t2d` was the wrong dtype.

```bash
PAIRED_DIR=${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired \
WORKSPACE=${WORKSPACE} \
MODELS=${MODELS} \
bash repro/scripts/training/smoke_train.sh
```

The script invokes `torchrun` for 90 seconds and then post-flight-greps the log for known failure markers.

### Pass criteria

- Exit code is `124` (timeout-killed = ran the full 90s without crashing). The launcher's automated check expects exactly this.
- Log contains `global_step=N` for `N ≥ 50`.
- Log contains real `train/position N acc=X` lines (per-position drafter accuracy actually computing).
- Log contains `lr=Y` showing LR-warmup progress (approaching `--lr` value by step ~100).
- Log does **NOT** contain any of:
  - `R54: hs prompt prefix mismatch`
  - `Loaded token ids ... don't match input ids`
  - `anchor_positions include padding locations`
  - `t2d has N non-zero values, expected M`
  - `d2t has N values, expected M`
  - `FileNotFoundError: token_freq.pt`

If any of those fire, fix the data-side issue first; do not proceed.

### Reference smoke evidence

```
[Step 110]  global_step=110, epoch=0, lr=2.95e-05
            train/loss=4.123  train/position 5 acc=0.050
            train/position 6 acc=0.044  train/position 7 acc=0.054
            train/expected accept/proposed=0.008
[Step 115]  global_step=115, epoch=0, lr=2.98e-05
PASS: smoke run clean — safe to proceed to full training
```

115 steps in 90s on a single GB10 (Spark), no warnings, lr warming as expected.

---

## 2.6 Production training run

Use the launcher script under `systemd-run --user` so the run survives any SSH session disconnect or local agent restart. **Do not** use bare `bash launcher.sh` over SSH for a multi-hour run — see `references/sigh​up-gotcha.md` for the empirical reason (SIGHUP propagates through SSH and kills the wrapping bash, which kills torchrun).

### Launcher

`repro/scripts/training/launch_full.sh`:

```bash
#!/usr/bin/env bash
set -eo pipefail

PAIRED=${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired
TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME="full_5L_paired_${TS}"
SAVE_DIR=${WORKSPACE}/dflash_minimax/checkpoints/${RUN_NAME}
LOG=${WORKSPACE}/dflash_minimax/logs/${RUN_NAME}.log
mkdir -p "$SAVE_DIR" "$(dirname "$LOG")"

source ${WORKSPACE}/venvs/vllm/bin/activate
cd ${WORKSPACE}/repos/speculators

# Production hyperparameters that produced the validated MiniMax-M2.7-DFlash.gguf
# (md5 785c5b5a6bcf8eecb545a1bebb75eb4e), now with 5.5x more paired data.
torchrun --master_port=29502 --nproc-per-node=1 \
    scripts/train.py \
    --speculator-type dflash \
    --verifier-name-or-path ${MODELS}/MiniMax-M2.7-FP8 \
    --data-path "${PAIRED}/prompts" \
    --hidden-states-path "${PAIRED}/hidden_states" \
    --save-path "$SAVE_DIR" \
    --epochs 17 \
    --total-seq-len 2048 \
    --max-anchors 512 \
    --num-workers 1 --prefetch-factor 2 \
    --on-missing skip \
    --target-layer-ids 2 16 30 45 59 \
    --draft-arch qwen3 \
    --draft-hidden-act silu \
    --mask-token-id 200054 \
    --block-size 8 \
    --hidden-states-dtype bfloat16 \
    --num-layers 5 \
    --draft-vocab-size 32768 \
    --lr 3e-5 \
    --scheduler-warmup-steps 100 \
    --save-best \
    --log-freq 5 \
  2>&1 | tee "$LOG"
```

### Hyperparameters explained

- `--num-layers 5` — DFlash adapter layers. Distinct from the 6 layer taps in the trace data (the 6th tap is the verifier's last hidden state, used as the cross-attention input).
- `--target-layer-ids 2 16 30 45 59` — must match §1's data generation. The 6th index (62, the last hidden) is auto-appended by the trainer; do not pass it here.
- `--block-size 8` — DFlash chunk length. Must match GGUF inference value in §3.
- `--mask-token-id 200054` — MiniMax-M2.7's `<|mask|>` token id. Mismatch silently degrades every block position.
- `--draft-vocab-size 32768` — drafter operates on this reduced vocab via §2.4's `d2t`/`t2d`.
- `--max-anchors 512` — production setting. Smoke runs use 64 for ~3× faster iteration; final accuracy is broadly similar but per-step compute differs.
- `--total-seq-len 2048` — anchor window. Larger means fewer anchors per row; 2048 is the production value.
- `--lr 3e-5`, `--scheduler-warmup-steps 100` — production. Cosine decay over `epochs × steps_per_epoch`.
- `--save-best` — only writes a checkpoint when val loss strictly decreases; updates `checkpoint_best -> N` symlink. Disk-friendly.

### Launch under systemd-run

```bash
HOST=spark-N    # the training host
ssh ${HOST} "systemctl --user reset-failed dflash-full 2>/dev/null; \
  systemd-run --user \
    --unit=dflash-full \
    --description='DFlash 5L FULL paired training' \
    --collect \
    bash ${WORKSPACE}/repos/speculators-repro/repro/scripts/training/launch_full.sh"
```

### Verify it's actually under user-scope (not session-scope)

```bash
ssh ${HOST} "systemctl --user status dflash-full --no-pager | head -20"
```

The CGroup line should look like:

```
CGroup: /user.slice/user-NNNN.slice/user@NNNN.service/app.slice/dflash-full.service
```

If it shows `session-NNN.scope`, you skipped systemd-run and the process is still SSH-bound. Stop and redo.

### Tail logs without endangering the run

A separate `ssh ${HOST} "journalctl --user -u dflash-full -f"` or `ssh ${HOST} "tail -f ${LOG}"` can be killed at any time without affecting the training. The unit lives in its own scope.

### Stopping cleanly

```bash
ssh ${HOST} "systemctl --user stop dflash-full"
```

---

## 2.7 What success looks like — chained accuracy progression

Real evidence from a production-equivalent run on the 6515-paired-file dataset, GB10 single-rank, hyperparams as in §2.6.

### Per-epoch chained accuracies (∏ p_i)

| Position | Epoch 1 | Epoch 2 | Epoch 3 | (target) Epoch ~17 |
|---|---|---|---|---|
| 1 | 14.00% | 18.50% | **20.50%** | ~21.8% |
| 2 | 1.428% | 2.238% | **2.686%** | ~4.34% |
| 3 | 0.120% | 0.208% | **0.263%** | ~0.568% |
| 4 | 0.00876% | 0.01642% | **0.0218%** | ~0.0705% |
| 5 | 0.000578% | 0.001157% | **0.00164%** | ~0.0085% |
| 6 | 3.58e-07 | 7.45e-07 | **0.000113%** | ~0.001% |
| 7 | 2.11e-08 | 4.55e-08 | **7.35e-08** | ~0.0001% |
| **E[acc]** | 0.156 | 0.207 | **0.235** | ~0.268 |

### What to track

1. **Chain-pos-1** is the single biggest signal. It should grow monotonically from ~10% (1 epoch warmup) to 18-22% (production-quality plateau).
2. **Chain-pos-2** should track ~`(chain-pos-1)²` × ~1.4-1.7. If it lags far behind that ratio, your drafter is overfitting position 1 only.
3. **E[acc]** is the single number that maps to runtime speedup ceiling: speedup ≤ `1 + E[acc]`. Production target ≥ 0.25 means a theoretical ceiling near `1.25×` over autoregressive baseline (real verify cost will reduce this — see §3 on benchmarking).
4. **`val_metrics.json` is overwritten on every "save-best" event.** To get the per-epoch progression, grep the log:

   ```bash
   grep -A12 "val/loss_epoch=" ${LOG}
   ```

### Wall-clock reference

On a single GB10 with `max-anchors=512`:
- ~14 minutes per training epoch (5,863 train rows, ~2,025 steps)
- ~45 seconds per validation epoch (652 val rows)
- **17 epochs total ≈ 4 hours**, plus ~20 min for verifier load + first-batch JIT

`max-anchors=64` (smoke config) is roughly 3× faster per step but converges to lower absolute accuracy in the same number of epochs — use only for plumbing checks.

---

## 2.8 Offline validation — proving the checkpoint isn't broken

Before trusting any checkpoint, run the offline-validation harness that reproduces the trainer's per-position accuracy on the held-out val split. **This is the single highest-value smoke test in the entire pipeline** — it tells you whether the checkpoint, when loaded fresh, gives the same numbers training reported. If it doesn't, you have a save/load bug that will cascade into GGUF and runtime.

```bash
cd ${WORKSPACE}/repos/speculators/scripts
source ${WORKSPACE}/venvs/vllm/bin/activate
python repro/scripts/training/dflash_offline_eval.py \
    --ckpt ${CHECKPOINTS}/full_5L_paired_<TS>/checkpoint_best \
    --data ${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired/prompts \
    --hs   ${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired/hidden_states \
    --verifier ${MODELS}/MiniMax-M2.7-FP8 \
    --max-batches 60
```

### Pass criteria

- Each per-position accuracy within `±5pp absolute` of `val_metrics.json` from the same checkpoint (skill says ±3pp is achievable with 60 batches; use ±5pp as the hard fail threshold).
- `full_acc` within `±2pp absolute`.
- All 7 positions marked ✓ in the output.

### Reference output (real, 60-val-batch run on the SMOKE 3-epoch checkpoint)

```
[eval] full_acc     = 0.0723    (training reported 0.0705)
[eval] position 1 acc = 0.1028    (training reported 0.0965, Δ=+0.0063) ✓
[eval] position 2 acc = 0.0752    (training reported 0.0801, Δ=-0.0049) ✓
[eval] position 3 acc = 0.0723    (training reported 0.0710, Δ=+0.0013) ✓
[eval] position 4 acc = 0.0672    (training reported 0.0655, Δ=+0.0017) ✓
[eval] position 5 acc = 0.0604    (training reported 0.0597, Δ=+0.0006) ✓
[eval] position 6 acc = 0.0669    (training reported 0.0580, Δ=+0.0089) ✓
[eval] position 7 acc = 0.0606    (training reported 0.0619, Δ=-0.0013) ✓
```

All within ±0.9pp of training. With this passing, the checkpoint is safe to convert to GGUF in §3 — any runtime accept-rate gap is now isolated to the conversion or runtime stack, not the trained weights.

### What to do if it fails

If a position is more than ±5pp off, **do not blame GGUF/runtime yet**. The bug is on the safetensors-load side. Diagnostic order:

1. Confirm `model.load_verifier_weights()` is being called (`verifier_lm_head.weight` is loaded from the verifier checkpoint, not left NaN-initialized).
2. Confirm `t2d` and `d2t` loaded with the model — the script does this implicitly via `from_pretrained` but check the load report.
3. Confirm `split_ratio=-0.1` (last 10%, the trainer's val convention) — `split_ratio=0.1` gives the FIRST 10%, which is a different sample distribution.
4. Re-read `references/dflash-drafter-offline-validation.md` for the full diagnostic table.

---

## 2.9 Speculators NaN patches (still required)

The patches in `patches/speculators/` (R27/R28/R29/R30/R31/R32) are still required for training stability on training-side dtype/NaN issues, even though the vLLM-side patches were reverted. They live as separate files in this repo and are applied per the top-level README before training:

```bash
cd ${WORKSPACE}/repos/speculators
for p in ${WORKSPACE}/repos/speculators-repro/patches/speculators/*.patch; do
  git apply "$p"
done
```

If you skip these, expect intermittent loss explosions to NaN around step 200-500.

---

## 2.10 Common failure modes

| Symptom | Root cause | Fix |
|---|---|---|
| `R54: hs prompt prefix mismatch for index N` | Pool file N's `token_ids` ≠ prompt[N].input_ids prefix | Re-pair pool against all candidate prompt sources by content hash (§2.3) |
| `anchor_positions include padding locations` | Downstream of R54 — all anchors degrade to pad position | Same as above; fix the prefix mismatch |
| `t2d has 536687232 non-zero values, expected 32768` | `t2d.npy` saved as int with sentinel `-1`, trainer expects bool mask | Use `build_vocab_maps.py` (§2.4); it emits the correct bool dtype |
| `t2d.sum() != draft_vocab_size` | Same as above | Same fix |
| `FileNotFoundError: token_freq.pt` | File written next to dataset dir instead of inside `prompts/` | The trainer looks in `prompts/`; place all three vocab files there |
| Trainer falls back to `draft_vocab_size = verifier_vocab_size` | `token_freq.pt` missing or unreadable | Verify path inside `prompts/`, regenerate from paired dataset |
| Loss explodes to NaN ~step 200-500 | Missing speculators NaN patches | Apply patches per §2.9 |
| Run dies when SSH session ends | Bare `bash launcher.sh` over SSH propagates SIGHUP | Always launch under `systemd-run --user` (§2.6) |
| Offline eval pos-N accuracy off by >5pp from `val_metrics.json` | `load_verifier_weights()` skipped, OR wrong split_ratio sign, OR vocab maps not loaded | See §2.8 diagnostic order |

---

## 2.11 What's next

After §2 you should have on disk:

```
${CHECKPOINTS}/full_5L_paired_<TS>/checkpoint_best/
├── config.json
├── config.py
├── model.safetensors          # ~2.1 GB for 5L Qwen3 hidden=3072
├── optimizer_state_dict.pt
├── scheduler_state_dict.pt
└── val_metrics.json           # last-best epoch's per-position metrics
```

Plus the offline-eval pass evidence from §2.8.

Continue to **[Section 3 — Inference](03-inference.md)** for GGUF conversion and the llama.cpp speculative-decode benchmark.

---

## See also

- `references/dflash-drafter-offline-validation.md` — full skill content for the validation harness, including the chain-gated metric framing and the "47× gap" cautionary tale
- `references/sighup-gotcha.md` — why you launch under systemd-run, with empirical evidence from a real run that lost 7 minutes of warmup to a SIGHUP-via-SSH kill
- `references/pool-completeness-debugging.md` — what to do if your pool ⊆ peer-staging check fails
