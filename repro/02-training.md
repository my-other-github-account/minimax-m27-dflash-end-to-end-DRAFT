# Section 2 — Training the DFlash Drafter

End-to-end reproduction of the DFlash drafter checkpoint that pairs with `MiniMax-M2.7-FP8` as the verifier. The output is a PyTorch checkpoint convertible to GGUF for `llama.cpp` inference.

> **What's in this section:** Inputs, the canonical paired-dataset structure (the most failure-prone part), how to rebuild a paired dataset from a heterogeneous pool, vocab-map generation, the last-known-good training config, smoke verification, and observed training metrics.

---

## 2.1 Inputs

- **Hidden-state pool** from §1: a directory of `hs_<N>.safetensors` files, each containing per-layer hidden states **and the `token_ids` that produced them**.
- **Prompt sources**: one or more HuggingFace `Dataset` directories (arrow + dataset_info.json) whose `input_ids` produced the pool. Each pool file's saved `token_ids` must equal the first `len(token_ids)` tokens of exactly one prompt's `input_ids`.
- **Verifier model**: `${MODELS}/MiniMax-M2.7-FP8`.
- **Speculators repo**: `${WORKSPACE}/repos/speculators` @ `67bafe6`.

---

## 2.2 The paired-dataset requirement (CRITICAL)

The trainer requires a **strict 1:1 index pairing** between hidden-state files and prompts:

```
<dataset_dir>/
├── prompts/
│   ├── data-00000-of-00001.arrow      (HF Dataset: rows have input_ids, loss_mask, ...)
│   ├── dataset_info.json
│   ├── state.json
│   ├── d2t.npy                        (int64, shape [draft_vocab_size])
│   ├── t2d.npy                        (bool,  shape [verifier_vocab_size])
│   └── token_freq.pt                  (dict[int,int] of token-id → count)
└── hidden_states/
    └── hs_<i>.safetensors             (one file per row i in prompts/)
```

The pairing rule the trainer enforces:

> `hs_<i>.safetensors`'s saved `token_ids` MUST equal the first N tokens of `prompts[i]["input_ids"]` (N = len(token_ids)).

If this is violated, you'll see this exact failure cascade:

```
UserWarning: R54: hs prompt prefix mismatch for index <i>
UserWarning: Loaded token ids tensor([...]) for index <j> don't match input ids tensor([...])
ValueError: anchor_positions include padding locations: [0, 0, 0, ...]
```

The padding-anchor `ValueError` is the **downstream** symptom — when the trainer can't find a valid prompt prefix, all anchor positions degrade to position 0 (a pad), and validation fails. **Fix the prefix-mismatch warnings, not the padding-anchor error.**

---

## 2.3 Rebuilding a paired dataset from a heterogeneous pool

If your pool was accumulated by an autonomous trace-generation loop (multiple prompt sources, monotonic file renaming by a validator), the pool's `hs_<N>.safetensors` index `N` no longer corresponds to row `N` of any single prompt source. You must **re-pair by content**.

### Algorithm

1. For every `hs_<N>.safetensors` in the pool: read its `token_ids` field, compute `sha256(json(token_ids.tolist()))`.
2. For every candidate prompt source: load the arrow file, for each row compute `sha256(json(input_ids.tolist()))`. Build a hash → (source, row_idx) map.
3. For each pool hash: find the matching prompt row. Emit a paired dataset where prompts are taken in pool-index order and hidden-states are symlinked back to the original pool files.

### Reference implementation

`scripts/training/build_paired_dataset.py` (in this repo) implements this. Run it as:

```bash
python scripts/training/build_paired_dataset.py \
    --pool ${DATA_ROOT}/preprocessed_5L_FP8/hs_clean_pool \
    --prompt-source name1=${DATA_ROOT}/preprocessed/source1/prompts \
    --prompt-source name2=${DATA_ROOT}/preprocessed/source2/prompts \
    --prompt-source name3=${DATA_ROOT}/preprocessed/source3/prompts \
    --output ${DATA_ROOT}/preprocessed_5L_FP8/train_all_paired
```

Outputs:
- `train_all_paired/prompts/data-*.arrow` — paired prompt rows in pool-index order
- `train_all_paired/hidden_states/hs_<i>.safetensors` — symlinks back to pool
- `train_all_paired/pairing_report.json` — match counts per source, unmatched count
- `train_all_paired/match_table.jsonl` — per-file provenance

### Empirical recovery rate

For the production 6515-file pool generated across the loop's prompt-source sequence (`combined_48k` → `bonus_seed4321` → `mixed_1777534621` → `mixed_1777545051`), **all 6515 files matched** once `train1176_prompts` was added as a 5th source — those 657 unmatched files were from an earlier preprocessed_5L_FP8 production curated subset.

```
matched:    6515 / 6515  (100.00%)
by_source:
  combined_48k       — 2,331
  mixed_1777545051   — 2,290
  mixed_1777534621   — 1,237
  train1176_prompts  —   657
```

### Pitfalls

- **First 5–10 tokens are useless as a discriminator**: they're the chat-template prefix `[200034, 200019, 28463, 10, 2985]` (`<|begin_of_document|><|user_turn|>`-equivalent) and are identical across all prompts. Use full-sequence sha256.
- **Cross-source duplicate prompts are common**: `bonus_seed4321` is a strict subset of `combined_48k`. The pairer's "first matching source" rule handles this correctly — pool files are attributed to the source they were *generated from*, but the prompt content is identical across sources, so the train-time content is right either way.
- **`list == tensor` is always False in Python**: when verifying matches, convert both to lists or both to tensors before comparing.
- **The pool's `token_ids` is the full prompt** (no truncation, no offset): we verified `len(token_ids) == len(prompt.input_ids) == prompt.seq_len` for known-good production data.

---

## 2.4 Generating vocab maps (`d2t.npy`, `t2d.npy`, `token_freq.pt`)

The DFlash drafter operates on a reduced vocabulary (default 32768). Three files in the `prompts/` subdir bridge it to the full verifier vocab (200064 for MiniMax-M2.7):

| File | Shape | Dtype | Meaning |
|---|---|---|---|
| `token_freq.pt` | `dict[int, int]` | — | Frequency count of each verifier token id under the loss mask |
| `t2d.npy` | `(verifier_vocab_size,)` | **bool** | `True` where verifier token is in the draft vocab; `sum() == draft_vocab_size` |
| `d2t.npy` | `(draft_vocab_size,)` | **int64** | **Offset** table: `verifier_token = draft_id + d2t[draft_id]` |

### CRITICAL format notes

- `t2d` is a **bool mask**, NOT an index map. If you emit it as int with `-1` for OOV and indices for in-vocab, the trainer's check `t2d.sum() == draft_vocab_size` will fail with errors like `t2d has 536687232 non-zero values, expected 32768`.
- `d2t` stores **offsets**, NOT target token ids. The canonical formula is `d2t[i] = selected_token_ids[i] - i` where `selected_token_ids` is the top-K-by-freq token ids sorted ascending.

### Canonical generation

Use speculators' own helper (in `speculators.train.utils.vocab_mapping`):

```python
from speculators.train.utils.vocab_mapping import build_vocab_mappings_from_distribution
d2t, t2d = build_vocab_mappings_from_distribution(
    token_freq_dict,                  # dict[int,int]
    target_vocab_size=verifier_vocab_size,   # 200064 for MiniMax-M2.7
    draft_vocab_size=draft_vocab_size,       # 32768
)
```

`scripts/training/build_vocab_maps.py` in this repo wraps this and emits all three files from a paired dataset's `prompts/` arrow. Loss-mask coverage of the top-32K tokens should be ≥98% for a healthy dataset; below that, the dataset is likely too narrow.

### Reference values

For the 6515-file `train_all_paired` dataset:
- 2.47 M loss-mask tokens, 59,171 unique
- Top-32K coverage: **98.39%**
- `d2t.unique() == 22,363` (production `train1176` had 22,083 — close, expected)

---

## 2.5 Last-known-good training config

The drafter `MiniMax-M2.7-DFlash.gguf` (MD5 `785c5b5a6bcf8eecb545a1bebb75eb4e`) currently in production was trained with:

```bash
torchrun --master_port=29501 --nproc-per-node=1 \
    ${WORKSPACE}/repos/speculators/scripts/train.py \
    --speculator-type dflash \
    --verifier-name-or-path ${MODELS}/MiniMax-M2.7-FP8 \
    --data-path ${DATA_ROOT}/preprocessed_5L_FP8/train1176/prompts \
    --hidden-states-path ${DATA_ROOT}/preprocessed_5L_FP8/train1176/hidden_states \
    --save-path ./checkpoints/dflash-drafter \
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
    --log-freq 5
```

Critical details:
- `--num-layers 5` for the DFlash adapter (5-layer drafter, distinct from the 6 layer taps in the trace data — the 6th tap is the verifier last hidden, used as the verifier-side cross-attention input).
- `--target-layer-ids 2 16 30 45 59` from §1; the auto-appended last-hidden index 62 is **not** passed here.
- `--block-size 8` (DFlash chunk length) — must match GGUF inference value.
- `--mask-token-id 200054` (MiniMax-M2.7 mask token id).
- `--draft-vocab-size 32768` — drafter operates on the reduced vocab via the `d2t`/`t2d` maps generated in §2.4.
- `--hidden-states-dtype bfloat16` — matches what §1 produced.
- `--max-anchors 512`, `--total-seq-len 2048` for the production run; smoke runs use `--max-anchors 64`, `--total-seq-len 4096`.

### Scaling to a larger paired pool

To train on `train_all_paired` (6515 files, 5.5× the production data), point `--data-path` and `--hidden-states-path` at the new directories. All other flags unchanged. Expect ~5.5× more steps per epoch and proportionally longer wall-clock; LR/warmup unchanged.

---

## 2.6 Smoke verification

Before launching a full multi-hour training run, verify the data path with a 90-second smoke run:

```bash
timeout 90 torchrun --master_port=29501 --nproc-per-node=1 \
    ${WORKSPACE}/repos/speculators/scripts/train.py \
    --speculator-type dflash \
    --verifier-name-or-path ${MODELS}/MiniMax-M2.7-FP8 \
    --data-path <paired_dir>/prompts \
    --hidden-states-path <paired_dir>/hidden_states \
    --save-path /tmp/dflash-smoke \
    --epochs 1 \
    --total-seq-len 2048 \
    --max-anchors 64 \
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
    --log-freq 5 \
  2>&1 | tee /tmp/dflash-smoke.log
```

**Pass criteria:**

- Exit code is `124` (timeout-killed, i.e. ran the full 90s without crashing).
- Log contains `global_step=N` for `N ≥ 50`.
- Log contains `train/position N acc=X` lines (real per-position drafter accuracy).
- Log contains `lr=Y` showing LR-warmup progress (should be approaching `--lr` value by step ~100).
- Log does **NOT** contain:
  - `R54: hs prompt prefix mismatch`
  - `Loaded token ids ... don't match input ids`
  - `anchor_positions include padding locations`
  - `t2d has N non-zero values, expected M`
  - `d2t has N values, expected M`

If any of these appear, do not proceed to the full run; fix the data first.

### Smoke evidence (2026-04-30, 6515-file paired dataset)

```
[Step 110]  global_step=110, epoch=0, lr=2.95e-05
            train/loss=4.123  train/position 5 acc=0.050
            train/position 6 acc=0.044  train/position 7 acc=0.054
            train/expected accept/proposed=0.008
[Step 115]  global_step=115, epoch=0, lr=2.98e-05
```

115 steps in 90s on a single H100, no warnings, lr warming as expected.

---

## 2.7 Production training metrics (last-known-good)

From `val/full_acc_epoch=0.143` of the verified production run (2026-04-29 21:16 PT, 1176-file dataset, 17 epochs):

| Position | Accuracy |
|---|---|
| 1 | 0.218 |
| 2 | 0.199 |
| 3 | 0.131 |
| 7 | 0.116 |
| prefix ≥ pos 1 | 0.218 |
| prefix ≥ pos 2 | 0.009 |

The chain-gated prefix accuracy collapses fast — see §3 on metric framing.

For the 6515-file `train_all_paired` run, expect **higher position-wise accuracies** (5.5× more data, more diverse prompt sources) but the prefix-collapse pattern will persist (it's a property of multi-token speculation, not the data).

---

## 2.8 Patches required

The NaN-bug patches in `patches/speculators/` (R27/R28/R29/R30/R31/R32) are still required for training stability on training-side dtype/NaN issues, even though the vLLM-side patches were reverted. Apply them per the top-level README before training. These remain valid — they fix bugs in the trainer itself, not in the trace pipeline.

---

## 2.9 GGUF conversion

After training, convert `checkpoint_best/` to GGUF for `llama.cpp` inference. See `repro/legacy/prep_for_pr22105_converter.py` for the speculators-checkpoint → PR-#22105-converter-input transformation, and the top-level README §"Building the patched llama.cpp" for the converter invocation.

---

## 2.10 Common failure modes (lessons learned)

| Symptom | Root cause | Fix |
|---|---|---|
| `R54: hs prompt prefix mismatch for index N` | Pool file N's `token_ids` != prompt[N].input_ids prefix | Re-pair pool against all candidate prompt sources by content hash (§2.3) |
| `anchor_positions include padding locations` | Downstream of R54 — all anchors degrade to pad position | Same as above; fix the prefix mismatch |
| `t2d has 536687232 non-zero values, expected 32768` | `t2d.npy` saved as int with sentinel `-1`, trainer expects bool mask | Use `build_vocab_mappings_from_distribution` (§2.4) — emits correct bool dtype |
| `t2d.sum() != draft_vocab_size` | Same as above | Same fix |
| `FileNotFoundError: token_freq.pt` | File written next to dataset dir instead of inside `prompts/` | The trainer looks in `prompts/`; place all three vocab files there |
| Trainer falls back to `draft_vocab_size = verifier_vocab_size` | `token_freq.pt` missing or unreadable | Verify path, regenerate from paired dataset |
| Loss explodes to NaN early | Missing speculators NaN patches | Apply patches per §2.8 |

---

Continue to [Section 3 — Inference](03-inference.md), or back to [Section 1 — Generation](01-generation.md).
