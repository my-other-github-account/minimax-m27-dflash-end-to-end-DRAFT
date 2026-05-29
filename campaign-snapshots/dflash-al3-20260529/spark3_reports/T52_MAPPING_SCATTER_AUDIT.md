# T52 mapping/scatter audit of T48 low-AL rebake

Date: 2026-05-29
Task: t_9b2b7536
Scope used: macmini local logs + spark-work read-only only. I did not touch spark-2, spark-3 trainer, spark-4, training, inference, Kasa, or source/model files.

## Bottom line

Verdict: `REMOTE_ARTIFACTS_NEEDED` for the full requested T52 audit because the exact T48 rebaked tensor and raw generation/tokenization files live on spark-2 and are not mirrored on spark-work.

However, the fast local/spark-work checks strongly overturn the specific T51 suspicion that duplicate raw `d2t` values caused T48 scatter collisions. In this codebase `d2t` is not a direct target-token-id list. It is a draft-index offset vector. The target token id is computed as:

`target_token_id = draft_index + d2t[draft_index]`

That convention is explicit in the checked source and in T48's own rebake command. Under that convention, duplicate raw `d2t` offset values are allowed and do not imply duplicate/missing scatter target rows.

## Evidence checked

### 1. What `d2t` and `t2d` are

Checked read-only on spark-work:

- Source file `/home/dnola/dflash-llama-c15/repro/scripts/inference/prep_for_pr22105_v2_with_d2t_rebake.py` says:
  - lines 13-17: fix is to scatter trained draft-vocab lm_head into rows indexed by `d2t[i] + i`, with non-mapped rows set to `-65504`.
  - lines 78-84: function docstring says target-vocab shape `[target_V, hidden]`, and `target_token_id = i + d2t[i]`.
  - lines 93-102: code computes `indices = torch.arange(draft_V) + d2t` and then `expanded[indices] = lm_head_draft`.
  - lines 131-152: `t2d` is skipped/dropped; runtime does not use it after rebake.

- Source file `/home/dnola/dflash-llama-c15/scripts/prep_full_for_buun_converter.py` says:
  - lines 6-9: read `d2t` as draft->target offsets, build full `lm_head` by scatter.
  - lines 120-141: if `lm_head` is draft-sized, compute `target_ids = torch.arange(draft_vocab) + d2t`, assert max target id is in vocab, fill new full-vocab lm_head, then scatter `new_lm_head[target_ids] = lm_head`.
  - line 125 logs `t2d.sum()` against `draft_vocab`, consistent with `t2d` being a bool target-vocab mask, not an inverse int table.
  - line 168 sets `draft_vocab_size = target_vocab` after rebake.

Spark-work mapping check for the same nominal path family:

```text
/home/dnola/iq4_full_run/iq4_v17_consolidated/prompts/d2t.npy
raw d2t: shape=(65536,), dtype=int64, min=2, max=134516, unique=34454, dup_values=12713, max_count=95
ids = arange + d2t: shape=(65536,), min=2, max=200051, unique=65536, dup_values=0, in_range=True
/home/dnola/iq4_full_run/iq4_v17_consolidated/prompts/t2d.npy
t2d: shape=(200064,), dtype=bool, sum=65536
t2d true positions exactly equal set(ids)=True
sha12: d2t cb58001fbf1d, t2d d5bcab2605bc
```

This spark-work dataset is 65,536-draft-vocab, not T48's 32,768-draft-vocab, so it is not the exact T48 artifact. It still validates the representation: raw `d2t` offsets are duplicate-heavy, but `arange + d2t` target ids are unique and exactly match the bool `t2d` mask.

T48 exact local log/report evidence:

- T48 report recorded the source `lm_head.weight=(32768,3072)` and target vocab/token count `200064`.
- T48 report recorded mapping as `d2t.shape=(32768,)`, target ids min=9 max=200051, unique=32768, and `t2d.shape=(200064,)`, sum=32768.
- T48 log lines 254-261 show the exact rebake command used:
  - load `/home/dnola/iq4_full_run/iq4_v17_consolidated/prompts/d2t.npy`
  - `ids=torch.arange(lh.shape[0]) + torch.from_numpy(d2t_np).long()`
  - assert `torch.unique(ids).numel()==lh.shape[0]`
  - `new=torch.full((200064, lh.shape[1]), -65504.0, dtype=lh.dtype)`
  - `new[ids]=lh`

Conclusion for Q1: `d2t` is a length-draft-vocab offset map, not a direct unique-token list. `t2d` being a bool target-vocab mask is intentional and consistent with `flatnonzero(t2d) == arange + d2t`; it is not expected to be an integer inverse map in the rebaked/full-vocab path.

### 2. Did duplicate/missing d2t entries cause T48 rows to be unfilled/zero/incorrect?

What can be answered without touching spark-2:

- Raw duplicate `d2t` values alone did not cause scatter collisions in T48, because T48 scattered by `arange + d2t`, not by raw `d2t`.
- T48 exact target ids were recorded as unique=32768. Therefore duplicate target-id count for the actual T48 scatter is 0 by the checked scatter convention.
- Expected full-vocab row classes for T48:
  - total target rows: 200064
  - learned/scattered rows: 32768
  - deliberate floor-filled rows: 200064 - 32768 = 167296
  - floor value: -65504.0
- These floor rows are intentional in the full-vocab rebake because buun expects the draft `output.weight` to match the full tokenizer vocabulary after rebake; the runtime should not select floor-filled rows under normal argmax/top-k unless all learned rows are worse.

What could not be completed without spark-2 exact artifacts:

- Exact zero-row count vs floor-row count inside `/home/dnola/v18_fix/t48/prepped_minimax_rebaked_v17d2t/model.safetensors`.
- Exact norm histogram of learned/scattered rows vs floor rows.
- Exact check of the three T48 generated content token IDs against the correct `target_ids = arange + d2t` set. T51's 56.05% coverage number used raw `d2t` values as token IDs, so it is not valid evidence of missing generated-token rows.

Files needed for exact completion after T50 frees spark-2:

- `/home/dnola/v18_fix/t48/prepped_minimax_rebaked_v17d2t/model.safetensors`
- `/home/dnola/v18_fix/t48/al_runs.jsonl`
- `/home/dnola/models/MiniMax-M2.7-tokenizer/` or the tokenizer files used by T48
- `/home/dnola/iq4_full_run/iq4_v17_consolidated/prompts/d2t.npy`
- `/home/dnola/iq4_full_run/iq4_v17_consolidated/prompts/t2d.npy`

### 3. Does buun/minimax conversion expect reduced rows + metadata, or full-scattered lm_head?

Available source/log evidence supports full-scattered `output.weight` for this buun path:

- T48 report says the first conversion with 32768-row output was rejected by buun:
  `check_tensor_dims: tensor 'output.weight' has wrong shape; expected 3072, 200064, got 3072, 32768`.
- T48 final GGUF metadata verified `tokenizer.ggml.tokens=200064` and `output.weight` GGUF reader shape `(3072,200064)`.
- `/home/dnola/dflash-llama-c15/scripts/prep_full_for_buun_converter.py` explicitly rebakes draft-vocab `lm_head` into `[target_vocab, hidden]`, drops d2t/t2d, and sets `draft_vocab_size=target_vocab` after rebake.
- `/home/dnola/dflash-llama-c15/repro/scripts/inference/prep_for_pr22105_v2_with_d2t_rebake.py` says runtime does not honor d2t/t2d after conversion and the rebake replaces their function.
- Runtime source `/home/dnola/llama.cpp.buun/src/llama-context.cpp` initializes sampling over `model.vocab.n_tokens()` full vocabulary. Runtime source `/home/dnola/llama.cpp.buun/src/llama-model.cpp` treats `output.weight` as the output tensor. No checked source evidence showed a reduced-vocab + mapping-metadata path for this T48 buun runtime.

Conclusion for Q3: for the T48 buun/minimax path, the expected artifact is a full-scattered 200064-row `lm_head`/`output.weight`, not a 32768-row reduced head plus mapping metadata.

## Verdict

Primary verdict for this run: `REMOTE_ARTIFACTS_NEEDED` because exact T48 row norms and generated-token coverage require spark-2 files listed above, and the task explicitly forbids contending with spark-2 while T50 is running.

Sub-verdict from completed checks: `CONVERSION_ARTIFACT_UNLIKELY` for the specific suspected duplicate-raw-d2t scatter collision. The low AL should not be blamed on `d2t unique=20681` unless a follow-up on spark-2 disproves the logged `target_ids = arange + d2t` uniqueness/assert path.

## Next exact spark-2 read-only check when idle

Run one CPU-only script on spark-2 that:

1. Loads T48 exact `d2t.npy`/`t2d.npy`.
2. Computes `ids = arange(len(d2t)) + d2t` and verifies `set(ids) == set(flatnonzero(t2d))`.
3. Opens `/home/dnola/v18_fix/t48/prepped_minimax_rebaked_v17d2t/model.safetensors` and computes:
   - learned/floor/zero row counts for `lm_head.weight`
   - row norm histogram for `ids` rows vs non-ids rows
4. Tokenizes prompts and completions from `/home/dnola/v18_fix/t48/al_runs.jsonl` with `/home/dnola/models/MiniMax-M2.7-tokenizer/` and checks token coverage against `ids`, not raw `d2t`.
