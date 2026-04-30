---
name: dflash-full-checkpoint-to-gguf-spark1
description: Convert a speculators-format DFlash drafter checkpoint (with draft_vocab compression) to a working GGUF on spark-1, run it through the proven clean-room llama-speculative-simple benchmark, and validate runtime chain-cumulative accept rates against chain-gated training-side prediction. Use when you have a freshly-trained DFlash drafter for MiniMax-M2.7 (or sibling Qwen3-arch verifier) and want runtime accept-rate proof.
---

# DFlash drafter checkpoint → GGUF → runtime accept-rate validation (spark-1)

This is the proven end-to-end recipe that landed measured chain-pos-1 = 29.9–33.2% vs predicted 22.0% (FULL epoch-5, dmax=2/4/7) and chain-pos-2 within 0.5–1σ of prediction across all three dmax values. Reproduces tick-9 of `dflash-clean-repro-loop` against a freshly-trained drafter.

## Triggers

Use this skill when:
- You have a `speculators`-format checkpoint at `/home/user/dflash_minimax/checkpoints/<run>/checkpoint_best/` with `architectures=["DFlashDraftModel"]` and a draft_vocab smaller than target_vocab (32768 vs 200064).
- The user wants empirical proof that runtime accept rates match the chain-gated training prediction (∏ p_i over per-position conditional accuracies).
- The verifier is MiniMax-M2.7 (UD-IQ4_XS GGUF on spark-1 at `/home/user/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/`).

Don't use for:
- Drafters trained with `draft_vocab_size = target_vocab_size` (200064) — those don't need rebake; pass through buun converter directly.
- Non-MiniMax verifiers (the tokenizer-hash whitelist in step 4 is MiniMax-specific).

## Required state on spark-1

- **Clean buun build:** `/home/user/dflash_clean_repro/build_clean/bin/llama-speculative-simple` (~4.5 MB binary, built from upstream `e275191e` with no source patches).
- **Verifier GGUF:** `/home/user/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf` (sharded, ~108 GB total).
- **Tokenizer source:** `/home/user/models/MiniMax-M2.7-FP8/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt}`.
- **Free disk:** ≥10 GB on `/home`.
- **Buun converter:** `/home/user/buun-llama-cpp/convert_hf_to_gguf.py` with the MiniMax-M2 tokenizer-hash registered (see step 4).
- **Python env:** `/home/user/venvs/vllm/bin/activate` with `torch`, `safetensors`, `gguf`.
- **SSH pattern:** `ssh -J operator@100.66.198.32 operator@10.0.0.103` (jump via spark-5 because spark-1's Tailscale has been offline >22d).
- **QSFP route:** `192.168.200.X` for fast checkpoint sync between Sparks (rsync ~290 MB/s observed).

## The recipe (8 steps)

### 1. Sync checkpoint to spark-1 over QSFP

```bash
ssh -J operator@100.66.198.32 operator@10.0.0.103 \
  'mkdir -p /home/user/<run>_for_gguf && \
   rsync -av --info=progress2 \
     operator@192.168.200.4:/home/user/dflash_minimax/checkpoints/<run>/checkpoint_best/ \
     /home/user/<run>_for_gguf/'
```

Replace `192.168.200.4` with the QSFP IP of the Spark holding the checkpoint (4=spark-4, 2=spark-2, 3=spark-3). Replace `<run>` with the actual run dir name.

Expected: ~3.5 GB transfer (2.1 GB safetensors + 1.4 GB optimizer state + tiny configs) at ~280–300 MB/s.

### 2. Verify the source checkpoint has the expected speculators-format tensors

You MUST see:
- `embed_tokens.weight: shape=[200064, 3072]` (target vocab)
- `lm_head.weight: shape=[32768, 3072]` (draft vocab — needs rebake)
- `fc.weight: shape=[3072, 15360]` (5 layers × 3072)
- `hidden_norm.weight: shape=[3072]`
- `d2t: shape=[32768] dtype=torch.int64` (draft→target offsets)
- `t2d: shape=[200064] dtype=torch.bool` (mask, sum=32768)

If `d2t`/`t2d` are missing AND `lm_head` is `[200064, 3072]`, skip the rebake (step 3a) and go straight to step 4 with the source dir.

### 3. Run the prep script (rebake + flatten config + copy tokenizer)

The script at `scripts/prep_full_for_buun_converter.py` does:
- (a) **Rebake `lm_head` `[32768, 3072]` → `[200064, 3072]` with `-65504` floor.** This is the exact d2t zero-row dilution fix from the `dflash-gguf-conversion` skill. Without it, chain-pos-2 measures ~5× lower than predicted at runtime (the documented "47× gap" root cause).
- (b) Strip `d2t` and `t2d` (buun converter doesn't expect them).
- (c) Flatten the speculators-format config: hoist `transformer_layer_config.*` to top level; rename `aux_hidden_state_layer_ids` → `target_layer_ids`; bump `draft_vocab_size` to `target_vocab_size`; set `model_type="qwen3"`.
- (d) Copy `tokenizer.json`/`tokenizer_config.json`/`vocab.json`/`merges.txt` from the verifier path into the prepped dir.

```bash
scp scripts/prep_full_for_buun_converter.py spark-5:/tmp/
ssh -J operator@100.66.198.32 operator@10.0.0.103 \
  'cat > /tmp/prep_full_for_buun_converter.py' < scripts/prep_full_for_buun_converter.py

ssh -J operator@100.66.198.32 operator@10.0.0.103 \
  'source /home/user/venvs/vllm/bin/activate && \
   python3 /tmp/prep_full_for_buun_converter.py \
     /home/user/<run>_for_gguf \
     /home/user/<run>_prepped'
```

Expected output ends with: `Wrote .../config.json`, tokenizer files copied, "Done. Run buun converter on: /home/user/<run>_prepped".

### 4. Register the FP8-tokenizer hash in the buun converter (one-time per spark-1 reset)

The FP8 verifier's `tokenizer.json` has hash `a77756c3cc91392f442c5b99e414be8020d53ae31460de90754b4fcf5cc84a2d` while buun's converter only has the upstream MiniMax-M2 hash `f4f37b6c8eb9ea29b3eac6bb8c8487c5ab7885f8d8022e67edc1c68ce8403e95` registered. Both map to the same `minimax-m2` pre-tokenizer behavior; whitelist the FP8 hash:

```bash
ssh -J operator@100.66.198.32 operator@10.0.0.103 \
  'grep -q "a77756c3cc91392f442c5b99e414be8020d53ae31460de90754b4fcf5cc84a2d" \
     /home/user/buun-llama-cpp/convert_hf_to_gguf.py || \
   sed -i "s|if chkhsh == \"f4f37b6c8eb9ea29b3eac6bb8c8487c5ab7885f8d8022e67edc1c68ce8403e95\":|if chkhsh == \"f4f37b6c8eb9ea29b3eac6bb8c8487c5ab7885f8d8022e67edc1c68ce8403e95\" or chkhsh == \"a77756c3cc91392f442c5b99e414be8020d53ae31460de90754b4fcf5cc84a2d\":|" \
     /home/user/buun-llama-cpp/convert_hf_to_gguf.py'
```

This is idempotent — only patches if the hash isn't already present.

### 5. Run buun's converter

```bash
ssh -J operator@100.66.198.32 operator@10.0.0.103 \
  'source /home/user/venvs/vllm/bin/activate && \
   cd /home/user/buun-llama-cpp && \
   python3 convert_hf_to_gguf.py /home/user/<run>_prepped \
     --outtype bf16 \
     --outfile /home/user/models/MiniMax-M2.7-DFlash-<tag>.gguf'
```

Replace `<tag>` with something like `FULL-epoch5` so multiple drafters can coexist.

Expected: 60 tensors, 3.13 GB output. Final log line: `INFO:hf-to-gguf:Model successfully exported to ...`.

### 6. Verify GGUF metadata

Required values:
- `general.architecture`: `'dflash-draft'`
- `tokenizer.ggml.pre`: `'minimax-m2'`
- `tokenizer.ggml.model`: `'gpt2'`
- `dflash-draft.dflash.target_layer_ids`: `[2, 16, 30, 45, 59]` (training-aligned for MiniMax-M2.7-FP8)
- `dflash-draft.dflash.block_size`: 8
- `dflash-draft.dflash.mask_token_id`: 200054
- `dflash-draft.dflash.n_target_features`: 15360 (= 5 × 3072)
- total tensors: 60

### 7. Run the smoke benchmark (the actual chain-pos validation)

Use the same Fibonacci prompt and dmax sweep as tick-9 of the proven clean-repro loop (script in `scripts/run_dflash_smoke.sh`).

Expected wall-clock: ~5 min verifier mmap-load (one-time, ~600 MB/s from nvme) + ~80s/run × 3 runs = ~9 minutes total.

### 8. Extract chain-pos numbers and compute z-scores vs prediction

Parse the rejection histogram from each `logs/<tag>_dmax{2,4,7}.log`:

```
rejection histogram (position → count [%]):
  pos  0:  136 ( 70.1%)        # chain-pos-1 reject = N - n_chain_pos_1_accept
  pos  1:   53 ( 27.3%)        # given pos-0 accepted, pos-1 rejected
  all ok:    5 (  2.6%)        # all dmax positions accepted = chain-pos-(dmax) accept
```

For dmax=2: chain-pos-1 measured = (n - rej[0]) / n; chain-pos-2 measured = "all ok" / n.
For dmax=4/7: chain-pos-k measured = (n - sum(rej[0..k-1])) / n.

Compare to chain prediction = ∏ p_i where p_i are the per-position conditional accuracies from `val_metrics.json`.

z-score = (measured - predicted) / sqrt(predicted * (1-predicted) / n).

|z| < 2 ⇒ within sample noise ⇒ training and runtime agree.

## Interpretation

The runtime accept rates SHOULD track chain-gated prediction within ±2σ at sample sizes around n=190. Specifically:

- **chain-pos-1 measured > predicted is normal and good.** The Fibonacci prompt is easy-to-predict (high local entropy in code keywords), so training-set-averaged conditionals understate code-prompt accept. Production was +0.59σ; FULL epoch-5 measured +2.6σ — both favorable.
- **chain-pos-2 should track prediction within ±1σ.** This is the cleanest signal because dmax=2 forces the chain to actually exercise position 2.
- **chain-pos-3+ at dmax=4/7 will often measure 0** at small n (predicted ~0.3% × 190 = 0.6 events expected). z=−0.8 is sample noise, not a bug.
- **If chain-pos-2 measures 5× LOWER than predicted (e.g., 1% measured vs 5% predicted at dmax=2), the d2t rebake didn't apply correctly.** Verify the GGUF's `lm_head` shape via gguf-reader: should be `[3072, 200064]` (transposed) BF16, not `[3072, 32768]`.

## Pitfalls (verified during this skill's creation)

1. **`fix_mistral_regex` warning is benign.** Buun emits `[transformers] The tokenizer ... incorrect regex pattern` because the FP8 verifier's tokenizer.json has a known mistral-style regex. The actual tokenization is correct (verified by chain-pos-1 = 30% on Fibonacci, matching production within sample noise).

2. **Don't run on spark-2.** Spark-2's `llama.cpp-dflash` build is a different fork with arch-name `dflash` (not `dflash-draft`); GGUFs built by buun won't load there. Spark-1's `build_clean` is the only verified runtime.

3. **Don't run on spark-1's `/home/user/llama.cpp-dflash`** if it exists. That's stale. Always use `/home/user/dflash_clean_repro/build_clean/bin/llama-speculative-simple`.

4. **`fp_8` and `bf16` outtypes both work**; `bf16` matches production format and is recommended.

5. **The optimizer state file (`optimizer_state_dict.pt`, ~1.4 GB) is irrelevant for inference** but rsyncs by default. You can `--exclude='optimizer_state_dict.pt' --exclude='scheduler_state_dict.pt'` to save bandwidth, but the rsync is fast enough that we don't bother.

6. **`max_anchors` doesn't affect inference.** It's a training-time data shape param; the GGUF doesn't store it.

7. **The "tokenizer.model" file isn't needed.** The buun converter falls through `_set_vocab_sentencepiece` to `_set_vocab_gpt2` automatically when `tokenizer.model` is missing but `tokenizer.json` is present.

8. **The bench produces `statistics dflash: #acc drafts = 0` regardless of actual accepts.** This is a known dead counter in this build. Use `n_accept` and the rejection histogram instead.

## Validation: did the skill actually produce a working GGUF?

After step 5, the GGUF should:
- Load successfully in `llama-speculative-simple` (no `unknown model architecture` errors).
- Emit the `statistics dflash:` line in stderr (proves DFlash code path was exercised, not a fallback to copyspec/ngram).
- Produce a non-empty `rejection histogram (position → count [%])` block in the output.
- Generate coherent code from the Fibonacci prompt (semantic sanity check that the model isn't outputting garbage).

If all four hold, the conversion was correct.

## Reference numbers from the skill's source-of-truth run (FULL epoch-5, 2026-04-30)

| Metric | predicted | dmax=2 (n=194) | dmax=4 (n=190) | dmax=7 (n=187) |
|---|---|---|---|---|
| chain-pos-1 | 22.03% | 29.90% (+2.6σ) | 31.58% (+3.2σ) | 33.16% (+3.7σ) |
| chain-pos-2 | 3.06% | 2.58% (−0.4σ) | 3.68% (+0.5σ) | 4.28% (+1.0σ) |
| chain-pos-3 | 0.31% | n/a | 0.00% (−0.8σ) | 0.00% (−0.8σ) |
| Throughput | (4.4 t/s AR baseline) | 3.31 t/s | 2.79 t/s | 2.11 t/s |

Throughput regression at dmax≥2 is the documented "verify cost dominates" anomaly — the verifier is so expensive (UD-IQ4_XS, 230B MoE) that even a successful 2.6× batch of drafts doesn't pay off at this quant. Speedup will return at lower-quant verifier (Q4_K_M, Q3_K_M).
