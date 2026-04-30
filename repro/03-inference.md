# Section 3 — Inference (DFlash speculative decoding via llama.cpp)

End-to-end recipe for converting the trained DFlash drafter to GGUF and running speculative-decode benchmarks against `MiniMax-M2.7-FP8` (UD-IQ4_XS quant) using llama.cpp PR #22105.

> **Status as of this commit:** prep-script with proper d2t rebake is done and verified. **Convert script still needs DFlash-specific tensor-mapping patches** before we can produce a clean GGUF. Once that's done, the benchmark + chain-cumulative comparison runs end-to-end.

---

## 3.0 The metric that defines success

Just like §2, this section leads with **chain-cumulative per-position accept rate** (∏ p_i), because that's what runtime measures. Per-position teacher-forced conditional from `val_metrics.json` is reported only as a sanity-check next to the chained values.

For the production drafter (md5 `785c5b5a6bcf8eecb545a1bebb75eb4e`) trained 17 epochs on the 1176-paired set:

| Position | Chained `∏ p_i` | (conditional `p_k`) |
|---|---|---|
| 1 | **28.88%** | 28.88% |
| 2 | **5.83%** | 20.17% |
| 3 | **0.80%** | 13.74% |
| 4 | **0.085%** | 10.56% |
| 5 | **0.0071%** | 8.42% |
| 6 | **0.00052%** | 7.32% |
| 7 | **3.75e-7** | 7.20% |
| **E[acc]** | **0.356** | — |

A working runtime should report `1 - hist[pos 0]` ≈ 28.88% chain-pos-1, `(rounds where pos-0 AND pos-1 accepted) / rounds` ≈ 5.83% chain-pos-2, etc. The **single biggest quality signal** is whether chain-pos-1 matches within sample noise (z<2).

For the FULL run on the 6515-paired-file dataset (still in progress as of this commit, currently at epoch 5/17):

| Position | Chained `∏ p_i` |
|---|---|
| 1 | **22.03%** |
| 2 | **3.06%** |
| 3 | **0.309%** |
| 4 | **0.0264%** |
| 5 | **0.00198%** |
| 6 | **1.38e-6** |
| 7 | **8.80e-8** |
| **E[acc]** | **0.254** |

Chain-pos-1 is on track to exceed production. Chain-pos-2 is currently below production (3.06% vs 5.83%) — expected, since position-2 takes longer to converge in DFlash training.

---

## 3.1 The d2t-zero-row dilution bug — read FIRST

Production drafters trained with **draft-vocab reduction** (`draft_vocab_size=32768 < target_vocab_size=200064`) have a separate `lm_head.weight` of shape `[32768, 3072]` plus a `d2t` offset table mapping draft → target IDs.

PR #22105's `convert_hf_to_gguf.py` does **not** honor `d2t` at sample time. The drafter's argmax must yield target-vocab IDs directly. So the GGUF must carry a target-vocab-shaped `lm_head.weight` `[200064, 3072]` produced by scattering the draft `lm_head` rows into target-indexed positions via `target_id = i + d2t[i]`.

**The naive scatter is wrong.** Setting non-mapped rows to zero produces logits of `0` for those rows regardless of hidden state. When the drafter's confidence drops at later block positions, real-row logits go negative and zero rows can win argmax — producing massively-degraded chain-pos-2+ accept rates.

**Empirical evidence**: a MiniMax-M2.7-FP8 5-layer DFlash drafter trained to `position 1 acc = 28.88%` and `position 2 acc = 20.17%` validated offline at 31.5%/18.4% (matched training within sample noise), but at runtime measured pos-0 marginal = 31.6% (matched ✓) and **pos-1 marginal = 4.2% (5× lower than expected)**. Root cause: zero-row dilution on positions 2+. The "47× apparent gap" investigation in `references/dflash-drafter-offline-validation.md` was driven partly by this real bug being conflated with the metric-mismatch bug.

**The fix:** non-mapped rows = `-65504` (BF16 most-negative finite value). They can never win argmax, drafter sees only the trained 32K rows in the relevant positions.

```python
NEG_FLOOR = -65504.0
expanded = torch.full((target_vocab_size, hidden), NEG_FLOOR, dtype=lm_head.dtype)
indices = torch.arange(draft_vocab_size, dtype=torch.int64) + d2t.to(torch.int64)
expanded[indices] = lm_head
```

`-1e9` works for F16/F32; BF16 saturates `-inf` to a value some kernels mishandle, so prefer the explicit `-65504` even for non-BF16 quants — it's the safe universal value.

---

## 3.2 Pre-processing the safetensors (mandatory)

Run `repro/scripts/inference/prep_for_pr22105_v2_with_d2t_rebake.py` before convert_hf_to_gguf.py:

```bash
python repro/scripts/inference/prep_for_pr22105_v2_with_d2t_rebake.py \
    --in  ${CHECKPOINTS}/full_5L_paired_<TS>/checkpoint_best \
    --out /tmp/dflash-prepped
```

What it does:
1. Reads `config.json` from speculators format (`transformer_layer_config` nested)
2. Flattens it into a Qwen3-style flat config plus a `dflash_config` block at top level
3. Reads `model.safetensors`, **rebakes `lm_head.weight` from `[32768, 3072]` to `[200064, 3072]`** with `-65504` floor for non-mapped rows
4. Drops `d2t` and `t2d` (no longer needed; rebake replaces their function)
5. Writes prepared `config.json` and `model.safetensors`

Reference output:

```
[1/3] wrote config.json (target_vocab=200064)
[2/3] reading model.safetensors  (62 source tensors)
[3/3] rebaking lm_head [draft_V] -> [target_V] with NEG_FLOOR=-65504
      lm_head_draft shape: (32768, 3072)
      d2t shape: (32768,), dtype: torch.int64
      rebaked lm_head shape: (200064, 3072), non-floor rows: 32768 (expected: 32768)
      writing 60 tensors (dropped: ['d2t', 't2d'])
```

---

## 3.3 Convert prepped safetensors to GGUF

> **⚠️ Known issue at this commit:** PR #22105's `convert_hf_to_gguf.py` `DFlashModel.modify_tensors` doesn't yet map `embed_tokens.weight`/`fc.weight`/`hidden_norm.weight` to the runtime's expected names (`token_embd.weight`, `dflash_fc.weight`, `dflash_hidden_norm.weight`). The convert step fails with `Can not map tensor 'model.embed_tokens.weight'`. Patches to `convert_hf_to_gguf.py`'s `DFlashModel.modify_tensors` are required and tracked under §3.7 below.

Once the converter is patched, the conversion command is:

```bash
cd ${WORKSPACE}/repos/llama.cpp-pr22105
source ${WORKSPACE}/venvs/vllm/bin/activate
python convert_hf_to_gguf.py /tmp/dflash-prepped \
    --outtype bf16 \
    --target-model-dir ${MODELS}/MiniMax-M2.7-FP8 \
    --outfile ${MODELS}/MiniMax-M2.7-DFlash.gguf
```

Expected runtime: ~60–90s. Output: ~3.1 GB BF16 GGUF, 60 tensors.

---

## 3.4 Build llama.cpp PR #22105 with required patches

PR #22105 hard-codes a 5-element `target_layer_ids` array and only emits DFlash extraction hooks in qwen3/qwen35/qwen35moe/openai-moe-iswa graph builders. For MiniMax-M2 you need the **dynamic-N + per-arch-hook** patches documented in `references/dflash-gguf-conversion.md`.

Verify all 9 patch markers are present in the source on the host where you'll run the benchmark:

```bash
HOST=spark-N
ssh ${HOST} 'cd ${WORKSPACE}/repos/llama.cpp-pr22105
echo "Edit 1 (std::array<int, 8>):       $(grep -c "std::array<int, 8>" src/llama-hparams.h)"
echo "Edit 1 (dflash_n_target_layer):    $(grep -c "dflash_n_target_layer_ids" src/llama-hparams.h)"
echo "Edit 2 (gguf_get_arr_n):           $(grep -c "gguf_get_arr_n.ml.metadata" src/llama-model.cpp)"
echo "Edit 4 (template inst <int, 8>):   $(grep -c "std::array<int, 8>" src/llama-model-loader.cpp)"
echo "Edit 5 (set_dflash assign begin+): $(grep -c "begin() + dflash_hparams.dflash_n_target_layer_ids" src/llama-context.cpp)"
echo "Edit 6 (encode n_embd dynamic):    $(grep -c "dflash_n_target_layer_ids \* hparams.n_embd" src/llama-context.cpp)"
echo "Edit 7 (dflash.cpp dynamic):       $(grep -c "dflash_n_target_layer_ids" src/models/dflash.cpp)"
echo "Edit 9 (qwen3 dynamic hook):       $(grep -c "dflash_extract_%zu" src/models/qwen3.cpp)"
'
# All counts should be ≥ 1
```

Build:

```bash
ssh ${HOST} 'cd ${WORKSPACE}/repos/llama.cpp-pr22105 && \
  export PATH=/usr/local/cuda/bin:$PATH && \
  cmake --build build --target llama-speculative-simple -j 8'
```

~3 min on a single GB10. Verify the binary contains the patched runtime strings:

```bash
ssh ${HOST} 'strings ${WORKSPACE}/repos/llama.cpp-pr22105/build/bin/libllama.so | grep -E "DFlash extract_layers|dflash_extract_%"'
```

Should show output. If not, the build didn't pick up your source — check the build output for compile errors.

---

## 3.5 Run the benchmark

```bash
TARGET=${MODELS}/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf
DRAFTER=${MODELS}/MiniMax-M2.7-DFlash.gguf

# Code-style prompt (predictable structure → upper-bound chain-pos-1)
PROMPT='Write a Python function that computes the nth Fibonacci number using memoization. Include type hints and docstring. The function should handle the edge cases for n=0 and n=1.'

cd ${WORKSPACE}/repos/llama.cpp-pr22105
./build/bin/llama-speculative-simple \
    -m "$TARGET" \
    -md "$DRAFTER" \
    -p "$PROMPT" \
    -n 300 \
    --draft-max 7 \
    --draft-min 1 \
    --temp 0 \
    -ngl 99 \
    -ngld 99 \
    --dflash \
    -c 4096 \
  2>&1 | tee /tmp/dflash-bench.log
```

Run under `systemd-run --user --unit=dflash-bench --collect bash -c "..."` per `references/sighup-gotcha.md` so it survives SSH disconnects.

Expected to take ~3–5 minutes including verifier load (102 GB UD-IQ4_XS via mmap from NVMe on UMA Spark, throughput ~600 MB/s during page-in).

---

## 3.6 Verify accept rate matches chain-cumulative prediction

The benchmark output ends with:

```
n_drafted = N_d   n_accept = N_a   accept = X.X%
rejection histogram (position → count):
  pos 0:  K0  (P0%)
  pos 1:  K1  (P1%)
  ...
  all ok: KA  (PA%)
```

Convert to **chain-pos-N marginals** (training nomenclature, off-by-one from runtime "pos N"):

```python
n_rounds = sum(K_i for all i) + KA
chain_pos_1 = (sum_K_i_after_pos_0) / n_rounds  # equivalent to 1 - K0/n_rounds
chain_pos_2 = (sum_K_i_after_pos_1 + KA) / n_rounds
chain_pos_k = (rounds where positions 1..k all accepted) / n_rounds
```

Compare against the chained predictions from §3.0. Use a binomial z-score:

```python
from math import sqrt
def z(p_obs, p_pred, n):
    se = sqrt(p_pred * (1 - p_pred) / n)
    return (p_obs - p_pred) / se if se > 0 else float('nan')
```

**Pass criteria:** `|z| < 2` for chain-pos-1 and chain-pos-2 (the only positions with statistical power at typical n_rounds ≈ 50–100). Higher positions need n>500 to distinguish from chain-cumulative predictions.

If chain-pos-1 matches but chain-pos-2 is dramatically below prediction, you have the **d2t zero-row dilution bug** — confirm `lm_head.weight` in your GGUF is `[200064, 3072]` and that `~32768` rows are non-floor (the rest at exactly `-65504`). Re-do §3.2 if not.

If both fail, see `references/dflash-drafter-offline-validation.md` Step 4 for the diagnostic table.

---

## 3.7 Open work / known issues

- **convert_hf_to_gguf.py DFlashModel tensor mapping is incomplete.** It can't map `embed_tokens.weight`, and emits `fc.weight`/`hidden_norm.weight` literally rather than `dflash_fc.weight`/`dflash_hidden_norm.weight`. Three options:
  1. **Patch the converter** (~20 min): override `modify_tensors` in `DFlashModel` to explicitly emit the right names.
  2. **Switch to buun-llama-cpp's converter** — the `dflash-llamacpp-implementation-options` skill flags this fork as architecturally cleaner; its converter handles DFlash tensor names natively.
  3. **Patch the prep script** to emit tensor names that the existing convert script can map (rename `embed_tokens.weight` → `model.embed_tokens.weight` plus the fc/hidden_norm hand-emit names that the GGUF runtime expects).
- **Empirical accept-rate validation pending** — once a clean GGUF builds, run §3.5 and §3.6, embed the chain-pos-1/2 z-scores in this section as the success evidence.
- **Speedup measurement deferred** — speedup is an orthogonal question (depends on verify-cost ratio). The accept-rate validation proves the drafter works; the speedup question runs on top once we have working benchmarks at multiple draft-max values via `speculative-decode-benchmark-sweep`.

---

## See also

- `references/dflash-gguf-conversion.md` — the full 9-patch playbook for PR #22105 + Kimi/non-Qwen targets, AND the d2t zero-row dilution detailed analysis (the source skill for §3.1)
- `references/dflash-drafter-offline-validation.md` — chain-gating metric framing and diagnostic table for accept-rate divergence
- `references/dflash-llamacpp-implementation-options.md` — buun-llama-cpp vs PR #22105 trade-offs, multi-node patched-binary verification recipe
- `repro/scripts/inference/prep_for_pr22105_v2_with_d2t_rebake.py` — the prep script with proper rebake
