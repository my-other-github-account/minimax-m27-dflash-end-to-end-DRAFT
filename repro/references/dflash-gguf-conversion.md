
# DFlash → GGUF Conversion for llama.cpp PR #22105

Use when: you have z-lab/Kimi-K2.5-DFlash (or any `DFlashDraftModel`/EAGLE-3 drafter that needs a target tokenizer) and need a GGUF for `llama-speculative-simple --dflash`. This is the exact working recipe — do NOT skip steps; several are non-obvious and cost >30 min to rediscover.

## Prerequisites (verify first)

1. PR #22105 branch (`ruixiang63:dflash`) fetched into a worktree (e.g. `~/llama.cpp-dflash`). HEAD should be `d1d2c81c` or similar with "dflash: add support for qwen3.5/3.6 moe models".
2. Safetensors of `z-lab/Kimi-K2.5-DFlash` fully downloaded to `~/models/Kimi-K2.5-DFlash/` (6.6 GiB, 2 shards + config.json + dflash.py).
3. Target tokenizer metadata from `moonshotai/Kimi-K2.5` in `~/models/Kimi-K2.5-target-meta/`:
   - `tiktoken.model` (2.7 MiB — required)
   - `tokenization_kimi.py` (custom tokenizer class)
   - `tool_declaration_ts.py` ← **often missed! `tokenization_kimi.py` imports from it.** Fetch with:
     `curl -sL -o tool_declaration_ts.py https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/tool_declaration_ts.py`
   - `config.json`, `configuration_deepseek.py`, `configuration_kimi_k25.py`, `tokenizer_config.json`, `chat_template.jinja`

## Step 1 — Python venv with exact versions

```bash
python3 -m venv ~/venv-llamacpp
source ~/venv-llamacpp/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install numpy safetensors sentencepiece gguf "transformers<5" protobuf tiktoken blobfile
```

**CRITICAL: `transformers<5`** — transformers 5.x removed `bytes_to_unicode` from `transformers.models.gpt2.tokenization_gpt2`. The kimi-k2 tiktoken branch in `convert_hf_to_gguf.py` imports it and will crash. 4.57.6 works.

## Step 2 — Fix HF cache permissions

If previous runs (especially any with sudo) polluted the cache:
```bash
sudo chown -R $USER:$USER ~/.cache/huggingface/modules
```
Symptom of this bug: `PermissionError: [Errno 13] Permission denied: '/home/.../.cache/huggingface/modules/transformers_modules/...'`

## Step 3 — Patch `convert_hf_to_gguf.py`

Two orthogonal patches are both required.

### Patch A: `trust_remote_code=True` on every AutoTokenizer call

Kimi's tokenizer is loaded via custom code (`tokenization_kimi.py`). The PR's convert script has bare `AutoTokenizer.from_pretrained(self.dir_model)` calls in multiple places (Qwen3 path, etc.). Add `trust_remote_code=True` to all:

```bash
cd ~/llama.cpp-dflash
sed -i 's|AutoTokenizer.from_pretrained(self.dir_model)|AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)|g' convert_hf_to_gguf.py
sed -i 's|AutoTokenizer.from_pretrained(dir_model)|AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)|g' convert_hf_to_gguf.py
```

Verify ~7 sites changed with `git diff`.

### Patch B: Override `DFlashModel.set_vocab` with kimi-k2 tiktoken fallback

Without this, conversion dies at `AttributeError: TikTokenTokenizer has no attribute vocab` because Qwen3Model's set_vocab chain eventually calls `_set_vocab_gpt2` → `get_vocab_base` which assumes BPE `tokenizer.vocab`. Tiktoken doesn't have that.

The fix: wrap `super().set_vocab()` in try/except, and on failure run the kimi-k2 tiktoken path directly (copied from `DeepseekV2Model.set_vocab` which already has this fallback).

See `scripts/patch_dflash_set_vocab.py` in this skill for the exact patch. Run it:
```bash
python3 patch_dflash_set_vocab.py  # edits convert_hf_to_gguf.py in place
```

## Step 4 — Run conversion

```bash
cd ~/llama.cpp-dflash
source ~/venv-llamacpp/bin/activate
python convert_hf_to_gguf.py \
  ~/models/Kimi-K2.5-DFlash \
  --outtype bf16 \
  --target-model-dir ~/models/Kimi-K2.5-target-meta \
  --outfile ~/models/Kimi-K2.5-DFlash.gguf \
  --verbose
```

Expected runtime: ~60–90s. Output should be ~6.9 GiB, 69 tensors, BF16.

## Verification

Success log lines to look for:
- `INFO:hf-to-gguf:DFLASH: Using tokenizer from target model:`
- `INFO:hf-to-gguf:DFLASH: default set_vocab failed (...); trying kimi-k2 tiktoken path` (expected on first try)
- `DEBUG:hf-to-gguf:chkhsh: 81212dc7cdb7e0c1074ca62c5aeab0d43c9f52b8a737be7b12a777c953027890` (kimi-k2 hash)
- `DEBUG:hf-to-gguf:tokenizer.ggml.pre: 'kimi-k2'`
- `INFO:gguf.gguf_writer:... n_tensors = 69, total_size = 7.0G`
- `INFO:hf-to-gguf:Model successfully exported to ...`

## Pitfalls

- **Don't forget `tool_declaration_ts.py`** — failure mode: `FileNotFoundError: ... tool_declaration_ts.py`. This is a sibling module imported by tokenization_kimi.py.
- **Don't use `hf download` for the target-meta files** — xet backend is slow (3 MiB/s). Use `curl -L` directly.
- **Don't skip the `trust_remote_code` patch** even if step B looks sufficient — the initial code path into `_set_vocab_gpt2` runs first and needs it.
- **Don't use a conda env with preinstalled transformers 5.x** — you'll chase cryptic import errors. Start fresh with a pinned venv.
- **HF cache collisions**: each run creates `~/.cache/huggingface/modules/transformers_modules/<model-dir-slug>/`. If you change the tokenizer files, `rm -rf` that directory before re-running.

## Runtime load fix — PR #22105 hardcodes 5 target_layer_ids (Kimi has 6)

**This bites after successful conversion** — the GGUF is valid, but `llama-speculative-simple --dflash` crashes at draft-model load with:

```
llama_model_load: error loading model hyperparameters: key dflash.target_layer_ids has wrong array length; expected 5, got 6
failed to load draft model, '.../Kimi-K2.5-DFlash.gguf'
```

**Why**: PR #22105 was written for Qwen3.5/3.6 MoE which has 5 target layers. It hardcodes `std::array<int, 5>` and passes `n=5` to `get_key_or_arr()`. The loader strictly checks `n == arr_info.length` and throws. Kimi-K2.5-DFlash has **6** target_layer_ids `[1,12,24,35,47,58]` (from `dflash_config.target_layer_ids` in `~/models/Kimi-K2.5-DFlash/config.json`, `block_count=6`).

**Fix** — 4 edits in the PR worktree (e.g. `~/llama.cpp-dflash`), then rebuild:

### Edit 1 — `src/llama-hparams.h`

Find the DFlash draft model block. Enlarge the array and add a count field:

```cpp
    // DFlash draft model
    std::array<int, 8> dflash_target_layer_ids = {};    // was: std::array<int, 5>
    uint32_t dflash_n_target_layer_ids = 0;              // NEW
    uint32_t dflash_block_size     = 16;
    uint32_t dflash_mask_token_id  = 0;
```

### Edit 2 — `src/llama-model.cpp`, `LLM_ARCH_DFLASH` case (~line 2385)

Replace the hardcoded `n=5` call with a dynamic read of the GGUF array length:

```cpp
// REPLACE the get_key_or_arr(..., 5, false) + 5-element log block with:
{
    // dynamically determine target_layer_ids array length
    // (supports 5 for qwen, 6 for kimi, etc.)
    const std::string key_tlid = "dflash.target_layer_ids";
    const int kid_tlid = gguf_find_key(ml.metadata, key_tlid.c_str());
    if (kid_tlid < 0) {
        throw std::runtime_error("DFlash model requires 'target_layer_ids' in GGUF metadata");
    }
    const uint32_t n_tlid = (uint32_t) gguf_get_arr_n(ml.metadata, kid_tlid);
    if (n_tlid == 0 || n_tlid > hparams.dflash_target_layer_ids.size()) {
        throw std::runtime_error(format("DFlash target_layer_ids length %u is out of range (1..%u)",
                                        n_tlid, (uint32_t) hparams.dflash_target_layer_ids.size()));
    }
    hparams.dflash_n_target_layer_ids = n_tlid;
    if (!ml.get_key_or_arr(LLM_KV_DFLASH_TARGET_LAYER_IDS, hparams.dflash_target_layer_ids, n_tlid, false)) {
        throw std::runtime_error("DFlash model requires 'target_layer_ids' in GGUF metadata");
    }
    std::string ids_str;
    for (uint32_t i = 0; i < n_tlid; ++i) {
        if (i) ids_str += ", ";
        ids_str += std::to_string(hparams.dflash_target_layer_ids[i]);
    }
    LLAMA_LOG_INFO("%s: DFlash extract_layers (%u) = [%s]\n", __func__, n_tlid, ids_str.c_str());
}
```

**Pitfalls in Edit 2**:
- The field on `llama_model_loader` is `ml.metadata` (bare `gguf_context *`), NOT `ml.meta.get()`. A `std::unique_ptr` wrapper exists elsewhere but the public field used throughout `llama-model.cpp` is `metadata`.
- Don't try `llm_kv(LLM_KV_DFLASH_TARGET_LAYER_IDS)` as a constructor — it's not one in this codebase and won't compile. Just hardcode the string `"dflash.target_layer_ids"` (matches `LLM_KV_NAMES` in `llama-arch.cpp`).

### Edit 3 — `src/llama-model.cpp`, tensor-creation site (~line 7006)

Swap `.size()` for the runtime count:

```cpp
// OLD: const int64_t n_target_layer_ids = (int64_t)hparams.dflash_target_layer_ids.size();
const int64_t n_target_layer_ids = (int64_t)hparams.dflash_n_target_layer_ids;
```

Critical — without this, `.size()` returns 8 (the array's max capacity) instead of the actual 6, and tensor shapes will be wrong.

### Edit 4 — `src/llama-model-loader.cpp` (~line 504)

Update the explicit template instantiation to match the new array size:

```cpp
// OLD: template bool llama_model_loader::get_key_or_arr<std::array<int, 5>>(...)
template bool llama_model_loader::get_key_or_arr<std::array<int, 8>>(enum llm_kv kid, std::array<int, 8> & result, uint32_t n, bool required); // store DFlash up to 8 layer ids
```

Without this, the linker fails with undefined reference to the new instantiation.

### Rebuild + deploy to RPC workers

```bash
cd ~/llama.cpp-dflash
export PATH=/usr/local/cuda/bin:$PATH
cmake --build build --target llama-speculative-simple rpc-server -j 8
```

~3 min on GB10. Then rsync `build/bin/` (the binaries AND the `libllama.so` / `libggml*.so` shared libs — all live side-by-side in `bin/`) to every RPC worker node over QSFP:

```bash
for ip in <NODE1_QSFP_IP> <NODE2_QSFP_IP> <NODE3_QSFP_IP>; do
  rsync -a ~/llama.cpp-dflash/build/bin/ $ip:${WORKSPACE}/llama.cpp-dflash/build/bin/
done
```

Then kill the old rpc-servers on the workers and relaunch them with the patched build — otherwise their `libggml-rpc.so` version may mismatch and RPC handshake silently breaks.

### Verification (Edit 1–4 only — more fixes below!)

In `/tmp/dflash_run.log` after Edit 1–4 you should see:

```
load_hparams: DFlash extract_layers (6) = [1, 12, 24, 35, 47, 58]
load_hparams: DFlash block_size = 8, mask_token_id = 163838
```

Models will load, compute buffers will reserve, warmup will fire. But then inference will crash — you're not done yet. Keep reading.

## Runtime crash #2 — `std::array<int,8>.size()` still lies at 4 more sites

**Symptom**: after Edits 1–4, target + draft both load (`set_dflash: DFlash extraction enabled for layers [2, 13, 25, 36, 48]` — note only **5** ids logged!), warmup completes, then first decode crashes with:

```
GGML_ASSERT(tensor != nullptr && "DFlash extraction tensor is null") failed
```

**Why**: the enlarged `std::array<int, 8>` has 8 slots; only `dflash_n_target_layer_ids` (=6) are meaningful, the rest are zero-padded. But four more PR #22105 sites iterate via `.begin()/.end()` or `.size()`:

- `src/llama-context.cpp` `set_dflash()` — `assign(begin(), end())` pulls in the 2 zero-padded slots.
- `src/llama-context.cpp` `set_dflash()` — the hardcoded printf prints only 5 ids (masking the bug in the log).
- `src/llama-context.cpp` `encode()` — `n_embd = dflash_target_layer_ids.size() * n_embd` multiplies by 8 not 6.
- `src/models/dflash.cpp` `build_inp_embd` — same `.size()` mistake for `n_embd_target_features`.

**Edit 5 — `src/llama-context.cpp` `set_dflash()`**:

```cpp
// OLD:
// dflash.extract_layer_indices.assign(
//         dflash_hparams.dflash_target_layer_ids.begin(),
//         dflash_hparams.dflash_target_layer_ids.end());
// Replace with:
dflash.extract_layer_indices.assign(
        dflash_hparams.dflash_target_layer_ids.begin(),
        dflash_hparams.dflash_target_layer_ids.begin() + dflash_hparams.dflash_n_target_layer_ids
        );
```

Also replace the hardcoded 5-arg `LLAMA_LOG_INFO("... [%d, %d, %d, %d, %d]", ...)` with a dynamic loop so the log prints all N:

```cpp
{
    std::string ids_str;
    for (size_t i = 0; i < dflash.extract_layer_indices.size(); ++i) {
        if (i > 0) ids_str += ", ";
        ids_str += std::to_string(dflash.extract_layer_indices[i]);
    }
    LLAMA_LOG_INFO("%s: DFlash extraction enabled for layers [%s]\n", __func__, ids_str.c_str());
}
```

**Edit 6 — `src/llama-context.cpp` `encode()` n_embd calc**:

```cpp
// Find the line that multiplies n_embd by the layer count (inside encode() when dflash_extract_enabled):
// OLD: n_embd = (int64_t) hparams.dflash_target_layer_ids.size() * hparams.n_embd;
n_embd = (int64_t) hparams.dflash_n_target_layer_ids * hparams.n_embd;
```

**Edit 7 — `src/models/dflash.cpp` `build_inp_embd`** (top of file):

```cpp
// OLD: const int64_t n_target_layer_ids = (int64_t) hparams.dflash_target_layer_ids.size();
const int64_t n_target_layer_ids = (int64_t) hparams.dflash_n_target_layer_ids;
const int64_t n_embd_target_features = n_target_layer_ids * n_embd;
```

Rebuild + rsync + restart RPC workers again.

## Runtime crash #3 — deepseek2 (Kimi) has NO graph hook in PR #22105

**Symptom**: after Edits 5–7, `set_dflash` now correctly prints `DFlash extraction enabled for layers [2, 13, 25, 36, 48, 59]` (all 6 ids). But inference still crashes with the same `GGML_ASSERT(tensor != nullptr && "DFlash extraction tensor is null")` in `llama_context::extract_dflash_features` at `llama-context.cpp:2498`.

**Why**: PR #22105 only emits the `dflash_extract_N` tensor-name callback inside the graph builders for **qwen3 / qwen35 / qwen35moe / openai-moe-iswa**. Kimi-K2.5 is **deepseek2** architecture — there is **no hook in `src/models/deepseek2.cpp`** at all. The graph callback never fires for any layer, so `dflash.extract_tensors[0..N-1]` stay nullptr, and the first decode asserts.

Additionally, the existing hooks in the 4 qwen/openai files hardcode a 5-entry `dflash_extract_names[]` array and cap the loop at `i < 5` — so even if you pointed Kimi at a qwen graph builder it would silently drop the 6th layer.

**Edit 8 — add DFlash extraction hook to `src/models/deepseek2.cpp`**

Right after `ggml_tensor * inpSA = inpL;` inside the main `for (int il = 0; il < effective_n_layers; ++il)` loop:

```cpp
for (int il = 0; il < effective_n_layers; ++il) {
    ggml_tensor * inpSA = inpL;

    // DFlash: Extract intermediate layer features from target model
    if (dflash && cparams.dflash_extract_enabled && !dflash->extract_layer_indices.empty()) {
        for (size_t i = 0; i < dflash->extract_layer_indices.size(); ++i) {
            if (dflash->extract_layer_indices[i] == il) {
                char nm[32];
                snprintf(nm, sizeof(nm), "dflash_extract_%zu", i);
                cb(inpL, nm, il);
                break;
            }
        }
    }

    // norm
    cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
    ...
```

**Edit 9 — make qwen3/qwen35/qwen35moe/openai-moe-iswa hooks dynamic**

In each of those 4 files, replace:

```cpp
// OLD:
if (dflash && cparams.dflash_extract_enabled && !dflash->extract_layer_indices.empty()) {
    static const char * dflash_extract_names[] = {
        "dflash_extract_0", "dflash_extract_1", "dflash_extract_2",
        "dflash_extract_3", "dflash_extract_4"
    };
    for (size_t i = 0; i < dflash->extract_layer_indices.size() && i < 5; ++i) {
        if (dflash->extract_layer_indices[i] == il) {
            cb(inpL, dflash_extract_names[i], il);
            break;
        }
    }
}
```

with the same dynamic `snprintf` pattern as deepseek2. A quick python script (see `scripts/patch_dflash_graph_hooks.py`) does all 5 edits in one shot.

Rebuild + rsync + restart RPC workers a THIRD time.

### Final verification

After Edits 1–9, a successful run logs:

```
load_hparams: DFlash extract_layers (6) = [1, 12, 24, 35, 47, 58]
set_dflash: DFlash extraction enabled for layers [2, 13, 25, 36, 48, 59]
main: DFlash chat template applied
sched_reserve: ...
<generation output>
eval time = ... ms / N tokens (... tok/s)
```

Then the run proceeds to KV cache alloc, compute buffer reserve, warmup, and finally `generated 256 tokens` with the decode tok/s line.

## Summary: PR #22105 needs 9 patches to work with any non-qwen target

If someone merges the PR as-is, **it will only work for qwen3/qwen3.5/qwen3.6-moe with exactly 5 target layers**. For Kimi-K2.5 / deepseek2 / any model with ≠5 target layers, all 9 edits are mandatory. These should probably be upstreamed as a follow-up PR — the `std::array<int, 5>` + hardcoded qwen-only graph hooks are the two root anti-patterns.

## Pitfall: d2t rebake into target-vocab lm_head (zero-row dilution)

When converting drafters with **draft-vocab reduction** (`draft_vocab_size < target_vocab_size`, e.g. 32768 < 200064 for MiniMax-M2), the safetensors `lm_head.weight` has shape `[draft_vocab_size, hidden]` plus a `d2t` tensor mapping draft indices → target token IDs:

```python
d2t[i] + i = target_token_id  # speculators stores d2t as offset, not absolute
```

For runtimes with **no native d2t support** (buun-llama-cpp, llama.cpp upstream — neither honors `d2t`/`t2d` at sample time), the GGUF must carry a target-vocab-shaped `lm_head` so argmax yields target-vocab IDs directly. The natural scatter is:

```python
expanded = torch.zeros(target_vocab_size, hidden, dtype=original.dtype)
for i in range(draft_vocab_size):
    expanded[d2t[i] + i] = original[i]
```

**This is wrong.** The non-mapped rows are zero vectors → their inner product with any hidden state is exactly 0 → their logits are constant 0 regardless of the drafter's prediction. When the drafter's confidence is low (typical at block positions ≥2), the real-row logits can go negative, and a zero row can win argmax.

**Empirical evidence**: a MiniMax-M2.7-FP8 5-layer DFlash drafter trained to `position 1 acc = 28.88%` and `position 2 acc = 20.17%` measured offline at 31.5%/18.4% (validating training) but at runtime measured pos-0 marginal = 31.6% (matched ✓) and **pos-1 marginal = 4.2% (5x lower than expected)**. Root cause: zero-row dilution on positions 2+.

**Fix — set non-mapped rows to a very-negative finite value:**
```python
NEG = -65504.0  # bf16 most-negative; survives any common quant
expanded = torch.full((target_vocab_size, hidden), NEG, dtype=original.dtype)
for i in range(draft_vocab_size):
    expanded[d2t[i] + i] = original[i]
```

`-1e9` works for F32/F16; BF16 saturates to `-inf` which some kernels mishandle, so prefer `-65504` for safety. `-inf` literal is tempting but propagates to NaN in some softmax/argmax paths.

**Verification**: re-run the offline validation harness (see `dflash-drafter-offline-validation` skill) after fixing — runtime pos-1 marginal should now match training pos-2 acc within ±3pp.

## Reuse checklist

Before running conversion, confirm with `ls`:
- [ ] `~/models/Kimi-K2.5-DFlash/model-00001-of-00002.safetensors` (4.5 GiB)
- [ ] `~/models/Kimi-K2.5-DFlash/model-00002-of-00002.safetensors` (2.1 GiB)
- [ ] `~/models/Kimi-K2.5-DFlash/config.json` with `"architectures": ["DFlashDraftModel"]`
- [ ] `~/models/Kimi-K2.5-target-meta/tiktoken.model`
- [ ] `~/models/Kimi-K2.5-target-meta/tokenization_kimi.py`
- [ ] `~/models/Kimi-K2.5-target-meta/tool_declaration_ts.py`
- [ ] `~/venv-llamacpp/bin/python -c 'import transformers; print(transformers.__version__)'` shows 4.x
- [ ] `grep -c 'trust_remote_code=True' ~/llama.cpp-dflash/convert_hf_to_gguf.py` shows 7+ occurrences on the AutoTokenizer lines
- [ ] `grep -A2 'class DFlashModel' ~/llama.cpp-dflash/convert_hf_to_gguf.py` — check `set_vocab` has try/except fallback

Before running `llama-speculative-simple --dflash`, confirm the runtime patches are in:
- [ ] `grep 'std::array<int, 8>' ~/llama.cpp-dflash/src/llama-hparams.h` returns a hit (Edit 1)
- [ ] `grep 'dflash_n_target_layer_ids' ~/llama.cpp-dflash/src/llama-hparams.h` returns a hit (Edit 1)
- [ ] `grep 'gguf_get_arr_n(ml.metadata' ~/llama.cpp-dflash/src/llama-model.cpp` returns a hit (Edit 2, dynamic length read)
- [ ] `grep 'std::array<int, 8>' ~/llama.cpp-dflash/src/llama-model-loader.cpp` returns a hit (Edit 4, template instantiation)
- [ ] `grep 'begin() + dflash_hparams.dflash_n_target_layer_ids' ~/llama.cpp-dflash/src/llama-context.cpp` returns a hit (Edit 5, set_dflash assign)
- [ ] `grep 'dflash_n_target_layer_ids \* hparams.n_embd' ~/llama.cpp-dflash/src/llama-context.cpp` returns a hit (Edit 6, encode n_embd)
- [ ] `grep 'dflash_n_target_layer_ids' ~/llama.cpp-dflash/src/models/dflash.cpp` returns a hit (Edit 7)
- [ ] `grep 'dflash_extract_%zu' ~/llama.cpp-dflash/src/models/deepseek2.cpp` returns a hit (Edit 8, deepseek2 graph hook — ONLY required if target is deepseek2 arch like Kimi-K2/K2.5)
- [ ] `grep -c 'dflash_extract_%zu' ~/llama.cpp-dflash/src/models/qwen3.cpp ~/llama.cpp-dflash/src/models/qwen35.cpp ~/llama.cpp-dflash/src/models/qwen35moe.cpp ~/llama.cpp-dflash/src/models/openai-moe-iswa.cpp` each return 1 (Edit 9, dynamic N support in existing hooks)
- [ ] Binaries rebuilt AND rsynced to every RPC worker
- [ ] RPC servers on workers were restarted after the rsync (old libggml-rpc.so must not still be loaded)
