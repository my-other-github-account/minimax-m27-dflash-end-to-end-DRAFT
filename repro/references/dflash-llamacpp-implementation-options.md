
# DFlash on llama.cpp — Implementation Options + Reporting Discipline

## When to use

Load this skill at the START of any DFlash-on-llama.cpp investigation. It contains three pieces of decisive operational knowledge that prevent:
- Wasting weeks on the wrong implementation when a better one exists
- Spending 24+ hours chasing a phantom CUDA-dispatch bug because of ambiguous metric reporting
- Running multi-day benchmark sweeps on an unpatched binary across a cluster

## Section 1 — DFlash implementation landscape (as of 2026-04-27)

There are **exactly two** serious DFlash implementations targeting llama.cpp. They have very different architectures.

### Option A — PR #22105 by `ruixiang63` (the "official" upstream-targeted PR)

- URL: https://github.com/ggml-org/llama.cpp/pull/22105
- Branch: `ruixiang63/llama.cpp:dflash`, HEAD ~`d1d2c81ca`
- Status: open DRAFT, not merged into ggml-org/llama.cpp master
- Status as of 2026-04-27: actively reviewed, mergeable=unstable
- **Architecture**: hidden-states extract via `extract_dflash_features` function called after each target decode. Uses `ggml_backend_tensor_get_async` per layer with manual sync.
- **Target arch support**: Qwen3, Qwen3.5, Qwen3.5-MoE, openai-moe-iswa **ONLY**. Does NOT support deepseek2/MLA/Kimi out of the box.
- **Performance** (author's bench, NVIDIA L40S 48GB, Qwen3-8B BF16 target + bf16 draft, thinking off):
  - Quicksort prompt: **8.08× speedup, 93.3% accept**
  - Pythagoras: 2.59× / 20.9% accept
  - Trip plan: 1.49× / 8.9% accept
  - This is the realistic ceiling for DFlash done right — ~90% accept on code-style prompts, lower on conversational
- **Known multi-backend RPC bug**: per-layer-sync race in `extract_dflash_features` produces 0.34% accept (vs ~36% post-fix) on cross-backend deployments. Requires the E9d global-sync-at-top patch documented in `dflash-speculative-debugging` skill. NOT yet upstreamed.

### Option B — `spiritbuun/buun-llama-cpp` (independent, more sophisticated)

- URL: https://github.com/spiritbuun/buun-llama-cpp
- Branch: `master`, default; many `experiment/SD-NNN-*` branches with feature work
- Status: independent fork, 468 stars, actively maintained (commits hourly), upstream of `TheTom/llama-cpp-turboquant`
- **Architecture**: uses `ggml_backend_sched_eval_callback` (`dflash_eval_callback`) installed via `cparams.cb_eval`. Captures hidden states **at the moment each tensor is computed**, on the actual backend that computed it, with the data ready. **No cross-backend race possible** — this is architecturally the correct design.
- Has additional features PR #22105 lacks:
  - GPU tape (`dflash_tape_gpu`) for persistent per-slot GPU buffers
  - Multi-slot DFlash (multiple sequences in flight simultaneously)
  - SWA (sliding-window attention) support for the drafter
  - Cross-attention ring buffer with VRAM cap
  - Hybrid/recurrent target compatibility
  - GPU cross-attention ring buffer optimizations
- **Target arch support**: Qwen3.5/3.6 family + gpt-oss officially. **BUT** the eval-callback approach is architecture-agnostic — buun captures any tensor named `l_out-<layer_id>` via the graph's existing `cb(cur, "l_out", il)` callback, which deepseek2.cpp ALREADY emits. So **deepseek2/Kimi works out of the box** without any port code if the GGUF metadata uses buun's keys (which match PR #22105's: `dflash.target_layer_ids`, `dflash.mask_token_id`, etc.).
- **Entry-point flag**: `--spec-type dflash` (different from PR #22105's `--dflash`)
- Same `convert_hf_to_gguf.py` DFlash output works on both — both fork's converters write the same GGUF metadata keys.

### Option C — there is no Option C

- No Moonshot/z-lab official llama.cpp fork exists
- No other PRs against ggml-org/llama.cpp mention DFlash (only EAGLE3 PR #18039 which is a sibling-not-replacement)
- The 2 forks of `ruixiang63/llama.cpp` (`amirrezasalimi`, `mbednarek360`) are personal experimentation; neither has Kimi support
- HF model cards (`spiritbuun/Qwen3.6-27B-DFlash-GGUF`, `lym00/*`, `mradermacher/*`) only ship draft GGUFs, not implementations

### Which to choose

| Situation | Choice |
|---|---|
| Target is Qwen3 or Qwen3.5 family | Either works; PR #22105 has clearer upstream lineage |
| Target is deepseek2/MLA/Kimi | **buun-llama-cpp** — the eval-callback design avoids the cross-backend sync race; PR #22105 requires hand-porting the deepseek2 hook + the E9d sync fix |
| Multi-node RPC cluster | **buun-llama-cpp** — eval-callback is fundamentally race-free; PR #22105's per-layer-sync pattern is racy and needs the E9d patch |
| Multi-slot serving (many concurrent sequences) | **buun-llama-cpp** — only fork that supports it |
| Need SWA-attention drafter (e.g. Qwen3.6) | **buun-llama-cpp** — required |
| You already have a working PR #22105 setup | Stay; don't reflavor mid-investigation |
| Starting from scratch on any non-Qwen target | **buun-llama-cpp** — fewer porting headaches |

### Migration notes (PR #22105 → buun-llama-cpp)

- GGUF files convert identically (same metadata keys); existing `Kimi-K2.5-DFlash.gguf` draft loads in both
- buun build: `cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_RPC=ON` (~5 sec configure, ~10-20 min build)
- Flag rename: `--dflash` → `--spec-type dflash`
- `set_dflash_capture(layer_ids, n_layers)` is the equivalent of PR #22105's `extract_layer_indices` — buun installs the eval callback automatically
- The base graph's existing `cb(cur, "l_out", il)` is sufficient for capture; do NOT add architecture-specific `dflash_extract_N` hooks
- For deepseek2 on buun: literally no source changes required if the layer-output `l_out` callback already exists (verify with `grep 'cb(cur, "l_out"' src/models/<arch>.cpp`)

## Section 2 — WHOLE-MODEL vs SINGLE-TENSOR-SURGERY reporting rule

When reporting accept rate, decode speed, or any DFlash result, **ALWAYS explicitly label**:

- **WHOLE-MODEL**: every tensor in the GGUF is at the named precision (e.g. "WHOLE-MODEL Q4_K_S = 53.4% accept" means all tensors are Q4_K_S)
- **SINGLE-TENSOR-SURGERY**: only one named tensor was modified, the rest of the model stays at a different (named) precision (e.g. "SINGLE-TENSOR `token_embd` Q3_K→F16 on Q3_K_S = 55.1% accept" means just that one tensor was promoted)

Saying "Q4 result: 35%" without WHOLE-MODEL or SINGLE-TENSOR is **forbidden** — it is **manipulative ambiguity** that misleads the user about what was actually measured. The two regimes have completely different implications:
- SINGLE-TENSOR results test dispatch path / tensor-specific quant impact for that ONE tensor (and tell you about the get_rows kernel dispatch cliff between F-types and k-quants)
- WHOLE-MODEL results test model-wide quant scheme impact (and tell you about the actual production-deployment ceiling)

**Real-world cost of conflation**: in the Kimi-K2.5 DFlash investigation (2026-04-26), the agent reported four "Q4 results" (Q4_0_EMBD=35.1%, Q4_K_EMBD=31.0%, IQ4_XS=51.5%, F16_token_embd=55.1%) without consistently labeling SINGLE-TENSOR vs WHOLE-MODEL. The user assumed all four were WHOLE-MODEL and concluded "Q4 doesn't help." When clarified that 3 of 4 were SINGLE-TENSOR-SURGERY and the user had been asking about WHOLE-MODEL the whole time, the trust hit was severe and the user labeled it manipulation. **Subsequent WHOLE-MODEL run produced 53.4% — the actual answer.** ~24 hours of investigation effort + significant trust damage.

### Mandatory format for any DFlash result

```
Variant: WHOLE-MODEL <quant_name>             <-- pick one
        SINGLE-TENSOR-SURGERY <tensor>         <-- pick one
        on <base_quant>                        <-- only for surgery
Accept rate: NN.NN%
Probe log: <path>
```

Refuse to use any unlabeled metric for reasoning until re-verified. If you see "Q4_K_S = X%" in a tick summary or state JSON, look up which variant actually ran and re-label before continuing.

### Non-negotiable rules to bake into any DFlash investigation loop

1. Every quant statement says WHOLE-MODEL or SINGLE-TENSOR-SURGERY explicitly. No exceptions.
2. Never use "too big to fit", "exceeds UMA", or "INDETERMINATE due to memory" as stopping reasons. mmap + `--cpu-moe` always available; see `unified-memory-oversize-models` skill.
3. Never declare a "ceiling" without comparing to a measured WHOLE-MODEL baseline. Single-tensor surgery results cannot establish a ceiling.

## Section 3 — Multi-node patched-binary verification

Multi-node llama.cpp clusters have a footgun: **the binary on each node is independently built**. When you patch source on one node and run from another node, you may run an UNPATCHED binary without realizing it.

### How this happens

- Source patches are typically authored on the head node (where you ssh and edit)
- Workers run `rpc-server` from THEIR locally-built binary (not the head's)
- If cluster topology flips (head changes because the model lives on a different node), the new head runs ITS binary, which may be a stale clone
- `rsync src/` to workers does NOT rebuild on workers — you must explicitly rebuild on each affected node

### Real-world example (Kimi-K2.5, 2026-04-26)

Cluster topology flipped from head=node4 to head=node2 because the new whole-model Q4 GGUFs were too big for node4's disk. Spark-2's `~/llama.cpp-dflash/` was a stock clone of PR #22105 from April 22 — never had the E9d sync barrier patch, never had the deepseek2 hook port, never had the array-size-5→8 patches. Three full whole-model 4-bit DFlash probes (IQ4_XS, Q4_0, Q4_K_S, ~16 minutes each plus ~3-4 hours each download + cold-load) ran on this UNPATCHED binary before anyone checked. Results were still "interesting" (51.5%, 42.86%, 53.4%) but they were not measuring what we thought they were measuring.

### Verification recipe (run BEFORE any DFlash benchmark on a multi-node cluster)

```bash
# On the head node where the model + binary will run:
HEAD_NODE=node2  # whichever node has the model

# 1. Verify binary mtime is newer than source patches
ssh $HEAD_NODE 'stat -c "%y %n" ~/llama.cpp-dflash/build/bin/llama-speculative-simple ~/llama.cpp-dflash/src/llama-context.cpp'

# 2. Verify patches are actually in the source on that node
ssh $HEAD_NODE 'grep -c "E9d" ~/llama.cpp-dflash/src/llama-context.cpp'    # should be ≥1
ssh $HEAD_NODE 'grep -c "D5_FEATURE_DUMP" ~/llama.cpp-dflash/src/llama-context.cpp'  # if instrumentation needed

# 3. Verify the COMPILED binary contains the patch's runtime artifacts (string literals from log lines, file paths, etc.)
ssh $HEAD_NODE 'strings ~/llama.cpp-dflash/build/bin/libllama.so | grep -E "features_llamacpp|D5_FEATURE_DUMP|D10_DECODE_DUMP"'
# Expected output: actual hits showing the patched runtime strings are present
```

If ANY check fails on the head node, rebuild before benchmarking:

```bash
# Rsync ALL relevant source dirs from canonical-patched node (e.g. node4) to head
rsync -av --delete ${WORKSPACE}/llama.cpp-dflash/src/ HEAD:${WORKSPACE}/llama.cpp-dflash/src/
rsync -av --delete ${WORKSPACE}/llama.cpp-dflash/common/ HEAD:${WORKSPACE}/llama.cpp-dflash/common/
rsync -av --delete ${WORKSPACE}/llama.cpp-dflash/examples/speculative-simple/ HEAD:${WORKSPACE}/llama.cpp-dflash/examples/speculative-simple/

# Rebuild on head
ssh HEAD 'cd ~/llama.cpp-dflash && export PATH=/usr/local/cuda/bin:$PATH && cmake --build build -j --target llama-speculative-simple'
# ~40 sec for incremental rebuild touching only llama-context.cpp.o + libllama.so + final exe

# IMPORTANT: also rebuild rpc-server on workers if patches affect graph build (they usually do)
# Workers' rpc-servers carry the OLD binary mmap'd; a mid-investigation rsync to workers does NOT hot-swap.
# pkill rpc-server on each worker, restart with the new binary.
```

### Don't trust "the binary built on date X" — verify what's IN it

`stat -c "%y"` shows mtime, not patch state. A stale binary may have a recent mtime if someone touched it. Always `strings` for patch markers, or run a probe with patch-specific log lines and confirm those lines actually appear.

### When changing cluster topology mid-investigation

If you flip head=A → head=B (e.g. because B has more disk for a new model variant):
1. Verify B's source matches A's source (rsync + diff)
2. Rebuild on B
3. Restart rpc-servers on workers (they need the new binary too if patches affect graph build)
4. Verify with the recipe above
5. THEN run probe

Skipping any of these = your benchmark is measuring something different than you think.

## Common entry-point pattern

When user asks "is there a better way to do DFlash on llama.cpp" or "let's start fresh":
1. Default answer: yes, **buun-llama-cpp** is the better starting point as of 2026-04-27 (sched-eval-callback, multi-slot, deepseek2 likely works without port)
2. Build cost: ~15-20 min on DGX Spark
3. Test with existing GGUF metadata: should work because both forks use identical `dflash.target_layer_ids` etc.
4. Migration is one-time and decisive — once on buun, the cross-backend race is structurally eliminated
