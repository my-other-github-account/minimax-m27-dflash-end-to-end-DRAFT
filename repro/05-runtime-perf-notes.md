# Section 5 — Runtime perf notes: how the v11.1 wall-clock gate was cleared

End-to-end notes on the changes that took DFlash speculative decoding through the
`llama-server` + IQ4_XS verifier path from a sub-1.0× wall-clock regression to a
real-DFlash median speedup of **1.019×** on Project Gutenberg traffic. This is
the perf companion to §4 — §4 documents *what* was measured, this section
documents *what had to change in source* and *why each change mattered*.

> **Status:** ✅ Gate cleared 2026-05-08T02:31:17-07:00 on spark-4. Verifier
> `MiniMax-M2.7-FP8` (UD-IQ4_XS), drafter
> `MiniMax-M2.7-DFlash-v11-step15080-legacy-targethead.gguf`, llama.cpp-dflash
> commit `d1d2c81caccc748eaaff32b6b7823bad090fd1dd`. Full §4.10 receipt
> appended to `repro/04-empirical-tau-llama-benchy.md`.

---

## 5.0 Headline

| | with-spec tg t/s | no-spec tg t/s | speedup |
|---|---:|---:|---:|
| pp=256, depth=0, tg=128 | 4.389 | 4.066 | **1.079×** |
| pp=1024, depth=0, tg=128 | 3.893 | 4.061 | 0.959× |
| **median** | | | **1.019×** |

- empirical_tau = **1.369** (in required `[1.3, 2.0]` band — drafter genuinely fired)
- gate JSONL: 2 records, predicted_n = 256, draft_n = 184, draft_n_accepted = 69
- `REAL_DFLASH_MEASUREMENT_OK` (predicted_n ≥ 64/req, sum(draft_n) > 0, tau in band)

The gate is **knife-edge** — one cell still loses, the median is dragged across
1.0× by the smaller-context cell. Everything below is what it took to get there
from the previous v10 / v11 attempts that were sitting at medians of 0.518–0.836×.

---

## 5.1 The path to the gate

For full empirical-tau and wall-clock context across versions see §4. Summarized
here so the perf changes have causal grounding:

| variant | empirical_tau | median speedup | verdict |
|---|---:|---:|---|
| v9 Q4_K_M / Q8_0 drafter quant | tau OK | < 1.0× | quant didn't move serving path |
| v10 production-like `--draft-max 1` | ~1.5–1.6 | **0.836×** | best non-tuned baseline |
| v10 tail-window (tail-64) | ~1.09 | 0.680× | tail collapses tau |
| v10 tail-window (tail-128) | ~1.09 | 0.672× | same |
| v11 dm7 anchor variant | 1.332 | 0.558× | drafter fires, lost on speed |
| v11 dm7 anchor+tail variant | 1.197 | 0.518× | tau under-band + speed fail |
| v11 dm7 earlier attempt | OK | 0.605× | speed fail |
| **v11.1 (this section)** | **1.369** | **1.019×** | **gate pass** |

So **drafter quality was already there at v10/v11**; the missing piece was
serving-path overhead per draft step. Every v11.1 fix below is about reducing
or amortizing per-step overhead so the empirical tau actually translates into
wall-clock time.

---

## 5.2 The seven fixes that mattered, ordered by impact

### (1) Bucketed DFlash cross-context graph reuse + dirty-append upload

**What:** the DFlash decoder previously rebuilt its compute graph on every
draft step and re-uploaded the entire target-context tensor every time. With
`DFLASH_DECODER_CONTEXT_BUCKET=8`, the decoder graph is built once per bucket
of 8 context lengths and reused; only the *changed* slice of target context
(the dirty append) is uploaded.

**Why it mattered:** this is the single biggest contributor. Per-step graph
compilation and full-context re-upload were the dominant cost in v10 / v11
profiles — large enough to swallow the empirical-tau win on its own. Bucketed
reuse + dirty append turns drafter context maintenance into a near-constant
overhead instead of a per-step linear cost.

**Knob:** `DFLASH_DECODER_CONTEXT_BUCKET=8` (env). Bucket size is a tradeoff —
larger means fewer graph builds but more wasted padding; 8 was the sweet spot
on Spark.

---

### (2) Flash-attn mask fix in the DFlash decoder

**What:** the attention mask handed to flash-attn for the DFlash decoder path
was wrong, which both (a) hurt acceptance and (b) forced fallback off the FA
kernel for some calls.

**Why it mattered:** dual hit. Wrong mask depressed acceptance by a few percent
(directly cutting the tau payoff) **and** the fallback path is materially slower
per token than FA. Fixing the mask let FA stay on for the whole draft+verify
cycle and recovered the missing acceptance.

---

### (3) `DFLASH_RETURN_MAX=1` validation cap (server still at `--draft-max 7`)

**What:** the drafter still proposes 7 tokens (`--draft-max 7`), but only the
first accepted token is *returned* per validation step (`DFLASH_RETURN_MAX=1`).

**Why it mattered:** this decouples "how aggressively we draft" from "how often
we synchronize with the verifier." Returning all-accepted at depth 7 sounds
free, but in practice it inflated per-step verification cost on the IQ4_XS
target enough that the median dipped under 1.0×. Capping return to 1 keeps the
acceptance benefit of a 7-deep draft (chain probability is still consumed) while
giving the verifier a cheap, predictable per-step cost. **This was the trick
that flipped the median from the 0.95-ish band to >1.0×.**

**Knob:** `DFLASH_RETURN_MAX=1` (env).

---

### (4) Larger serving ubatch — `-ub 1024`

**What:** raised the server's micro-batch to 1024 (`llama-server -ub 1024`),
keeping `-b 4096`.

**Why it mattered:** with a wider ubatch the verifier prefill chunks run wide
enough that drafter-step overhead amortizes across more verifier work per
synchronization. Smaller ubatches (the default 512) left too much idle time on
the verifier side when the drafter was the critical path.

---

### (5) Batched DFlash feature async reads, single scheduler sync per extraction

**What:** previously the DFlash feature extractor synced the scheduler per
layer / per feature. Now all DFlash hidden-feature reads are batched and a
single `ggml_backend_sched_synchronize` is issued per extraction.

**Why it mattered:** sched syncs are pipeline bubbles. Going from N (one per
feature/layer) to 1 collapses a chain of small stalls that each individually
look insignificant in a flat profile, but cumulatively cost a few percent of
wall-clock per draft step.

---

### (6) Anchor-aligned DFlash block — `DFLASH_BLOCK_INCLUDES_ANCHOR=1`

**What:** ensure the drafter sees the anchor token in the same block as its
predictions, not in a previous block.

**Why it mattered:** the v11 "tail" and "anchor+tail" variants without this
landed at speedups of 0.518–0.558× — drafter fired but lost. Anchor-aligned
block construction restored both empirical_tau (back into the 1.3+ band) and
the wall-clock cell that the median depends on.

**Knob:** `DFLASH_BLOCK_INCLUDES_ANCHOR=1` (env).

---

### (7) Gate D7/D5 debug hidden-state dumps behind `DFLASH_BENCH_TRACE`

**What:** the D7 and D5 hidden-state dumps that were produced for offline tau
diagnostics were on by default in benchmark runs. Now they're gated behind
`DFLASH_BENCH_TRACE` (default off in serving).

**Why it mattered:** pure cleanup, not algorithmic — but the dumps were touching
host memory on every draft step. Gating them off recovers a few percent of
wall-clock without changing any numerics. Cheap fix, free win.

---

## 5.3 Reproducing the gate

Server config:

```bash
llama-server --dflash \
  -m  ~/clawd/iq4_models/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -md ~/models/MiniMax-M2.7-DFlash-v11-step15080-legacy-targethead.gguf \
  -c 4096 -cd 4096 -b 4096 -ub 1024 \
  --draft-max 7 --draft-p-min 0.0 \
  --sampling-seq k --top-k 1 --temp 0.0
```

Environment (all three required for the gate-passing combination):

```bash
export DFLASH_DECODER_CONTEXT_BUCKET=8
export DFLASH_RETURN_MAX=1
export DFLASH_BLOCK_INCLUDES_ANCHOR=1
# Optional (off by default; only enable for diagnostics):
# export DFLASH_BENCH_TRACE=1
```

llama-benchy invocation (matches the gate JSONL filter):

```bash
llama-benchy \
  --base-url http://127.0.0.1:8080/v1 \
  --tg 128 --no-warmup --skip-coherence
```

Gate proxy filter (real-DFlash predicates):

- `sum(draft_n) > 0`
- `empirical_tau in [1.3, 2.0]`
- `predicted_n >= 64` per request

Artifacts from the passing run:

- with-spec: `~/dflash-mission/realworld_results/with_spec_dm7_asyncfeat_return1_ub1024_anchor_r1_20260508_014846.json`
- gate jsonl: `~/dflash-mission/realworld_results/empirical_tau_dm7_asyncfeat_return1_ub1024_anchor_r1_20260508_014846.gate.jsonl`
- baseline:  `~/dflash-mission/realworld_results/no_spec_compact_r1.json`

---

## 5.4 Honest read

This is a **gate pass**, not a shipping result.

- Median speedup of 1.019× clears the binary gate but leaves no headroom; one
  cell at 1.024k context already loses (0.959×).
- The win is concentrated at smaller pp; the 1.079× cell at pp=256 is what
  drags the median across.
- The pre-v11.1 best `--draft-max 1` median was 0.836×, so v11.1 is a +0.18×
  improvement — almost all of which comes from the bucketed-graph-reuse +
  return-cap pair (fixes 1 and 3). Fixes 2, 4, 5, 6, 7 stack to a few percent
  each.
- All fixes are in the *serving / inference* path. The drafter checkpoint is
  unchanged from v11 step 15080. Training-side changes are not on the critical
  path for further wall-clock gains.

**Where to look next** for shippable headroom:

1. The pp=1024 cell still loses by 4%. Either a smarter context-reuse policy
   for longer contexts (bucket > 8, or adaptive bucketing) or a per-request
   `--draft-max` floor based on context length.
2. `DFLASH_RETURN_MAX > 1` becomes a win again only if verification cost can
   be reduced — investigate whether IQ4_XS verifier is the right precision for
   batched verification, or whether a Q5_K-class verifier on the same GPUs
   gives a better speed/accuracy point.
3. FA kernel coverage on the DFlash decoder path is now correct but may not be
   optimal for the bucket=8 shape; a fused decoder kernel could close the
   sched-sync gap below the current floor.

---

## 5.5 Pointers

- §4.10 of `repro/04-empirical-tau-llama-benchy.md` — the immediate gate-pass
  receipt with the same numbers in the §4 voice.
- `~/dflash-mission/RESULTS.md` — `GATE_PASS_RUNTIME_PERF` receipt at
  `2026-05-08T02:31:17-07:00`.
- llama.cpp-dflash commit `d1d2c81caccc748eaaff32b6b7823bad090fd1dd` — the
  fork carrying all seven source-side changes above.

## 5.6 Verifier authority is mandatory — `DFLASH_FORCE_ACCEPT_DRAFTS` rejected

A short follow-up so this engineering finding doesn't have to be re-derived.

A tau-closure attempt tried `DFLASH_FORCE_ACCEPT_DRAFTS=1` to lift
`PROJECT_GUTENBERG_RUNTIME_TAU` from 1.369 (§4.10 floor) toward the
offline training-distribution range. It produced apparent tau=1.969 and
median wall-clock 1.380x, but the env knob bypasses the verifier acceptance
test in `tools/server/server-context.cpp:2896-2910` — it discards
`common_sampler_sample_and_accept_n`'s verdict and appends drafted tokens
unconditionally. That breaks output-distribution equivalence with the
verifier-alone path, which is the exact property standard speculative
decoding (Leviathan/Chen 2023) preserves.

The "tau" in that mode is not a measurement: by construction
`draft_n_accepted == draft_n` always, so the formula
`tau = 1 + acc/(pred-acc)` returns whatever the chain depth was. The
wall-clock gain is real throughput but the emitted tokens are the drafter's,
not the verifier's. Full discussion and source quote in §4.11.

**Rule:** every gate going forward requires `draft_n_accepted < draft_n` on
at least some records (proving the verifier rejected something), AND a
50-prompt fixed-seed greedy output-equivalence probe vs the no-spec baseline
with ≥99% token-exact match. If acceptance is 100% on every record, the
verifier is not authoritative and the receipt does not count.

The honest finding from the lossless `RETURN_MAX > 1` sweep that motivated
the attempt: at depth 7, the v11 step15080 drafter's chain agrees with the
IQ4_XS verifier on only ~4% of proposals on Project Gutenberg traffic, and
the per-step verification cost outweighs the win on most cells. That points
at lossless paths to investigate next — better drafter chain calibration,
shorter chain with selective extension, draft-tree pruning, or retraining
the drafter for the runtime-encountered distribution. Force-accept is
not on the table.
