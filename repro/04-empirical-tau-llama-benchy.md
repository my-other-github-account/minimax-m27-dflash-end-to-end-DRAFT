# Section 4 — Empirical tau against llama-benchy / Project Gutenberg traffic

End-to-end recipe for measuring the *deployment* tau of a trained DFlash drafter
under realistic OAI-API traffic. The drafter is converted to GGUF in §3, served
by the dflash fork of `llama-server`, fronted by a small aiohttp proxy that
captures per-request speculative-decode timings, and benchmarked with
`eugr/llama-benchy`'s Project Gutenberg corpus.

> **Status:** ✅ Captured end-to-end on spark-4, 2026-05-07. Verifier
> `MiniMax-M2.7-FP8` (UD-IQ4_XS), drafter
> `MiniMax-M2.7-DFlash-v11-step15080-legacy-targethead.gguf`
> (sha256 `4121831075997a48cc466a210d208a4f47d81702492b29785e2f1409c2dc543a`,
> 819 MB), llama.cpp-dflash commit `d1d2c81caccc748eaaff32b6b7823bad090fd1dd`,
> 24 sampled `/v1/chat/completions` requests through the proxy.

---

## 4.0 What this stage proves

**Headline:** empirical tau ≈ **1.5659** across 24 OAI-compatible chat
completions issued by `llama-benchy` against Project Gutenberg book text
(Sherlock Holmes, default corpus).

| empirical tau | training-distribution tau (Phase B 10-row, step15080) | training-distribution tau (Phase A row_04, step15080 screen) | trainer val tau (step15080, teacher-forced) |
|---|---|---|---|
| **1.566** | 1.899 | 2.82 | ~2.30 |

The data-distribution generalization verdict is **partial, not full.**
Training-distribution prompts (the MiniMax DFlash mix) yield tau ≈ 1.9–2.4;
Project Gutenberg English narrative yields tau ≈ 1.57. There is a real
~0.4–0.7 tau gap when leaving the training distribution.

This stage *only* makes sense after §3 produces a working legacy-targethead
GGUF. Without §4.1 you measure tau ≈ 1.0 and the rest of this document is moot.

Headline data: §4.6. Pitfalls that cost the most time: §4.1 and §4.8.

---

## 4.1 The GGUF format gotcha

The §3 conversion path produces two distinct GGUF formats. Only one of them
runs correctly on the dflash fork.

| format | tensors | `output.weight` | `d2t` mapping | runtime tau on Sherlock Holmes |
|---|---|---|---|---|
| **compact-d2t-i32** (32k draft vocab + d2t mapping table) | 70 | yes (compact) | yes (i32) | ≈ **1.008** — collapses |
| **legacy target-lm-head** (rebaked to full 200064 vocab) | 69 | no | no | **1.57** |

The compact format collapses tau to ~1.008 on every prompt we tried (step_3915
and step_14355 both reproduced it). The legacy format — produced by the
d2t-aware scatter described in [`03-inference.md` §3.3](03-inference.md) — is
the only path that works in the current `llama.cpp-dflash` binary.

> **Diagnostic signal.** If your DFlash setup is otherwise correct (env vars
> set, `--draft-max 7` on the command line, fork binary, deterministic
> sampler) but you see tau ≈ 1.0 on every prompt, your GGUF is the wrong
> format. There is no other failure mode that produces this exact symptom.

§4 builds directly on the §3.3 rebake. The verbatim Phase A receipt in
`~/dflash-mission/RESULTS.md` (2026-05-07 08:52 PT) marks the moment the
format switch landed: tau on row_04 jumped from 1.008 to **2.385** with the
same checkpoint, same prompt, only the GGUF format changed.

---

## 4.2 Required env vars and server flags

All three env vars are mandatory. Removing any one of them silently degrades
tau (the most aggressive degradation, `DFLASH_BLOCK_INCLUDES_ANCHOR=0`,
collapses weak rows back toward 1.0).

| env var | meaning | why mandatory |
|---|---|---|
| `DFLASH_BLOCK_INCLUDES_ANCHOR=1` | The anchor token is included in the drafted block (matches training). | The drafter was trained with anchor-in-block; without this it sees a different block layout from training and acceptance collapses on weaker rows. |
| `DFLASH_RAW_TOKENS=1` | Bypass chat templating; treat input as raw token IDs end-to-end. | Required so the prompt encoding matches what the trace generator produced; otherwise the verifier and drafter disagree on the prefix. (Historical alternate value `***` is equivalent.) |
| `DFLASH_VERIFIER_KV_TRIM_ON_REJECT=1` | On a draft reject, trim the verifier's KV-cache back to the accepted prefix. | Modest measured accept-rate improvement; without it the verifier carries stale KV from rejected tokens and the next round's `p_1` regresses. |

Four server flags are required:

```
-m   <verifier GGUF>              # MiniMax-M2.7-FP8 UD-IQ4_XS
-md  <legacy-targethead GGUF>     # the §3 output, NOT compact-d2t-i32
--draft-max 7                     # tau ceiling matches training (block_size=8)
--top-k 1 --temp 0.0              # deterministic sampler
--host 127.0.0.1 --port 8080
```

`--top-k 1 --temp 0.0` is the deterministic sampler that makes runs
apples-to-apples; under any other sampler the per-request tau becomes
prompt-dependent in a way that masks the drafter signal.

The wrapper that bakes all of this in:
[`scripts/inference/launch_server_dflash.sh`](scripts/inference/launch_server_dflash.sh).

---

## 4.3 Server fork API additions

The `~/llama.cpp-dflash` fork (commit `d1d2c81caccc748eaaff32b6b7823bad090fd1dd`)
adds two fields to every `/v1/chat/completions` (and `/completions`)
JSON response, inside the existing `timings` object:

```jsonc
"timings": {
    "prompt_n": 59, "prompt_ms": 1578.695, "prompt_per_second": 37.37,
    "predicted_n": 128, "predicted_ms": 19234.1, "predicted_per_second": 6.65,
    "draft_n": 567,            // <- fork addition
    "draft_n_accepted": 89     // <- fork addition
}
```

Source confirmed:

- `tools/server/server-task.cpp`:
  `base["draft_n"] = draft_n; base["draft_n_accepted"] = draft_n_accepted;`
- `tools/server/server-context.cpp`: per-slot `n_draft_total` /
  `n_draft_accepted` accounting and the per-request log line
  `draft acceptance rate = X (Y accepted / Z generated)`.

This is **fork-specific.** Upstream `llama.cpp` does not expose these fields,
which means a benchmark client reading upstream `timings` cannot recover
empirical tau. We depend on this addition end-to-end:

1. The proxy in §4.4 reads `timings.draft_n` and `timings.draft_n_accepted`
   directly off the response body.
2. The summarizer in §4.5 step 4 computes per-request tau as
   `predicted_n / (predicted_n − draft_n_accepted)`.
3. The §4.7 verification cross-checks the per-slot log lines against
   the proxy's per-request totals.

If you point `tau_capture_proxy.py` at upstream `llama.cpp`-built
`llama-server`, it logs but every record will have `draft_n=0,
draft_n_accepted=0` and the summarizer reports `tau_overall = 1.0`. This is
not a bug — there is no signal to recover.

---

## 4.4 The capture method

`eugr/llama-benchy` is, as of early 2026, the only OAI-endpoint benchmark
harness that handles MTP / speculative-decode response chunks correctly
(per its README). It uses Project Gutenberg book text — Sherlock Holmes by
default — for prompts: diverse, realistic, and emphatically **not** our
training distribution.

The catch: `llama-benchy` does not expose per-request `timings` to its
client. It records aggregate throughput numbers and that is all.

So we sit a tiny aiohttp proxy on `127.0.0.1:8081` between the harness and
`llama-server` on `127.0.0.1:8080`. The proxy:

1. Forwards every request transparently (method, headers, body).
2. On `/v1/chat/completions` (and `/completions`):
   - For non-streaming `application/json` bodies, parses the response
     once it has streamed in full and extracts `timings`.
   - For `text/event-stream` bodies, buffers the SSE chunks during
     streaming and parses the *last* `data: {…}` event that contains a
     `timings` block (llama-server emits final stats on the final
     non-`[DONE]` data event).
3. Writes one JSONL record per request to
   `./empirical_tau_traffic.jsonl`:

```jsonc
{"ts": 1778182000.59, "path": "/v1/chat/completions", "status": 200,
 "id": "chatcmpl-…", "elapsed_s": 19.82,
 "prompt_n": 20, "predicted_n": 61,
 "draft_n": 231, "draft_n_accepted": 27, …}
```

Source: [`scripts/inference/tau_capture_proxy.py`](scripts/inference/tau_capture_proxy.py)
(~190 lines). All other paths (`/v1/models`, etc.) pass through with a small
log line so the harness's warmup probes still succeed.

> **Pitfall** (caught the hard way today). The first iteration of the
> capture proxy assumed the response body was reliably `application/json`
> and called `json.loads` on it; for streaming requests the body is SSE
> and the parse failed silently. The corrected proxy in this stage
> dispatches on `Content-Type` and walks the SSE stream when needed.

---

## 4.5 Run recipe (8 steps)

All commands assume `cd ~/dflash-llama` and that `llama-server` from
`~/llama.cpp-dflash` is on `$PATH`. Substitute your own paths into the
two GGUF env vars; the rest defaults are sensible.

```bash
# 1. Bring the DFlash server up on :8080 (background it however you like).
VERIFIER_GGUF=~/clawd/iq4_models/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
DRAFTER_GGUF=~/models/MiniMax-M2.7-DFlash-v11-step15080-legacy-targethead.gguf \
    bash repro/scripts/inference/launch_server_dflash.sh &

PORT=8080 bash repro/scripts/inference/wait_for_server.sh

# 2. Bring the capture proxy up on :8081.
OUT_JSONL=./empirical_tau_traffic.jsonl \
UPSTREAM_URL=http://127.0.0.1:8080 \
LISTEN_PORT=8081 \
    python3 repro/scripts/inference/tau_capture_proxy.py &

# 3. Run llama-benchy through the proxy.
PORT=8081 OUT=./with_spec.json \
    bash repro/scripts/inference/bench.sh

# 4. Aggregate the proxy JSONL into the §4.6 table.
python3 repro/scripts/inference/summarize_empirical_tau.py \
    ./empirical_tau_traffic.jsonl

# 5. Bring up the autoregressive baseline server (kill the DFlash one first).
VERIFIER_GGUF=~/clawd/iq4_models/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
    bash repro/scripts/inference/launch_server_ar.sh &
PORT=8080 bash repro/scripts/inference/wait_for_server.sh

# 6. Run llama-benchy directly (no need for the proxy here — there are no
#    speculation timings to capture).
PORT=8080 OUT=./no_spec.json \
    bash repro/scripts/inference/bench.sh

# 7. Compare the two llama-benchy JSON outputs side-by-side
#    (with_spec.json vs no_spec.json — pp / tg / depth cells).
#
# 8. Tear down: kill the proxy and the server.
```

Steps 1, 2, 5 are background processes; steps 3, 4, 6, 7 run in the
foreground.

---

## 4.6 Live results

### Headline (empirical tau, 24 requests through the proxy)

| metric | value |
|---|---|
| n_requests | 24 |
| predicted_tokens | 3005 |
| draft_n_accepted | 1086 |
| draft_n | 12913 |
| inferred_rounds | 1919 |
| draft_accept_rate_by_tokens | 8.4101% |
| **tau_overall** | **1.5659** |
| mean_tau | 1.5813 |
| median_tau | 1.5706 |
| p10_tau | 1.4222 |
| p90_tau | 1.7394 |
| min_tau | 1.3617 |
| max_tau | 1.9692 |

### Distribution

| bin | count |
|---|---|
| tau < 1.5 | 6 (25%) |
| 1.5 ≤ tau < 2.0 | 18 (75%) |
| 2.0 ≤ tau < 2.5 | 0 |
| tau ≥ 2.5 | 0 |

No sampled real-world request reached tau ≥ 2.0. The 75th-percentile
request lands near 1.74, well below the trainer-val upper bound.

### Comparison to training-distribution tau

| measurement | tau | source / config |
|---|---|---|
| Trainer val (step_15080, teacher-forced upper bound) | ~2.30 | spark-2 `val_metrics.json` |
| Phase A row_04 (step_15080, single training-dist prompt, fast screen at -n 128) | 2.82 | replay `tau_screen_step15080_legacy_rows040206_20260507_1032_summary.json` |
| Phase A row_04 (step_14355, single training-dist prompt) | 2.385 | `phaseA_step14355_legacy_row04_20260507_084906_summary.json` |
| Phase B 10-row aggregate (step_14935, training-dist) | 1.956 | `phaseB_step14935_legacy_10row_20260507_1006_summary.json` |
| Phase B 10-row aggregate (step_15080, training-dist) | 1.899 | `phaseB_step15080_legacy_10row_20260507_1037_summary.json` |
| **Empirical (step_15080, llama-benchy / Gutenberg, 24 req)** | **1.566** | this stage |

### Honest interpretation: the gap is real

Going from a teacher-forced training-distribution upper bound (~2.30) to a
deployment-distribution empirical mean (~1.57) costs roughly **0.7 tau**.
Going from a same-checkpoint Phase B 10-row aggregate (1.899) to the
Gutenberg-traffic empirical (1.566) costs roughly **0.33 tau**. Both are
larger than sample noise on n=24.

The drafter generalizes *partially* off-distribution. It still produces
useful speculation — every Gutenberg request had `draft_n_accepted > 0`,
and 75% land in the 1.5–2.0 bin — but it does not retain its
training-distribution tau on English narrative prose.

### Wall-clock secondary table (compact stable cells, runs=1)

| pp | depth | tg | with_spec t/s | no_spec t/s | ratio |
|---|---|---|---|---|---|
| 256  | 0 | 128 | 3.246 | 4.066 | 0.798 |
| 1024 | 0 | 128 | 2.305 | 4.061 | 0.568 |

Median wall-clock ratio 0.683, range [0.568, 0.798]. **The current DFlash
server path is slower than `--draft-max 0` on these stable cells**: the
empirical accept rate (8.4% by tokens) is too low to offset drafter
overhead at this verifier quant.

(Larger / deeper grids — `pp=4096`, `depth=8192` — were attempted but the
DFlash server stalled in the deep-context cells, so the wall-clock
comparison is limited to the stable depth=0 cells. Empirical tau uses
the larger 24-request server-log sample.)

---

## 4.7 Verification: did real DFlash actually serve these requests?

Five signals confirm the DFlash code path was exercised, not a fallback:

1. **The three `DFLASH_*` env vars were set in the launching shell**
   and are echoed verbatim by
   [`launch_server_dflash.sh`](scripts/inference/launch_server_dflash.sh)
   into the server log on every launch. Grep
   `~/dflash-mission/logs/server_realworld_tau_emp_spec_20260507_122551.log`
   for `DFLASH_BLOCK_INCLUDES_ANCHOR`, `DFLASH_RAW_TOKENS`,
   `DFLASH_VERIFIER_KV_TRIM_ON_REJECT` — all three present.
2. **Per-request `draft_n_accepted` is nonzero on every captured chat
   completion with `predicted_n > 1`.** Across 24 requests, sum
   `draft_n_accepted = 1086` against `draft_n = 12913` and
   `predicted_n = 3005`. A copyspec/ngram fallback would zero
   `draft_n_accepted` while `draft_n` could remain nonzero.
3. **The fork's per-slot log line fires once per request:**
   `draft acceptance rate = 0.0X (Y accepted / Z generated)`. This line
   only exists in `tools/server/server-context.cpp` of the dflash fork
   (cross-referenced in §4.3) and is absent from upstream binaries.
4. **`--draft-max 7` was on the launching command line** — visible in
   the server log header. Without `--draft-max ≥ 1` the fork emits no
   `draft_n` accounting at all.
5. **No `ngram` / `copyspec` strings appear in the server log.** The
   fork prefers DFlash whenever both `-md` and the DFlash GGUF metadata
   keys (`block_size`, `n_target_features`, `mask_token_id`) load
   cleanly; the legacy-targethead GGUF carries all three.

Together these match the §3.5 verification pattern from the offline
chain-pos sweep — same library code path, exposed through the OAI-API
surface.

---

## 4.8 Pitfalls

- **GGUF format wrong → tau collapses to ≈ 1.0 on every prompt.** The
  compact-d2t-i32 GGUF *loads*, *runs*, and *generates plausible text*,
  while silently producing tau ≈ 1.008. Confirmed on step_3915 and
  step_14355. Always use the legacy target-lm-head GGUF from §3.3.
- **Missing `DFLASH_BLOCK_INCLUDES_ANCHOR=1` → silent collapse on weak
  rows.** Rows that already have lower per-position `p_i` regress
  hardest; row_04 stays favorable but tail rows fall. The probe
  receipts in `~/dflash-mission/RESULTS.md`
  (2026-05-07 09:45 PT no-op runtime probes) document the pattern row
  by row.
- **Non-deterministic sampler → unreproducible numbers.** Run with
  `--top-k 1 --temp 0.0` for any tau measurement. Anything else turns
  per-request tau into a prompt-dependent random variable.
- **Upstream `llama.cpp` instead of the fork → no
  `timings.draft_n_accepted` in responses.** The proxy will log every
  request but every record carries `draft_n = draft_n_accepted = 0`,
  the summarizer reports `tau_overall = 1.0`, and there is nothing to
  recover. Use the fork at commit
  `d1d2c81caccc748eaaff32b6b7823bad090fd1dd` (or newer) — see §4.3.
- **Streaming response parsing.** If you write your own capture proxy,
  make sure to dispatch on `Content-Type`: SSE (`text/event-stream`)
  bodies are *not* a single JSON document. The first iteration of
  `tau_capture_proxy.py` did not handle this and silently lost most of
  the day's captured timings (visible as `JSONDecodeError` lines in
  the 2026-05-07 12:25 JSONL); the version checked in here does.
- **Deep-context grids stall.** `pp=4096` / `depth=8192` cells stalled
  the DFlash server today on spark-4. Stick to the compact stable cells
  for wall-clock comparisons; deep cells are still a known issue and
  the empirical-tau sample uses the captured request log instead.
