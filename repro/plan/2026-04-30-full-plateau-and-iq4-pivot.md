# 2026-04-30: FULL training plateau and the IQ4 trace-gen pivot

> Status snapshot at write-time: FULL training on spark-4 at epoch 13/17, plateaued at val p_1 ≈ 22.9% (vs production 28.88%). IQ4-llama.cpp single-machine trace-gen experiment 50% complete on spark-3 (500/1000 traces) with full pipeline scaffolded.

## 1. The problem: FULL is plateauing 6pp under production with the same recipe

**FULL training run** (`full_5L_paired6515_20260430_111332`, on spark-4, `dflash-full.service` under `systemd-run --user`):

- Architecture: 5L, hidden=3072, mlp=1536, draft_vocab=32768, target_layer_ids=[2,16,30,45,59], block_size=8, mask_token=200054
- Recipe: 17 epochs, lr=3e-5, scheduler-warmup-steps=100, max_anchors=512, total_seq_len=2048, --save-best
- **Data: 6,515 paired samples** (the new "all paired" pool — see §2)

**Production reference drafter** (`fp8_5L_heavy_es_5layer_20260429_094013/checkpoint_best`, the one shipped in `MiniMax-M2.7-DFlash.gguf` md5 `785c5b5a6bcf...`):

- **Same architecture, same recipe**, same 17 epochs
- **Data: 1,176 paired samples** (hand-curated)

Per-epoch trajectory of FULL's p_1 (val):

| epoch | p_1 | Δ |
|---|---|---|
| 1 | 14.00% | – |
| 2 | 18.50% | +4.50pp |
| 3 | 20.50% | +2.00pp |
| 4 | 21.80% | +1.30pp |
| 5 | 22.00% | +0.20pp |
| 6 | 22.20% | +0.20pp |
| 7 | 22.50% | +0.30pp |
| 8 | 22.60% | +0.10pp |
| 9 | 22.90% | +0.30pp |
| 10 | 22.80% | -0.10pp |
| 11 | 22.90% | +0.10pp |
| 12 | 22.80% | -0.10pp |
| 13 | 22.89% | +0.10pp ← current best |

Last 5 epochs gained +0.4pp total, oscillating ±0.1pp. **Plateau is essentially complete.**

## 2. Side-by-side comparison

### Conditional p_i (per-position, val):

| pos | epoch-5 | epoch-13 | production | gap (epoch-13 vs prod) |
|---|---|---|---|---|
| 1 | 22.03% | 22.89% | **28.88%** | -5.99pp (-20.7% rel) |
| 2 | 13.90% | 14.42% | 20.17% | -5.75pp (-28.5% rel) |
| 3 | 10.08% | 10.26% | 13.74% | -3.48pp (-25.3% rel) |
| 4 | 8.56% | 8.68% | 10.56% | -1.88pp (-17.8% rel) |
| 5 | 7.49% | 7.78% | 8.42% | -0.64pp (-7.6% rel) |
| 6 | 6.95% | 6.98% | 7.32% | -0.34pp (-4.6% rel) |
| 7 | 6.40% | 6.55% | 7.20% | -0.65pp (-9.0% rel) |

### Chain-cumulative ∏ p_i (the metric that actually matters):

| pos | epoch-5 chain | epoch-13 chain | production chain | rel gap |
|---|---|---|---|---|
| 1 | 22.03% | 22.89% | 28.88% | -20.7% |
| 2 | 3.06% | 3.30% | 5.83% | **-43.3%** |
| 3 | 0.31% | 0.34% | 0.80% | -57.7% |
| 4 | 0.026% | 0.029% | 0.085% | -65.2% |
| 5 | 0.0020% | 0.0023% | 0.0072% | -67.9% |
| 6 | 0.000139% | 0.000158% | 0.000522% | -69.7% |
| 7 | 0.0000089% | 0.0000103% | 0.0000376% | -72.5% |

**Chain-cumulative gap compounds** through positions because each position's relative gap multiplies. Chain-pos-2 is 43% under production; chain-pos-7 is 72% under.

## 3. Diagnosis: data, not capacity, not config

### Why not capacity:

- Train acc and val acc are tracking together (no diverging gap). Train p_1 oscillates noisily 0.14–0.35 across mini-batches, val sits steady at 0.229. If model were capacity-bound, train would run away above val.
- Same 5L/3072/1536 architecture hit 28.88% in production. The model class can reach that ceiling.

### Why not config:

- Recipe is byte-for-byte identical to production: same lr, scheduler, warmup, max_anchors, total_seq_len, save-best, target_layer_ids, draft_vocab_size, block_size, mask_token, num_hidden_layers.
- p_1 trajectory is smoothly converged — no late-stage instability, no oscillation pattern that would indicate LR or scheduler issues.
- Speculators repo commit between the two runs is the same (67bafe6 with dflash submodule).

### Why data is the suspect:

- **5.5× more samples → worse results.** That's the smoking gun. More data with same recipe should monotonically improve, or at worst plateau at the production level. Not regress.
- **Pool audit history:** the 6,515 set was assembled by a TP=4 multi-Spark vLLM datagen with retries; staging dirs accumulated 17,657 files (2.7× duplicates) before dedup. Even after dedup-by-token-id-sha256, there's no guarantee that the "winning" hidden-state copy for each sample is the cleanest.
- **Relative-gap shape across positions** (pos-1 -21%, pos-2 -29%, pos-3 -25%) is consistent with label-noise: noisy hidden-state targets degrade per-position teacher forcing, and the noise compounds in chain-cumulative.
- **Production used 1,176 hand-curated samples**, FULL uses 6,515 from a less-curated pipeline. Quantity ≠ quality.

### What WOULD prove the data hypothesis:

- Re-train on the production 1,176 set with current code → if it hits 28.88%, data is the bottleneck. If it hits 22.9%, look for code drift.
- Curate a quality-filtered subset of FULL's 6,515 → if it beats FULL-13 at the same epoch count, low-quality samples are dragging down the average.

## 4. The IQ4 pivot — testing "small + clean > large + noisy"

Independently of the FULL plateau diagnosis, the user asked about an alternative trace-gen pipeline:
**Replace the 4-Spark vLLM TP=4 FP8 trace-gen with single-machine llama.cpp inference at IQ4_XS quantization, collect 1000 traces, train a SMOKE drafter on them, compare to FP8-trace SMOKE baseline.**

If this works, drafter training becomes self-hostable on one GB10 spark instead of needing the cluster, AND the consistency of single-machine quantized inference might produce *cleaner* labels than the multi-machine FP8 pipeline (which is now suspected of label noise per §3).

### Status (delegated subagent on spark-3, interrupted at 78 tool calls / 58 min):

✅ Done:
- Synced IQ4_XS verifier (108 GB across 4 shards) from spark-1 → spark-3 over QSFP at ~290 MB/s
- Synced buun-llama-cpp source from spark-1 (vanilla `e275191e`)
- Restored `dump-hiddens` example to `examples/dump-hiddens/` and patched CMakeLists
- Built buun on spark-3 with CUDA `sm_121a-real`, target `dump-hiddens` succeeded
- Synced FP8 prompts arrow dataset from spark-4 (`/home/user/iq4_tracegen/prompts_fp8/`, 6515 rows)
- Synced speculators repo from spark-4 → spark-3
- Wrote `gen_traces.py` + batched `gen_traces_batch.py` driver scripts
- **Generated 500/1000 traces** at `/home/user/iq4_tracegen/traces/hidden_states/hs_<i>.safetensors` — same prompt indices as FP8 reference, IQ4-rendered hiddens at layers `[2,16,30,45,59,LAST]`
- Wrote `sanity_check.py` and `iq4_smoke_train.sh` (not yet executed)

🟡 In flight:
- Trace generation is **~50% complete**; the driver was running OK (3-10 sec/trace depending on seq_len) before the subagent context was interrupted. Last log lines show successful traces at indices 500-502.

❌ Not done:
- Remaining 500 traces
- Sanity check IQ4 vs FP8 (cosine, norm, token-id agreement)
- Build `dataset_iq4/{prompts,hidden_states}/` for training
- Run SMOKE training (3 epochs, max_anchors=64) on the IQ4 traces
- Compare to FP8 SMOKE baseline (val_loss=6.541, p_1=9.70%, 1176 traces, same recipe)

### Spark-3 artifacts (for resume):

```
/home/user/iq4_tracegen/
├── buun-llama-cpp/             # built buun with dump-hiddens
├── prompts_fp8/                # 6515 prompt arrow dataset (we use first 1000)
├── repos/speculators/          # synced from spark-4
├── scripts/                    # gen_traces*.py, sanity_check.py, iq4_smoke_train.sh
├── traces/hidden_states/       # 500 hs_<i>.safetensors files (so far)
├── checkpoints/                # empty
└── logs/                       # tracegen.log, smoke.log (1 test trace, not real smoke)
~/clawd/iq4_models/UD-IQ4_XS/   # IQ4 verifier shards (108 GB)
```

## 5. Decision: what to actually do next

Three orthogonal experiments, each ~30–60 min wall-clock:

### A. **Diagnostic re-run on production 1,176 data** (priority 1)

Re-run the FULL training recipe against the **original production 1,176 paired set** on spark-2 (idle). Same hparams, same epochs (or 5 epochs is enough to clear the question — production hit p_1=21.8% at epoch 4 with that data).

**Success criterion:** if 5-epoch p_1 ≥ 21.5%, data is the bottleneck. If ≤ 19%, there's code drift between the production training session and now.

**Effort:** ~25 min wall-clock (same per-epoch time as FULL, smaller dataset).

**Spark:** spark-2 (currently idle, 1.8 TB free, GPU cold).

### B. **Resume IQ4 trace-gen + run SMOKE** (priority 2)

Resume the spark-3 subagent's work. Generate remaining 500 traces, sanity-check, train SMOKE.

**Success criterion:** IQ4 SMOKE p_1 ∈ [6%, 13%] (within 30% of FP8 SMOKE's 9.70% baseline). Pipeline runs end-to-end.

**Bonus:** if IQ4 SMOKE *beats* FP8 SMOKE, that's strong evidence for the "single-machine cleaner" hypothesis and worth following with an IQ4 FULL.

**Effort:** ~45 min wall-clock for remaining 500 traces (~3-10 sec each) + ~30 min SMOKE training.

**Spark:** spark-3 (idle, 556 GB free, work already half-done).

### C. **Quality-curated subset of FULL's 6,515** (priority 3)

Build a per-sample quality metric (e.g., norm-consistency check vs reference statistics, or hidden-state SNR proxy) and pick the top-K samples for K ∈ {1176, 2500, 4000}. Train each subset for 5 epochs and compare p_1 to FULL-5's 22.0%.

**Success criterion:** at least one subset beats FULL-5 by ≥1pp p_1, demonstrating that signal-quality matters.

**Effort:** ~1 h to design + execute the curation + train one subset.

**Spark:** spark-2 or spark-3 depending on what's free.

### Don't bother:

- **Training FULL longer than 17 epochs.** Last 5 epochs gained 0.4pp. Projected gain to epoch 30 would be ~+1pp. Not worth the wall-clock vs the alternatives above.
- **Architecture changes.** Capacity isn't the issue.
- **LR tuning.** Recipe matches production; production reached 28.88%.

## 6. Live state and other in-flight work

- **FULL training (spark-4):** epoch 13/17 done, currently on epoch 14. ETA finish ~15:30 PDT. Will produce final checkpoint regardless of plateau diagnosis.
- **Cron watcher (`dflash-full-completion-watcher` job `69f74aa7a13f`):** every 15 min, will auto-detect FULL completion → run the proven §3 recipe → produce GGUF + benchmark + report chain-cumulative measured-vs-predicted z-scores. **Will run regardless of the plateau** — it's a methodology check (does runtime match training prediction?), not a quality check.
- **Genomics backup (spark-1):** still chugging on USB-3 at ~165 MB/s. Untouched. Don't disturb.
- **Repo:** main HEAD `06a76c9` (§3 inference doc). This plan adds `repro/plan/`.
- **Bundle:** `/tmp/minimax-m27-dflash-end-to-end-DRAFT.bundle`, push still blocked on auth for `my-other-github-account`.

## 7. Open questions to revisit

1. **Why does the FULL pool's data quality differ from production's 1,176?** If it's TP=4 multi-Spark batch drift, the IQ4 single-machine path could be the *better* long-term answer regardless of total sample count.
2. **Is there a hidden-state quality metric** that correlates with downstream drafter accuracy, so future trace-gen pipelines can self-filter?
3. **Does production-1,176 + IQ4-1000 mixed training** outperform either alone? Diversity of trace source might help generalization.

## 8. Memory note

The user prefers empirical verification over estimates. All claims in this document are backed by:
- Per-epoch val_metrics.json on spark-4 (FULL trajectory)
- Production val_metrics.json on spark-2 (`fp8_5L_heavy_es_5layer_20260429_094013/checkpoint_best/val_metrics.json`)
- Spark-3 subagent's logs/scripts/traces at `/home/user/iq4_tracegen/`
- Pool audit numbers from `repro/scripts/training/audit_pool_completeness.py`

If any number above ages out, re-pull from those paths to verify before acting.
