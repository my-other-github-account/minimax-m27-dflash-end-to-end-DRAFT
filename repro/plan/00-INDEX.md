# Plan directory

> **🚨 HARD RULE: NEVER TOUCH SPARK-5.** See `00-NEVER-touch-spark-5.md` — applies to every workflow in this project, no exceptions, no negotiations.

Forward-looking notes: doctrine, current state, what's plateaued, what to try next, and what's already in flight.

## Doctrine (read these first — they govern everything)

- **`00-NEVER-touch-spark-5.md`** — operational hard rule. spark-5 is jumphost-only for SSH proxy to spark-1. Do not run workloads on it, do not evict processes on it, do not modify state on it.
- **`00-resumability-doctrine.md`** — every long-running pipeline must be killable at any moment with the guarantee that re-running picks up exactly where it left off. Hard rules for atomic writes, flushed logs, skip-existing defaults, state.json with hash-verified updates, and resume tests. Project doctrine.

## Dated decisions and diagnoses

- **`2026-04-30-full-plateau-and-iq4-pivot.md`** — diagnosis of the FULL training plateau at p_1 ≈ 22.9% (final epoch 13/17) vs production's 28.88% on the same recipe, with the conclusion that data quality (not capacity, not config) is the bottleneck. Documents the IQ4-llama.cpp single-machine trace-gen pivot as a candidate replacement for the FP8 4-machine pipeline.
- **`2026-04-30-iq4-worker-orchestration.md`** — orchestration recipe for parallel trace-gen workers across spark-2/3/4 (NEVER spark-5). Includes status check, stop, resume, add-new-worker, verify-non-overlapping, pool-for-training. Active range plan: workers A/B/C cover `[0, 6515)` with no overlap.
- **`2026-04-30-iq4-gguf-only-end-to-end.md`** — full GGUF-only training pipeline working end-to-end on spark-1. Covers the bridge-tensor extraction from GGUF (Q8_0/Q6_K/F32 → bf16), the train/val split shuffle bug fix, results comparison vs FP8 SMOKE baseline (v2 epoch 4: p_1=13.0% > FP8 SMOKE 9.7%), and the spark-1 disk cleanup that ripped non-genomics data → spark-6 over QSFP. **Pipeline status: working.**

## Conventions

- One file per dated decision/diagnosis. Don't edit old plan files in-place; supersede them with a new dated file that links back.
- Each plan file should answer: (1) what's the problem, (2) what's the evidence, (3) what's the proposed move, (4) what's the success/failure criterion, (5) what's in flight right now.
- Cross-link to `repro/01-generation.md`, `02-training.md`, `03-inference.md` and to skills under `~/.hermes/skills/` when relevant.
- Doctrine files (filenames starting with `00-`) are higher-priority than dated files; they encode standing rules every pipeline must follow.
