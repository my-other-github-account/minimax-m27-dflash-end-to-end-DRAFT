# Plan directory

Forward-looking notes: doctrine, current state, what's plateaued, what to try next, and what's already in flight.

## Doctrine (read these first — they govern everything)

- **`00-resumability-doctrine.md`** — every long-running pipeline must be killable at any moment with the guarantee that re-running picks up exactly where it left off. Hard rules for atomic writes, flushed logs, skip-existing defaults, state.json with hash-verified updates, and resume tests. Project doctrine.

## Dated decisions and diagnoses

- **`2026-04-30-full-plateau-and-iq4-pivot.md`** — diagnosis of the FULL training plateau at p_1 ≈ 22.9% (epoch 13/17) vs production's 28.88% on the same recipe, with the conclusion that data quality (not capacity, not config) is the bottleneck. Documents the IQ4-llama.cpp single-machine trace-gen pivot as a candidate replacement for the FP8 4-machine pipeline, partially executed (500/1000 traces done before subagent context was interrupted).

## Conventions

- One file per dated decision/diagnosis. Don't edit old plan files in-place; supersede them with a new dated file that links back.
- Each plan file should answer: (1) what's the problem, (2) what's the evidence, (3) what's the proposed move, (4) what's the success/failure criterion, (5) what's in flight right now.
- Cross-link to `repro/01-generation.md`, `02-training.md`, `03-inference.md` and to skills under `~/.hermes/skills/` when relevant.
- Doctrine files (filenames starting with `00-`) are higher-priority than dated files; they encode standing rules every pipeline must follow.
