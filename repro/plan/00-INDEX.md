# Plan directory

Forward-looking notes: where the project is, what's plateaued, what to try next, and what's already in flight. Each file in this directory is dated and stands on its own.

## Files

- **`2026-04-30-full-plateau-and-iq4-pivot.md`** — diagnosis of the FULL training plateau at p_1 ≈ 22.9% (epoch 13/17) vs production's 28.88% on the same recipe + status of the IQ4-llama.cpp trace-gen experiment intended to test whether smaller, cleaner trace data beats larger, noisier trace data.

## Conventions

- One file per dated decision/diagnosis. Don't edit old plan files in-place; supersede them with a new dated file that links back.
- Each plan file should answer: (1) what's the problem, (2) what's the evidence, (3) what's the proposed move, (4) what's the success/failure criterion, (5) what's in flight right now.
- Cross-link to `repro/01-generation.md`, `02-training.md`, `03-inference.md` and to skills under `~/.hermes/skills/` when relevant.
