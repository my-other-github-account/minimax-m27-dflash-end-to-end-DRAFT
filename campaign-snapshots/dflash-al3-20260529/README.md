# MiniMax-M2.7 DFlash — End-to-End Campaign Bundle (DRAFT)

This branch is a **knowledge + artifact snapshot** of the dflash-al3 speculative-decoding campaign
(getting a DFlash drafter for MiniMax-M2.7 to a real accepted-length AL >= 3.0 with measured speedup).
It is checked in so the investigation state is not lost. It does **not** contain model weights / GGUFs
(those live on the spark cluster); it contains every task spec, the full orchestration thread, the
spark-side analysis reports, and a written findings summary.

## Contents

- `kanban/` — all 90 task specs + comments + events from the dflash-al3 board
  (`00_BOARD_OVERVIEW.txt` is the index). This is the full audit trail: every measurement task,
  its exact runner args, results, and the driver/worker discussion thread.
- `spark3_reports/` — text reports, parse JSON, launch scripts, and AL-run logs harvested from
  `spark-3:/home/dnola/v18_fix/` (T28/T54/T65/T69/T70/T71/T73/T75/T78/T80/T83/T86 + eval scripts).
- `analysis/FINDINGS.md` — **the important part**: the session analysis that exists nowhere else —
  train-side AL_LO/AL_HI derivation, the AL-inflation (denominator) red flag, the wrong-model catch,
  and the open verification plan.
- `analysis/AL_bounds.py` — reproducible script computing AL_LO / AL_HI from per-position accept accs.

## Status at snapshot time (2026-05-29)

- Current-best **measured** runtime AL on OUR step500 adapter: ~2.43 (5-prompt repro, dmax13) to
  2.92 (thin 3-prompt, dmax15). A separate "3.39" came from the **older step2500 export, NOT our
  adapter**, at dmax17.
- **These numbers are under active falsification.** Runtime AL exceeds the step500 model's own
  teacher-forced ceiling (AL_HI ≈ 2.20), which is a red flag for a measurement artifact, not a win.
  See `analysis/FINDINGS.md`.
- Two verification tasks open: `t_e6dde2c8` (in-dist Tulu3 AL + denominator audit + lossless-output
  check) and `t_d5077e99` (measured wall-clock speedup vs no-draft baseline).

## NOT included (by design)
- GGUF / safetensors weights, hidden-state traces, the consolidated `.arrow` dataset (all large,
  all on-cluster). Paths to them are recorded in the reports so they can be re-fetched.
