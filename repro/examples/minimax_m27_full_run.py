"""
End-to-end run: train a DFlash drafter for MiniMax-M2.7 from scratch.

This script is the canonical reference for the full library workflow:
    1. Pick a verifier (model family + GGUF/HF paths)
    2. Generate ~6,500 self-describing fp8 traces over a prompts arrow dataset
    3. trainer.prepare() — assemble paired dataset + vocab maps
    4. trainer.smoke() — 90s plumbing check
    5. trainer.train() — the real 17-epoch run (~4 hours on one GB10)
    6. trainer.offline_eval() — verify the saved checkpoint reproduces training
       accuracy to within ±5pp

Total wall-clock: ~6-7 hours on a single DGX Spark (most of it in step 5).

Prerequisites
-------------
    pip install -e /path/to/dflash-llama
    # plus speculators (for steps 4-6) — pip install speculators
    # plus a working llama-dump-hiddens binary
    # plus the verifier model (GGUF for trace gen, HF for the trainer)
    # plus a tokenized arrow prompts dataset

Edit the constants block, then::

    python minimax_m27_full_run.py
"""
import os
from pathlib import Path

from dflash_llama import DFlashTrainer, TraceGenerator, load_verifier

# ---- environment paths — edit these ----
GGUF       = "/home/user/clawd/iq4_models/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
LLAMA_BIN  = "/home/user/iq4_tracegen/buun-llama-cpp/build/bin/llama-dump-hiddens"
PROMPTS    = "/home/user/iq4_tracegen/prompts_tulu3"
HF_VERIFIER = "/home/user/models/MiniMax-M2.7-FP8"
WORK       = Path("/home/user/dflash_runs/minimax_m27_full")
N_TRACES   = 6500
EPOCHS     = 17
# ----------------------------------------

WORK.mkdir(parents=True, exist_ok=True)

verifier = load_verifier("minimax-m2.7-iq4-xs", gguf_path=GGUF, hf_path=HF_VERIFIER)
print(f"verifier loaded: {verifier.name}  hidden={verifier.hidden_size}  "
      f"vocab={verifier.vocab_size}  layer_ids={verifier.layer_ids}")

# 1. Trace generation
print(f"\n[1/4] generating {N_TRACES} traces over rows [0, {N_TRACES})")
gen = TraceGenerator(
    verifier=verifier,
    storage="fp8_per_tensor_scale",
    backend="llamacpp_gguf",
    backend_kwargs={"binary": LLAMA_BIN},
)
gen_report = gen.generate(
    prompts=PROMPTS,
    output_dir=str(WORK / "traces"),
    rows=range(0, N_TRACES),
    state_path=str(WORK / "trace_state.json"),
    max_seq_len=2048,
)
print(f"  ✓ {gen_report['completed']} completed, {gen_report['failed']} failed")

# 2. Construct trainer + prepare paired dataset + vocab maps
print(f"\n[2/4] building paired dataset + vocab maps")
trainer = DFlashTrainer(
    traces_dir=str(WORK / "traces"),
    verifier=verifier,
    num_layers=5,
    draft_vocab_size=32768,
    paired_dir=str(WORK / "paired"),
)
prep = trainer.prepare()
print(f"  ✓ paired n_rows={prep['assemble']['n_rows']}  "
      f"coverage={prep['vocab_maps']['top_k_coverage_pct']}%")

# 3. Smoke first (90 seconds), abort if it doesn't pass
print(f"\n[3/4] 90-second smoke train")
smoke = trainer.smoke(timeout_sec=90, save_to=str(WORK / "smoke_ckpt"))
if not smoke.passed:
    raise SystemExit(
        f"smoke failed: exit={smoke.exit_code}, global_step={smoke.global_step}; "
        f"see {smoke.log_path}"
    )
print(f"  ✓ smoke passed: global_step={smoke.global_step}")

# 4. Full run
print(f"\n[4/4] full {EPOCHS}-epoch training (~4 hours)")
ckpt_dir = WORK / "checkpoints"
result = trainer.train(
    save_to=str(ckpt_dir),
    epochs=EPOCHS,
    lr=3e-5,
    max_anchors=512,
    log_freq=5,
    scheduler_warmup_steps=100,
    save_best=True,
)
print(f"  ✓ training finished: {result}")

# 5. Offline eval against saved checkpoint
best_ckpt = ckpt_dir / "checkpoint_best"
print(f"\noffline eval against {best_ckpt}")
eval_report = trainer.offline_eval(checkpoint=str(best_ckpt), max_batches=60)
print(f"  ✓ {eval_report}")

print(f"\nDone. Checkpoint: {best_ckpt}")
print(f"Next: convert to GGUF for llama.cpp speculative-decode (see repro/03-inference.md)")
