"""
Tiny smoke example — the smallest possible end-to-end demo of dflash_llama.

What this does (in ~5 minutes on a single Spark/GPU):
    1. Generate 10 self-describing fp8 traces from the first 10 prompts of
       a tokenized arrow dataset, using a small llama.cpp GGUF verifier.
    2. Assemble them into a paired prompts dataset.
    3. Build the vocab maps the speculators trainer expects.
    4. Run a 90-second smoke train (proves the pipeline plumbs end-to-end).

This is not a real training run — chain-pos-1 accuracy will be ~5%. The point is
to verify every step of the pipeline works on your specific environment in
under 5 minutes before committing to a multi-hour real run.

Prerequisites
-------------
    pip install -e /path/to/dflash-llama
    # plus a working llama-dump-hiddens binary (a buun-llama-cpp build)
    # plus a tokenized arrow prompts dataset

Edit the paths below to match your environment, then::

    python tiny_smoke.py

You should see all 4 steps log a final ✓ and the script exits 0 in <6 minutes.
"""
import os
from pathlib import Path

from dflash_llama import (
    DFlashTrainer,
    TraceGenerator,
    list_verifiers,
    load_verifier,
)

# --- edit these for your environment ---
GGUF       = "/home/user/clawd/iq4_models/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
LLAMA_BIN  = "/home/user/iq4_tracegen/buun-llama-cpp/build/bin/llama-dump-hiddens"
PROMPTS    = "/home/user/iq4_tracegen/prompts_tulu3"   # any HF Dataset dir with input_ids, loss_mask
HF_VERIFIER = "/home/user/models/MiniMax-M2.7-FP8"     # optional, only needed for smoke train
WORK       = Path("/tmp/dflash_llama_tiny_smoke")
# ---------------------------------------

print(f"available verifiers: {list_verifiers()}")

verifier = load_verifier("minimax-m2.7-iq4-xs", gguf_path=GGUF, hf_path=HF_VERIFIER)
print(f"[1/4] verifier {verifier.name} loaded "
      f"(hidden={verifier.hidden_size}, vocab={verifier.vocab_size})")

# 1. Generate 10 traces
gen = TraceGenerator(
    verifier=verifier,
    storage="fp8_per_tensor_scale",
    backend="llamacpp_gguf",
    backend_kwargs={"binary": LLAMA_BIN},
)
traces_dir = WORK / "traces"
traces_dir.mkdir(parents=True, exist_ok=True)
gen_report = gen.generate(
    prompts=PROMPTS,
    output_dir=str(traces_dir),
    rows=range(0, 10),
    state_path=str(WORK / "state.json"),
    max_seq_len=2048,
)
print(f"[2/4] generated {gen_report['completed']} traces "
      f"(skipped={gen_report['skipped']}, failed={gen_report['failed']})")

# 2. + 3. Build the paired dataset + vocab maps via DFlashTrainer.prepare()
trainer = DFlashTrainer(
    traces_dir=str(traces_dir),
    verifier=verifier,
    num_layers=5,
    draft_vocab_size=32768,
    paired_dir=str(WORK / "paired"),
)
prep = trainer.prepare(force=True)
print(f"[3/4] prepared paired dataset: "
      f"n_rows={prep['assemble']['n_rows']}  "
      f"vocab coverage={prep['vocab_maps']['top_k_coverage_pct']}%")

# 4. 90-second smoke train (uses speculators trainer via torchrun shell-out)
smoke = trainer.smoke(timeout_sec=90, save_to=str(WORK / "smoke_ckpt"))
print(f"[4/4] smoke result: passed={smoke.passed} "
      f"global_step={smoke.global_step} exit_code={smoke.exit_code}")

print("\nAll steps OK — pipeline is plumbed end-to-end.")
print(f"Workspace: {WORK}")
