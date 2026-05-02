"""Example: benchmark a DFlash drafter — per-position + chain-cumulative
accept rates with z-scores against training prediction.

Run on spark-1::

    python repro/examples/03_benchmark.py

Wall-clock: ~10 minutes (verifier load is ~3 min, then 3 dmax runs of
~90-120s each at the default n_tokens=384).
"""
from dflash_llama import benchmark

VERIFIER = "/home/user/clawd/iq4_models/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
DRAFTER  = "/home/user/models/MiniMax-M2.7-DFlash-bf16only.gguf"

# val_metrics.json from training — provides the chain-cumulative ∏p_i
# prediction baseline so the report includes z-scores.
VAL_METRICS = "/home/user/bf16only_for_gguf/val_metrics.json"

report = benchmark(
    verifier_gguf=VERIFIER,
    drafter_gguf=DRAFTER,
    val_metrics=VAL_METRICS,
    dmax_sweep=[2, 4, 7],
    n_tokens=384,
    drafter_label="bf16only-may1-ep14",
    progress=True,                  # tqdm bar across the dmax sweep
    log_dir="/home/user/bf16only_bench",
)

# Markdown report (this is the chat-format table)
print(report.markdown())

# Save full structured JSON for cross-run diffs
report.to_json("/home/user/bf16only_bench/report.json")
print("\n[saved /home/user/bf16only_bench/report.json]")

# Programmatic access:
for dmax, run in report.runs.items():
    print(f"dmax={dmax}: chain-pos-1={run['chained'][0]*100:.2f}%, "
          f"chain-pos-2={run['chained'][1]*100:.2f}% "
          f"(predicted {report.training_chained[1]*100:.2f}%)")
