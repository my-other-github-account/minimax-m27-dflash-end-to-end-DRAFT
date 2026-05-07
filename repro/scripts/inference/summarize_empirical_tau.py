"""
summarize_empirical_tau.py — aggregate the JSONL emitted by
``tau_capture_proxy.py`` into the empirical-tau receipt printed in
repro/04-empirical-tau-llama-benchy.md §4.6.

Per-request tau is computed from llama-server fork timings as::

    rounds_i = predicted_n_i - draft_n_accepted_i
    tau_i    = predicted_n_i / rounds_i  (== 1 + accepted/rounds)

Overall (token-weighted) tau is the same formula on the summed totals.

Usage::

    python3 summarize_empirical_tau.py path/to/empirical_tau_traffic.jsonl
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: summarize_empirical_tau.py <traffic.jsonl>",
              file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    completions = [
        r for r in rows
        if r.get("path", "").endswith("/chat/completions")
        and r.get("status") == 200
        and isinstance(r.get("predicted_n"), int)
        and r["predicted_n"] > 1
        and isinstance(r.get("draft_n_accepted"), int)
        and isinstance(r.get("draft_n"), int)
    ]
    if not completions:
        print("no usable /chat/completions records found", file=sys.stderr)
        return 1

    sum_pred = sum(r["predicted_n"] for r in completions)
    sum_acc = sum(r["draft_n_accepted"] for r in completions)
    sum_drafted = sum(r["draft_n"] for r in completions)
    rounds_total = sum_pred - sum_acc
    tau_overall = sum_pred / rounds_total if rounds_total > 0 else float("nan")

    per_request_tau: list[float] = []
    for r in completions:
        rnd = r["predicted_n"] - r["draft_n_accepted"]
        if rnd > 0:
            per_request_tau.append(r["predicted_n"] / rnd)

    mean = statistics.fmean(per_request_tau)
    median = statistics.median(per_request_tau)
    p10 = _percentile(per_request_tau, 0.10)
    p50 = _percentile(per_request_tau, 0.50)
    p90 = _percentile(per_request_tau, 0.90)
    tmin = min(per_request_tau)
    tmax = max(per_request_tau)

    bins = {"tau<1.5": 0, "1.5<=tau<2.0": 0,
            "2.0<=tau<2.5": 0, "tau>=2.5": 0}
    for t in per_request_tau:
        if t < 1.5:
            bins["tau<1.5"] += 1
        elif t < 2.0:
            bins["1.5<=tau<2.0"] += 1
        elif t < 2.5:
            bins["2.0<=tau<2.5"] += 1
        else:
            bins["tau>=2.5"] += 1

    accept_rate = (sum_acc / sum_drafted) if sum_drafted else float("nan")

    print(f"# Empirical tau — {path.name}")
    print()
    print("| metric | value |")
    print("|---|---|")
    print(f"| n_requests | {len(completions)} |")
    print(f"| predicted_tokens | {sum_pred} |")
    print(f"| draft_n_accepted | {sum_acc} |")
    print(f"| draft_n | {sum_drafted} |")
    print(f"| inferred_rounds | {rounds_total} |")
    print(f"| draft_accept_rate_by_tokens | {accept_rate:.4%} |")
    print(f"| **tau_overall** | **{tau_overall:.4f}** |")
    print(f"| mean_tau | {mean:.4f} |")
    print(f"| median_tau | {median:.4f} |")
    print(f"| p10_tau | {p10:.4f} |")
    print(f"| p50_tau | {p50:.4f} |")
    print(f"| p90_tau | {p90:.4f} |")
    print(f"| min_tau | {tmin:.4f} |")
    print(f"| max_tau | {tmax:.4f} |")
    print()
    print("| bin | count |")
    print("|---|---|")
    for k, v in bins.items():
        print(f"| {k} | {v} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
