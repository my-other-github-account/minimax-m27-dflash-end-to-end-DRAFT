#!/usr/bin/env python3
"""
Compute chain-cumulative measured-vs-predicted accept rates from the
llama-speculative-simple rejection histograms, and print z-scores.

Usage:
  python3 compute_chain_z.py VAL_METRICS_JSON LOG_DMAX2 LOG_DMAX4 LOG_DMAX7

VAL_METRICS_JSON: speculators training output, e.g.
  /home/user/dflash_minimax/checkpoints/<run>/checkpoint_best/val_metrics.json
LOG_DMAX{2,4,7}: speculative-simple stderr+stdout log files.
"""
import sys, json, math, re
from pathlib import Path

def chain_predict(metrics):
    p = []
    for i in range(1, 8):
        k = f"position {i} acc_epoch"
        if k in metrics:
            p.append(metrics[k])
    chain = []
    acc = 1.0
    for pi in p:
        acc *= pi
        chain.append(acc)
    return p, chain

def parse_log(path):
    """Returns dict with n (drafts), rej_pos (list), all_ok, tps."""
    text = Path(path).read_text()
    rej = []
    all_ok = None
    n = None
    tps = None
    in_hist = False
    for line in text.splitlines():
        if "rejection histogram" in line:
            in_hist = True
            continue
        if in_hist:
            m = re.match(r"\s+pos\s+(\d+):\s+(\d+)\s+\(\s*([\d.]+)%\)", line)
            if m:
                rej.append(int(m.group(2)))
                continue
            m = re.match(r"\s+all ok:\s+(\d+)", line)
            if m:
                all_ok = int(m.group(1))
                in_hist = False
                continue
            if line.strip() == "":
                continue
        m = re.search(r"statistics dflash:.*#calls\(b,g,a\)\s*=\s*\d+\s+(\d+)", line)
        if m:
            n = int(m.group(1))
        m = re.search(r"decoded\s+\d+\s+tokens in\s+[\d.]+\s+seconds, speed:\s+([\d.]+)\s+t/s", line)
        if m:
            tps = float(m.group(1))
    if n is None:
        # Fallback: total drafts = sum of all positions including all-ok
        n = sum(rej) + (all_ok or 0)
    return {"n": n, "rej_pos": rej, "all_ok": all_ok, "tps": tps}

def chain_measured(n, rej_pos):
    out = []
    cum_rej = 0
    for r in rej_pos:
        cum_rej += r
        out.append((n - cum_rej) / n if n > 0 else 0.0)
    return out

def fmt_z(meas, pred, n):
    if pred <= 0 or pred >= 1 or n == 0: return f"{meas*100:>6.2f}%"
    se = math.sqrt(pred * (1-pred) / n)
    z = (meas - pred) / se
    return f"{meas*100:>6.2f}% (z={z:+.2f})"

def main():
    val_path = sys.argv[1]
    logs = sys.argv[2:]  # in order: dmax2, dmax4, dmax7
    metrics = json.loads(Path(val_path).read_text())
    p, chain_pred = chain_predict(metrics)

    print(f"Source: {val_path}")
    print(f"Per-position conditional p_i: " + " ".join(f"{x*100:.2f}%" for x in p))
    print(f"Chain-cumulative prediction (∏ p_i):")
    for i, c in enumerate(chain_pred, 1):
        print(f"  pos{i}: {c*100:.4f}%")
    print()

    runs = []
    labels = ["dmax=2", "dmax=4", "dmax=7"]
    for log, label in zip(logs, labels):
        runs.append((label, parse_log(log)))

    header = f"{'Metric':<12} {'predicted':>10}"
    for label, _ in runs:
        header += f"  {label+' meas':>20}"
    print(header)
    print("-" * len(header))

    # chain-pos-1 .. chain-pos-7
    for k in range(7):
        if k >= len(chain_pred):
            break
        row = f"{'chain-pos'+str(k+1):<12} {chain_pred[k]*100:>9.3f}%"
        any_data = False
        for label, run in runs:
            cm = chain_measured(run["n"], run["rej_pos"])
            if k < len(cm):
                row += f"  {fmt_z(cm[k], chain_pred[k], run['n']):>20}"
                any_data = True
            else:
                row += f"  {'n/a':>20}"
        if any_data:
            print(row)

    print()
    tps_row = f"{'Throughput':<12} {'(AR base)':>10}"
    for label, run in runs:
        tps = run.get("tps")
        tps_row += f"  {(str(tps)+' t/s' if tps else '?'):>20}"
    print(tps_row)

if __name__ == "__main__":
    main()
