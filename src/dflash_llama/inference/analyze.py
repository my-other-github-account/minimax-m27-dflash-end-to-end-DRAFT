"""Parse ``llama-speculative-simple`` logs into structured per-position +
chained acceptance metrics, with z-scores against training prediction.

Public surface::

    parse_speculative_log(log_path) -> dict
    SpeculativeReport             — aggregate report across dmax sweep
    chain_pred_from_val(val_path) -> (per_pos, chained, val_loss)
    z_score(measured, predicted, n_samples)
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


def chain_pred_from_val(val_path: str | Path) -> tuple[list[float], list[float], Optional[float]]:
    """Read training val_metrics.json → (per-position p_i, chained ∏p_i, val_loss).

    Returns lists of length 7 (block_size-1).
    """
    m = json.load(open(val_path))
    p = []
    i = 1
    while True:
        key = f"position {i} acc_epoch"
        if key not in m:
            break
        p.append(float(m[key]))
        i += 1
    chain, acc = [], 1.0
    for pi in p:
        acc *= pi
        chain.append(acc)
    return p, chain, m.get("loss_epoch")


def z_score(measured: float, predicted: float, n: int) -> float:
    """Binomial-proportion z-score: (measured-predicted) / sqrt(p(1-p)/n)."""
    if predicted <= 0 or predicted >= 1 or n <= 0:
        return float("nan")
    se = math.sqrt(predicted * (1 - predicted) / n)
    return (measured - predicted) / se if se > 0 else float("nan")


_PROMPT_TPS_RE = re.compile(
    r"prompt eval time =\s+([\d.]+)\s+ms /\s+(\d+) tokens "
    r"\(\s+[\d.]+\s+ms per token,\s+([\d.]+)\s+tokens per second\)"
)
_GEN_TPS_RE = re.compile(
    r"\beval time =\s+([\d.]+)\s+ms /\s+(\d+) runs"
)
_HIST_BLOCK_RE = re.compile(
    r"rejection histogram[^\n]*\n((?:  pos\s+\d+:\s+\d+[^\n]*\n)+)\s*all ok:\s+(\d+)"
)
_POS_LINE_RE = re.compile(r"pos\s+\d+:\s+(\d+)")
_N_DRAFTED_RE = re.compile(r"n_drafted\s*=\s*(\d+)")
_N_ACCEPT_RE = re.compile(r"n_accept\s*=\s*(\d+)")
_DFLASH_STATS_RE = re.compile(
    r"statistics dflash:\s*#calls\([^)]*\)\s*=\s*\d+\s+(\d+)\s+\d+,\s*"
    r"#gen drafts\s*=\s*(\d+)"
)


def parse_speculative_log(log_path: str | Path) -> dict:
    """Parse a llama-speculative-simple log file.

    Returns dict with::

        n_iter           # total speculative rounds (= sum(rej) + all_ok)
        rej              # list of rejection counts per position
        all_ok           # rounds where every drafted token was accepted
        n_drafted        # raw token count proposed (= sum(dmax * rounds))
        n_accept         # raw token count accepted
        prompt_tps       # prompt-eval throughput (tokens/sec)
        dflash_path_fired  # bool — DFlash stats counter present
        dflash_n_calls   # number of DFlash calls in inner statistics block
    """
    txt = Path(log_path).read_text()

    m_hist = _HIST_BLOCK_RE.search(txt)
    if not m_hist:
        return {"_error": "no rejection histogram block found", "raw_len": len(txt)}

    rej = [int(x) for x in _POS_LINE_RE.findall(m_hist.group(1))]
    all_ok = int(m_hist.group(2))
    n_iter = sum(rej) + all_ok

    n_drafted = int(_N_DRAFTED_RE.search(txt).group(1)) if _N_DRAFTED_RE.search(txt) else None
    n_accept = int(_N_ACCEPT_RE.search(txt).group(1)) if _N_ACCEPT_RE.search(txt) else None

    prompt_tps = None
    m_p = _PROMPT_TPS_RE.search(txt)
    if m_p:
        prompt_tps = float(m_p.group(3))

    m_d = _DFLASH_STATS_RE.search(txt)
    dflash_path_fired = m_d is not None
    dflash_n_calls = int(m_d.group(1)) if m_d else None

    return {
        "n_iter": n_iter,
        "rej": rej,
        "all_ok": all_ok,
        "n_drafted": n_drafted,
        "n_accept": n_accept,
        "prompt_tps": prompt_tps,
        "dflash_path_fired": dflash_path_fired,
        "dflash_n_calls": dflash_n_calls,
    }


def chain_measured(parsed: dict) -> list[float]:
    """Convert rejection histogram → measured chain-cumulative accept rate per position.

    chain[k] = (n_iter - sum(rej[0..k])) / n_iter
    """
    n = parsed["n_iter"]
    if n <= 0:
        return []
    out = []
    cum = 0
    for r in parsed["rej"]:
        cum += r
        out.append((n - cum) / n)
    return out


def per_position_conditional(parsed: dict) -> list[dict]:
    """Per-position conditional p_k: P(accept pos k | reached pos k).

    Returns one dict per measured position with keys
    ``{position, n_reached, n_accepted, p_k}``.
    """
    rows = []
    n_reached = parsed["n_iter"]
    if n_reached <= 0:
        return rows
    for k, rej_k in enumerate(parsed["rej"]):
        n_acc_k = n_reached - rej_k
        p_k = n_acc_k / n_reached if n_reached > 0 else 0.0
        rows.append({
            "position": k + 1,           # training nomenclature: 1-indexed
            "n_reached": n_reached,
            "n_accepted": n_acc_k,
            "p_k": p_k,
        })
        n_reached -= rej_k
    return rows


# --- aggregate report -----------------------------------------------

@dataclass
class SpeculativeReport:
    """Aggregate across a dmax sweep against a single drafter checkpoint."""

    drafter_label: str
    val_loss: Optional[float] = None
    training_per_pos: list[float] = field(default_factory=list)
    training_chained: list[float] = field(default_factory=list)
    runs: dict[int, dict] = field(default_factory=dict)  # dmax -> parsed log + derived

    def add_run(self, dmax: int, parsed: dict):
        derived = dict(parsed)
        derived["per_position"] = per_position_conditional(parsed)
        derived["chained"] = chain_measured(parsed)
        # z-scores per position using training prediction where available
        for row in derived["per_position"]:
            k = row["position"] - 1
            if k < len(self.training_per_pos):
                row["training_p_k"] = self.training_per_pos[k]
                row["z"] = z_score(row["p_k"], self.training_per_pos[k], row["n_reached"])
            else:
                row["training_p_k"] = None
                row["z"] = None
        derived["chained_z"] = []
        for k, ch in enumerate(derived["chained"]):
            pred = self.training_chained[k] if k < len(self.training_chained) else None
            derived["chained_z"].append({
                "position": k + 1,
                "chained": ch,
                "predicted": pred,
                "z": z_score(ch, pred, parsed["n_iter"]) if pred else None,
            })
        self.runs[dmax] = derived

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Optional[str | Path] = None, indent: int = 2) -> str:
        s = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            Path(path).write_text(s)
        return s

    def markdown(self) -> str:
        """Multi-line markdown table — same format we report in chat."""
        out = []
        out.append(f"## DFlash speculative-decode report — {self.drafter_label}")
        if self.val_loss is not None:
            out.append(f"_Training val_loss = {self.val_loss:.4f}_")
        if self.training_per_pos:
            tp = "  ".join(f"p_{i+1}={p*100:.2f}%" for i, p in enumerate(self.training_per_pos))
            out.append(f"_Training per-position conditional: {tp}_")
        out.append("")

        dmaxes = sorted(self.runs.keys())
        if not dmaxes:
            out.append("(no runs recorded)")
            return "\n".join(out)

        # Per-position conditional table
        out.append("### Per-position conditional p_k (accept rate given chain reached k)")
        header = "| Position | training | " + " | ".join(f"dmax={d}" for d in dmaxes) + " |"
        sep = "|---" * (2 + len(dmaxes)) + "|"
        out.append(header)
        out.append(sep)
        max_pos = max(len(self.runs[d]["per_position"]) for d in dmaxes)
        for k in range(max_pos):
            row = [f"pos {k+1}"]
            tpk = self.training_per_pos[k] if k < len(self.training_per_pos) else None
            row.append(f"{tpk*100:.2f}%" if tpk is not None else "—")
            for d in dmaxes:
                pp = self.runs[d]["per_position"]
                if k < len(pp):
                    pk = pp[k]["p_k"]
                    z = pp[k].get("z")
                    n = pp[k]["n_reached"]
                    if z is not None and not math.isnan(z):
                        row.append(f"{pk*100:.2f}% (n={n}, z={z:+.2f})")
                    else:
                        row.append(f"{pk*100:.2f}% (n={n})")
                else:
                    row.append("—")
            out.append("| " + " | ".join(row) + " |")
        out.append("")

        # Chained
        out.append("### Chain-cumulative ∏ p_i (training prediction)")
        out.append(header)
        out.append(sep)
        max_pos2 = max(len(self.runs[d]["chained_z"]) for d in dmaxes)
        for k in range(max_pos2):
            row = [f"chain-pos-{k+1}"]
            tpc = self.training_chained[k] if k < len(self.training_chained) else None
            row.append(f"{tpc*100:.3f}%" if tpc is not None else "—")
            for d in dmaxes:
                ch = self.runs[d]["chained_z"]
                if k < len(ch):
                    cm = ch[k]["chained"]
                    z = ch[k].get("z")
                    if z is not None and not math.isnan(z):
                        row.append(f"{cm*100:.2f}% (z={z:+.2f})")
                    else:
                        row.append(f"{cm*100:.2f}%")
                else:
                    row.append("—")
            out.append("| " + " | ".join(row) + " |")
        out.append("")

        # Throughput / sample sizes
        out.append("### Sample sizes & DFlash sanity")
        out.append("| dmax | n_rounds | n_drafted | n_accept | prompt_tps | dflash_fired |")
        out.append("|---|---|---|---|---|---|")
        for d in dmaxes:
            r = self.runs[d]
            out.append(
                f"| {d} | {r['n_iter']} | {r['n_drafted']} | {r['n_accept']} | "
                f"{r['prompt_tps']:.2f} t/s | {'✓' if r['dflash_path_fired'] else '✗'} |"
            )
        return "\n".join(out)


__all__ = [
    "parse_speculative_log",
    "chain_pred_from_val",
    "chain_measured",
    "per_position_conditional",
    "z_score",
    "SpeculativeReport",
]
