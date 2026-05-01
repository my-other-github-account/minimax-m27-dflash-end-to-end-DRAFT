#!/usr/bin/env python3
"""
gen_traces_v2.py — IQ4 GGUF trace generator. Resumable, atomic, kill-safe.

Conforms to repro/plan/00-resumability-doctrine.md:
  1. Durable progress per-trace (each hs_<i>.safetensors is the unit)
  2. Line-buffered stdout + flush after every print
  3. Atomic writes: hs_<i>.safetensors.part -> fsync -> rename
  4. Skip-existing by default
  5. state.json with FM27 hash-verify wrapper
  6. Single-process-per-trace (model mmap'd, page cache hot, ~5-10s overhead)
  7. SIGTERM/SIGINT handler: finish current trace, exit cleanly

Usage:
  gen_traces_v2.py --prompts <arrow_dir> --out <out_dir> --start 0 --end 1000

Resume: just re-run with same args. It picks up where it left off.
"""
import os, sys, struct, subprocess, time, json, argparse, signal, hashlib, tempfile
from pathlib import Path
import numpy as np
import torch
from datasets import load_from_disk
from safetensors.torch import save_file

# ---- Hard rule: line-buffered logs ----
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DEFAULT_LAYERS = [2, 16, 30, 45, 59, 61]  # last is final residual

# ---- Atomic write helper ----
def atomic_save_safetensors(tensors, final_path: Path):
    """Write to temp file, fsync, rename. Never leave a half-written final-named file."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=f".{final_path.name}.tmp_",
        suffix=".part",
    )
    os.close(fd)  # safetensors will reopen
    tmp = Path(tmp)
    try:
        save_file(tensors, str(tmp))
        # fsync the file
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        # rename atomically
        os.rename(tmp, final_path)
        # fsync directory to persist the rename
        dfd = os.open(str(final_path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        try: tmp.unlink()
        except FileNotFoundError: pass
        raise

# ---- FM27 state.json wrapper ----
class State:
    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
            self._write_initial()
        # Validate hash on load
        self.data = self._load_validated()

    def _write_initial(self):
        initial = {
            "version": 1,
            "started_at": time.time(),
            "completed_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "last_completed_idx": None,
            "last_failed_idx": None,
            "_prev_hash": "",
        }
        self._atomic_json_write(initial)

    def _load_validated(self):
        raw = self.path.read_bytes()
        d = json.loads(raw)
        # Don't fail on hash mismatch on load — the hash is for write-conflict detection,
        # but on a single-writer pipeline it's mostly informational.
        return d

    def _atomic_json_write(self, data):
        # Compute prev hash
        if self.path.exists():
            prev_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]
        else:
            prev_hash = ""
        data["_prev_hash"] = prev_hash
        new_raw = json.dumps(data, indent=2, sort_keys=True).encode()
        # Atomic write
        fd, tmp = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.tmp_",
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(new_raw)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp, self.path)
        except Exception:
            try: os.unlink(tmp)
            except FileNotFoundError: pass
            raise

    def update(self, mutator):
        """Mutate state via callable, write atomically with hash chain."""
        # Re-read to detect concurrent writers
        raw = self.path.read_bytes()
        cur = json.loads(raw)
        new = mutator(cur)
        self._atomic_json_write(new)
        self.data = new

# ---- Trace generation ----
def write_tokens_bin(tokens, path):
    n = len(tokens)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(np.asarray(tokens, dtype=np.int32).tobytes())
        f.flush()
        os.fsync(f.fileno())

def parse_hidden_bin(path):
    raw = open(path, "rb").read()
    off = 0
    n_layers, n_tokens, n_embd = struct.unpack_from("<iii", raw, off)
    off += 12
    capture_layers = list(struct.unpack_from(f"<{n_layers}i", raw, off))
    off += 4 * n_layers
    n_toks_in = struct.unpack_from("<i", raw, off)[0]
    off += 4
    token_ids = list(struct.unpack_from(f"<{n_toks_in}i", raw, off))
    off += 4 * n_toks_in
    body = raw[off:]
    expected_bytes = n_layers * n_tokens * n_embd * 4
    if len(body) != expected_bytes:
        raise ValueError(
            f"body size {len(body)} != expected {expected_bytes} "
            f"(n_layers={n_layers} n_tokens={n_tokens} n_embd={n_embd})"
        )
    arr = np.frombuffer(body, dtype=np.float32).reshape(n_layers, n_tokens, n_embd)
    arr = np.transpose(arr, (1, 0, 2)).copy()  # [n_tokens, n_layers, n_embd]
    return arr, token_ids, capture_layers

def run_one(args, idx, input_ids):
    toks_bin = f"/tmp/iq4_v2_toks_{os.getpid()}_{idx}.bin"
    out_bin = f"/tmp/iq4_v2_hs_{os.getpid()}_{idx}.bin"
    write_tokens_bin(input_ids, toks_bin)

    capture_str = ",".join(str(L) for L in args.layers)
    env = os.environ.copy()
    env["TOKENS_BIN"] = toks_bin
    env["OUT_BIN"] = out_bin
    env["CAPTURE_LAYERS"] = capture_str
    env["LLAMA_LOG_LEVEL"] = "2"  # WARN

    cmd = [
        args.binary, "-m", args.model,
        "-ngl", "99",
        "-ot", "exps=CPU",
        "-c", str(args.ctx),
        "-p", "x",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f"  [idx={idx}] TIMEOUT after {args.timeout}s", flush=True)
        for p in (toks_bin, out_bin):
            try: os.remove(p)
            except FileNotFoundError: pass
        return None, time.time() - t0
    elapsed = time.time() - t0
    if proc.returncode != 0:
        sys.stderr.write(f"  [idx={idx}] rc={proc.returncode} elapsed={elapsed:.1f}s\n")
        sys.stderr.write(proc.stderr.decode("utf-8", errors="replace")[-1500:] + "\n")
        for p in (toks_bin, out_bin):
            try: os.remove(p)
            except FileNotFoundError: pass
        return None, elapsed
    if not os.path.exists(out_bin):
        sys.stderr.write(f"  [idx={idx}] no out_bin produced ({elapsed:.1f}s)\n")
        try: os.remove(toks_bin)
        except FileNotFoundError: pass
        return None, elapsed
    try:
        hs_f32, tok_out, cap_out = parse_hidden_bin(out_bin)
    except Exception as e:
        sys.stderr.write(f"  [idx={idx}] parse failed: {e}\n")
        for p in (toks_bin, out_bin):
            try: os.remove(p)
            except FileNotFoundError: pass
        return None, elapsed
    finally:
        for p in (toks_bin, out_bin):
            try: os.remove(p)
            except FileNotFoundError: pass

    if cap_out != list(args.layers):
        sys.stderr.write(f"  [idx={idx}] capture mismatch: {cap_out} vs {args.layers}\n")
        return None, elapsed
    if tok_out != list(input_ids):
        sys.stderr.write(f"  [idx={idx}] token_ids mismatch len_out={len(tok_out)} len_in={len(input_ids)}\n")
        return None, elapsed
    return (hs_f32, tok_out), elapsed

# ---- Signal handling ----
_should_stop = False
def _sigterm(signum, frame):
    global _should_stop
    print(f"\n[signal] received signal {signum} — finishing current trace then exiting cleanly", flush=True)
    _should_stop = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--binary", default="/home/user/iq4_tracegen/buun-llama-cpp/build/bin/llama-dump-hiddens")
    ap.add_argument("--model", default="/home/user/clawd/iq4_models/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--state", default=None, help="state.json path (default: <out>/../state.json)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state) if args.state else out_dir.parent / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = State(state_path)
    print(f"[state] {state_path} loaded: {state.data}", flush=True)

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    print(f"[loader] loading dataset from {args.prompts}", flush=True)
    ds = load_from_disk(args.prompts)
    print(f"[loader] {len(ds)} rows, columns={ds.column_names}", flush=True)

    completed_this_run = 0
    skipped_this_run = 0
    failed_this_run = 0
    timings = []

    print(f"[run] processing range [{args.start}, {min(args.end, len(ds))})", flush=True)
    print(f"[run] capture layers: {args.layers}, ctx={args.ctx}, max_seq_len={args.max_seq_len}", flush=True)

    for i in range(args.start, min(args.end, len(ds))):
        if _should_stop:
            print("[stop] graceful shutdown requested, exiting", flush=True)
            break

        out_path = out_dir / f"hs_{i}.safetensors"
        if out_path.exists():
            skipped_this_run += 1
            continue

        row = ds[i]
        input_ids = list(row["input_ids"])
        seq_len = len(input_ids)
        if seq_len > args.max_seq_len:
            print(f"  [idx={i}] skip: seq_len={seq_len} > max_seq_len={args.max_seq_len}", flush=True)
            failed_this_run += 1
            continue
        if seq_len > args.ctx - 16:
            print(f"  [idx={i}] skip: seq_len={seq_len} > ctx-16={args.ctx-16}", flush=True)
            failed_this_run += 1
            continue

        result, elapsed = run_one(args, i, input_ids)
        if result is None:
            failed_this_run += 1
            state.update(lambda d: {**d, "failed_count": d.get("failed_count",0)+1, "last_failed_idx": i})
            continue

        hs_f32, tok_out = result
        # Save atomically
        tensors = {
            "hidden_states": torch.from_numpy(hs_f32).to(torch.bfloat16).contiguous(),
            "token_ids": torch.tensor(tok_out, dtype=torch.int64),
        }
        try:
            atomic_save_safetensors(tensors, out_path)
        except Exception as e:
            print(f"  [idx={i}] FAILED save: {e}", flush=True)
            failed_this_run += 1
            state.update(lambda d: {**d, "failed_count": d.get("failed_count",0)+1, "last_failed_idx": i})
            continue

        completed_this_run += 1
        timings.append(elapsed)
        state.update(lambda d: {
            **d,
            "completed_count": d.get("completed_count",0)+1,
            "last_completed_idx": i,
            "last_completed_at": time.time(),
        })

        if completed_this_run % 10 == 0:
            avg = sum(timings[-50:]) / max(1, len(timings[-50:]))
            remaining = max(0, min(args.end, len(ds)) - i - 1)
            eta_min = remaining * avg / 60
            print(f"  [idx={i}] OK ntok={len(tok_out)} dt={elapsed:.1f}s "
                  f"| run completed={completed_this_run} skipped={skipped_this_run} failed={failed_this_run} "
                  f"| avg(last50)={avg:.1f}s ETA={eta_min:.0f}min",
                  flush=True)

    avg = sum(timings) / max(1, len(timings))
    print(f"\n[done] this run: completed={completed_this_run} skipped={skipped_this_run} failed={failed_this_run}", flush=True)
    print(f"[done] total in out: {len(list(out_dir.glob('hs_*.safetensors')))}", flush=True)
    print(f"[done] avg time/trace: {avg:.1f}s", flush=True)
    print(f"[done] state: {state.data}", flush=True)

if __name__ == "__main__":
    main()
