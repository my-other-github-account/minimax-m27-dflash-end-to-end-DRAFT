# Resumability is a hard requirement (project doctrine)

> **🚨 OPERATIONAL HARD RULE (PROJECT-WIDE): NEVER TOUCH SPARK-5.**
> spark-5 is jumphost-only for spark-1 SSH (since spark-1's Tailscale has been offline 22d+). Do NOT evict processes on, run workloads on, or modify state on spark-5 under any circumstance. This rule is older and more important than every other rule in this document. Any work that would land there must instead go to spark-2/3/4/6 or be queued for later.

> Effective immediately. Applies to every long-running pipeline, batch job, training run, trace generation, conversion, sync, and cron loop in this project.

## The rule

**Every process that runs longer than 30 seconds must be killable at any moment with the guarantee that re-running picks up exactly where it left off, with zero work redone and zero work lost beyond the in-flight unit.**

If a power flake, ssh disconnect, OOM kill, or `Ctrl-C` happens during a 4-hour job, we lose at most one work unit (one trace, one batch, one epoch step) — never hours.

## Why this matters

We've already lost time to:

1. Subagent context-window interruption mid-trace-gen → process kept running blindly, hung on something obscure with no log output, scratch dir got full of stale `.bin` files (50% complete state but no clean way to know what was actually finalized).
2. SIGHUP killing FULL training launches when the local watcher died → had to migrate to `systemd-run --user`.
3. `gen_traces_batch.py` design that loaded all 1000 in one process — when it stalled at trace 503, no progress was visible because logs were buffered, and there was no "trace 502 is done, 503 in flight" record on disk.

The pattern is always the same: large monolithic processes that hold state in RAM, write logs through pipes that get buffered, and produce final output only at the end. They're fast when they work and catastrophic when they don't.

## The hard rules

### 1. Durable progress every small N

Whatever the unit of work is (a trace, an epoch step, a sample), write a durable on-disk artifact every N units, with N ≤ 10. Smaller is better. The artifact must be:

- A real file on a real filesystem (not just an in-memory log buffer)
- Self-contained for that unit (one trace per file, not "all traces concatenated")
- Atomically written (see rule 3)

For batch jobs: emit the per-unit artifact AS the unit completes, not in a final flush.

### 2. Logs flush every line

Every Python script: `print(..., flush=True)` for every status line. Or `sys.stdout.reconfigure(line_buffering=True)` at the top.

Every shell pipeline: use `tee` not redirection, and prefix the command with `stdbuf -oL` or `python3 -u`.

Every C++ binary launched as subprocess: pass `bufsize=0` or `bufsize=1` to `subprocess.run`/`Popen`.

If we can't see live progress in the log file via `tail -f`, the rule is broken.

### 3. Atomic writes

Never write directly to the final path. Pattern:

```python
import os, tempfile
def atomic_write(path, write_fn):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise
```

Same pattern for safetensors: write to `*.safetensors.part`, fsync, rename.

A killed process must never leave a half-written final-named file that fools the next run into thinking work is done.

### 4. Skip-existing by default

Every batch tool must check whether the final output for unit `i` already exists. If yes, skip, increment `skipped`, move on. Never recompute completed work without an explicit `--force` flag.

### 5. Per-workflow `state.json`

Every multi-step pipeline writes a JSON state file at predictable path (e.g., `<workdir>/state.json`) updated after each meaningful action. Use the FM27 hash-verify pattern:

```python
def update_state(path, mutator):
    raw = open(path, "rb").read()
    old_hash = hashlib.sha256(raw).hexdigest()[:16]
    state = json.loads(raw)
    state = mutator(state)
    state["_prev_hash"] = old_hash
    new_raw = json.dumps(state, indent=2, sort_keys=True).encode()
    atomic_write(path, lambda f: f.write(new_raw))
```

Resume reads `state.json` first, validates `_prev_hash` matches what's actually on disk, and uses the recorded progress to decide what to do next.

### 6. Avoid all-in-one-process designs

If the design says "load model once, run all 1000 traces", at least split it into chunks of 50 (or whatever) so a stall is bounded. Better: chunk to 1 (single-process-per-unit) unless model load amortization is genuinely the bottleneck (and even then, use a long-lived worker that writes per-unit checkpoints).

If the cold-start cost is, say, 30s per process and you have 1000 units, that's an extra 8 hours of overhead — but only if the per-unit cost is comparable. If the per-unit cost is 5s, single-process-per-unit overhead is 6× wallclock and not worth it. In that case: long-lived worker, durable per-unit checkpoint after every unit, kill-safe by design.

### 7. Resume tests are mandatory

Before declaring a pipeline done, do this test:

1. Start the pipeline.
2. Wait for it to make some progress (e.g., 10% done).
3. `kill -9` it.
4. Re-run with same args.
5. Verify it completes correctly with the right final output count.

If it doesn't pass this test, the pipeline isn't done.

## Patterns we use

### Single-trace-per-process (proven)

`gen_traces.py` is correct: each trace = one subprocess invocation of `llama-dump-hiddens`, writes one `hs_<i>.safetensors`, exits. Skip-existing on outer loop. Killable at any time, lose at most the in-flight trace.

Pays a ~5-10s mmap-page-cache-warmup cost per call but the model file is hot in page cache after the first invocation, so cost stays bounded. Net: 1000 traces in ~1.5 hours instead of theoretical 50 minutes batched, but never loses progress.

### Long-lived worker with per-unit checkpoint (when amortization matters)

If we ever need the batched pattern for amortization, the contract is:

- Worker reads `state.json` on startup, finds last completed unit.
- For each new unit: process → write `hs_<i>.safetensors.part` → fsync → rename → update state.json → flush stdout.
- Heartbeat to log every unit, including timing.
- SIGTERM handler: finish current unit, exit cleanly. Don't try to be cute with multi-unit transactions.

### Cron loop with durable state

`hermes-session-proxy-cron` skill pattern. Each tick reads state.json, does ONE meaningful action, writes state.json with hash verification, exits. Can be killed mid-tick and the next tick picks up.

## What this means for in-flight work

- ✅ FULL training on spark-4: epoch checkpoints under `checkpoint_best/N/` are durable. `--save-best` writes after each new-best val. Killing it loses the current epoch's progress (≤14 min). Acceptable.
- ❌ The original `gen_traces_batch.py` on spark-3 violated this — that's why it stalled invisibly for 2 hours. Killed and replaced with `gen_traces.py` (single-process-per-trace).
- ✅ The `hermes-session-proxy-cron` based loops (`dflash-tracegen-tp4-loop`, etc.) follow the pattern by design: state.json + COMMANDS.md + TASKS.md, FM27 hash-verify, one action per tick.

## Audit checklist for new pipelines

Before merging any new pipeline script:

- [ ] Every print has `flush=True` or stdout is line-buffered
- [ ] Every batch loop checks `if final_output.exists(): skipped += 1; continue` at the top
- [ ] Every output file is written atomically (tmp + fsync + rename)
- [ ] If there's a state.json, all writes go through the FM27 hash-verify wrapper
- [ ] Resume test passed: start → kill -9 at 10% → re-run → completes correctly
- [ ] No "load all data into RAM, process, write at end" pattern
- [ ] Logs visible via `tail -f` show real-time progress
- [ ] Process can be killed cleanly with `pkill -TERM`; SIGTERM handler completes current unit and exits
