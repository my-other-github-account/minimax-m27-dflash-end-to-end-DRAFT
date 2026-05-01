# How to launch additional IQ4 trace-gen workers

> **🚨 HARD RULE: NEVER TOUCH SPARK-5. EVER.**
> spark-5 is jumphost-only for spark-1 SSH (since spark-1's Tailscale has been offline 22d+). Do NOT evict processes on spark-5, do NOT run workloads on spark-5, do NOT modify state on spark-5 under any circumstance. Anything that would land there must instead go to spark-2/3/4 or be queued for later.

Workers A, B, C are running on spark-3, spark-2, spark-4 respectively as of 2026-04-30 17:52 PDT, processing disjoint shards of the 6,515-prompt FP8 reference dataset.

Existing 500 traces (indices 0–499) were validated against FP8 reference (cosine ≥ 0.92 all layers, token-ids 100% match) before scaling up.

## Status check (any time)

```bash
# Worker A on spark-3
ssh spark-3 'tmux ls; tail -10 /home/user/iq4_tracegen/logs/worker_A_*.log; ls /home/user/iq4_tracegen/traces/hidden_states/ | wc -l; cat /home/user/iq4_tracegen/state_worker_A.json'
```

## Stop a worker (clean — finishes current trace, exits)

```bash
ssh spark-N 'tmux send-keys -t iq4_worker_<ID> C-c'    # SIGINT, gracefully stops after current trace
# or
ssh spark-N 'pkill -TERM -f gen_traces_v2'             # same effect
```

## Hard kill (loses current trace only — atomic writes ensure no corruption)

```bash
ssh spark-N 'tmux kill-session -t iq4_worker_<ID>'
```

## Resume after stop (no work lost, picks up at next missing index)

Just re-run with the same env:

```bash
ssh spark-N 'WORKER_ID=A START=500 END=2500 bash /home/user/iq4_tracegen/scripts/iq4_worker.sh'
```

## Add a future worker (when shards open up or new shards are needed)

🚨 Reminder: **NEVER on spark-5.** Available targets are spark-2/3/4 (and possibly spark-6 if the user explicitly approves evicting the local MiniMax bot).

The general recipe for any new worker on spark-N (N ∈ {2,3,4,6}):

```bash
# 1. Sync verifier + buun source + prompts + scripts (~5 min over QSFP)
ssh spark-N 'mkdir -p ~/iq4_tracegen/{traces/hidden_states,logs,scripts} ~/clawd/iq4_models/UD-IQ4_XS && \
             tmux new-session -d -s iq4_sync "rsync -av operator@192.168.200.3:/home/user/clawd/iq4_models/UD-IQ4_XS/ ~/clawd/iq4_models/UD-IQ4_XS/ && \
             rsync -av --exclude=build --exclude=.git operator@192.168.200.3:/home/user/iq4_tracegen/buun-llama-cpp/ ~/iq4_tracegen/buun-llama-cpp/ && \
             rsync -av operator@192.168.200.3:/home/user/iq4_tracegen/prompts_fp8/ ~/iq4_tracegen/prompts_fp8/ && \
             rsync -av operator@192.168.200.3:/home/user/iq4_tracegen/scripts/ ~/iq4_tracegen/scripts/ && \
             touch /tmp/iq4_sync.done"'

# 2. Build buun (~5-10 min)
ssh spark-N 'tmux new-session -d -s iq4_build "export PATH=/usr/local/cuda/bin:\$PATH && \
             cd ~/iq4_tracegen/buun-llama-cpp && \
             cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=121a-real && \
             cmake --build build -j 8 --target llama-dump-hiddens && \
             touch /tmp/iq4_build.done"'

# 3. Pick a non-overlapping START/END (verify with the audit recipe below) and launch
ssh spark-N 'WORKER_ID=<unique_letter> START=<some_start> END=<some_end> bash ~/iq4_tracegen/scripts/iq4_worker.sh'
```

## Verify shards are non-overlapping

```bash
# Print every worker's claimed range, plus the actual indices on each machine.
# If any indices appear on multiple machines, traces can collide on rsync-pool
# (atomic-write rule means the LAST writer wins, no corruption — but it IS wasted work).
for h in spark-2 spark-3 spark-4 spark-6; do
  ssh -o ConnectTimeout=5 $h "echo === \$(hostname) ===; \
    cat ~/iq4_tracegen/state_worker_*.json 2>/dev/null | python3 -c 'import sys,json,glob; \
      [print(\"  worker={} range=last_completed_idx={} count={}\".format(p.split(\"_\")[-1].split(\".\")[0], json.load(open(p))[\"last_completed_idx\"], json.load(open(p))[\"completed_count\"])) for p in glob.glob(\"/home/user/iq4_tracegen/state_worker_*.json\")]'; \
    ls ~/iq4_tracegen/traces/hidden_states/ 2>/dev/null | wc -l" 2>&1
done
```

## Pool traces from all workers into one training set

When you want to train on the union, on whatever machine has the training rig:

```bash
mkdir -p ~/iq4_train/pool/hidden_states
# CRITICAL: do NOT include spark-5
for src in 192.168.200.3 192.168.200.2 192.168.200.4; do
  rsync -av operator@${src}:/home/user/iq4_tracegen/traces/hidden_states/ ~/iq4_train/pool/hidden_states/
done
ls ~/iq4_train/pool/hidden_states/ | wc -l   # should equal sum of all workers' counts

# Then re-run filter_dataset.py to build the dense indexed dataset:
python3 ~/iq4_train/scripts/filter_dataset.py \
    --src-prompts ~/iq4_train/prompts \
    --src-traces ~/iq4_train/pool/hidden_states \
    --out-prompts ~/iq4_train/dataset_iq4_pool/prompts \
    --out-traces ~/iq4_train/dataset_iq4_pool/hidden_states

# Then train on the pooled dataset (re-launch iq4_train_smoke.sh with adjusted paths)
```

## Range plan (ACTIVE 2026-04-30 17:52 PDT)

| Worker | Spark | Range | Count | Status |
|---|---|---|---|---|
| (already done) | (was spark-3, now in pool of every worker) | `[0, 500)` | 500 | ✅ |
| **A** | **spark-3** | `[500, 2500)` | 2000 | 🟢 running |
| **B** | **spark-2** | `[2500, 4500)` | 2000 | 🟢 running |
| **C** | **spark-4** | `[4500, 6515)` | 2015 | 🟢 running |
| **Total** | | **`[0, 6515)`** | **6515** | covers full pool |

Workers DO NOT share filesystem — each writes locally to `/home/user/iq4_tracegen/traces/hidden_states/` on its own machine. Pool by rsync only when training.

## Why this design

- **Resumable per doctrine**: state.json with FM27 hash, atomic safetensors writes, skip-existing default, line-buffered logs, SIGTERM-handler to finish current trace.
- **Independently stoppable**: every worker is a tmux session named `iq4_worker_<ID>`. `tmux kill-session` or `pkill -TERM` is fine. State on disk reflects truth.
- **Non-overlapping by construction**: disjoint START/END ranges. No coordination needed.
- **Loss bound**: power flake or kill loses ≤1 trace per worker (the in-flight one). Re-launch picks up at next missing index automatically.
