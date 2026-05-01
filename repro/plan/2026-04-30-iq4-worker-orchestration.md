# How to launch additional IQ4 trace-gen workers

Worker A is already running on spark-3, processing index range `[500, 2500)`.
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

## Add worker B on spark-5

This requires evicting `baby`'s idle llama-server (PID 972562, 116 GB GPU mem, 0% util):

```bash
# 1. Confirm with baby first if they need it. Then:
ssh spark-5 'pkill -TERM -f /home/baby/src/llama.cpp/build-cuda/bin/llama-server'
sleep 5
ssh spark-5 'pgrep -af llama-server'  # verify gone

# 2. Sync IQ4 verifier + buun build + prompts dataset to spark-5 (~15 min over QSFP)
ssh spark-5 'mkdir -p ~/iq4_tracegen/{traces,logs,scripts} ~/clawd/iq4_models/UD-IQ4_XS && \
             rsync -av --info=progress2 operator@192.168.200.3:/home/user/clawd/iq4_models/UD-IQ4_XS/ ~/clawd/iq4_models/UD-IQ4_XS/ && \
             rsync -av --info=progress2 --exclude=build --exclude=.git operator@192.168.200.3:/home/user/iq4_tracegen/buun-llama-cpp/ ~/iq4_tracegen/buun-llama-cpp/ && \
             rsync -av --info=progress2 operator@192.168.200.3:/home/user/iq4_tracegen/prompts_fp8/ ~/iq4_tracegen/prompts_fp8/ && \
             rsync -av operator@192.168.200.3:/home/user/iq4_tracegen/scripts/ ~/iq4_tracegen/scripts/'

# 3. Build buun on spark-5 (~5-10 min)
ssh spark-5 'export PATH=/usr/local/cuda/bin:$PATH && cd ~/iq4_tracegen/buun-llama-cpp && \
             cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
                   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=121a-real && \
             cmake --build build -j 8 --target llama-dump-hiddens'

# 4. Launch worker B on shard [2500, 4500)
ssh spark-5 'WORKER_ID=B START=2500 END=4500 bash ~/iq4_tracegen/scripts/iq4_worker.sh'
```

## Add worker C on spark-6

Same pattern. Need to evict the local MiniMax bot llama-server (port 18080):

```bash
ssh spark-6 'pkill -TERM -f /opt/llama.cpp/build-cuda/bin/llama-server'
# (then steps 2-4 above with WORKER_ID=C START=4500 END=6515)
```

## Pool traces from all workers into one training set

When you want to train on the union, on whatever machine has spark-2's training rig:

```bash
# On the training machine:
mkdir -p ~/iq4_train/pool/hidden_states
for src in 192.168.200.3 192.168.200.5 192.168.200.6; do
  rsync -av operator@${src}:/home/user/iq4_tracegen/traces/hidden_states/ ~/iq4_train/pool/hidden_states/
done
ls ~/iq4_train/pool/hidden_states/ | wc -l   # total traces

# Then re-run filter_dataset.py to build the dense indexed dataset:
python3 /home/user/iq4_train/scripts/filter_dataset.py \
    --src-prompts ~/iq4_train/prompts \
    --src-traces ~/iq4_train/pool/hidden_states \
    --out-prompts ~/iq4_train/dataset_iq4_pool/prompts \
    --out-traces ~/iq4_train/dataset_iq4_pool/hidden_states

# Then train on the pooled dataset (re-launch iq4_train_smoke.sh with adjusted paths)
```

## Range plan (non-overlapping)

| Worker | Spark | Range | Count |
|---|---|---|---|
| (already done) | (was spark-3, now in pool) | `[0, 500)` | 500 |
| A | spark-3 (active) | `[500, 2500)` | 2000 |
| B | spark-5 (when greenlit) | `[2500, 4500)` | 2000 |
| C | spark-6 (when greenlit) | `[4500, 6515)` | 2015 |
| **Total** | | **`[0, 6515)`** | **6515** |

Workers DO NOT share filesystem — each writes locally to its own machine's `/home/user/iq4_tracegen/traces/hidden_states/`. Pool by rsync at the moment of training.

## Why this design

- **Resumable per doctrine**: state.json with FM27 hash, atomic safetensors writes, skip-existing default, line-buffered logs, SIGTERM-handler to finish current trace.
- **Independently stoppable**: every worker is a tmux session named `iq4_worker_<ID>`. `tmux kill-session` or `pkill -TERM` is fine. State on disk reflects truth.
- **Non-overlapping by construction**: disjoint START/END ranges. No coordination needed.
- **Loss bound**: power flake or kill loses ≤1 trace per worker (the in-flight one). Re-launch picks up at next missing index automatically.
