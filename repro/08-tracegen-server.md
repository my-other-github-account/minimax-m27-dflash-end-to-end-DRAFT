# §8 — Persistent batched-decode trace server (~2.5× speedup)

## TL;DR

Replace the one-prompt-per-process spawn pattern with a **persistent
batched-decode trace server**. Same self-describing fp8 output format;
numerically equivalent hidden states (max abs diff = 0 against the
single-prompt reference); **2.56× throughput on MiniMax-M2.7 UD-IQ4_XS**
on a DGX Spark GB10.

| backend | throughput on 100-prompt MiniMax-M2.7 sample | notes |
|---|---|---|
| `llamacpp_gguf` (REMOVED — historical, fork-per-prompt) | ~26.7 traces/min | reloaded 215 GB of weights per prompt; deleted from the library |
| `tracegen_client` (this doc, single-seq) | ~33 traces/min | one persistent worker, one seq at a time |
| `tracegen_client` (this doc, batch width 4, same-length) | **~68 traces/min** | shipped — verified 10/10 bit-identical |

Hardware: 1× DGX Spark GB10, sm_121a, 124 GB VRAM, MoE experts on CPU
(`-ot exps=CPU`), bf16 → fp8 saturating-cast storage. Prompt sample:
`50 <= seq_len <= 96`, `np.random.default_rng(20260518).permutation(...)[:100]`
drawn from `iq4_tracegen_v13_pool/traces/`.

## Why batching is fast for this workload

A single trace generation request loads tokens → runs prefill → captures
hidden states at the configured layers → writes a `hs_<i>.safetensors`
file. The library used to ship a `llamacpp_gguf` backend that spawned a
fresh `llama-dump-hiddens` process per prompt. That meant:

- The entire ~215 GB MiniMax-M2.7 UD-IQ4_XS GGUF is `mmap`-faulted into
  the CPU-side MoE expert buffer **on every request**.
- The CUDA graph for the dense layers is reserved and discarded **on
  every request**.
- The fp8 cast + safetensors write happens serially with the next
  prompt's process launch.

The persistent server keeps the model loaded once and runs one
`llama_decode()` over multiple sequences with distinct `seq_id`s. For
prompts of equal length, prefill compute fuses cleanly across the batch.

## How equivalence is preserved

Multi-prompt decoding in llama.cpp routes each prompt to its own KV-cache
slot keyed by `seq_id`. As long as `n_seq_max >= batch_width`, the
sequences never share state. The fix that unlocks correct batching is a
single line in the worker:

```cpp
// vendor/dump-hiddens/dump_hiddens_worker.cpp
auto cparams = common_context_params_to_llama(params);
cparams.n_seq_max = 8;  // enable batched decoding up to width 8
llama_context * ctx = llama_init_from_model(model, cparams);
```

Without this, `llama_decode` returns `init: invalid seq_id[1][0] = 1 >= 1`
the moment a batch contains a second sequence — Codex hit this bug
silently and produced corrupted batches that almost matched the
single-prompt output but had divergence at exactly the token where the
shorter sequence dropped from the batch. With `n_seq_max=8`, the
hidden-state tensor coming out of the batched path is **bit-identical
(max abs diff = 0.0, cosine ≥ 1.0001)** to the single-prompt reference
across all 10 prompts we validated.

For non-equal-length prompts, the worker has a length-bucketed
`decode_batch_padded_pair` path. The default policy is **same-length
grouping only**, which keeps numerics provably identical. The padded-pair
path is opt-in and not currently used by the default trace generator.

## CLI

Start the server (long-running):

```bash
# In one shell (or under tmux / systemd / a process manager):
dflash-llama trace-server \
    --gguf-path /path/to/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
    --layer-ids 2,16,30,45,59,61 \
    --socket unix:///tmp/dflash_tracegen.sock \
    --ctx 4096 --ngl 99 \
    --binary $(bash scripts/build_llama_dump_hiddens.sh | tail -1 | xargs dirname)/llama-dump-hiddens-worker \
    --override-tensor 'exps=CPU' \
    --log /tmp/dflash_tracegen_worker.log
```

Generate against it (in another shell, or as a follow-on script):

```bash
dflash-llama generate \
    --verifier minimax-m2.7-iq4-xs \
    --gguf-path /path/to/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
    --backend tracegen_client \
    --socket unix:///tmp/dflash_tracegen.sock \
    --prompts /path/to/prompts_arrow_dir \
    --rows 0:1000 \
    --out /path/to/output/traces \
    --state /path/to/output/state.json
```

Or auto-start the server from `generate` itself with
`--auto-start-server` — useful for one-shot jobs:

```bash
dflash-llama generate \
    --verifier minimax-m2.7-iq4-xs \
    --gguf-path /path/to/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
    --backend tracegen_client \
    --auto-start-server \
    --server-log /tmp/dflash_tracegen_worker.log \
    --prompts /path/to/prompts_arrow_dir \
    --rows 0:1000 \
    --out /path/to/output/traces
```

## Python API

```python
from dflash_llama import TraceClient, load_verifier, TraceGenerator

verifier = load_verifier(
    "minimax-m2.7-iq4-xs",
    gguf_path="/path/to/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf",
)
gen = TraceGenerator(
    verifier=verifier,
    storage="fp8_per_tensor_scale",
    backend="tracegen_client",
    backend_kwargs={
        "socket_path": "unix:///tmp/dflash_tracegen.sock",
        "auto_start": True,  # spawn the server lazily if no socket is listening
        "binary": "llama-dump-hiddens-worker",
        "request_timeout": 600.0,
    },
)
gen.generate(
    prompts="/path/to/prompts_arrow_dir",
    output_dir="/path/to/output/traces",
    rows=range(0, 1000),
    state_path="/path/to/output/state.json",
    max_seq_len=2048,
)
```

For a long-lived service, talk to the server directly without going
through `TraceGenerator`:

```python
from dflash_llama import TraceClient

with TraceClient(
    socket_path="unix:///tmp/dflash_tracegen.sock",
    auto_start=True,
    gguf_path="/path/to/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf",
    layer_ids=[2, 16, 30, 45, 59, 61],
    binary="llama-dump-hiddens-worker",
) as client:
    # one prompt at a time
    result = client.dump_hiddens(input_ids=ids, max_seq_len=2048)
    hs = result["hidden_states"]  # torch.Tensor [seq, n_layers, hidden]

    # batched (all prompts MUST be the same length on the default path)
    results = client.dump_hiddens_many(
        batch_inputs=[ids1, ids2, ids3, ids4],
        max_seq_len=2048,
    )
```

## Numerical equivalence check

For every commit that touches `dump_hiddens_worker.cpp`, the batched
output must be re-verified against the single-prompt reference on the
fixed 100-prompt sample. The check is intentionally weaker than sha256
(GPU reductions are non-deterministic across batch composition) but
strong enough to catch real state-leakage bugs:

| metric | threshold | what it catches |
|---|---|---|
| max element-wise abs diff | < 1e-2 | KV-cache cross-contamination |
| mean element-wise abs diff | < 1e-3 | systematic bias from a wrong attention path |
| cosine similarity (flattened) | > 0.9999 | distributional drift |

Empirically, the shipped batched path lands at **max=0, mean=0,
cos≈1.0001** for every prompt in the sample, i.e. truly bit-identical
on the standard 100-prompt sample (the cosine reading slightly above
1.0 is float epsilon from the dot/normalize formulation, not a real
deviation).

## Reproduction harness

The benchmark script that produced the headline number is
`repro/scripts/bench_tracegen_speedup.py`. It:

1. Selects the standard 100-prompt sample (deterministic seed).
2. Generates 10 reference traces with the single-prompt server.
3. Generates the same 10 traces with the batched server, width 4.
4. Asserts the four numerical-equivalence metrics above.
5. If the gate passes, runs a 100-prompt timing benchmark and prints
   `traces/min`.

Run it on a fresh Spark:

```bash
python repro/scripts/bench_tracegen_speedup.py \
    --gguf-path /path/to/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
    --pool-dir /path/to/iq4_tracegen_v13_pool/traces \
    --binary $(bash scripts/build_llama_dump_hiddens.sh | tail -1 | xargs dirname)/llama-dump-hiddens-worker
```

It emits `results.json` with the rate, per-prompt equivalence metrics,
and shape info — ready to drop into `repro/artifacts/`.

## What I learned the hard way

- `cparams.n_seq_max` defaults to 1. With the batched-worker code path
  but `n_seq_max=1`, you don't get a clean error — you get **silently
  corrupted hidden states for every batch position ≥ 1**, which happen
  to match the reference for short prompts and diverge later. Pass the
  numerical-equivalence gate, not just smoke tests.
- Each batch triggers a fresh `sched_reserve: reserving...` for the
  current compute-graph shape (~45 ms on GB10). At 100 prompts × width 4
  = 25 batches, that's ~1.1 s of overhead — negligible.
- MoE experts on CPU (`-ot exps=CPU`) is the dominant cost; batching
  the dense layers on GPU still helps because the per-token dense-side
  compute fuses across the batch, and the per-request socket round-trip
  amortizes.
- Don't trust sha256 mismatches at face value. Compute max/mean/cosine
  instead and decide.
