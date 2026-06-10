# MiniMax DFlash strict-accept reproduction (1.1874x provenance patch)

This recipe makes the banked MiniMax DFlash strict-accept result reproducible from public Lucebox source plus a trained adapter checkpoint. It performs integration only; fresh GPU verification is intentionally left to a downstream card. The shipped/headline mode is strict argmax accept; eta/top-k modes are retained only as experimental provenance knobs.

## Summary

Shipped/headline width-3 cell:

- mode: strict argmax accept
- knob: env unset or `DFLASH_ACCEPT_MODE=strict`
- AL_true: 1.8402077151335312
- full wall speedup: 1.0687912232731116x
- measurement setting: GB10/DGX-Spark class system, max_ctx 2304, max_tokens 256, temperature 0, reasoning ON, 50-prompt set referenced by md5 only

Experimental top-k provenance cell (not the default/reproduction gate): `DFLASH_ACCEPT_MODE=topk`, `DFLASH_ACCEPT_TOPK=3` measured AL_true 2.135437881873727 / full wall speedup 1.1873573820278733x.

Verifier-authoritative accept semantics:

```text
accept draft token t at position k if:
  target argmax == t
  OR eta mode: p_target(t) >= eta * max_j p_target(j)
  OR top-k mode: t is in target top-k

The verifier remains authoritative. Accepted draft tokens are committed only through the target-side verification path; the drafter proposes candidates but does not own final token correctness.
```

## Provenance md5s

| artifact | md5 |
| --- | --- |
| patched `dflash_server` binary | `30af929df903978121e3cd4323981d52` |
| served drafter GGUF | `63cdd84c69ef6086a9d15820b866ef8b` |
| target first GGUF shard | `019759eeec4be4931592eeefbd54d2ba` |
| 50-prompt set | `4af2fbf40afcc97af9633a2ff49a4139` |

Weights and prompts are identified by hash only here.

## Expected numbers from the banked run

`gate_status=VOID` means the metric was measured and retained, but that cell fails the no-repeat-loop quality gate.

| cell | width | accept knobs | AL_true | full_wall_speedup | gate_status | notes |
| --- | ---: | --- | ---: | ---: | --- | --- |
| target_only | - | none | 1.000000000 | - | reference | greedy target baseline, 18.846784950 tok/s |
| w3_strict | 3 | strict | 1.840207715 | 1.068791223x | PASS | original argmax-only accept |
| w3_eta0p7 | 3 | `DFLASH_ACCEPT_MODE=eta`, `DFLASH_ACCEPT_ETA=0.7` | 1.861798252 | 1.051382693x | PASS | lossy eta accept |
| w3_eta0p5 | 3 | `DFLASH_ACCEPT_MODE=eta`, `DFLASH_ACCEPT_ETA=0.5` | 1.888838544 | 1.080021909x | PASS | lossy eta accept |
| w3_eta0p3 | 3 | `DFLASH_ACCEPT_MODE=eta`, `DFLASH_ACCEPT_ETA=0.3` | 1.941738594 | 1.108468307x | VOID | one manually confirmed repeat-loop completion |
| w3_top2 | 3 | `DFLASH_ACCEPT_MODE=topk`, `DFLASH_ACCEPT_TOPK=2` | 2.020620231 | 1.141072749x | PASS | quality-passing |
| w3_top3 | 3 | `DFLASH_ACCEPT_MODE=topk`, `DFLASH_ACCEPT_TOPK=3` | 2.135437882 | 1.187357382x | PASS | best quality-passing width-3 cell |
| w4_top3 | 4 | `DFLASH_ACCEPT_MODE=topk`, `DFLASH_ACCEPT_TOPK=3` | 2.342140026 | 1.145009360x | PASS | higher AL, lower wall speed than w3_top3 |

The banked result uses server-log `thinking=true` evidence for reasoning mode. A response-side reasoning-token counter in that run reported zero tokens, so the log-level proof is the authoritative reasoning-mode check for the historical table.

## 1. Build public Lucebox with the patch

```bash
git clone https://github.com/Luce-Org/lucebox-hub.git lucebox-hub
cd lucebox-hub
git checkout bdc706a
git submodule update --init --recursive

git apply /path/to/minimax-m27-dflash-end-to-end-DRAFT/patches/lucebox/01-typical-topk-accept.patch

cmake -S server -B server/build-sm121 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build server/build-sm121 --target dflash_server -j
```

If building on a non-sm121 GPU, adjust `CMAKE_CUDA_ARCHITECTURES` for that device. The banked numbers above were measured on a GB10/DGX-Spark class system.

## 2. Convert the trained adapter checkpoint to GGUF

The checkpoint argument can be either the checkpoint directory or its `model_lucebox_layout.safetensors` file. The command is generic over training step; it is not hardcoded to step 20000.

```bash
cd /path/to/minimax-m27-dflash-end-to-end-DRAFT
PYTHONPATH=src dflash-llama export-lucebox \
  --checkpoint /path/to/adapter_checkpoint/model_lucebox_layout.safetensors \
  --out /path/to/output/draft.gguf \
  --verifier-meta-dir /path/to/verifier_meta \
  --buun-repo /path/to/buun-llama-cpp \
  --force-block-size 8 \
  --verify
```

Equivalent Python API:

```python
from dflash_llama.inference import export_lucebox_to_gguf, verify_gguf_metadata

gguf = export_lucebox_to_gguf(
    checkpoint="/path/to/adapter_checkpoint/model_lucebox_layout.safetensors",
    output_path="/path/to/output/draft.gguf",
    verifier_meta_dir="/path/to/verifier_meta",
    buun_repo="/path/to/buun-llama-cpp",
    force_block_size=8,
)
print(verify_gguf_metadata(gguf, expected_block_size=8))
```

## 3. Serve Lucebox DFlash

Shipped/headline strict cell (also the default: the wrapper leaves accept env vars unset for strict mode):

```bash
PYTHONPATH=src dflash-llama serve-lucebox \
  --binary /path/to/lucebox-hub/server/build-sm121/dflash_server \
  --target /path/to/target-first-shard.gguf \
  --draft /path/to/output/draft.gguf \
  --max-ctx 2304 \
  --default-max-tokens 256 \
  --think-max-tokens 256 \
  --hard-limit-reply-budget 0 \
  --model-name dflash \
  --prefix-cache-slots 0 \
  --accept-mode strict \
  --host 127.0.0.1 \
  --port 8080
```

Experimental top-k provenance cell (not the default/reproduction gate):

```bash
PYTHONPATH=src dflash-llama serve-lucebox \
  --binary /path/to/lucebox-hub/server/build-sm121/dflash_server \
  --target /path/to/target-first-shard.gguf \
  --draft /path/to/output/draft.gguf \
  --max-ctx 2304 \
  --default-max-tokens 256 \
  --think-max-tokens 256 \
  --hard-limit-reply-budget 0 \
  --model-name dflash \
  --prefix-cache-slots 0 \
  --accept-mode topk \
  --accept-topk 3 \
  --host 127.0.0.1 \
  --port 8080
```

For strict mode the wrapper leaves the accept environment unset, which selects the stock longest target-argmax-prefix behavior. Eta/top-k are experimental non-default knobs mapped to environment variables consumed by the patch. No GPU work is performed by this documentation card.

## 4. Summarize AR-vs-spec rows

`dflash_llama.inference.benchmark` exposes the 50-prompt summary helpers used by the typaccept protocol:

```python
import json
from dflash_llama.inference.benchmark import summarize_ar_vs_spec_50

rows = [json.loads(line) for line in open("/path/to/ar_vs_spec_rows.jsonl")]
summary = summarize_ar_vs_spec_50(rows)
print(json.dumps(summary, indent=2))
```

The summary computes:

- `AL_true = n_pred / (n_pred - n_acc)`
- `full_wall_speedup = sum(ar_wall_sec) / sum(spec_wall_sec)`
- quality pass/fail counts from: nonempty content, positive reasoning-token evidence, and no repeated 9-gram loop.

For the historical banked run, use the table above as the expected target. For any fresh run, do not claim reproduction unless the full 50-prompt suite passes the quality gates and the md5s/provenance match the intended artifacts.
