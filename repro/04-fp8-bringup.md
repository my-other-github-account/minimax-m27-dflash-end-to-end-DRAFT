# §4 — FP8 Bringup on a Spark (sm_120/121)

End-to-end recipe to get TransformerEngine FP8 production training working on a fresh DGX Spark (or any sm_120/121 host). The four blockers documented here were all hit during 2026-05-04 / 2026-05-05 production work; the difference between knowing them and rediscovering them is roughly a day.

> **Verified on:** NVIDIA GB10 (DGX Spark), aarch64, Ubuntu 22.04, Python 3.12, torch 2.10, TransformerEngine main @ 2026-05-04. Targets sm_121.

This doc assumes you've already worked through §0 (`repro/00-spark-from-scratch.md`) and have a working `dflash-llama` install + verifier model staged. We're adding the FP8 layer here.

---

## Step 1: 30-second readiness check

Before building anything, confirm the host actually wants FP8:

```bash
dflash-llama check-fp8
```

On a Spark you'll see:

```json
{
  "compute_capability": "12.1",
  "compute_capability_int": 121,
  "device_name": "NVIDIA GB10",
  "torch_available": true,
  "cuda_available": true,
  "supports_fp8cs": true,
  "supports_mxfp8_te": false,
  "supports_block_fp8": false,
  "recommends": "current_fp8 (Float8CurrentScaling HYBRID + fused LayerNormMLP)"
}
⚠  Spark-class arch detected (sm_120/121). Reminders:
   - kind='block_fp8' is silently NON-CONVERGENT here (TE #2382). REFUSED.
   - kind='mxfp8' via TE is BLOCKED here (cuBLAS layout, TE #2668). REFUSED.
   - kind='current_fp8' + fused te.LayerNormMLP IS the production recipe.
   - Always use FP8Recipe(use_split_accumulator=True) on all 3 GEMMs.
   - Launch via python (NOT torchrun) on single-GPU to avoid the silent-bf16 trap.
```

If you don't see `supports_fp8cs: true`, stop. FP8 won't work.

---

## Step 2: source-build TransformerEngine (the four blockers)

There is no aarch64 prebuilt wheel for TransformerEngine. You will be building from source. With the right environment, this takes ~2 minutes. With the wrong environment, it takes 30+ minutes (and may OOM the JIT compiler).

### Build environment template (`scripts/te-build-env.sh`)

```bash
#!/usr/bin/env bash
# Source this before building TransformerEngine on a Spark.
set -euo pipefail

# (1) Single-arch build. Stock TE builds for ALL of sm_70/75/80/86/89/90/100/120
#     which is 30+ minutes and 16 GB of intermediate object files. We only need
#     sm_121.
export NVTE_CUDA_ARCHS=121

# (2) torch sees the same arch list. The trailing 'a' is required if you ALSO
#     plan to build torchao mx_formats (see Step 5); otherwise plain "12.1" is
#     fine. We set "12.1a" to keep both paths open.
export TORCH_CUDA_ARCH_LIST="12.1a"

# (3) Tell TE to skip JAX/PaddlePaddle frontends.
export NVTE_FRAMEWORK=pytorch

# (4) cudnn + nccl headers must be on CPATH and LIBRARY_PATH. The pip-installed
#     versions in the vllm venv work fine — point at them.
VLLM_VENV=${VLLM_VENV:-$HOME/repos/vllm/.venv}
CUDNN_DIR=$(${VLLM_VENV}/bin/python -c \
    'import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))')
NCCL_DIR=$(${VLLM_VENV}/bin/python -c \
    'import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))' || true)
export CPATH="${CUDNN_DIR}/include:${NCCL_DIR}/include:${CPATH:-}"
export LIBRARY_PATH="${CUDNN_DIR}/lib:${NCCL_DIR}/lib:${LIBRARY_PATH:-}"

# (5) ninja from the vllm venv on PATH (faster than the system one, and matches
#     the torch ABI vllm was built against).
export PATH="${VLLM_VENV}/bin:${PATH}"

echo "[te-build] arch=${NVTE_CUDA_ARCHS} torch_arch=${TORCH_CUDA_ARCH_LIST}"
echo "[te-build] cudnn=${CUDNN_DIR}"
echo "[te-build] nccl=${NCCL_DIR}"
```

### Build into a sibling venv (the `.pth` link trick)

Because dflash-llama lives in its own venv but TE depends tightly on the torch ABI, the cleanest pattern is:

```bash
# Sibling venv just for TE
python3.12 -m venv ~/repos/te-venv
source ~/repos/te-venv/bin/activate
pip install --upgrade pip wheel ninja

# Same torch as your dflash-llama venv. Reuse it via LD_LIBRARY_PATH /
# .pth-linkage rather than reinstalling.
DFLASH_VENV=~/repos/dflash-llama/.venv
echo "${DFLASH_VENV}/lib/python3.12/site-packages" \
    > ~/repos/te-venv/lib/python3.12/site-packages/dflash-link.pth

# Source the build env
source /path/to/dflash-llama/scripts/te-build-env.sh

# Clone + build
git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
pip install . -v 2>&1 | tee te-build.log
```

Watch for `[te-build] arch=121` early in the log. If you see `arch=70;75;80;...` you forgot to source the env script and you're in for a 30-minute wait.

---

## Step 3: the cuBLAS LD_PRELOAD trick

TransformerEngine on sm_121 needs `cublasLtGroupedMatrixLayoutInit_internal`, which only ships in cuBLAS ≥ 13.2. The system cuBLAS on most Sparks is 12.x. Symptom at runtime:

```
undefined symbol: cublasLtGroupedMatrixLayoutInit_internal
```

The fix is to install the newer cuBLAS to a sibling directory and `LD_PRELOAD` it at runtime:

```bash
# Install newer cuBLAS into te/extra/
TE_EXTRA=~/repos/te-venv/extra
pip install --target="${TE_EXTRA}" 'nvidia-cublas>=13.2'
```

### Runtime env template (`scripts/te-runtime-env.sh`)

```bash
#!/usr/bin/env bash
# Source this before any FP8 training run on a Spark.
set -euo pipefail

TE_EXTRA=${TE_EXTRA:-$HOME/repos/te-venv/extra}
NEW_CUBLAS=${TE_EXTRA}/nvidia/cublas/lib

# (1) Newer cuBLAS — symbol mismatch otherwise.
export LD_PRELOAD="${NEW_CUBLAS}/libcublasLt.so.13${LD_PRELOAD:+:${LD_PRELOAD}}"

# (2) flex_attention pattern-matcher trips dynamo in speculators' attention
#     path. Disable both. (TORCH_COMPILE_DISABLE alone isn't enough on
#     torch 2.10.)
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

# (3) Make sure Hugging Face's transformers doesn't try to be clever.
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "[te-runtime] LD_PRELOAD=${LD_PRELOAD}"
```

---

## Step 4: 30-second readiness verification

```bash
source scripts/te-runtime-env.sh
python - <<'PY'
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import (
    Float8CurrentScaling, Format, MMParams,
)

# (1) Recipe construction succeeds with split-accumulator on all three GEMMs.
mm = MMParams(use_split_accumulator=True)
recipe = Float8CurrentScaling(
    fp8_format=Format.HYBRID,
    fp8_gemm_fprop=mm,
    fp8_gemm_dgrad=mm,
    fp8_gemm_wgrad=mm,
)
print("[ok] Float8CurrentScaling constructed")

# (2) Forward + backward at production-realistic shapes.
device = "cuda"
B, S, H, I = 2, 1024, 3072, 6144      # MiniMax-M2.7 drafter shape
x = torch.randn(B, S, H, device=device, dtype=torch.bfloat16, requires_grad=True)
ln_mlp = te.LayerNormMLP(
    hidden_size=H, ffn_hidden_size=I,
    eps=1e-6, normalization="RMSNorm", activation="swiglu", bias=False,
).to(device)
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    y = ln_mlp(x)
loss = y.sum()
loss.backward()
print(f"[ok] fwd+bwd at ({B},{S},{H})→({B},{S},{I}); loss={loss.item():.3f}")
print(f"[ok] grad has no NaNs: {not torch.isnan(x.grad).any().item()}")
PY
```

If this prints two `[ok]` lines without a stack trace, your FP8 stack is healthy. Now run a real DFlash training:

```bash
source scripts/te-runtime-env.sh
dflash-llama train \
    --verifier minimax-m2.7-iq4-xs \
    --hf-repo MiniMaxAI/MiniMax-M2 \
    --gguf-repo unsloth/MiniMax-M2-GGUF --gguf-quant UD-IQ4_XS \
    --traces data/traces_v12 \
    --output data/ckpt_v12 \
    --epochs 15 --lr 3e-5 --max-anchors 512 \
    --fp8-recipe current_fp8 \
    --drafter-intermediate-size 6144
```

Watch for `[FP8] enabled: kind=current_fp8 split_accumulator=True format=HYBRID` in the log within the first 5 seconds. If it isn't there, you've hit the silent-bf16 trap (see [§2 Pitfalls](02-training.md#pitfalls)).

---

## Step 5: speculators trainer patches

The flags `--fp8-recipe-kind`, `--fp8-split-accumulator`, `--te-use-fused`, `--drafter-intermediate-size`, `--nan-skip` are emitted by `dflash-llama train` whenever `fp8_recipe` is set, but they are only **understood** by speculators if you've patched its trainer:

```bash
python scripts/patch_speculators_for_fp8.py ~/repos/speculators
```

The script is idempotent — running it twice is safe. It adds:

1. **TrainerConfig fields**: `fp8_recipe_kind`, `fp8_split_accumulator`, `fp8_format`, `te_use_fused`, `drafter_intermediate_size`, `nan_skip`.
2. **`_maybe_wrap_te(model)` helper**: replaces `nn.Linear` with `te.Linear` and fuses Qwen3-style `post_attention_layernorm + mlp` into `te.LayerNormMLP`. Logs the wrap stats so you can see in the log how many modules were converted.
3. **`fp8_autocast` forward wrapper**: opens a `te.fp8_autocast` block around each forward pass with the configured recipe.
4. **NaN-skip optimizer guard**: if any gradient is NaN/Inf after backward + clip, skip `optimizer.step()` and `scheduler.step()`, increment `global_step`, continue. Cheap insurance.

Re-run after any `git pull` in the speculators repo; the patcher is idempotent.

---

## Optional: torchao MXFP8 (research only)

If you want MXFP8 on Spark for research, you'll need to bypass TransformerEngine entirely:

```bash
TORCH_CUDA_ARCH_LIST="12.1a" pip install -v --no-build-isolation \
    git+https://github.com/pytorch/ao.git
```

The trailing `a` in `12.1a` is mandatory. torchao's `mx_formats` re-quantizes both dim0 and dim1 so every GEMM is TN, sidestepping the cuBLAS issue. **This path is not wired into the dflash-llama trainer** — it's documented here purely so you don't waste time trying to use TE's `MXFP8BlockScaling` (which is refused at recipe-construction time on sm_120/121).

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `undefined symbol: cublasLtGroupedMatrixLayoutInit_internal` | system cuBLAS too old | `LD_PRELOAD=...te/extra/.../libcublasLt.so.13` |
| no `[FP8]` line in log | silent-bf16 trap | `use_torchrun=False` (default in 0.2.0+) |
| NaN at step ~40 | fprop split-accumulator False | `FP8Recipe(use_split_accumulator=True)` (default) |
| loss flatlines forever, no NaN | Float8BlockScaling on sm_120/121 | use `kind="current_fp8"` instead |
| cublasLt error at first matmul | MXFP8 via TE | refused; use torchao or stay on `current_fp8` |
| TE build takes 30+ minutes | building all archs | `NVTE_CUDA_ARCHS=121` |
| `RuntimeError: CUDA error: ...` from flex_attention | dynamo pattern-matcher | `TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1` |
