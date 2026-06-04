# MiniMax-M2 DFlash lucebox AL=3.5276 reproduction bundle (-DRAFT)

This is a from-scratch, publication-staging reproduction bundle for the MiniMax-M2 DFlash drafter on the Lucebox/Luce DFlash server.

Scope of this draft:

- Reproduced: MiniMax-M2 DFlash drafter conversion to lucebox safetensors layout.
- Reproduced: Lucebox/Luce server command for target + drafter serving.
- Reproduced: 7-prompt greedy measurement with `AL_true_mean_commit=3.52755905511811`, accepted `326/1016`, p1..p7 `[0.677,0.504,0.354,0.197,0.228,0.213,0.173]`.
- Pending: 256-token stability (R5), speedup-vs-baseline (R6), accept-verify audit (R4).

Important publication note: this is a draft staging directory only. No GitHub push was performed by this worker.

## 0. Prereqs / hardware

Observed reproduced hardware/software environment:

- DGX Spark / GB10 class host with NVIDIA Blackwell `sm_121` target.
- CUDA build of Lucebox server.
- Python 3 for conversion and measurement scripts.
- Target GGUF shards and drafter safetensors available locally; weights are not included in this repo.

Source evidence:

- Lucebox tree used on spark-3: `/home/<USER>/lucebox-latest-20260603`
- Server build command required by the task and used for this tree: `cmake --build server/build-sm121 --target dflash_server -j2`
- Server binary recorded for the R1 run: md5 `fa08acfdb72b0df2463b934161b75caa`
- Independent spark-6 reproduction binary: md5 `7e27408f4d1db727777f9ff7a03c0ed9`

## 1. Get the target GGUF

Weights are not bundled. Obtain MiniMax-M2.7 GGUF UD-IQ4_XS shards and place them in the layout used by the reproduced run:

```bash
mkdir -p /home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS
# Put the four shards here:
# /home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf
# /home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00002-of-00004.gguf
# /home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00003-of-00004.gguf
# /home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00004-of-00004.gguf
md5sum /home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf
```

Expected md5 for shard 00001, observed on spark-3:

```text
019759eeec4be4931592eeefbd54d2ba  MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf
```

Source evidence: `artifacts/manifest_hashes.txt` copied from spark-3.

## 2. Get and convert the drafter

Weights are not bundled. Put the source drafter checkpoint at the path used by the reproduced runs:

```bash
mkdir -p /home/<USER>/t_5567a42d/golden_step_00020000
# Put model.safetensors here, plus config.json/config.py if you want the exact observed layout.
md5sum /home/<USER>/t_5567a42d/golden_step_00020000/model.safetensors
```

Expected source drafter md5, observed on spark-3:

```text
924758ef130170f01271b413154d626c  model.safetensors
```

Convert to lucebox layout using the copied transform script. The command shape is directly derived from `scripts/transform_minimax_dflash_safetensors.py`, which reads `sys.argv[1]` as input and `sys.argv[2]` as output:

```bash
python3 scripts/transform_minimax_dflash_safetensors.py \
  /home/<USER>/t_5567a42d/golden_step_00020000/model.safetensors \
  /home/<USER>/t_5567a42d/golden_step_00020000/model_lucebox_layout.safetensors

md5sum /home/<USER>/t_5567a42d/golden_step_00020000/model_lucebox_layout.safetensors
```

Expected converted drafter md5:

```text
30c27f646590b562608774bee82d2108  model_lucebox_layout.safetensors
```

Bundled config files were copied from spark-3:

- `drafter_config/config.json` from `/home/<USER>/t_5567a42d/golden_step_00020000/config.json`
- `drafter_config/config.py` from `/home/<USER>/t_5567a42d/golden_step_00020000/config.py`

## 3. Build lucebox `dflash_server` at the reproduced revert state

Get the Lucebox/Luce source corresponding to the observed tree:

```bash
# Upstream project evidence in the observed README points to Lucebox/Lucebox materials.
# Exact reproduced local tree path was:
# /home/<USER>/lucebox-latest-20260603
cd /home/<USER>/lucebox-latest-20260603
```

Important: the observed tree on spark-3 had no `.git` directory when inspected, so this bundle records the exact state by source path, patch, and fingerprints rather than inventing a commit SHA. See `MANIFEST.md` for source fingerprints.

Apply the source revert patch that captures Step A from `spark-3:/home/<USER>/t_77088062/NOT_RECONCILED_MINIMAX_REVERT_ROPE.md`:

```bash
cd /home/<USER>/lucebox-latest-20260603
patch -p1 < /path/to/this/repo/patches/source-revert-to-fa08acf-combined.patch
```

The patch encodes:

- `server/src/qwen35/qwen35_backend.cpp`: restore 0-based full K span, `for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;`
- `server/src/common/dflash_spec_decode.cpp`: restore the same 0-based full K span.
- Restore the pre-draftmask graph files corresponding to `*.pre-draftmaskfix-t_d81951a4` for `draft_graph.{h,cpp}`, `step_graph.h`, `dflash_draft_graph.cpp`, and `dflash_spec_decode.cpp`.

Build the server for `sm_121`:

```bash
cmake --build server/build-sm121 --target dflash_server -j2
```

Expected R1 baseline binary md5 recorded in the spark-3 report:

```text
fa08acfdb72b0df2463b934161b75caa  dflash_server
```

## 4. Serve target + drafter

`scripts/serve.sh` is directly derived from `spark-3:/home/<USER>/t_77088062/server_cmd_revert.sh`.

Default command:

```bash
./scripts/serve.sh 2>&1 | tee /tmp/server_8097_revert.log
```

Equivalent explicit command from the reproduced run:

```bash
cd /home/<USER>/lucebox-latest-20260603/server
export CUDA_VISIBLE_DEVICES=0
export DFLASH_PERPOS_PROBE=1
export DFLASH_DRAFTATTN_PROBE=1
stdbuf -oL -eL ./build-sm121/dflash_server \
  /home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  --draft /home/<USER>/t_5567a42d/golden_step_00020000/model_lucebox_layout.safetensors \
  --port 8097 \
  --host 127.0.0.1 \
  --max-ctx 2048 \
  --default-max-tokens 128 \
  --hard-limit-reply-budget 0 \
  --model-name minimax-luce-golden-revert
```

## 5. Measure AL on the 7 prompts

`scripts/run_measure_7.py` was copied from `spark-3:/home/<USER>/t_77088062/run_measure_7.py`. It posts these 7 prompts to `/v1/completions` with `max_tokens=64`, `temperature=0`, `top_k=1`, then parses `[spec-decode]` telemetry from the server log.

Run:

```bash
export LUCE_LOG=/tmp/server_8097_revert.log
export PORT=8097
export MODEL=minimax-luce-golden-revert
export TAG=revert
export OUTDIR=$PWD/results
./scripts/measure.sh
```

Expected summary, reproduced on spark-3 and spark-6:

```text
AL_true_mean_commit = 3.52755905511811
per_pos_argmax_match_weighted_p1_p7 = [0.6771653464566929, 0.5039369921259843, 0.35433083464566933, 0.19685030708661416, 0.22834636220472443, 0.21259853543307086, 0.17322849606299215]
accepted/draft_total = 326/1016
```

Short rounded form:

```text
AL_true=3.5276
p1..p7=[0.677,0.504,0.354,0.197,0.228,0.213,0.173]
accepted=326/1016
```

## 6. Known issues / honest status

Reproduced:

- AL 3.5276 on 7 prompts, greedy, max_tokens=64.
- Same summary on spark-3 R1 and spark-6 R2.
- Drafter converted to lucebox layout with md5 `30c27f646590b562608774bee82d2108`.

Pending:

- R5: 256-token server stability; this draft does not claim 256-token stability.
- R6: speedup-vs-AR baseline; this draft does not claim speedup.
- R4: accept-verify audit.
- Runtime-vs-train p2..p7 reconciliation. The spark-3 report says the +1 synthetic RoPE convention made analytic RoPE rows match the z-lab position audit but regressed AL from 3.5276 to 2.4615, so it is not kept.

## 7. Evidence files in this bundle

- `MANIFEST.md`: external artifacts, hashes, and source-state notes.
- `RESULTS.md`: R1/R2/original result evidence.
- `artifacts/NOT_RECONCILED_MINIMAX_REVERT_ROPE.md`: copied spark-3 report.
- `artifacts/measure_revert_7prompts_summary.json`: copied spark-3 summary.
- `artifacts/measure_repro6_7prompts_summary.json`: copied spark-6 summary.
- `artifacts/server_cmd_revert.sh`: copied spark-3 server command.
- `artifacts/server_cmd_repro6.sh`: copied spark-6 server command.
- `patches/source-revert-to-fa08acf-combined.patch`: source-revert patch bundle.

## 8. Commands used to assemble this staging bundle

These are included so reviewers can audit provenance. They were run from macmini and copied only scripts/configs/log artifacts, not weights:

```bash
scp spark-3:/home/<USER>/t_5567a42d/transform_minimax_dflash_safetensors.py scripts/
scp spark-3:/home/<USER>/t_5567a42d/golden_step_00020000/config.json drafter_config/
scp spark-3:/home/<USER>/t_5567a42d/golden_step_00020000/config.py drafter_config/
scp spark-3:/home/<USER>/t_77088062/run_measure_7.py scripts/
scp spark-3:/home/<USER>/t_77088062/NOT_RECONCILED_MINIMAX_REVERT_ROPE.md artifacts/
scp spark-3:/home/<USER>/t_77088062/measure_revert_7prompts_summary.json artifacts/
scp spark-6:/home/<USER>/t_1d157fda/measure_repro6_7prompts_summary.json artifacts/
```

No command in this README was invented as an observed run. Commands are either copied from a source artifact (`server_cmd_revert.sh`, `run_measure_7.py`) or directly derived from scripts/config paths cited above.
