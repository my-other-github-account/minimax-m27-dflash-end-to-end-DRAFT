# MANIFEST: external artifacts (weights not bundled)

This draft repo intentionally does not include model weights. It includes only scripts, configs, patches, and measured-result artifacts.

## Target GGUF

Source path used in reproduced runs:
`/home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/`

Files observed on spark-3:

- `MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf` size `7.9M`, md5 `019759eeec4be4931592eeefbd54d2ba`
- `MiniMax-M2.7-UD-IQ4_XS-00002-of-00004.gguf` size `47G` (md5 not recomputed for this staging bundle; only shard 00001 hash was required by the task)
- `MiniMax-M2.7-UD-IQ4_XS-00003-of-00004.gguf` size `47G` (md5 not recomputed for this staging bundle)
- `MiniMax-M2.7-UD-IQ4_XS-00004-of-00004.gguf` size `8.6G` (md5 not recomputed for this staging bundle)

Provenance: spark-3 command output copied to `artifacts/manifest_hashes.txt`.

## Drafter checkpoint and converted lucebox layout

Source path used in reproduced runs:
`/home/<USER>/t_5567a42d/golden_step_00020000/`

- `model.safetensors` md5 `924758ef130170f01271b413154d626c` (source checkpoint, not bundled)
- `model_lucebox_layout.safetensors` md5 `30c27f646590b562608774bee82d2108` (converted output, not bundled)
- `config.json` md5 `cc1af41d15ce13b4568fcca2cc07e01e` (bundled in `drafter_config/`)
- `config.py` md5 `9a907f2605b2fb9e4b2d011645c13d94` (bundled in `drafter_config/`)

Conversion script copied from spark-3:
`/home/<USER>/t_5567a42d/transform_minimax_dflash_safetensors.py` -> `scripts/transform_minimax_dflash_safetensors.py`

Conversion command shape, directly derived from that script's argv contract (`src=sys.argv[1]`, `out=sys.argv[2]`):

```bash
python3 scripts/transform_minimax_dflash_safetensors.py \
  golden_step_00020000/model.safetensors \
  golden_step_00020000/model_lucebox_layout.safetensors
md5sum golden_step_00020000/model_lucebox_layout.safetensors
# expected: 30c27f646590b562608774bee82d2108
```

## Lucebox / Luce source state

Observed source tree on spark-3:
`/home/<USER>/lucebox-latest-20260603`

Important limitation: that tree was not a git worktree when inspected (`.git` absent), so this bundle records exact source state by path/date, patches, and file md5s rather than an upstream commit SHA. Do not invent a commit. To reproduce exactly, start from the Lucebox source state corresponding to that tree and apply `patches/source-revert-to-fa08acf-combined.patch`.

Source fingerprints from spark-3 after identifying the intended baseline snapshots:

- `src/qwen35/qwen35_backend.cpp.pre-draftpos-t_1da1242d` md5 `96f117ebcb2cd85dfba2aa83907ea835`
- `src/common/dflash_spec_decode.cpp.pre-draftmaskfix-t_d81951a4` md5 `e4c84ae816926dc7ec45ca0d6a177a61`; after applying the documented 0-based K-span hunk in this bundle's patch, the reconstructed file md5 is `d84da1cb0a0013940e3498320c4fa7f3`
- `src/draft/draft_graph.cpp.pre-draftmaskfix-t_d81951a4` md5 `8d5a88e2fc5f6ab6602ea4a723fb855e`
- `src/draft/draft_graph.h.pre-draftmaskfix-t_d81951a4` md5 `6070f39b7d37955d9f7b4122416f43de`
- `src/common/step_graph.h.pre-draftmaskfix-t_d81951a4` md5 `a0ee144266c1c8ca3c52961c18263f1e`
- `src/common/dflash_draft_graph.cpp.pre-draftmaskfix-t_d81951a4` md5 `3484b2cee124c206a2c90e2c061076ce`

Reproduced run report records baseline server binary md5:
`fa08acfdb72b0df2463b934161b75caa`

Spark-6 independent reproduction used a locally named binary with md5:
`7e27408f4d1db727777f9ff7a03c0ed9  build-sm121-local/dflash_server`

## Included local files

- `README.md` end-to-end reproduction guide
- `RESULTS.md` measured AL evidence
- `scripts/transform_minimax_dflash_safetensors.py`
- `scripts/serve.sh`
- `scripts/measure.sh`
- `scripts/run_measure_7.py`
- `patches/*.patch`
- `drafter_config/config.json`
- `drafter_config/config.py`
- `artifacts/*` copied evidence from spark-3 and spark-6
