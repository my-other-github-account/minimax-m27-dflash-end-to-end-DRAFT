"""Export a trained DFlash drafter checkpoint to a buun-loadable GGUF.

The recipe applied here is the proven path validated on spark-1
(2026-04-30 / 2026-05-02). See ``repro/03-inference.md`` for the full
narrative and the ``dflash-minimax-buun-gguf-spark1`` skill for the
manual command sequence.

Public surface::

    export_to_gguf(checkpoint, output_path, *, ...) -> Path
    prep_for_buun_converter(src_dir, out_dir, ...)  -> Path  (re-exported)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Re-export the prep helper so callers can stage manually if they want
import importlib.util as _ilu

_PREP_PATH = Path(__file__).resolve().parents[3] / "scripts" / "prep_full_for_buun_converter.py"
if _PREP_PATH.exists():
    _spec = _ilu.spec_from_file_location("_dflash_prep", str(_PREP_PATH))
    _mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    prep_for_buun_converter = _mod.prep_for_buun_converter
else:  # pragma: no cover — only triggers in a malformed install
    def prep_for_buun_converter(*a, **kw):
        raise RuntimeError(
            f"prep_for_buun_converter script not found at {_PREP_PATH!s}. "
            "Reinstall dflash-llama via 'uv pip install -e .'."
        )


# --- buun-converter tokenizer-hash registration ---------------------

# The MiniMax-M2.7-FP8 quant ships a slightly different tokenizer.json than
# the upstream MiniMax-M2 model that buun's converter knows about, so the
# converter must whitelist its hash. Idempotent.
FP8_TOKENIZER_HASH = "a77756c3cc91392f442c5b99e414be8020d53ae31460de90754b4fcf5cc84a2d"
UPSTREAM_TOKENIZER_HASH = "f4f37b6c8eb9ea29b3eac6bb8c8487c5ab7885f8d8022e67edc1c68ce8403e95"


def register_minimax_fp8_tokenizer_hash(buun_repo: str | Path,
                                         hash_to_add: str = FP8_TOKENIZER_HASH) -> bool:
    """Whitelist the FP8 tokenizer hash in buun's ``convert_hf_to_gguf.py``.

    Returns True if a change was made, False if already registered.
    """
    conv = Path(buun_repo) / "convert_hf_to_gguf.py"
    if not conv.exists():
        raise FileNotFoundError(f"buun converter not found: {conv}")

    text = conv.read_text()
    if hash_to_add in text:
        return False  # already registered

    needle = f'if chkhsh == "{UPSTREAM_TOKENIZER_HASH}":'
    replacement = (f'if chkhsh == "{UPSTREAM_TOKENIZER_HASH}" '
                   f'or chkhsh == "{hash_to_add}":')
    if needle not in text:
        raise RuntimeError(
            f"Could not find upstream MiniMax-M2 hash anchor in {conv}. "
            "buun layout may have changed; manually whitelist or update this helper."
        )
    new_text = text.replace(needle, replacement, 1)
    conv.write_text(new_text)
    return True


# --- main API -------------------------------------------------------

def export_to_gguf(
    checkpoint: str | Path,
    output_path: str | Path,
    *,
    verifier_meta_dir: Optional[str | Path] = None,
    buun_repo: str | Path = "/home/user/buun-llama-cpp",
    venv_python: Optional[str | Path] = None,
    outtype: str = "bf16",
    rebake_floor: float = -65504.0,
    prepped_dir: Optional[str | Path] = None,
    register_tokenizer_hash: bool = True,
    verbose: bool = True,
) -> Path:
    """Convert a speculators-format DFlash drafter to a buun-loadable GGUF.

    Parameters
    ----------
    checkpoint : path
        speculators-format checkpoint dir (config.json, model.safetensors, ...)
    output_path : path
        Where to write the GGUF.
    verifier_meta_dir : path, optional
        Directory holding tokenizer files. Default: read from
        ``checkpoint/config.json["speculators_config"]["verifier"]["name_or_path"]``.
    buun_repo : path
        Path to a buun-llama-cpp checkout containing ``convert_hf_to_gguf.py``
        with the DFlashDraftModel converter class registered. Default
        matches spark-1 layout.
    venv_python : path, optional
        Python interpreter to invoke buun's converter with. Default:
        autodetect (``/home/user/venvs/vllm/bin/python3`` on spark-1, else
        ``sys.executable``).
    outtype : {"bf16", "f16", "f32"}
        Quantization for the output GGUF (passed to buun's converter).
    rebake_floor : float
        Floor value for non-mapped rows of the rebaked lm_head. Use the
        default ``-65504`` unless you understand the d2t zero-row dilution
        bug — see ``dflash-gguf-conversion`` skill.
    prepped_dir : path, optional
        Where to stage the prepped checkpoint. Default:
        ``<output_path>.prep/``.
    register_tokenizer_hash : bool
        If True (default), idempotently whitelist the FP8 tokenizer hash
        in buun's converter. Set False on systems where you've already
        applied a manual patch.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pathlib.Path : the GGUF path that was written.
    """
    checkpoint = Path(checkpoint)
    output_path = Path(output_path)
    buun_repo = Path(buun_repo)

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {checkpoint}")
    if not (buun_repo / "convert_hf_to_gguf.py").exists():
        raise FileNotFoundError(
            f"buun converter not found: {buun_repo}/convert_hf_to_gguf.py. "
            "Pass buun_repo= to point at a buun-llama-cpp clone with DFlashDraftModel."
        )

    if venv_python is None:
        for cand in ("/home/user/venvs/vllm/bin/python3", sys.executable):
            if cand and Path(cand).exists():
                venv_python = cand
                break
    venv_python = str(venv_python)

    if prepped_dir is None:
        prepped_dir = output_path.with_suffix(output_path.suffix + ".prep")
    prepped_dir = Path(prepped_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(*a):
        if verbose:
            print(*a, flush=True)

    # Step 1: stage prepped ckpt
    log(f"[1/3] Prepping checkpoint → {prepped_dir}")
    prep_for_buun_converter(
        src_dir=checkpoint,
        out_dir=prepped_dir,
        verifier_meta_dir=verifier_meta_dir,
        rebake_floor=rebake_floor,
        verbose=verbose,
    )

    # Step 2: register tokenizer hash if needed
    if register_tokenizer_hash:
        changed = register_minimax_fp8_tokenizer_hash(buun_repo)
        log(f"[2/3] Tokenizer hash registration: "
            f"{'ADDED' if changed else 'already present'}")
    else:
        log("[2/3] Tokenizer hash registration: SKIPPED (register_tokenizer_hash=False)")

    # Step 3: run buun's converter
    log(f"[3/3] Running buun converter → {output_path}")
    cmd = [
        venv_python, str(buun_repo / "convert_hf_to_gguf.py"),
        str(prepped_dir),
        "--outtype", outtype,
        "--outfile", str(output_path),
    ]
    log(f"  + {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(buun_repo),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"buun convert_hf_to_gguf.py failed (rc={proc.returncode}).\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    if verbose:
        # Print the last 10 lines of stderr (where buun puts its progress)
        tail = "\n".join(proc.stderr.splitlines()[-10:])
        log(f"  buun output (tail):\n{tail}")

    if not output_path.exists():
        raise RuntimeError(
            f"Converter returned 0 but {output_path} not found. "
            f"stdout was:\n{proc.stdout}\nstderr was:\n{proc.stderr}"
        )

    log(f"✓ GGUF written: {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)")
    return output_path


def verify_gguf_metadata(gguf_path: str | Path,
                         expected_arch: str = "dflash-draft",
                         expected_block_size: int = 8) -> dict:
    """Quick sanity-check on a freshly converted DFlash GGUF.

    Returns the parsed metadata dict. Raises ValueError if anything looks wrong.
    """
    try:
        from gguf import GGUFReader
    except ImportError as e:
        raise ImportError("Install gguf: 'pip install gguf'") from e

    r = GGUFReader(str(gguf_path))
    keys = [
        "general.architecture",
        "tokenizer.ggml.pre",
        "dflash-draft.dflash.target_layer_ids",
        "dflash-draft.dflash.block_size",
        "dflash-draft.dflash.mask_token_id",
        "dflash-draft.dflash.n_target_features",
    ]
    out = {}
    for k in keys:
        f = r.get_field(k)
        if f is None:
            out[k] = None
            continue
        if f.types[0].name == "STRING":
            out[k] = bytes(f.parts[f.data[0]]).decode()
        elif f.types[0].name == "ARRAY":
            out[k] = [int(f.parts[d][0]) for d in f.data]
        else:
            out[k] = int(f.parts[f.data[0]][0])
    out["total_tensors"] = len(r.tensors)
    out["file_size_bytes"] = Path(gguf_path).stat().st_size

    # Validate
    if out["general.architecture"] != expected_arch:
        raise ValueError(
            f"Expected arch={expected_arch!r}, got {out['general.architecture']!r}. "
            "buun expects 'dflash-draft' (NOT 'dflash')."
        )
    if out.get("dflash-draft.dflash.block_size") != expected_block_size:
        raise ValueError(
            f"Expected block_size={expected_block_size}, "
            f"got {out.get('dflash-draft.dflash.block_size')}"
        )
    if not out.get("dflash-draft.dflash.target_layer_ids"):
        raise ValueError("target_layer_ids missing or empty")
    return out


__all__ = [
    "export_to_gguf",
    "prep_for_buun_converter",
    "register_minimax_fp8_tokenizer_hash",
    "verify_gguf_metadata",
    "FP8_TOKENIZER_HASH",
]
