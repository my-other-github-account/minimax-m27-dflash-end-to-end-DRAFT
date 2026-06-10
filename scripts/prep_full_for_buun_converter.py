#!/usr/bin/env python3
"""Prep a speculators-format DFlash checkpoint for buun-llama-cpp's
``convert_hf_to_gguf.py`` DFlashDraftModel converter.

Steps:
1. Read d2t (draft->target offsets) and lm_head ``[draft_V, hidden]``
2. Build new lm_head ``[target_V, hidden]`` = ``full(-65504)`` then scatter
   draft rows to target_idx (the -65504 floor is the d2t-zero-row-dilution fix
   from the ``dflash-gguf-conversion`` skill — without it, runtime chain-pos-2
   measures ~5x lower than the training prediction)
3. Drop d2t and t2d from output
4. Copy verifier-owned token embedding/output tensors when the verifier meta
   directory includes a tiny ``model.safetensors`` with those shared tensors.
   The Lucebox adapter checkpoint itself is adapter-only, but the banked GGUF
   gate is a served drafter GGUF that carries ``token_embd.weight`` and
   ``output.weight`` for byte-identical load compatibility.
5. Flatten config: hoist ``transformer_layer_config.*`` keys to top level,
   rename ``aux_hidden_state_layer_ids`` → ``target_layer_ids``,
   set ``draft_vocab_size`` = ``target_vocab_size``
6. Copy tokenizer files from the verifier directory

Usage::

  python3 scripts/prep_full_for_buun_converter.py SRC_DIR OUT_DIR \\
      [--verifier-meta-dir DIR] [--rebake-floor -65504.0]

Library equivalent::

  from dflash_llama.inference import prep_for_buun_converter
  prep_for_buun_converter(src_dir, out_dir, verifier_meta_dir=...)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _resolve_verifier_path(cfg: dict, verifier_meta_dir: Optional[str | Path]) -> Path:
    """Resolve the verifier metadata directory used for tokenizer/shared tensors."""
    if verifier_meta_dir is not None:
        verifier_path = Path(verifier_meta_dir)
    else:
        candidate = cfg.get("speculators_config", {}).get("verifier", {}).get("name_or_path")
        verifier_path = Path(candidate) if candidate else Path("")
    return verifier_path


def _resolve_d2t_path(
    src_dir: Path,
    verifier_path: Path,
    d2t_path: Optional[str | Path],
) -> Optional[Path]:
    """Resolve the draft-to-target vocabulary map used for sparse output heads."""
    if d2t_path is not None:
        path = Path(d2t_path)
        if not path.exists():
            raise FileNotFoundError(f"--d2t-path does not exist: {path}")
        return path

    candidates = [
        src_dir / "d2t.npy",
        src_dir / "prompts" / "d2t.npy",
        verifier_path / "d2t.npy",
        verifier_path / "prompts" / "d2t.npy",
    ]
    # Historical MiniMax public repro layout: verifier_meta_v11 lives under the
    # run root, while the matching vocab-map sidecar is under
    # iq4_v17_consolidated/prompts/.  Prefer this over generic prompts/ maps
    # because the latter may belong to a different training run.
    run_root = verifier_path.parent
    candidates.extend([
        run_root / "iq4_v17_consolidated" / "prompts" / "d2t.npy",
        run_root / "prompts" / "d2t.npy",
        run_root / "prompts_dense" / "d2t.npy",
        run_root / "prompts_dense_v2" / "d2t.npy",
    ])
    for path in candidates:
        if path.exists():
            return path
    return None


def _target_ids_from_d2t(
    d2t_path: Path,
    draft_vocab: int,
    target_vocab: int,
) -> torch.Tensor:
    d2t_np = np.load(d2t_path)
    if d2t_np.shape != (draft_vocab,):
        raise ValueError(f"{d2t_path} shape {d2t_np.shape} != ({draft_vocab},)")
    d2t_long = torch.from_numpy(d2t_np.astype(np.int64, copy=False)).to(torch.long)
    offset_ids = torch.arange(draft_vocab, dtype=torch.long) + d2t_long
    absolute_ids = d2t_long

    def valid(ids: torch.Tensor) -> bool:
        return (
            int(ids.min()) >= 0
            and int(ids.max()) < target_vocab
            and int(torch.unique(ids).numel()) == draft_vocab
        )

    if valid(offset_ids):
        return offset_ids
    if valid(absolute_ids):
        return absolute_ids
    raise ValueError(
        f"{d2t_path} is neither valid offsets nor valid absolute ids for "
        f"draft_vocab={draft_vocab}, target_vocab={target_vocab}"
    )


def _copy_verifier_io_tensors(
    new_tensors: dict,
    src_dir: Path,
    verifier_path: Path,
    target_vocab: int,
    draft_vocab: int,
    hidden_size: int,
    d2t_path: Optional[str | Path] = None,
    rebake_floor: float = -65504.0,
    verbose: bool = True,
) -> list[str]:
    """Copy verifier token embedding/output tensors into the prepped checkpoint.

    Lucebox adapter banks may be adapter-only and omit token embeddings and the
    sparse DFlash output head. The served drafter GGUF gate still carries
    ``token_embd.weight`` plus ``output.weight``. ``token_embd.weight`` is the
    verifier embedding. ``output.weight`` is sparse: rows outside the draft
    vocabulary map are the rebake floor and mapped rows copy the verifier
    ``lm_head.weight``. A full verifier lm_head is only used as a fallback when
    no d2t sidecar can be resolved.
    """
    def log(*a):
        if verbose:
            print(*a, flush=True)

    verifier_st = verifier_path / "model.safetensors"
    if not verifier_st.exists():
        log(f"Verifier shared tensor file absent: {verifier_st} (skipping token_embd/output copy)")
        return []

    copied: list[str] = []
    with safe_open(str(verifier_st), "pt") as vf:
        keys = set(vf.keys())

        if "embed_tokens.weight" not in new_tensors:
            src_key = "model.embed_tokens.weight"
            if src_key in keys:
                tensor = vf.get_tensor(src_key)
                if list(tensor.shape) != [target_vocab, hidden_size]:
                    raise ValueError(
                        f"verifier tensor {src_key} has shape {list(tensor.shape)}, "
                        f"expected [{target_vocab}, {hidden_size}]"
                    )
                new_tensors["embed_tokens.weight"] = tensor
                copied.append("embed_tokens.weight")
                log(f"  copied verifier {src_key} -> embed_tokens.weight {list(tensor.shape)}")
            else:
                log(f"  verifier tensor {src_key} absent in {verifier_st}")
        else:
            log("  keeping checkpoint-provided embed_tokens.weight")

        if "lm_head.weight" in new_tensors:
            log("  keeping checkpoint-provided lm_head.weight")
        elif "lm_head.weight" in keys:
            verifier_lm_head = vf.get_tensor("lm_head.weight")
            if list(verifier_lm_head.shape) != [target_vocab, hidden_size]:
                raise ValueError(
                    f"verifier tensor lm_head.weight has shape {list(verifier_lm_head.shape)}, "
                    f"expected [{target_vocab}, {hidden_size}]"
                )
            resolved_d2t = _resolve_d2t_path(src_dir, verifier_path, d2t_path)
            if resolved_d2t is not None:
                target_ids = _target_ids_from_d2t(resolved_d2t, draft_vocab, target_vocab)
                sparse_head = torch.full(
                    (target_vocab, hidden_size),
                    float(rebake_floor),
                    dtype=verifier_lm_head.dtype,
                )
                sparse_head[target_ids] = verifier_lm_head[target_ids]
                new_tensors["lm_head.weight"] = sparse_head
                copied.append("lm_head.weight")
                log(
                    f"  rebaked verifier lm_head.weight -> sparse lm_head.weight "
                    f"using {resolved_d2t} ({draft_vocab}/{target_vocab} rows)"
                )
            else:
                new_tensors["lm_head.weight"] = verifier_lm_head
                copied.append("lm_head.weight")
                log("  copied full verifier lm_head.weight -> lm_head.weight (no d2t sidecar found)")
        else:
            log(f"  verifier tensor lm_head.weight absent in {verifier_st}")

    return copied


def prep_for_buun_converter(
    src_dir: str | Path,
    out_dir: str | Path,
    verifier_meta_dir: Optional[str | Path] = None,
    d2t_path: Optional[str | Path] = None,
    rebake_floor: float = -65504.0,
    verbose: bool = True,
) -> Path:
    """Run the prep recipe in-process.

    Parameters
    ----------
    src_dir : path
        Speculators-format checkpoint (must contain ``config.json``,
        ``model.safetensors``, optionally ``val_metrics.json``).
    out_dir : path
        Where to write the prepped checkpoint.
    verifier_meta_dir : path, optional
        Directory containing tokenizer files (``tokenizer.json``,
        ``tokenizer_config.json``, etc). If omitted, falls back to
        ``cfg["speculators_config"]["verifier"]["name_or_path"]``.
    d2t_path : path, optional
        Draft-to-target vocabulary map sidecar. Needed to reconstruct the sparse
        served-drafter output head when the source checkpoint is adapter-only.
    rebake_floor : float
        Value used for non-mapped rows of the rebaked lm_head. ``-65504.0``
        (largest-magnitude finite bf16 negative) is the documented correct
        value; do not change without reading the d2t-rebake skill.
    """
    src = Path(src_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def log(*a):
        if verbose:
            print(*a, flush=True)

    # 1. Read config
    with open(src / "config.json") as f:
        cfg = json.load(f)

    target_vocab = cfg["transformer_layer_config"]["vocab_size"]
    draft_vocab = cfg["draft_vocab_size"]
    hidden_size = cfg["transformer_layer_config"]["hidden_size"]
    target_layer_ids = cfg["aux_hidden_state_layer_ids"]
    log(f"target_vocab={target_vocab}, draft_vocab={draft_vocab}, hidden={hidden_size}")
    log(f"target_layer_ids={target_layer_ids}")

    verifier_path = _resolve_verifier_path(cfg, verifier_meta_dir)

    # 2. Load tensors, rebake lm_head, drop d2t/t2d.
    # Lucebox adapter banks may store the adapter under the explicit
    # model_lucebox_layout.safetensors name; standard speculators checkpoints
    # use model.safetensors.  Accept both so a file-path Lucebox export does not
    # require manual renaming before conversion.
    src_st = src / "model.safetensors"
    if not src_st.exists():
        lucebox_st = src / "model_lucebox_layout.safetensors"
        if lucebox_st.exists():
            src_st = lucebox_st
        else:
            raise FileNotFoundError(
                f"expected {src / 'model.safetensors'} or {lucebox_st}"
            )
    new_tensors = {}
    with safe_open(str(src_st), "pt") as f:
        keys = list(f.keys())
        d2t = f.get_tensor("d2t") if "d2t" in keys else None
        t2d = f.get_tensor("t2d") if "t2d" in keys else None
        lm_head = f.get_tensor("lm_head.weight") if "lm_head.weight" in keys else None

        if d2t is not None and lm_head is not None and lm_head.shape[0] == draft_vocab:
            log(f"REBAKE: lm_head {list(lm_head.shape)} -> [{target_vocab}, {hidden_size}] "
                f"with {rebake_floor} floor")
            if t2d is not None:
                t_true = int(t2d.sum().item())
                ok = "OK" if t_true == draft_vocab else "MISMATCH"
                log(f"  t2d.sum() = {t_true} (expected {draft_vocab}: {ok})")

            # Auto-detect d2t encoding: speculators-format trainers may store either
            # OFFSETS (target_id = i + d2t[i], typical of older runs) or ABSOLUTE
            # token ids (target_id = d2t[i], observed in newer runs e.g. v11+).
            d2t_long = d2t.to(torch.long)
            offset_ids = torch.arange(draft_vocab, dtype=torch.long) + d2t_long
            absolute_ids = d2t_long
            if 0 <= int(offset_ids.min()) and int(offset_ids.max()) < target_vocab:
                log("  d2t encoding = OFFSET (target_id = i + d2t[i])")
                target_ids = offset_ids
            elif 0 <= int(absolute_ids.min()) and int(absolute_ids.max()) < target_vocab:
                log("  d2t encoding = ABSOLUTE (target_id = d2t[i])")
                target_ids = absolute_ids
            else:
                raise AssertionError(
                    f"d2t neither valid offsets ({int(offset_ids.min())}..{int(offset_ids.max())}) "
                    f"nor valid absolute ids ({int(absolute_ids.min())}..{int(absolute_ids.max())}) "
                    f"for target_vocab={target_vocab}"
                )
            assert target_ids.min() >= 0, f"negative target id {target_ids.min()}"
            assert target_ids.max() < target_vocab, (
                f"target_id {target_ids.max()} >= vocab {target_vocab}"
            )

            new_lm_head = torch.full(
                (target_vocab, hidden_size),
                float(rebake_floor),
                dtype=lm_head.dtype,
            )
            new_lm_head[target_ids] = lm_head
            new_tensors["lm_head.weight"] = new_lm_head
            log(f"  rebaked: scattered {draft_vocab} rows into {target_vocab}-row tensor")
        elif lm_head is not None:
            log(f"lm_head already at target shape {list(lm_head.shape)} - no rebake")
            new_tensors["lm_head.weight"] = lm_head

        # Copy other tensors except d2t/t2d/lm_head (already handled)
        for k in keys:
            if k in ("d2t", "t2d", "lm_head.weight"):
                continue
            new_tensors[k] = f.get_tensor(k)

    copied_io = _copy_verifier_io_tensors(
        new_tensors,
        src_dir=src,
        verifier_path=verifier_path,
        target_vocab=target_vocab,
        draft_vocab=draft_vocab,
        hidden_size=hidden_size,
        d2t_path=d2t_path,
        rebake_floor=rebake_floor,
        verbose=verbose,
    )
    if copied_io:
        log(f"Added verifier shared tensors: {', '.join(copied_io)}")

    log(f"Output tensors: {len(new_tensors)} (was {len(keys)})")

    # 3. Save flattened safetensors
    save_file(new_tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    log(f"Wrote {out / 'model.safetensors'}")

    # 4. Flatten config for buun converter
    new_cfg = {"architectures": ["DFlashDraftModel"]}
    tlc = cfg.get("transformer_layer_config", {})
    for k, v in tlc.items():
        new_cfg[k] = v
    new_cfg["block_size"] = cfg["block_size"]
    new_cfg["mask_token_id"] = cfg["mask_token_id"]
    new_cfg["target_layer_ids"] = target_layer_ids
    new_cfg["aux_hidden_state_layer_ids"] = target_layer_ids
    new_cfg["draft_vocab_size"] = target_vocab  # after rebake
    new_cfg["model_type"] = "qwen3"
    new_cfg["max_anchors"] = cfg.get("max_anchors", 512)

    with open(out / "config.json", "w") as f:
        json.dump(new_cfg, f, indent=2)
    log(f"Wrote {out / 'config.json'}")

    # Also copy val_metrics.json if present (downstream eval reads it)
    if (src / "val_metrics.json").exists():
        shutil.copy2(src / "val_metrics.json", out / "val_metrics.json")
        log("  copied val_metrics.json")

    # 5. Copy tokenizer source files from the same verifier metadata directory.
    log(f"Verifier path for tokenizer: {verifier_path}")
    copied = []
    for tk in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "vocab.json", "merges.txt", "tokenizer.model"):
        src_tk = verifier_path / tk
        if src_tk.exists():
            shutil.copy2(src_tk, out / tk)
            log(f"  copied {tk}")
            copied.append(tk)
    if not copied:
        raise RuntimeError(
            f"No tokenizer files found at {verifier_path}. "
            f"Pass --verifier-meta-dir to override."
        )

    log(f"\nDone. Run buun converter on: {out}")
    return out


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--verifier-meta-dir", default=None,
                    help="Directory holding tokenizer.json etc (default: read from config)")
    ap.add_argument("--d2t-path", default=None,
                    help="draft-to-target vocab map .npy for adapter-only Lucebox output-head rebake")
    ap.add_argument("--rebake-floor", type=float, default=-65504.0,
                    help="Floor value for non-mapped rows in rebaked lm_head (default: -65504.0)")
    args = ap.parse_args()
    prep_for_buun_converter(args.src_dir, args.out_dir,
                             verifier_meta_dir=args.verifier_meta_dir,
                             d2t_path=args.d2t_path,
                             rebake_floor=args.rebake_floor)


if __name__ == "__main__":
    _cli()
