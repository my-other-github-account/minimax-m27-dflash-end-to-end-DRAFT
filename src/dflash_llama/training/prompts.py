"""``assemble_prompts_arrow`` — turn a dir of self-describing traces into an
HF Dataset that the speculators trainer can read.

Replaces the v2 ``build_paired_dataset.py`` flow, which had to sha256-pair
hidden-state files against a separately-saved prompts dataset. Because our
trace files already include ``input_ids`` and ``loss_mask``, this is now
just an enumeration step — no hashing, no symlinks, no pairing report.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from datasets import Dataset
from safetensors import safe_open

from ..version import SCHEMA_VERSION


def _stem_index(path: Path) -> int:
    """``hs_42.safetensors`` -> 42; falls back to 0 if non-numeric."""
    s = path.stem.split("_")[-1]
    return int(s) if s.isdigit() else 0


def assemble_prompts_arrow(
    traces_dir: Union[str, Path],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    glob: str = "hs_*.safetensors",
    require_schema: bool = True,
    link_hidden_states: bool = True,
) -> dict:
    """Walk a trace directory and emit an HF Dataset of prompts.

    Output layout (``output_dir`` defaults to ``traces_dir.parent / "paired"``)::

        <output_dir>/
          prompts/                  HF Dataset (input_ids, loss_mask, source_name, source_row_idx)
          hidden_states/            symlinks to the original trace files (hs_<i>.safetensors)
          assembly_report.json      counts + per-source breakdown

    The trainer's ``--data-path`` is ``<output_dir>/prompts``; its
    ``--hidden-states-path`` is ``<output_dir>/hidden_states``.

    Returns a small dict report.
    """
    import json

    traces_dir = Path(traces_dir)
    if not traces_dir.exists():
        raise FileNotFoundError(traces_dir)
    out_dir = Path(output_dir) if output_dir is not None else traces_dir.parent / "paired"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "prompts"
    hs_link_dir = out_dir / "hidden_states"
    hs_link_dir.mkdir(exist_ok=True)

    files = sorted(traces_dir.glob(glob), key=_stem_index)
    if not files:
        raise FileNotFoundError(f"no traces matching {glob} under {traces_dir}")

    rows: list[dict] = []
    by_source: dict[str, int] = {}
    skipped: list[str] = []

    for out_idx, path in enumerate(files):
        with safe_open(str(path), framework="pt") as f:
            keys = set(f.keys())
            meta = dict(f.metadata() or {})
            if require_schema and meta.get("schema_version") != SCHEMA_VERSION:
                skipped.append(f"{path.name}: schema_version={meta.get('schema_version')!r}")
                continue
            if "input_ids" not in keys or "loss_mask" not in keys:
                skipped.append(f"{path.name}: missing input_ids/loss_mask")
                continue
            input_ids = f.get_tensor("input_ids").to_list() if hasattr(f.get_tensor("input_ids"), "to_list") else f.get_tensor("input_ids").tolist()
            loss_mask = f.get_tensor("loss_mask").tolist()

        src_name = meta.get("source_name", traces_dir.name)
        try:
            src_row_idx = int(meta.get("source_row_idx", _stem_index(path)))
        except ValueError:
            src_row_idx = _stem_index(path)

        rows.append({
            "input_ids": list(input_ids),
            "loss_mask": [bool(x) for x in loss_mask],
            "source_name": src_name,
            "source_row_idx": src_row_idx,
        })
        by_source[src_name] = by_source.get(src_name, 0) + 1

        if link_hidden_states:
            link = hs_link_dir / f"hs_{out_idx}.safetensors"
            if link.exists() or link.is_symlink():
                link.unlink()
            try:
                os.symlink(path.resolve(), link)
            except OSError:
                # Fallback: hardlink
                try:
                    os.link(path, link)
                except OSError:
                    # Last-ditch: copy
                    import shutil
                    shutil.copy2(path, link)

    if not rows:
        raise RuntimeError(
            f"assemble_prompts_arrow found 0 valid traces under {traces_dir} "
            f"(skipped={len(skipped)}); first few skips: {skipped[:5]}"
        )
    ds = Dataset.from_list(rows)
    if prompts_dir.exists():
        # remove old shard files so save_to_disk doesn't choke
        import shutil

        shutil.rmtree(prompts_dir)
    ds.save_to_disk(str(prompts_dir))

    report = {
        "traces_dir": str(traces_dir),
        "output_dir": str(out_dir),
        "n_traces": len(files),
        "n_rows": len(rows),
        "n_skipped": len(skipped),
        "by_source": by_source,
        "skipped_first_5": skipped[:5],
    }
    (out_dir / "assembly_report.json").write_text(json.dumps(report, indent=2))
    return report


__all__ = ["assemble_prompts_arrow"]
