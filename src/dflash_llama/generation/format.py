"""SelfDescribingFormat — atomic, schema-versioned safetensor traces.

Every trace file written by this library carries enough metadata to be
trained against without any post-hoc pairing step. The on-disk schema is:

  hidden_states         (seq, n_layers, hidden) — fp8_e4m3fn or bfloat16
  hidden_states_scale   () or (n_layers,) float32 — per-tensor scale, only
                         present when storage == "fp8_per_tensor_scale"
  token_ids             (seq,) int64 — verifier-emitted token ids
  input_ids             (seq,) int64 — prompt input_ids (for trainer pairing)
  loss_mask             (seq,) bool  — anchor mask for the trainer

  metadata (safetensor __metadata__):
    schema_version, source_name, source_row_idx, gen_timestamp,
    storage, dtype, hidden_states_dtype_storage, n_layers, seq_len,
    hidden_size, layer_ids (json), abs_max (informational)

The fp8 path uses **saturating per-tensor scaling**:

    abs_max = hidden_states.abs().max()
    scale   = max(abs_max / FP8_E4M3FN_MAX, 1.0)
    fp8     = (hidden_states / scale).clamp(-MAX, MAX).to(float8_e4m3fn)

On load we multiply by ``scale`` and return bf16. By construction zero NaN
is possible — the direct ``tensor.to(float8_e4m3fn)`` cast that v2 used
silently produces NaN for any value above 448, which is the bug we are
fixing here.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

import torch
from safetensors.torch import load_file, save_file

from ..version import SCHEMA_VERSION

# fp8_e4m3fn finite range. Real spec max is ~448.0; we use exactly 448.0
# for clamp endpoints (it round-trips exactly through the dtype).
FP8_E4M3FN_MAX: float = 448.0

VALID_STORAGE = ("fp8_per_tensor_scale", "bf16")


# -----------------------------------------------------------------------
#  fp8 saturating cast
# -----------------------------------------------------------------------
def saturating_fp8_cast(hs: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Cast ``hs`` to ``float8_e4m3fn`` with per-tensor saturating scaling.

    Returns ``(fp8_tensor, scale)`` where ``hs ≈ fp8_tensor.to(fp32) * scale``.

    The scale is always ``>= 1.0``. For tensors whose abs-max already fits
    inside ±448 we still return scale=1.0 and just round-trip the values.
    The clamp protects against the NaN encoding (all-1 exponent) that
    direct ``to(float8_e4m3fn)`` would produce for out-of-range inputs.
    """
    if hs.numel() == 0:
        return hs.to(torch.float8_e4m3fn), 1.0
    # Force fp32 for the abs_max calculation so the scale is well-defined
    # even for input dtypes (bf16) that can't precisely represent 448.
    abs_max = float(hs.detach().to(torch.float32).abs().max().item())
    scale = max(abs_max / FP8_E4M3FN_MAX, 1.0)
    scaled = hs.to(torch.float32) / scale
    scaled = scaled.clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX)
    fp8 = scaled.to(torch.float8_e4m3fn)
    return fp8, scale


def saturating_fp8_recover(fp8: torch.Tensor, scale: float, *, dtype=torch.bfloat16) -> torch.Tensor:
    """Inverse of ``saturating_fp8_cast`` — multiply back by scale."""
    out = fp8.to(torch.float32) * float(scale)
    return out.to(dtype)


# -----------------------------------------------------------------------
#  Atomic save helper
# -----------------------------------------------------------------------
def _atomic_save_safetensors(
    tensors: dict, metadata: dict, final_path: Path
) -> None:
    """Write to a temp file, fsync, rename. Never leave a half-written final."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=f".{final_path.name}.tmp_",
        suffix=".part",
    )
    os.close(fd)
    tmp = Path(tmp_str)
    try:
        save_file(tensors, str(tmp), metadata=metadata)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.rename(tmp, final_path)
        # fsync the directory entry
        dfd = os.open(str(final_path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


# -----------------------------------------------------------------------
#  Public API
# -----------------------------------------------------------------------
def save_trace(
    path: Union[str, Path],
    *,
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    source_name: str,
    source_row_idx: int,
    storage: str = "fp8_per_tensor_scale",
    layer_ids: Optional[list[int]] = None,
    extra_metadata: Optional[dict] = None,
) -> dict:
    """Save a self-describing trace to ``path``. Returns the metadata written.

    ``hidden_states`` has shape ``(seq, n_layers, hidden)`` in any float dtype.
    For ``storage == "fp8_per_tensor_scale"`` we cast it through
    ``saturating_fp8_cast`` and store the scale alongside.

    ``token_ids``, ``input_ids`` (both ``(seq,) int64``) and ``loss_mask``
    (``(seq,) bool``) are stored as-is.
    """
    if storage not in VALID_STORAGE:
        raise ValueError(f"storage must be one of {VALID_STORAGE}, got {storage!r}")
    if hidden_states.dim() != 3:
        raise ValueError(
            f"hidden_states must be (seq, n_layers, hidden); got {tuple(hidden_states.shape)}"
        )
    seq_len, n_layers, hidden_size = hidden_states.shape
    for name, t in (("token_ids", token_ids), ("input_ids", input_ids), ("loss_mask", loss_mask)):
        if t.dim() != 1 or t.shape[0] != seq_len:
            raise ValueError(
                f"{name} must be (seq={seq_len},); got {tuple(t.shape)}"
            )

    # Coerce 1-D tensors to canonical dtypes.
    token_ids = token_ids.to(torch.int64).contiguous()
    input_ids = input_ids.to(torch.int64).contiguous()
    loss_mask = loss_mask.to(torch.bool).contiguous()

    abs_max = float(hidden_states.detach().to(torch.float32).abs().max().item()) \
        if hidden_states.numel() > 0 else 0.0

    tensors: dict[str, torch.Tensor] = {
        "token_ids": token_ids,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
    }

    if storage == "fp8_per_tensor_scale":
        fp8, scale = saturating_fp8_cast(hidden_states.contiguous())
        tensors["hidden_states"] = fp8.contiguous()
        # Store as a single-element fp32 tensor so safetensors can persist it.
        tensors["hidden_states_scale"] = torch.tensor([scale], dtype=torch.float32)
        hs_dtype_storage = "float8_e4m3fn"
    else:  # bf16
        tensors["hidden_states"] = hidden_states.to(torch.bfloat16).contiguous()
        hs_dtype_storage = "bfloat16"

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_name": str(source_name),
        "source_row_idx": str(int(source_row_idx)),
        "gen_timestamp": f"{time.time():.6f}",
        "storage": storage,
        "hidden_states_dtype_storage": hs_dtype_storage,
        "n_layers": str(int(n_layers)),
        "seq_len": str(int(seq_len)),
        "hidden_size": str(int(hidden_size)),
        "abs_max": f"{abs_max:.6e}",
        "layer_ids": json.dumps(list(layer_ids) if layer_ids is not None else []),
    }
    if extra_metadata:
        for k, v in extra_metadata.items():
            metadata[str(k)] = str(v)

    final = Path(path)
    _atomic_save_safetensors(tensors, metadata, final)
    return metadata


def load_trace(path: Union[str, Path]) -> dict:
    """Load a self-describing trace.

    Returns a dict with keys:
      - hidden_states (bf16, scale already applied)
      - token_ids (int64), input_ids (int64), loss_mask (bool)
      - metadata (dict[str, str])
      - hidden_states_scale (float, 1.0 for bf16 storage)
    """
    from safetensors import safe_open

    p = Path(path)
    out: dict = {}
    with safe_open(str(p), framework="pt") as f:
        meta = dict(f.metadata() or {})
        keys = set(f.keys())
        out["metadata"] = meta
        token_ids = f.get_tensor("token_ids") if "token_ids" in keys else None
        input_ids = f.get_tensor("input_ids") if "input_ids" in keys else None
        loss_mask = f.get_tensor("loss_mask") if "loss_mask" in keys else None
        hs_raw = f.get_tensor("hidden_states") if "hidden_states" in keys else None
        scale_t = f.get_tensor("hidden_states_scale") if "hidden_states_scale" in keys else None

    if hs_raw is None:
        raise ValueError(f"{p} has no hidden_states tensor")
    if token_ids is None:
        raise ValueError(f"{p} has no token_ids tensor")

    storage = meta.get("storage", "bf16")
    if storage == "fp8_per_tensor_scale":
        scale = float(scale_t.flatten()[0].item()) if scale_t is not None else 1.0
        hs = saturating_fp8_recover(hs_raw, scale)
    else:
        scale = 1.0
        hs = hs_raw.to(torch.bfloat16)

    out["hidden_states"] = hs
    out["hidden_states_scale"] = scale
    out["token_ids"] = token_ids.to(torch.int64) if token_ids is not None else None
    out["input_ids"] = input_ids.to(torch.int64) if input_ids is not None else None
    out["loss_mask"] = loss_mask.to(torch.bool) if loss_mask is not None else None
    return out


def validate_trace(path: Union[str, Path]) -> dict:
    """Cheap structural validation. Raises ValueError on schema problems.
    Returns the metadata dict on success.
    """
    d = load_trace(path)
    meta = d["metadata"]
    required = {"schema_version", "source_name", "source_row_idx", "storage", "n_layers"}
    missing = required - set(meta)
    if missing:
        raise ValueError(f"{path}: trace metadata missing required keys {missing}")
    if meta["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {meta['schema_version']!r} != {SCHEMA_VERSION!r}"
        )
    hs = d["hidden_states"]
    if torch.isnan(hs).any().item():
        raise ValueError(f"{path}: hidden_states contains NaN")
    return meta


__all__ = [
    "SCHEMA_VERSION",
    "FP8_E4M3FN_MAX",
    "VALID_STORAGE",
    "saturating_fp8_cast",
    "saturating_fp8_recover",
    "save_trace",
    "load_trace",
    "validate_trace",
]
