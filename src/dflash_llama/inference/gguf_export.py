"""GGUF export of a trained DFlash drafter — PLACEHOLDER.

The v2 ``prep_full_for_buun_converter.py`` flow is complex (modifies the
GGUF converter, tweaks tensor names, runs a custom buun build). It is
deferred to a follow-up session.

When implemented, this module should expose::

    def export_to_gguf(checkpoint, output_path, *, ...): ...

raising no NotImplementedError once it lands.
"""
from __future__ import annotations


def export_to_gguf(checkpoint: str, output_path: str, **kwargs) -> None:
    raise NotImplementedError(
        "GGUF export is not implemented in this version. "
        "See repro/03-inference.md for the v2 buun-converter recipe."
    )


__all__ = ["export_to_gguf"]
