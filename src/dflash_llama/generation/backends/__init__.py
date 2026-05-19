"""Verifier-execution backends.

The library has a single backend: ``TracegenClientBackend``, which talks to a
persistent ``TraceServer`` and unlocks the batched ``run_many`` fast path. The
old per-prompt-subprocess backend (``LlamaCppGGUFBackend``) was removed in
favor of the persistent server's ~2-5× higher throughput.

See ``repro/08-tracegen-server.md`` for the end-to-end recipe.
"""
from .base import BaseBackend
from .tracegen_client import TracegenClientBackend

__all__ = ["BaseBackend", "TracegenClientBackend"]
