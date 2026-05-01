"""Self-describing trace generation."""
from .format import (
    SCHEMA_VERSION,
    FP8_E4M3FN_MAX,
    saturating_fp8_cast,
    saturating_fp8_recover,
    save_trace,
    load_trace,
    validate_trace,
)
from .trace_generator import TraceGenerator
from .backends import BaseBackend, LlamaCppGGUFBackend

__all__ = [
    "TraceGenerator",
    "SCHEMA_VERSION",
    "FP8_E4M3FN_MAX",
    "saturating_fp8_cast",
    "saturating_fp8_recover",
    "save_trace",
    "load_trace",
    "validate_trace",
    "BaseBackend",
    "LlamaCppGGUFBackend",
]
