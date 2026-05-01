"""dflash_llama — self-describing fp8 trace generation + DFlash drafter training."""
from .version import __version__, SCHEMA_VERSION
from .verifiers import load_verifier, list_verifiers, register_verifier, BaseVerifier
from .generation import TraceGenerator
from .generation.format import load_trace, save_trace, saturating_fp8_cast
from .training import DFlashTrainer, SelfDescribingTraceDataset
from .training.prompts import assemble_prompts_arrow
from .training.vocab_maps import build_vocab_maps

__all__ = [
    # version + schema
    "__version__",
    "SCHEMA_VERSION",
    # verifier registry
    "load_verifier",
    "list_verifiers",
    "register_verifier",
    "BaseVerifier",
    # generation
    "TraceGenerator",
    "save_trace",
    "load_trace",
    "saturating_fp8_cast",
    # training
    "DFlashTrainer",
    "SelfDescribingTraceDataset",
    "assemble_prompts_arrow",
    "build_vocab_maps",
]
