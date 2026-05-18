"""dflash_llama — self-describing fp8 trace generation + DFlash drafter training."""
from .version import __version__, SCHEMA_VERSION
from .verifiers import (
    load_verifier,
    list_verifiers,
    list_experimental_verifiers,
    register_verifier,
    BaseVerifier,
    generic_verifier,
    auto_layer_ids,
    minimax_m27,
    minimax_m27_iq4_xs,
)
from .hub import cache_root, resolve_hf_repo, resolve_gguf_repo
from .generation import TraceGenerator
from .generation.format import load_trace, save_trace, saturating_fp8_cast
from .tracegen import TraceClient, TraceServer
from .training import DFlashTrainer, SelfDescribingTraceDataset
from .training.prompts import assemble_prompts_arrow
from .training.vocab_maps import build_vocab_maps
from .inference import (
    export_to_gguf,
    LlamaServer,
    benchmark,
    SpeculativeReport,
    parse_speculative_log,
    chain_pred_from_val,
)

__all__ = [
    # version + schema
    "__version__",
    "SCHEMA_VERSION",
    # verifier registry
    "load_verifier",
    "list_verifiers",
    "list_experimental_verifiers",
    "register_verifier",
    "BaseVerifier",
    "generic_verifier",
    "auto_layer_ids",
    "minimax_m27",
    "minimax_m27_iq4_xs",
    # hub / model slug resolution
    "cache_root",
    "resolve_hf_repo",
    "resolve_gguf_repo",
    # generation
    "TraceGenerator",
    "TraceClient",
    "TraceServer",
    "save_trace",
    "load_trace",
    "saturating_fp8_cast",
    # training
    "DFlashTrainer",
    "SelfDescribingTraceDataset",
    "assemble_prompts_arrow",
    "build_vocab_maps",
    # inference
    "export_to_gguf",
    "LlamaServer",
    "benchmark",
    "SpeculativeReport",
    "parse_speculative_log",
    "chain_pred_from_val",
]
