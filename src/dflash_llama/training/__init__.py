"""DFlash training pipeline."""
from .dataset import SelfDescribingTraceDataset
from .vocab_maps import build_vocab_maps, build_vocab_maps_from_counts, count_token_frequencies
from .prompts import assemble_prompts_arrow
from .smoke import run_smoke_test, SmokeResult
from .trainer import DFlashTrainer
from .eval import offline_eval
from .fp8 import (
    FP8Recipe,
    make_te_recipe,
    wrap_with_te,
    fp8_autocast_ctx,
    current_arch,
)

__all__ = [
    "SelfDescribingTraceDataset",
    "DFlashTrainer",
    "build_vocab_maps",
    "build_vocab_maps_from_counts",
    "count_token_frequencies",
    "assemble_prompts_arrow",
    "run_smoke_test",
    "SmokeResult",
    "offline_eval",
    # FP8 production training
    "FP8Recipe",
    "make_te_recipe",
    "wrap_with_te",
    "fp8_autocast_ctx",
    "current_arch",
]
