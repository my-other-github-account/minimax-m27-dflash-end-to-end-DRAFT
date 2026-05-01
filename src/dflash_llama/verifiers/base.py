"""BaseVerifier — model-family configuration shared by generation and training."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class BaseVerifier:
    """Static config for a verifier model family.

    The same dataclass is consumed by:
      - TraceGenerator (needs hidden_size, layer_ids, gguf_path or hf_path)
      - DFlashTrainer  (needs vocab_size, mask_token_id, hf_path for --verifier-name-or-path)

    Subclasses fill in the family-specific defaults. Every required field has
    a default so a generic dict-style ``BaseVerifier(**cfg)`` works for tests.
    """

    name: str = "generic"
    family: str = "generic"
    hidden_size: int = 0
    vocab_size: int = 0
    mask_token_id: int = 0
    num_hidden_layers: int = 0
    layer_ids: Sequence[int] = field(default_factory=tuple)
    # The drafter the trainer emits; kept here so the trainer doesn't need to
    # guess. For all current llama-family verifiers the drafter is "qwen3".
    drafter_arch: str = "qwen3"
    drafter_hidden_act: str = "silu"
    block_size: int = 8
    # Path to the verifier weights — exactly one of these is usually populated.
    gguf_path: Optional[str] = None
    hf_path: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layer_ids"] = list(self.layer_ids)
        return d

    def trainer_target_layer_ids(self) -> list[int]:
        """The list passed to the speculators trainer's --target-layer-ids.

        The speculators trainer auto-appends the final layer (it always
        consumes ``num_hidden_layers + 1`` taps), so we pass everything
        EXCEPT the final residual tap that we baked into our traces.
        """
        if not self.layer_ids:
            return []
        # Drop the final tap; the trainer auto-appends it.
        return list(self.layer_ids[:-1])

    def __post_init__(self):
        # Light validation. Layer ids should be sorted and unique.
        ids = list(self.layer_ids)
        if ids != sorted(ids):
            raise ValueError(f"{self.name}: layer_ids must be sorted, got {ids}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.name}: layer_ids must be unique, got {ids}")
        if self.gguf_path is not None and not isinstance(self.gguf_path, (str, Path)):
            raise TypeError("gguf_path must be a string path or None")
        if self.hf_path is not None and not isinstance(self.hf_path, (str, Path)):
            raise TypeError("hf_path must be a string path or None")
        # Coerce to str for downstream uses
        if isinstance(self.gguf_path, Path):
            self.gguf_path = str(self.gguf_path)
        if isinstance(self.hf_path, Path):
            self.hf_path = str(self.hf_path)
