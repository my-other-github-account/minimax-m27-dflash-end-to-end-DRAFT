"""Tests for the new verifier factories: DeepSeek-V4 + Nemotron-3."""
from __future__ import annotations

import pytest

from dflash_llama import (
    BaseVerifier,
    list_verifiers,
    load_verifier,
    register_verifier,
)
from dflash_llama.verifiers.experimental import (
    deepseek_v4_flash,
    deepseek_v4_pro,
    nemotron3_nano_30b_a3b,
    nemotron3_super_120b,
)


@pytest.fixture(autouse=False)
def _register_experimental_dsv4_nemo3():
    """Opt-in fixture — register experimental factories under their
    canonical names for the duration of a test, then restore."""
    from dflash_llama.verifiers import _REGISTRY
    snapshot = dict(_REGISTRY)
    register_verifier("deepseek-v4-flash", deepseek_v4_flash)
    register_verifier("deepseek-v4-pro", deepseek_v4_pro)
    register_verifier("nemotron3-super-120b", nemotron3_super_120b)
    register_verifier("nemotron3-nano-30b-a3b", nemotron3_nano_30b_a3b)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


# ----- DeepSeek-V4-Flash --------------------------------------------------

def test_deepseek_v4_flash_defaults():
    v = deepseek_v4_flash(gguf_path="/fake.gguf")
    assert isinstance(v, BaseVerifier)
    assert v.name == "deepseek-v4-flash"
    assert v.family == "deepseek_v4"
    assert v.hidden_size == 4096
    assert v.num_hidden_layers == 43
    assert v.vocab_size == 129280
    assert v.mask_token_id == 1  # EOS
    assert tuple(v.layer_ids) == (2, 11, 21, 32, 41, 42)
    assert v.gguf_path == "/fake.gguf"


def test_deepseek_v4_flash_layer_ids_override(_register_experimental_dsv4_nemo3):
    v = load_verifier(
        "deepseek-v4-flash",
        gguf_path="/x",
        layer_ids=[1, 10, 20, 30, 40, 42],
    )
    assert tuple(v.layer_ids) == (1, 10, 20, 30, 40, 42)
    assert v.hidden_size == 4096  # other defaults intact


def test_deepseek_v4_flash_full_shape_override(_register_experimental_dsv4_nemo3):
    v = load_verifier(
        "deepseek-v4-flash",
        gguf_path="/x",
        hidden_size=5120,
        num_hidden_layers=48,
        vocab_size=200064,
        mask_token_id=200054,
        layer_ids=[2, 12, 24, 36, 46, 47],
    )
    assert v.hidden_size == 5120
    assert v.num_hidden_layers == 48
    assert v.vocab_size == 200064
    assert v.mask_token_id == 200054
    assert tuple(v.layer_ids) == (2, 12, 24, 36, 46, 47)


def test_deepseek_v4_pro():
    v = deepseek_v4_pro(gguf_path="/fake.gguf")
    assert v.name == "deepseek-v4-pro"
    assert v.family == "deepseek_v4"
    assert v.hidden_size == 4096  # same shape as Flash by default
    assert tuple(v.layer_ids) == (2, 11, 21, 32, 41, 42)


# ----- Nemotron-3 ---------------------------------------------------------

def test_nemotron3_super_120b_defaults():
    v = nemotron3_super_120b(hf_path="/fake/hf")
    assert v.name == "nemotron3-super-120b"
    assert v.family == "nemotron_h"
    assert v.hidden_size == 4096
    assert v.num_hidden_layers == 88
    assert v.vocab_size == 131072
    assert v.mask_token_id == 0  # PAD
    assert tuple(v.layer_ids) == (3, 22, 44, 66, 86, 87)


def test_nemotron3_nano_30b_a3b_defaults():
    v = nemotron3_nano_30b_a3b(hf_path="/fake/hf")
    assert v.name == "nemotron3-nano-30b-a3b"
    assert v.family == "nemotron_h"
    assert v.hidden_size == 2688
    assert v.num_hidden_layers == 52
    assert v.vocab_size == 131072
    assert v.mask_token_id == 0
    assert tuple(v.layer_ids) == (2, 13, 26, 39, 50, 51)


def test_nemotron3_super_layer_override(_register_experimental_dsv4_nemo3):
    v = load_verifier(
        "nemotron3-super-120b",
        hf_path="/x",
        layer_ids=[5, 25, 50, 75, 85, 87],
    )
    assert tuple(v.layer_ids) == (5, 25, 50, 75, 85, 87)
    assert v.num_hidden_layers == 88


# ----- Registry membership ------------------------------------------------

def test_experimental_factories_listed_in_experimental_namespace():
    """Moved factories must be visible via list_experimental_verifiers(),
    NOT the default list_verifiers()."""
    from dflash_llama import list_experimental_verifiers
    exp = list_experimental_verifiers()
    for name in ("deepseek_v4_flash", "deepseek_v4_pro",
                 "nemotron3_super_120b", "nemotron3_nano_30b_a3b"):
        assert name in exp
    # Default registry keeps these out
    default = list_verifiers()
    for forbidden in ("deepseek-v4-flash", "deepseek-v4-pro",
                      "nemotron3-super-120b", "nemotron3-nano-30b-a3b"):
        assert forbidden not in default


# ----- Autodetect ---------------------------------------------------------

def test_autodetect_deepseek_v4(tmp_path):
    """A config.json with model_type=deepseek_v4 should pick the new factory."""
    import json
    cfg_dir = tmp_path / "dsv4"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "vocab_size": 129280,
        "eos_token_id": 1,
    }))
    v = load_verifier(None, hf_path=str(cfg_dir))
    assert v.family == "deepseek_v4"
    assert v.hidden_size == 4096
    assert v.num_hidden_layers == 43


def test_autodetect_nemotron_super(tmp_path):
    """A config.json with model_type=nemotron_h + hidden=4096 picks Super-120B."""
    import json
    cfg_dir = tmp_path / "n3super"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps({
        "model_type": "nemotron_h",
        "hidden_size": 4096,
        "num_hidden_layers": 88,
        "vocab_size": 131072,
        "pad_token_id": 0,
    }))
    v = load_verifier(None, hf_path=str(cfg_dir))
    assert v.family == "nemotron_h"
    assert v.hidden_size == 4096
    assert v.num_hidden_layers == 88


def test_autodetect_nemotron_nano(tmp_path):
    """A config.json with model_type=nemotron_h + hidden=2688 picks Nano-30B."""
    import json
    cfg_dir = tmp_path / "n3nano"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps({
        "model_type": "nemotron_h",
        "hidden_size": 2688,
        "num_hidden_layers": 52,
        "vocab_size": 131072,
        "pad_token_id": 0,
    }))
    v = load_verifier(None, hf_path=str(cfg_dir))
    assert v.family == "nemotron_h"
    assert v.hidden_size == 2688
    assert v.num_hidden_layers == 52


def test_autodetect_nemotron_unknown_size_falls_through_to_generic(tmp_path):
    """Unknown Nemotron variant should fall through to generic, not pick a wrong factory."""
    import json
    cfg_dir = tmp_path / "n3weird"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps({
        "model_type": "nemotron_h",
        "hidden_size": 5120,           # unknown size
        "num_hidden_layers": 100,
        "vocab_size": 131072,
        "pad_token_id": 0,
    }))
    v = load_verifier(None, hf_path=str(cfg_dir))
    # Generic fallback preserves the actual config's shape
    assert v.hidden_size == 5120
    assert v.num_hidden_layers == 100
    # And the family tag matches the source model_type
    assert v.family == "nemotron_h"
