"""Tests for the verifier-config registry."""
from __future__ import annotations

import pytest

from dflash_llama.verifiers import (
    BaseVerifier,
    list_verifiers,
    list_experimental_verifiers,
    load_verifier,
    register_verifier,
)


@pytest.fixture(autouse=False)
def _register_experimental():
    """Opt-in fixture — register the experimental factories under their
    canonical names for the duration of a test, then restore.
    Snapshots the registry so leakage is impossible."""
    from dflash_llama.verifiers import _REGISTRY
    from dflash_llama.verifiers.experimental import (
        kimi_k25, qwen3, qwen3_4b, qwen3_14b,
    )
    snapshot = dict(_REGISTRY)
    register_verifier("kimi-k2.5", kimi_k25)
    register_verifier("qwen3", qwen3)
    register_verifier("qwen3-4b", qwen3_4b)
    register_verifier("qwen3-14b", qwen3_14b)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def test_minimax_m27_shape():
    v = load_verifier("minimax-m2.7", gguf_path="/dummy")
    assert v.family == "minimax_m2"
    assert v.hidden_size == 3072
    assert v.vocab_size == 200064
    assert v.mask_token_id == 200054
    assert v.num_hidden_layers == 62
    assert tuple(v.layer_ids) == (2, 16, 30, 45, 59, 61)
    assert v.gguf_path == "/dummy"


def test_minimax_m27_iq4_xs_alias():
    v = load_verifier("minimax-m2.7-iq4-xs", gguf_path="/path/to/UD-IQ4_XS")
    # Same shape as the bf16 base
    assert v.hidden_size == 3072
    assert v.vocab_size == 200064
    assert v.mask_token_id == 200054
    assert v.name == "minimax-m2.7-iq4-xs"


def test_kimi_k25_shape(_register_experimental):
    v = load_verifier("kimi-k2.5", hf_path="/dummy")
    assert v.family == "kimi_k25"
    assert v.hidden_size == 7168
    assert v.vocab_size == 163840
    assert v.mask_token_id == 163838
    assert v.num_hidden_layers == 61
    assert tuple(v.layer_ids) == (1, 12, 24, 35, 47, 58)


def test_qwen3_4b_and_14b(_register_experimental):
    a = load_verifier("qwen3-4b")
    assert a.family == "qwen3"
    assert a.hidden_size == 2560
    assert a.num_hidden_layers == 36

    b = load_verifier("qwen3-14b")
    assert b.hidden_size == 5120
    assert b.num_hidden_layers == 48


def test_generic_qwen3_with_overrides(_register_experimental):
    v = load_verifier("qwen3", hidden_size=1024, num_hidden_layers=24)
    assert v.hidden_size == 1024
    assert v.num_hidden_layers == 24


def test_unknown_verifier_raises():
    with pytest.raises(KeyError):
        load_verifier("totally-fake-model")


def test_experimental_factories_listable():
    """The experimental namespace exposes the factories we moved out."""
    exp = list_experimental_verifiers()
    for name in ("kimi_k25", "qwen3_4b", "qwen3_14b",
                 "deepseek_v4_flash", "deepseek_v4_pro",
                 "nemotron3_super_120b", "nemotron3_nano_30b_a3b"):
        assert name in exp, f"missing experimental factory: {name}"


def test_experimental_factories_not_in_default_registry():
    """Default `list_verifiers()` must not advertise unverified families."""
    names = list_verifiers()
    for forbidden in ("kimi-k2.5", "qwen3-4b", "qwen3-14b",
                      "deepseek-v4-flash", "deepseek-v4-pro",
                      "nemotron3-super-120b", "nemotron3-nano-30b-a3b"):
        assert forbidden not in names, (
            f"{forbidden} leaked into default registry — it must stay in "
            f"dflash_llama.verifiers.experimental"
        )


def test_layer_ids_must_be_sorted():
    with pytest.raises(ValueError):
        BaseVerifier(layer_ids=(5, 1, 3))


def test_trainer_target_layer_ids_drops_last():
    v = load_verifier("minimax-m2.7")
    # The trainer auto-appends the final tap
    assert v.trainer_target_layer_ids() == [2, 16, 30, 45, 59]


def test_list_verifiers_contains_known_names():
    names = list_verifiers()
    assert "minimax-m2.7" in names
    assert "minimax-m2.7-iq4-xs" in names
    # Experimental names are NOT in the default registry — they must be
    # opted into via register_verifier().
    assert "kimi-k2.5" not in names
    assert "qwen3-4b" not in names
    assert "qwen3-14b" not in names


def test_to_dict_serialisable():
    v = load_verifier("minimax-m2.7", gguf_path="/x")
    d = v.to_dict()
    assert d["hidden_size"] == 3072
    assert isinstance(d["layer_ids"], list)


def test_register_verifier_roundtrip():
    """register_verifier adds a custom factory that load_verifier can find."""
    from dflash_llama import (
        BaseVerifier,
        list_verifiers,
        load_verifier,
        register_verifier,
    )

    name = "test-rt-verifier-9b"

    def factory(*, hf_path=None, gguf_path=None, **kw):
        return BaseVerifier(
            name=name,
            hidden_size=4096,
            num_hidden_layers=32,
            vocab_size=131072,
            mask_token_id=131071,
            layer_ids=[2, 8, 16, 24, 30, 31],
            hf_path=hf_path,
            gguf_path=gguf_path,
            **kw,
        )

    register_verifier(name, factory)
    assert name in list_verifiers()
    v = load_verifier(name)
    assert v.name == name
    assert v.hidden_size == 4096


def test_top_level_public_api_exports():
    """All advertised public names import from the dflash_llama root."""
    import dflash_llama

    expected = {
        "TraceGenerator", "DFlashTrainer", "SelfDescribingTraceDataset",
        "load_verifier", "list_verifiers", "register_verifier", "BaseVerifier",
        "save_trace", "load_trace", "saturating_fp8_cast",
        "assemble_prompts_arrow", "build_vocab_maps",
        "__version__", "SCHEMA_VERSION",
    }
    for n in expected:
        assert hasattr(dflash_llama, n), f"missing top-level export: {n}"
        assert n in dflash_llama.__all__, f"{n} not in __all__"
