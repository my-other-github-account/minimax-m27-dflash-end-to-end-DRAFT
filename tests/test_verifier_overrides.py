"""Tests for verifier shape overrides — layer_ids, generic, factory passthrough."""
from __future__ import annotations

import pytest

from dflash_llama import (
    BaseVerifier,
    auto_layer_ids,
    generic_verifier,
    list_verifiers,
    load_verifier,
    register_verifier,
)


@pytest.fixture(autouse=False)
def _register_kimi():
    from dflash_llama.verifiers import _REGISTRY
    from dflash_llama.verifiers.experimental import kimi_k25
    snapshot = dict(_REGISTRY)
    register_verifier("kimi-k2.5", kimi_k25)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def test_layer_ids_override_minimax():
    """Passing layer_ids to a known family should override the factory default."""
    v = load_verifier(
        "minimax-m2.7-iq4-xs",
        gguf_path="/fake/path.gguf",
        layer_ids=[5, 10, 20, 40, 60, 61],
    )
    assert tuple(v.layer_ids) == (5, 10, 20, 40, 60, 61)
    # other shape fields untouched
    assert v.hidden_size == 3072
    assert v.num_hidden_layers == 62
    assert v.vocab_size == 200064


def test_shape_override_minimax():
    """Each shape kwarg should be overridable."""
    v = load_verifier(
        "minimax-m2.7-iq4-xs",
        gguf_path="/fake/path.gguf",
        hidden_size=4096,
        num_hidden_layers=80,
        vocab_size=131072,
        mask_token_id=131071,
        layer_ids=[2, 8, 16, 24, 40, 79],
    )
    assert v.hidden_size == 4096
    assert v.num_hidden_layers == 80
    assert v.vocab_size == 131072
    assert v.mask_token_id == 131071
    assert tuple(v.layer_ids) == (2, 8, 16, 24, 40, 79)


def test_default_layer_ids_unchanged_without_override(_register_kimi):
    """Without any override, the canonical defaults must remain intact."""
    v = load_verifier("minimax-m2.7-iq4-xs", gguf_path="/fake.gguf")
    assert tuple(v.layer_ids) == (2, 16, 30, 45, 59, 61)
    assert v.hidden_size == 3072
    assert v.vocab_size == 200064

    k = load_verifier("kimi-k2.5", gguf_path="/fake.gguf")
    assert tuple(k.layer_ids) == (1, 12, 24, 35, 47, 58)


def test_layer_ids_override_kimi(_register_kimi):
    v = load_verifier(
        "kimi-k2.5",
        gguf_path="/fake/path.gguf",
        layer_ids=[3, 15, 30, 45, 55, 60],
    )
    assert tuple(v.layer_ids) == (3, 15, 30, 45, 55, 60)


def test_generic_requires_shape_kwargs():
    """name='generic' must reject calls without the 4 required shape kwargs."""
    with pytest.raises(ValueError, match="hidden_size"):
        load_verifier("generic", hf_path="/x")


def test_generic_full_descriptor():
    """name='generic' with full kwargs builds an arbitrary verifier."""
    v = load_verifier(
        "generic",
        name_override="llama-3.1-8b",
        hf_path="/data/llama-3.1-8b",
        gguf_path="/data/llama-3.1-8b.gguf",
        hidden_size=4096,
        num_hidden_layers=32,
        vocab_size=128256,
        mask_token_id=128255,
        layer_ids=[2, 8, 16, 24, 30, 31],
    )
    assert v.name == "llama-3.1-8b"
    assert v.hidden_size == 4096
    assert v.num_hidden_layers == 32
    assert v.vocab_size == 128256
    assert v.mask_token_id == 128255
    assert tuple(v.layer_ids) == (2, 8, 16, 24, 30, 31)
    assert v.hf_path == "/data/llama-3.1-8b"
    assert v.gguf_path == "/data/llama-3.1-8b.gguf"


def test_generic_auto_layer_ids():
    """When layer_ids omitted, num_layer_taps drives auto_layer_ids spread."""
    v = load_verifier(
        "generic",
        name_override="my-model-32layer",
        hidden_size=4096,
        num_hidden_layers=32,
        vocab_size=128000,
        mask_token_id=127999,
        num_layer_taps=6,
    )
    assert len(v.layer_ids) == 6
    assert v.layer_ids[-1] == 31  # final residual
    assert all(0 <= L <= 31 for L in v.layer_ids)
    # sorted + unique guaranteed by BaseVerifier post_init
    assert list(v.layer_ids) == sorted(set(v.layer_ids))


def test_auto_layer_ids_function():
    """auto_layer_ids spreads taps and always includes the final residual."""
    ids = auto_layer_ids(num_hidden_layers=62, num_taps=6)
    assert len(ids) == 6
    assert ids[-1] == 61
    assert ids[0] >= 1

    # Edge cases
    ids2 = auto_layer_ids(num_hidden_layers=32, num_taps=2)
    assert len(ids2) == 2
    assert ids2[-1] == 31

    with pytest.raises(ValueError, match="num_taps must be >= 2"):
        auto_layer_ids(num_hidden_layers=32, num_taps=1)
    with pytest.raises(ValueError, match="num_taps .* must be <="):
        auto_layer_ids(num_hidden_layers=8, num_taps=10)


def test_unknown_verifier_error_mentions_generic():
    """The 'unknown verifier' error message should hint at name='generic'."""
    with pytest.raises(KeyError, match="generic"):
        load_verifier("never-heard-of-this-model", gguf_path="/x")


def test_generic_verifier_factory_direct():
    """generic_verifier() should be importable + work standalone."""
    v = generic_verifier(
        name="my-7b",
        hidden_size=3584,
        num_hidden_layers=28,
        vocab_size=151936,
        mask_token_id=151643,
        layer_ids=[1, 7, 14, 21, 26, 27],
    )
    assert isinstance(v, BaseVerifier)
    assert tuple(v.layer_ids) == (1, 7, 14, 21, 26, 27)
    assert v.hidden_size == 3584


def test_factory_warns_on_dropped_kwarg():
    """A factory that doesn't accept a kwarg should emit a RuntimeWarning."""
    # Register a strict factory that doesn't take **kwargs and only accepts
    # the standard hf/gguf paths.
    from dflash_llama import register_verifier

    def strict_factory(*, hf_path=None, gguf_path=None):
        return BaseVerifier(
            name="strict", family="strict",
            hidden_size=128, num_hidden_layers=4, vocab_size=1000, mask_token_id=999,
            layer_ids=[0, 3], hf_path=hf_path, gguf_path=gguf_path,
        )

    register_verifier("strict-test", strict_factory)
    with pytest.warns(RuntimeWarning, match="does not accept"):
        v = load_verifier("strict-test", layer_ids=[1, 2, 3])
    # The override was dropped — defaults stand
    assert tuple(v.layer_ids) == (0, 3)


def test_list_includes_known():
    names = list_verifiers()
    assert "minimax-m2.7" in names
    assert "minimax-m2.7-iq4-xs" in names
    # Experimental names are NOT in the default registry — see
    # list_experimental_verifiers() / dflash_llama.verifiers.experimental
    assert "kimi-k2.5" not in names
    assert "qwen3-4b" not in names
    assert "qwen3-14b" not in names
