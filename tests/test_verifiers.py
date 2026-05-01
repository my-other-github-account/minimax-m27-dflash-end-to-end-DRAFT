"""Tests for the verifier-config registry."""
from __future__ import annotations

import pytest

from dflash_llama.verifiers import (
    BaseVerifier,
    list_verifiers,
    load_verifier,
)


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


def test_kimi_k25_shape():
    v = load_verifier("kimi-k2.5", hf_path="/dummy")
    assert v.family == "kimi_k25"
    assert v.hidden_size == 7168
    assert v.vocab_size == 163840
    assert v.mask_token_id == 163838
    assert v.num_hidden_layers == 61
    assert tuple(v.layer_ids) == (1, 12, 24, 35, 47, 58)


def test_qwen3_4b_and_14b():
    a = load_verifier("qwen3-4b")
    assert a.family == "qwen3"
    assert a.hidden_size == 2560
    assert a.num_hidden_layers == 36

    b = load_verifier("qwen3-14b")
    assert b.hidden_size == 5120
    assert b.num_hidden_layers == 48


def test_generic_qwen3_with_overrides():
    v = load_verifier("qwen3", hidden_size=1024, num_hidden_layers=24)
    assert v.hidden_size == 1024
    assert v.num_hidden_layers == 24


def test_unknown_verifier_raises():
    with pytest.raises(KeyError):
        load_verifier("totally-fake-model")


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
    assert "kimi-k2.5" in names
    assert "qwen3-4b" in names
    assert "qwen3-14b" in names


def test_to_dict_serialisable():
    v = load_verifier("minimax-m2.7", gguf_path="/x")
    d = v.to_dict()
    assert d["hidden_size"] == 3072
    assert isinstance(d["layer_ids"], list)
