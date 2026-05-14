from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _fake_liger_fused_linear_cross_entropy_apply(
    _input,
    weight,
    target,
    bias=None,
    ce_weight=None,  # noqa: ARG001
    ignore_index=-100,
    lse_square_scale=0.0,  # noqa: ARG001
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,  # noqa: ARG001
    return_z_loss=False,
    accum_dtype=None,  # noqa: ARG001
    use_token_scaling=False,  # noqa: ARG001
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    logits = _input @ weight.t()
    if bias is not None:
        logits = logits + bias
    loss = torch.nn.functional.cross_entropy(
        logits,
        target,
        reduction=reduction,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    predicted_tokens = torch.argmax(logits, dim=-1) if return_predicted_tokens else None
    token_accuracy = ((predicted_tokens == target).float() if return_token_accuracy else None)
    z_loss = torch.zeros_like(loss) if return_z_loss else None
    return loss, z_loss, token_accuracy, predicted_tokens


@pytest.fixture
def fake_liger(monkeypatch):
    from dflash_llama.training import liger_wrap

    monkeypatch.setattr(liger_wrap, "LIGER_AVAILABLE", True)
    fake_fn = type(
        "FakeLigerFusedLinearCrossEntropyFunction",
        (),
        {"apply": staticmethod(_fake_liger_fused_linear_cross_entropy_apply)},
    )
    monkeypatch.setattr(
        liger_wrap,
        "LigerFusedLinearCrossEntropyFunction",
        fake_fn,
        raising=False,
    )
    monkeypatch.setattr(liger_wrap, "LIGER_VERSION", "fake-test", raising=False)
    monkeypatch.setattr(
        liger_wrap,
        "liger_rotary_pos_emb",
        lambda q, k, cos, sin, position_ids=None, unsqueeze_dim=1: (q, k),
        raising=False,
    )
    monkeypatch.setattr(
        liger_wrap,
        "LigerRMSNorm",
        type("FakeLigerRMSNorm", (torch.nn.Module,), {
            "__init__": lambda self, hidden_size, eps=1e-6: torch.nn.Module.__init__(self) or setattr(self, "weight", torch.nn.Parameter(torch.ones(hidden_size))) or setattr(self, "variance_epsilon", eps),
            "forward": lambda self, x: x,
        }),
        raising=False,
    )
    return liger_wrap


def test_liger_fused_linear_ce_matches_reference(fake_liger):
    torch.manual_seed(0)
    batch = 1
    block_size = 8
    length = 16
    hidden_size = 12
    vocab_size = 32

    hidden_states = torch.randn(batch, length, hidden_size, dtype=torch.float32)
    lm_head_weight = torch.randn(vocab_size, hidden_size, dtype=torch.float32)
    targets = torch.randint(0, vocab_size, (batch, length), dtype=torch.long)
    loss_mask = torch.ones(batch, length, dtype=torch.bool)
    loss_mask[:, ::block_size] = 0

    ref_loss, ref_metrics, ref_pred = fake_liger.dflash_weighted_ce_reference(
        hidden_states,
        lm_head_weight,
        targets,
        loss_mask,
        block_size=block_size,
    )
    liger_loss, liger_metrics, liger_pred = fake_liger.dflash_weighted_ce_liger(
        hidden_states,
        lm_head_weight,
        targets,
        loss_mask,
        block_size=block_size,
    )

    assert torch.allclose(liger_loss, ref_loss, atol=1e-4, rtol=1e-4)
    assert torch.equal(liger_pred, ref_pred)
    assert torch.allclose(liger_metrics["full_acc"], ref_metrics["full_acc"], atol=1e-6, rtol=1e-6)
    for pos in range(1, block_size):
        key = f"position {pos} acc"
        assert torch.allclose(liger_metrics[key], ref_metrics[key], atol=1e-6, rtol=1e-6)


def test_liger_weighted_ce_respects_position_weights(fake_liger):
    torch.manual_seed(0)
    batch = 1
    block_size = 8
    length = 16
    hidden_size = 10
    vocab_size = 24
    hidden_states = torch.randn(batch, length, hidden_size, dtype=torch.float32)
    lm_head_weight = torch.randn(vocab_size, hidden_size, dtype=torch.float32)
    targets = torch.randint(0, vocab_size, (batch, length), dtype=torch.long)
    loss_mask = torch.ones(batch, length, dtype=torch.bool)
    loss_mask[:, ::block_size] = 0

    weighted_loss, _weighted_metrics, _weighted_pred = fake_liger.dflash_weighted_ce_liger(
        hidden_states,
        lm_head_weight,
        targets,
        loss_mask,
        block_size=block_size,
        gamma=4.0,
    )
    flatter_loss, _flatter_metrics, _flatter_pred = fake_liger.dflash_weighted_ce_liger(
        hidden_states,
        lm_head_weight,
        targets,
        loss_mask,
        block_size=block_size,
        gamma=1e9,
    )

    assert not torch.allclose(weighted_loss, flatter_loss, atol=1e-6, rtol=1e-6)


def test_valid_anchor_positions_drops_invalid_slots(fake_liger):
    anchors = torch.tensor([17, 9, 0, 0], dtype=torch.long)
    valid = torch.tensor([True, True, False, False])

    selected = fake_liger._valid_anchor_positions(anchors, valid)

    assert torch.equal(selected, torch.tensor([17, 9], dtype=torch.long))


def test_valid_anchor_positions_requires_at_least_one_anchor(fake_liger):
    anchors = torch.zeros(4, dtype=torch.long)
    valid = torch.zeros(4, dtype=torch.bool)

    with pytest.raises(ValueError, match="No valid anchors"):
        fake_liger._valid_anchor_positions(anchors, valid)
