from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


class _FakeLigerOutput:
    def __init__(self, loss, predicted_tokens):
        self.loss = loss
        self.predicted_tokens = predicted_tokens


class _FakeLigerFusedLinearCrossEntropyLoss:
    def __init__(self, ignore_index=-100, reduction="none", return_predicted_tokens=False, **kwargs):  # noqa: ARG002
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.return_predicted_tokens = return_predicted_tokens

    def __call__(self, lin_weight, _input, target, bias=None):
        logits = _input @ lin_weight.t()
        if bias is not None:
            logits = logits + bias
        loss = torch.nn.functional.cross_entropy(
            logits,
            target,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )
        predicted_tokens = torch.argmax(logits, dim=-1)
        return _FakeLigerOutput(loss=loss, predicted_tokens=predicted_tokens)


@pytest.fixture
def fake_liger(monkeypatch):
    from dflash_llama.training import liger_wrap

    monkeypatch.setattr(liger_wrap, "LIGER_AVAILABLE", True)
    monkeypatch.setattr(liger_wrap, "LigerFusedLinearCrossEntropyLoss", _FakeLigerFusedLinearCrossEntropyLoss, raising=False)
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
