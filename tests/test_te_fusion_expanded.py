from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


def _fake_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):  # noqa: ARG001
    return q, k


def _fake_eager_attention_forward(
    module,
    q,
    k,
    v,
    attention_mask,
    dropout=0.0,
    scaling=1.0,
    sliding_window=None,
    **kwargs,
):  # noqa: ARG001
    tail = slice(-q.shape[-2], None)
    out = (q + k[:, :, tail, :] + v[:, :, tail, :]) / 3.0
    return out * scaling, None


ALL_ATTENTION_FUNCTIONS = {}
eager_attention_forward = _fake_eager_attention_forward
apply_rotary_pos_emb = _fake_apply_rotary_pos_emb


class FakeLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, params_dtype=None, device=None, **kwargs):  # noqa: ARG002
        super().__init__()
        dtype = params_dtype or torch.float32
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class FakeLayerNormLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        eps=1e-5,
        bias=True,
        normalization="RMSNorm",
        params_dtype=None,
        device=None,
        return_layernorm_output=False,
        **kwargs,
    ):  # noqa: ARG002
        super().__init__()
        assert normalization == "RMSNorm"
        dtype = params_dtype or torch.float32
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.return_layernorm_output = return_layernorm_output
        self.layer_norm_weight = nn.Parameter(torch.ones(in_features, device=device, dtype=dtype))
        self.register_parameter("layer_norm_bias", None)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        normed = normed.to(x.dtype) * self.layer_norm_weight.to(x.dtype)
        out = torch.nn.functional.linear(normed, self.weight, self.bias)
        if self.return_layernorm_output:
            return out, normed
        return out


class FakeLayerNormMLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        eps=1e-5,
        bias=True,
        normalization="RMSNorm",
        activation="swiglu",
        params_dtype=None,
        device=None,
        **kwargs,
    ):  # noqa: ARG002
        super().__init__()
        assert normalization == "RMSNorm"
        assert activation == "swiglu"
        dtype = params_dtype or torch.float32
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.eps = eps
        self.layer_norm_weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.fc1_weight = nn.Parameter(
            torch.empty(ffn_hidden_size * 2, hidden_size, device=device, dtype=dtype)
        )
        self.fc2_weight = nn.Parameter(
            torch.empty(hidden_size, ffn_hidden_size, device=device, dtype=dtype)
        )
        nn.init.normal_(self.fc1_weight, std=0.02)
        nn.init.normal_(self.fc2_weight, std=0.02)
        if bias:
            self.fc1_bias = nn.Parameter(torch.zeros(ffn_hidden_size * 2, device=device, dtype=dtype))
            self.fc2_bias = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))
        else:
            self.register_parameter("fc1_bias", None)
            self.register_parameter("fc2_bias", None)

    def forward(self, x):
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        normed = normed.to(x.dtype) * self.layer_norm_weight.to(x.dtype)
        fc1 = torch.nn.functional.linear(normed, self.fc1_weight, self.fc1_bias)
        gate, up = fc1.chunk(2, dim=-1)
        swiglu = torch.nn.functional.silu(gate) * up
        return torch.nn.functional.linear(swiglu, self.fc2_weight, self.fc2_bias)


class FakeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FakeMLP(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class FakeAttention(nn.Module):
    def __init__(self, hidden_size=8, num_heads=2):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.layer_idx = 0
        self.head_dim = hidden_size // num_heads
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = False
        self.sliding_window = None
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = FakeRMSNorm(self.head_dim)
        self.k_norm = FakeRMSNorm(self.head_dim)

    def forward(
        self,
        hidden_states,
        target_hidden,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_output, attn_weights = eager_attention_forward(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_size=8, ffn_hidden_size=16):
        super().__init__()
        self.self_attn = FakeAttention(hidden_size=hidden_size)
        self.mlp = FakeMLP(hidden_size, ffn_hidden_size)
        self.input_layernorm = FakeRMSNorm(hidden_size)
        self.post_attention_layernorm = FakeRMSNorm(hidden_size)

    def forward(self, hidden_states, target_hidden, position_embeddings, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class FakeDFlashModel(nn.Module):
    def __init__(self, hidden_size=8, ffn_hidden_size=16, num_layers=2, vocab_size=32):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [FakeDecoderLayer(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size) for _ in range(num_layers)]
        )
        self.norm = FakeRMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states, target_hidden):
        bsz, seq_len, hidden_size = hidden_states.shape
        cos = torch.ones(bsz, seq_len * 2, hidden_size // 2, dtype=hidden_states.dtype)
        sin = torch.zeros_like(cos)
        position_embeddings = (cos, sin)
        for layer in self.layers:
            hidden_states = layer(hidden_states, target_hidden, position_embeddings)
        return self.lm_head(self.norm(hidden_states))


@pytest.fixture
def fake_te(monkeypatch):
    from dflash_llama.training import te_wrap

    monkeypatch.setattr(te_wrap, "TE_AVAILABLE", True)
    monkeypatch.setattr(
        te_wrap,
        "te",
        SimpleNamespace(
            Linear=FakeLinear,
            LayerNormLinear=FakeLayerNormLinear,
            LayerNormMLP=FakeLayerNormMLP,
        ),
        raising=False,
    )
    monkeypatch.setenv("TE_USE_FUSED", "1")
    return te_wrap


def test_wrap_with_te_preserves_forward_with_fake_te(fake_te):
    torch.manual_seed(0)
    model = FakeDFlashModel()
    wrapped = copy.deepcopy(model)
    hidden_states = torch.randn(1, 4, 8)
    target_hidden = torch.randn(1, 6, 8)

    expected = model(hidden_states, target_hidden)
    fake_te.wrap_with_te(wrapped, fp8=True)
    actual = wrapped(hidden_states, target_hidden)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_fusion_coverage_reports_zero_unfused_linears(fake_te):
    model = FakeDFlashModel(num_layers=2)
    fake_te.wrap_with_te(model, fp8=True)
    summary = fake_te.fusion_coverage(model)["summary"]
    assert summary["nn_linear"] == 0
    assert summary["unfused"] == 0
    assert summary["te_layernorm_mlp"] == 2
    assert summary["te_layernorm_linear"] == 3
    assert summary["te_linear"] == 6


def test_state_dict_rename_round_trip_with_fake_te(fake_te):
    torch.manual_seed(0)
    unfused_model = FakeDFlashModel(num_layers=2)
    wrapped_model = FakeDFlashModel(num_layers=2)
    fake_te.wrap_with_te(wrapped_model, fp8=True)

    unfused_sd = unfused_model.state_dict()
    mapped = fake_te.unfused_to_fused_state_dict(unfused_sd)
    missing, unexpected = wrapped_model.load_state_dict(mapped, strict=False)
    assert not missing
    assert not unexpected

    round_trip = fake_te.fused_to_unfused_state_dict(wrapped_model.state_dict())
    assert torch.equal(round_trip["norm.weight"], unfused_sd["norm.weight"])
    for idx in range(2):
        key = f"layers.{idx}.input_layernorm.weight"
        assert torch.equal(round_trip[key], unfused_sd[key])
