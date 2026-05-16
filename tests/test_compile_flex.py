from __future__ import annotations

import pytest


@pytest.fixture
def prepared_trainer(synthetic_trace_dir, tmp_path):
    from dflash_llama.training import DFlashTrainer
    from dflash_llama.verifiers import load_verifier

    verifier = load_verifier(
        "generic",
        name_override="synthetic-qwen3-tiny",
        hidden_size=64,
        num_hidden_layers=6,
        vocab_size=2048,
        mask_token_id=0,
        layer_ids=(0, 1, 2, 3, 4, 5),
        hf_path="/dummy/hf",
    )
    paired = tmp_path / "paired"
    trainer = DFlashTrainer(
        traces_dir=str(synthetic_trace_dir),
        verifier=verifier,
        num_layers=2,
        draft_vocab_size=128,
        paired_dir=str(paired),
    )
    trainer.prepare()
    return trainer


def test_train_dryrun_compile_flex_sets_env_and_removes_disable_vars(
    prepared_trainer,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    res = prepared_trainer.train(
        save_to=str(tmp_path / "ckpt"),
        dry_run=True,
        epochs=1,
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        compile_flex_attention=True,
    )
    env = res["env"]
    assert env["DFLASH_COMPILE_FLEX"] == "1"
    assert "TORCHDYNAMO_DISABLE" not in env
    assert "TORCH_COMPILE_DISABLE" not in env


def test_smoke_dryrun_compile_flex_sets_env_and_removes_disable_vars(
    prepared_trainer,
    monkeypatch,
):
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    res = prepared_trainer.smoke(
        dry_run=True,
        timeout_sec=15,
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        compile_flex_attention=True,
    )
    assert res.env is not None
    assert res.env["DFLASH_COMPILE_FLEX"] == "1"
    assert "TORCHDYNAMO_DISABLE" not in res.env
    assert "TORCH_COMPILE_DISABLE" not in res.env


def test_train_dryrun_c15_env_combines_compile_flex_and_fp8_param_storage(
    prepared_trainer,
    tmp_path,
):
    res = prepared_trainer.train(
        save_to=str(tmp_path / "ckpt"),
        dry_run=True,
        epochs=1,
        fp8_recipe_kind="current_fp8",
        te_use_fused=True,
        te_fp8_params=True,
        compile_flex_attention=True,
    )
    env = res["env"]
    assert env["TE_FP8_PARAMS"] == "1"
    assert env["DFLASH_COMPILE_FLEX"] == "1"
    assert "TORCHDYNAMO_DISABLE" not in env
    assert "TORCH_COMPILE_DISABLE" not in env
