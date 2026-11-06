"""CPU integration tests for the MTP capture-stage contract on the GLM path.

Covers the hidden_state_stage guards added with the teacher capture manifest
port: pre-norm captures are normalized exactly once, post-norm captures are
never re-normalized, the vLLM-aligned path is rejected for post-norm
captures/capabilities, and unknown stages are rejected.
"""

from pathlib import Path

import pytest
import torch
from torch import nn
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig,
)

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.mtp import MTPDraftModel, MTPSpeculatorConfig
from speculators.models.mtp.capabilities import get_mtp_capability
from speculators.proposals import GreedyTokenProposalConfig

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "glm_mtp"
SEQ_LEN = 32


def load_tiny_config() -> GlmMoeDsaConfig:
    config = GlmMoeDsaConfig.from_pretrained(FIXTURE_DIR / "tiny_config.json")
    config._attn_implementation = "sdpa"
    return config


def make_model() -> MTPDraftModel:
    torch.manual_seed(0)
    config = MTPSpeculatorConfig(
        transformer_layer_config=load_tiny_config(),
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["GlmMoeDsaForCausalLM"],
            ),
        ),
    )
    model = MTPDraftModel(config)
    nn.init.normal_(model.embed_tokens.weight, std=0.02)
    nn.init.normal_(model.lm_head.weight, std=0.02)
    for parameter in model.mtp_layers[0].parameters():
        if parameter.is_floating_point():
            nn.init.normal_(parameter, std=0.02)
    model.eval()
    return model


def make_batch(model: MTPDraftModel) -> dict:
    generator = torch.Generator().manual_seed(1)
    return {
        "input_ids": torch.randint(0, 128, (1, SEQ_LEN), generator=generator),
        "verifier_last_hidden_states": torch.randn(
            1, SEQ_LEN, model.config.hidden_size, generator=generator
        ),
        "loss_mask": torch.ones(1, SEQ_LEN, dtype=torch.long),
        "position_ids": torch.arange(SEQ_LEN).unsqueeze(0),
    }


def run(model: MTPDraftModel, batch: dict, **kwargs):
    with torch.no_grad():
        return model(
            batch["input_ids"],
            batch["verifier_last_hidden_states"],
            position_ids=batch["position_ids"],
            loss_mask=batch["loss_mask"],
            **kwargs,
        )


def test_glm_capability_is_post_norm():
    model = make_model()
    capability = model.mtp_capability
    assert capability is not None
    assert capability.model_type == "glm_moe_dsa"
    assert capability.hidden_state_stage == "post_norm"
    assert capability is get_mtp_capability("glm_moe_dsa")


def test_pre_norm_capture_is_normalized_exactly_once():
    model = make_model()
    batch = make_batch(model)
    hidden_states = batch["verifier_last_hidden_states"]

    with torch.no_grad():
        post_norm = model.verifier_norm(hidden_states)
    pre_tokens, pre_loss, _ = run(model, batch, hidden_state_stage="pre_norm")
    post_batch = dict(batch)
    post_batch["verifier_last_hidden_states"] = post_norm
    post_tokens, post_loss, _ = run(model, post_batch, hidden_state_stage="post_norm")

    assert torch.equal(post_norm, model.verifier_norm(hidden_states))
    torch.testing.assert_close(pre_loss, post_loss)


def test_post_norm_capture_never_calls_verifier_norm(monkeypatch):
    model = make_model()
    batch = make_batch(model)

    def fail(_tensor):
        raise AssertionError("post_norm captures must not be re-normalized")

    monkeypatch.setattr(model.verifier_norm, "forward", fail)
    _, loss, _ = run(model, batch, hidden_state_stage="post_norm")
    assert torch.isfinite(loss)


def test_vllm_rejects_post_norm_capture():
    model = make_model()
    batch = make_batch(model)

    with pytest.raises(ValueError, match="double-normalize"):
        run(model, batch, hidden_state_stage="post_norm", train_forward="vllm")


def test_vllm_guard_is_absolute_for_post_norm_capability():
    model = make_model()
    batch = make_batch(model)

    # An explicit pre_norm stage must not reopen the vLLM path for a
    # post-norm capability.
    with pytest.raises(ValueError, match="double-normalize"):
        run(model, batch, hidden_state_stage="pre_norm", train_forward="vllm")


def test_unknown_stage_is_rejected():
    model = make_model()
    batch = make_batch(model)

    with pytest.raises(ValueError, match="hidden_state_stage"):
        run(model, batch, hidden_state_stage="final_norm")


def test_indexer_stays_frozen_after_capability_freeze():
    model = make_model()
    indexer_params = [
        parameter
        for name, parameter in model.named_parameters()
        if ".indexer." in f".{name}."
    ]
    assert indexer_params
    assert all(not parameter.requires_grad for parameter in indexer_params)


def make_steps_model(steps: int) -> MTPDraftModel:
    torch.manual_seed(0)
    transformer_config = load_tiny_config()
    config = MTPSpeculatorConfig(
        transformer_layer_config=transformer_config,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=steps)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["GlmMoeDsaForCausalLM"],
            ),
        ),
    )
    model = MTPDraftModel(config)
    nn.init.normal_(model.embed_tokens.weight, std=0.02)
    nn.init.normal_(model.lm_head.weight, std=0.02)
    for parameter in model.mtp_layers[0].parameters():
        if parameter.is_floating_point():
            nn.init.normal_(parameter, std=0.02)
    model.eval()
    return model


def test_glm_dispatch_honors_configured_steps():
    """S0 review fix: the GLM branch must not silently pin MTP to 1 step."""
    five = make_steps_model(5)
    assert five.config.num_speculative_steps == 5
    batch = make_batch(five)
    _, loss, metrics = run(five, batch)
    for step in range(5):
        assert f"loss_step_{step}" in metrics
        assert f"supervised_tokens_step_{step}" in metrics
    assert "loss_step_5" not in metrics
    assert torch.isfinite(loss)


def test_glm_single_step_contract_still_available():
    """The .23-validated K=1 path stays reachable and unchanged."""
    one = make_steps_model(1)
    batch = make_batch(one)
    _, loss, metrics = run(one, batch)
    assert "loss_step_0" in metrics
    assert "loss_step_1" not in metrics
    assert torch.isfinite(loss)


def test_glm_dispatch_ignores_batch_level_step_override():
    """The config is the single source of truth for the step count."""
    five = make_steps_model(5)
    batch = make_batch(five)
    # A stale batch-level value must not shrink the trained depth.
    with torch.no_grad():
        _, _, metrics = five(
            batch["input_ids"],
            batch["verifier_last_hidden_states"],
            position_ids=batch["position_ids"],
            loss_mask=batch["loss_mask"],
            num_speculative_steps=1,
        )
    assert "loss_step_4" in metrics


def test_glm_one_step_matches_single_step_wrapper_bitwise():
    """config=1 dispatch must reproduce the .23-validated K=1 path exactly."""
    from speculators.models.mtp.glm_mtp import forward_glm_single_step

    one = make_steps_model(1)
    batch = make_batch(one)
    _, loss_dispatch, metrics_dispatch = run(one, batch)
    with torch.no_grad():
        _, loss_wrapper, metrics_wrapper = forward_glm_single_step(
            one,
            batch["input_ids"],
            batch["verifier_last_hidden_states"],
            position_ids=batch["position_ids"],
            loss_mask=batch["loss_mask"],
        )
    assert torch.equal(loss_dispatch, loss_wrapper)
    assert torch.equal(metrics_dispatch["loss_step_0"], metrics_wrapper["loss_step_0"])


def test_glm_short_sequence_with_five_steps_skips_not_crashes():
    """C1: a sample shorter than steps+2 must skip with a finite zero loss
    (the .23 "filter unsupervised samples and continue" contract), never
    crash the training step with a ValueError or a metrics KeyError."""
    five = make_steps_model(5)
    generator = torch.Generator().manual_seed(3)
    batch = {
        "input_ids": torch.randint(0, 128, (1, 6), generator=generator),
        "verifier_last_hidden_states": torch.randn(
            1, 6, five.config.hidden_size, generator=generator
        ),
        "loss_mask": torch.ones(1, 6, dtype=torch.long),
        "position_ids": torch.arange(6).unsqueeze(0),
    }
    _, loss, metrics = run(five, batch)

    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    assert int(metrics["supervised_tokens"]) == 0
    assert "loss_step_0" not in metrics


def test_glm_all_zero_loss_mask_yields_finite_zero_loss():
    """C2: no supervised tokens must not produce a 0/0 NaN loss."""
    one = make_steps_model(1)
    batch = make_batch(one)
    batch["loss_mask"] = torch.zeros_like(batch["loss_mask"])

    _, loss, metrics = run(one, batch)

    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    assert int(metrics["supervised_tokens_step_0"]) == 0


@pytest.mark.parametrize("steps", [2, 3, 4, 5])
def test_glm_multi_step_loss_decomposition(steps):
    """T2: for every depth, per-step weighted losses sum to the reported
    loss_sum and the beta^i schedule covers exactly the trained steps."""
    model = make_steps_model(steps)
    batch = make_batch(model)
    _, loss, metrics = run(model, batch)

    per_step = sum(metrics[f"loss_step_{i}"] for i in range(steps))
    torch.testing.assert_close(per_step, metrics["loss_sum"])
    torch.testing.assert_close(metrics["loss_sum"], loss.detach())
    assert torch.isfinite(loss)
    assert "loss_step_" + str(steps) not in metrics
