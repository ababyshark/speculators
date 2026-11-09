"""GLM MTP robustness fixes from the 16K field report (R1/R3/R4):

- R1: an empty recovery batch (document_ids all -1) returns a graph-connected
  zero loss instead of crashing on the packed-documents check.
- R3: the causal mask is built with row-chunked fills, element-wise equal to
  the original two-buffer construction.
- R4: chunk_size resolves explicit arg > SPECULATORS_MTP_CHUNK_SIZE env > 64
  and is forwarded to every chunked helper.
"""

from pathlib import Path

import pytest
import torch
from torch import nn
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig,
)

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.mtp import MTPDraftModel, MTPSpeculatorConfig, glm_mtp
from speculators.models.mtp.glm_mtp import (
    _CAUSAL_MASK_ROW_CHUNK,
    _glm_causal_mask,
    forward_glm_mtp,
)
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
        "hidden_states": torch.randn(
            1, SEQ_LEN, model.config.hidden_size, generator=generator
        ),
        "loss_mask": torch.ones(1, SEQ_LEN, dtype=torch.long),
        "position_ids": torch.arange(SEQ_LEN).unsqueeze(0),
    }


def make_empty_batch(model: MTPDraftModel) -> dict:
    """Mirror train/data.py create_empty_batch: all-pad placeholder with
    document_ids == -1 everywhere (produced on teacher request failure)."""
    return {
        "input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long),
        "hidden_states": torch.zeros(1, SEQ_LEN, model.config.hidden_size),
        "loss_mask": torch.zeros(1, SEQ_LEN, dtype=torch.bool),
        "position_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long),
        "document_ids": -torch.ones(1, SEQ_LEN, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# R1: empty recovery batch
# ---------------------------------------------------------------------------


def test_empty_recovery_batch_returns_zero_loss():
    model = make_model()
    batch = make_empty_batch(model)
    losses, loss, metrics = forward_glm_mtp(model, **batch)
    assert losses == []
    assert loss.item() == 0.0
    assert metrics["loss_sum"].item() == 0.0
    assert metrics["loss"].item() == 0.0
    assert metrics["supervised_tokens"].item() == 0.0
    assert metrics["loss_total"].item() == 1.0


def test_empty_recovery_batch_zero_is_graph_connected():
    model = make_model()
    batch = make_empty_batch(model)
    hidden = batch["hidden_states"].clone().requires_grad_(True)
    losses, loss, _ = forward_glm_mtp(
        model,
        input_ids=batch["input_ids"],
        hidden_states=hidden,
        loss_mask=batch["loss_mask"],
        position_ids=batch["position_ids"],
        document_ids=batch["document_ids"],
    )
    assert loss.requires_grad
    loss.backward()
    assert hidden.grad is not None
    assert hidden.grad.abs().sum().item() == 0.0


def test_empty_batch_guard_returns_before_position_check():
    """create_empty_batch position_ids are all zero, which the packed/reset
    position validation rejects; the guard must return before reaching it."""
    model = make_model()
    batch = make_empty_batch(model)
    losses, _, _ = forward_glm_mtp(model, **batch)
    assert losses == []


def test_real_packed_documents_still_rejected():
    model = make_model()
    batch = make_batch(model)
    with pytest.raises(ValueError, match="Packed documents"):
        forward_glm_mtp(
            model,
            input_ids=batch["input_ids"],
            hidden_states=batch["hidden_states"],
            position_ids=batch["position_ids"],
            loss_mask=batch["loss_mask"],
            document_ids=torch.cat(
                [
                    torch.zeros(1, 16, dtype=torch.long),
                    torch.ones(1, 16, dtype=torch.long),
                ],
                dim=1,
            ),
        )


def test_right_padded_single_document_still_accepted():
    model = make_model()
    batch = make_batch(model)
    seq_len = batch["input_ids"].shape[1]
    document_ids = torch.cat(
        [
            torch.zeros(1, seq_len - 8, dtype=torch.long),
            -torch.ones(1, 8, dtype=torch.long),
        ],
        dim=1,
    )
    with torch.no_grad():
        losses, loss, metrics = forward_glm_mtp(
            model,
            input_ids=batch["input_ids"],
            hidden_states=batch["hidden_states"],
            position_ids=batch["position_ids"],
            loss_mask=batch["loss_mask"],
            document_ids=document_ids,
        )
    assert len(losses) == 0  # logits list stays empty in the GLM path
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# R3: chunked causal mask construction
# ---------------------------------------------------------------------------


def _reference_causal_mask(length, dtype, device):
    """The original two-buffer construction kept for equivalence checks."""
    causal = torch.ones((length, length), dtype=torch.bool, device=device).triu(1)
    mask = torch.zeros((1, 1, length, length), dtype=dtype, device=device)
    return mask.masked_fill(causal, torch.finfo(dtype).min)


@pytest.mark.parametrize("length", [1, 2, 5, 8, 33, 64, 65, 130])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_causal_mask_matches_reference(length, dtype):
    mask = _glm_causal_mask(length, dtype, torch.device("cpu"))
    assert mask.shape == (1, 1, length, length)
    assert mask.dtype == dtype
    assert torch.equal(mask, _reference_causal_mask(length, dtype, torch.device("cpu")))


def test_causal_mask_multichunk_equivalence(monkeypatch):
    monkeypatch.setattr(glm_mtp, "_CAUSAL_MASK_ROW_CHUNK", 4)
    assert glm_mtp._CAUSAL_MASK_ROW_CHUNK == 4
    length = 10  # spans three chunks (4/4/2)
    mask = _glm_causal_mask(length, torch.float32, torch.device("cpu"))
    assert torch.equal(
        mask, _reference_causal_mask(length, torch.float32, torch.device("cpu"))
    )


def test_causal_mask_semantics():
    length = 7
    mask = _glm_causal_mask(length, torch.float32, torch.device("cpu"))
    min_val = torch.finfo(torch.float32).min
    for i in range(length):
        for j in range(length):
            expected = 0.0 if j <= i else min_val
            assert mask[0, 0, i, j].item() == expected


def test_causal_mask_default_chunk_constant():
    # 4096-row chunks: ~64MB bool stripe at 16K vs 256MB full-size bool.
    assert _CAUSAL_MASK_ROW_CHUNK == 4096


# ---------------------------------------------------------------------------
# R4: chunk_size resolution and forwarding
# ---------------------------------------------------------------------------


def _spy_chunked_ce(monkeypatch):
    calls = []
    original = glm_mtp._glm_chunked_ce

    def spy(model, output, labels, valid, chunk_size=64, **kwargs):
        calls.append(chunk_size)
        return original(model, output, labels, valid, chunk_size=chunk_size, **kwargs)

    monkeypatch.setattr(glm_mtp, "_glm_chunked_ce", spy)
    return calls


def test_chunk_size_defaults_to_64(monkeypatch):
    model = make_model()
    batch = make_batch(model)
    monkeypatch.delenv("SPECULATORS_MTP_CHUNK_SIZE", raising=False)
    calls = _spy_chunked_ce(monkeypatch)
    with torch.no_grad():
        forward_glm_mtp(model, **batch)
    assert calls and all(size == 64 for size in calls)


def test_chunk_size_env_override(monkeypatch):
    model = make_model()
    batch = make_batch(model)
    monkeypatch.setenv("SPECULATORS_MTP_CHUNK_SIZE", "16")
    calls = _spy_chunked_ce(monkeypatch)
    with torch.no_grad():
        forward_glm_mtp(model, **batch)
    assert calls and all(size == 16 for size in calls)


def test_chunk_size_explicit_arg_beats_env(monkeypatch):
    model = make_model()
    batch = make_batch(model)
    monkeypatch.setenv("SPECULATORS_MTP_CHUNK_SIZE", "16")
    calls = _spy_chunked_ce(monkeypatch)
    with torch.no_grad():
        forward_glm_mtp(model, **batch, chunk_size=8)
    assert calls and all(size == 8 for size in calls)


def test_chunk_size_invalid_raises():
    model = make_model()
    batch = make_batch(model)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        forward_glm_mtp(model, **batch, chunk_size=0)


def test_trainer_kwargs_forward_chunk_size():
    train_kwargs, val_kwargs = MTPDraftModel.get_trainer_kwargs(
        num_speculative_steps=5, mtp_chunk_size=32
    )
    assert train_kwargs["chunk_size"] == 32
    assert val_kwargs["chunk_size"] == 32


def test_trainer_kwargs_omit_chunk_size_by_default():
    train_kwargs, _ = MTPDraftModel.get_trainer_kwargs(num_speculative_steps=5)
    assert "chunk_size" not in train_kwargs
