"""Checkpointer recovery helpers: zero-momentum fill for zero-grad params and
fp32 optimizer-state preservation.

The DSA indexer's top-k is non-differentiable: its params never receive grads,
AdamW lazily skips creating their state, and a strict resume crashes on the
missing entries (save-lenient/load-strict asymmetry). The fill functions
reproduce the true behavior (those params were never updated) by inserting
zero moments. convert_float_dtype(sd, None) must keep fp32 moments fp32 —
casting them to bf16 compounds precision loss across resumes.
"""

import torch
from torch import nn

from speculators.train.checkpointer import (
    _fill_missing_named_optimizer_states,
    _fill_missing_optimizer_states,
    cast_model_state_dict,
    convert_float_dtype,
)


def test_fill_missing_optimizer_states_single_gpu() -> None:
    torch.manual_seed(0)
    trained = nn.Parameter(torch.randn(4, 4))
    untouched = nn.Parameter(torch.randn(3, 3))  # zero-grad (e.g. DSA indexer)
    opt = torch.optim.AdamW([trained, untouched], lr=1e-3)
    loss = (trained**2).sum()
    loss.backward()
    opt.step()  # creates state for `trained` only

    sd = opt.state_dict()
    filled = _fill_missing_optimizer_states(opt, sd)
    assert filled == 1
    # the contract: the GIVEN dict is mutated in place (DCP flows reload it)
    # 2 params in the single group; label 1 = `untouched`
    assert 1 in sd["state"]
    assert (sd["state"][1]["exp_avg"] == 0).all()
    assert (sd["state"][1]["exp_avg_sq"] == 0).all()
    assert "step" in sd["state"][1]
    # idempotent on an already-filled dict
    assert _fill_missing_optimizer_states(opt, sd) == 0


def test_fill_missing_named_optimizer_states_dcp() -> None:
    torch.manual_seed(0)
    model = nn.Module()
    model.a = nn.Parameter(torch.randn(2, 2))
    model.b = nn.Parameter(torch.randn(5))
    full_sd: dict = {
        "state": {
            "a": {
                "step": torch.tensor(7.0),
                "exp_avg": torch.randn(2, 2),
                "exp_avg_sq": torch.randn(2, 2),
            }
        }
    }
    filled = _fill_missing_named_optimizer_states(model, full_sd)
    assert filled == 1
    assert "b" in full_sd["state"]
    assert (full_sd["state"]["b"]["exp_avg"] == 0).all()
    assert float(full_sd["state"]["b"]["step"]) == 7.0


def test_fill_skips_frozen_params() -> None:
    """Frozen params (requires_grad=False, e.g. the frozen experts) were never
    in the optimizer anyway — the fill must not invent state for them."""
    model = nn.Module()
    model.p = nn.Parameter(torch.randn(3))
    model.frozen = nn.Parameter(torch.randn(3), requires_grad=False)
    full_sd: dict = {"state": {}}
    # no AdamW-style sample -> no-op
    assert _fill_missing_named_optimizer_states(model, full_sd) == 0
    full_sd["state"]["p"] = {
        "step": torch.tensor(2.0),
        "exp_avg": torch.randn(3),
        "exp_avg_sq": torch.randn(3),
    }
    assert _fill_missing_named_optimizer_states(model, full_sd) == 0
    assert "frozen" not in full_sd["state"]


def test_convert_float_dtype_none_preserves_fp32() -> None:
    sd = {"w": torch.randn(4, dtype=torch.float32), "i": torch.tensor([1, 2]), "n": 3}
    out = convert_float_dtype(sd, None)
    assert out["w"].dtype == torch.float32
    assert out["i"].dtype == torch.int64
    # bf16 cast still works for model weights
    out2 = convert_float_dtype(
        {"w": torch.randn(4, dtype=torch.float32)}, torch.bfloat16
    )
    assert out2["w"].dtype == torch.bfloat16


def test_cast_model_state_dict_preserves_router_bias_fp32() -> None:
    """The router's e_score_correction_bias (~15.8 magnitude, 0.08 informative
    spread) must survive checkpoint casting in fp32: bf16 rounding flips 75%
    of the top-8 expert selections at serving (measured)."""
    sd = {
        "mtp_layers.0.mlp.gate.e_score_correction_bias": torch.randn(
            256, dtype=torch.float32
        )
        + 15.8,
        "mtp_layers.0.self_attn.q_b_proj.weight": torch.randn(
            8, 4, dtype=torch.float32
        ),
        "mtp_layers.0.input_proj.weight": torch.randn(4, 8, dtype=torch.float32),
    }
    out = cast_model_state_dict(sd, torch.bfloat16)
    assert out["mtp_layers.0.mlp.gate.e_score_correction_bias"].dtype == torch.float32
    assert torch.equal(
        out["mtp_layers.0.mlp.gate.e_score_correction_bias"],
        sd["mtp_layers.0.mlp.gate.e_score_correction_bias"],
    )
    assert out["mtp_layers.0.self_attn.q_b_proj.weight"].dtype == torch.bfloat16
    assert out["mtp_layers.0.input_proj.weight"].dtype == torch.bfloat16
    # dtype=None keeps everything
    out2 = cast_model_state_dict(sd, None)
    assert all(
        v.dtype == torch.float32 for v in out2.values() if isinstance(v, torch.Tensor)
    )
