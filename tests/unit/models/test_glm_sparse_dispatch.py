"""Sparse expert dispatch (_sparse_experts_forward): mathematical equivalence
with the HF reference dispatch, autograd correctness, and degenerate-routing
robustness. CPU-only; the implementation is device-agnostic.

The reference is the dense all-experts computation with a routing mask —
mathematically identical to HF's per-expert boolean-index loop but trivially
correct. The sparse form (argsort + bincount + per-expert GEMM + index_add_)
must match it for arbitrary routing, including skewed (load-imbalanced) and
empty-expert cases hit in real MTP training.
"""

import inspect
from types import MethodType

import pytest
import torch
from torch import nn

from speculators.models.mtp.model_definitions import _sparse_experts_forward


class _DummyExperts(nn.Module):
    """Same attribute surface as GlmMoeDsaExperts (grouped weights)."""

    def __init__(self, num_experts: int, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * inter, hidden) * 0.1
        )
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.1)
        self.act_fn = nn.SiLU()


def _reference_dispatch(
    x: torch.Tensor,
    topk_index: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    act_fn: nn.Module,
) -> torch.Tensor:
    """Dense all-experts computation with a routing mask (reference oracle)."""
    n, h = x.shape
    num_experts = w1.shape[0]
    g = torch.einsum("nh,eih->nei", x, w1)
    gate, up = g.chunk(2, dim=-1)
    hmid = act_fn(gate) * up
    y = torch.einsum("nei,eih->neh", hmid, w2.transpose(1, 2))
    routing = torch.zeros(n, num_experts, dtype=x.dtype)
    # scatter_ADD matches HF's per-slot job semantics (and the sparse impl's
    # index_add_) if topk indices ever repeat; real routers emit unique topk.
    routing.scatter_add_(1, topk_index, topk_weights.to(x.dtype))
    return torch.einsum("ne,neh->nh", routing, y)


@pytest.fixture
def experts() -> _DummyExperts:
    torch.manual_seed(7)
    module = _DummyExperts(num_experts=8, hidden=16, inter=24)
    module.forward = MethodType(_sparse_experts_forward, module)
    return module


@pytest.mark.parametrize(("batch", "seq"), [(1, 12), (3, 7)])
def test_matches_reference_random_routing(
    experts: _DummyExperts, batch: int, seq: int
) -> None:
    torch.manual_seed(0)
    x = torch.randn(batch, seq, experts.gate_up_proj.shape[-1])
    topk_index = torch.topk(torch.rand(batch, seq, 8), 4, dim=-1).indices
    topk_weights = torch.softmax(torch.randn(batch, seq, 4), dim=-1)
    out = experts(x, topk_index, topk_weights)
    ref = _reference_dispatch(
        x.reshape(-1, x.shape[-1]),
        topk_index.reshape(-1, 4),
        topk_weights.reshape(-1, 4),
        experts.gate_up_proj,
        experts.down_proj,
        experts.act_fn,
    ).reshape(x.shape)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_matches_reference_skewed_routing(experts: _DummyExperts) -> None:
    """Concentrated routing (few experts take all jobs) — the load-imbalance
    path where padded/grouped alternatives have historically failed."""
    torch.manual_seed(1)
    x = torch.randn(2, 10, experts.gate_up_proj.shape[-1])
    topk_index = torch.full((2, 10, 4), 3)
    topk_index[:, :, 2] = 5  # only experts 3 and 5 receive jobs
    topk_weights = torch.full((2, 10, 4), 0.25)
    out = experts(x, topk_index, topk_weights)
    ref = _reference_dispatch(
        x.reshape(-1, x.shape[-1]),
        topk_index.reshape(-1, 4),
        topk_weights.reshape(-1, 4),
        experts.gate_up_proj,
        experts.down_proj,
        experts.act_fn,
    ).reshape(x.shape)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_matches_reference_empty_experts(experts: _DummyExperts) -> None:
    """More experts than jobs: many experts must receive zero jobs."""
    torch.manual_seed(2)
    x = torch.randn(1, 3, experts.gate_up_proj.shape[-1])
    scores = torch.zeros(1, 3, 8)
    scores[..., :2] = 10.0  # only experts 0/1 are ever selected
    topk_index = torch.topk(scores, 2, dim=-1).indices
    topk_weights = torch.full((1, 3, 2), 0.5)
    out = experts(x, topk_index, topk_weights)
    ref = _reference_dispatch(
        x.reshape(-1, x.shape[-1]),
        topk_index.reshape(-1, 2),
        topk_weights.reshape(-1, 2),
        experts.gate_up_proj,
        experts.down_proj,
        experts.act_fn,
    ).reshape(x.shape)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(out).all()


def test_gradients_flow_to_input_only_when_weights_frozen(
    experts: _DummyExperts,
) -> None:
    """The training config freezes gate_up/down: grads must flow to the input
    but not the frozen weights (the frozen-weight OOM guard)."""
    x = torch.randn(1, 6, experts.gate_up_proj.shape[-1], requires_grad=True)
    experts.gate_up_proj.requires_grad_(False)
    experts.down_proj.requires_grad_(False)
    topk_index = torch.topk(torch.rand(1, 6, 8), 2, dim=-1).indices
    topk_weights = torch.full((1, 6, 2), 0.5)
    experts(x, topk_index, topk_weights).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert experts.gate_up_proj.grad is None
    assert experts.down_proj.grad is None


def test_gradients_flow_to_weights_when_trainable(experts: _DummyExperts) -> None:
    x = torch.randn(1, 6, experts.gate_up_proj.shape[-1], requires_grad=True)
    topk_index = torch.topk(torch.rand(1, 6, 8), 2, dim=-1).indices
    topk_weights = torch.full((1, 6, 2), 0.5)
    experts(x, topk_index, topk_weights).sum().backward()
    assert experts.gate_up_proj.grad is not None
    assert experts.down_proj.grad is not None


def test_single_d2h_sync_contract() -> None:
    """The implementation must materialize the expert offsets exactly once
    (tolist) — per-call nonzero/boolean-index syncs were the original
    minutes-per-step pathology."""
    source = inspect.getsource(_sparse_experts_forward)
    body = source[source.index('"""') + 3 :]
    body = body[body.index('"""') + 3 :]  # strip docstring
    assert body.count(".tolist()") <= 2  # offsets + counts, ONE sync total
    assert "nonzero" not in body
