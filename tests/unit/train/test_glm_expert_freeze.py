"""Training-side guards for the GLM MTP config: grouped-expert freezing and
FSDP unit selection.

- freeze_grouped_expert_weights: the experts' combined bf16 grads (~19.3 GiB)
  cannot coexist with the unsharded params on 61 GiB cards; the freeze must
  hit every module exposing the grouped weights and only those.
- _fsdp_wrap_candidates: the experts submodule must be wrapped as its own FSDP
  unit (a single ~10B layer all-gathers ~20 GiB otherwise); already-sharded
  (DTensor) submodules must be skipped to keep the HCCL communicator count
  within limits.
"""

import torch
from torch import nn

from speculators.train.cli import freeze_grouped_expert_weights
from speculators.train.distributed import _fsdp_wrap_candidates


class _GroupedExperts(nn.Module):
    def __init__(self, e: int = 4, h: int = 8, i: int = 6) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(e, 2 * i, h))
        self.down_proj = nn.Parameter(torch.randn(e, h, i))


class _FakeMoeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _GroupedExperts()


class _FakeLayer(nn.Module):
    def __init__(self, with_experts: bool = True) -> None:
        super().__init__()
        self.mlp = _FakeMoeLayer() if with_experts else nn.Linear(8, 8)


def test_freeze_grouped_expert_weights() -> None:
    model = nn.Sequential(_FakeLayer(), _FakeLayer())
    frozen = freeze_grouped_expert_weights(model)
    assert frozen == 2
    for m in model.modules():
        if isinstance(m, _GroupedExperts):
            assert not m.gate_up_proj.requires_grad
            assert not m.down_proj.requires_grad
    # unrelated params untouched
    other = nn.Linear(4, 4)
    freeze_grouped_expert_weights(other)
    assert other.weight.requires_grad


def test_fsdp_wrap_candidates_plain_layer() -> None:
    layer = _FakeLayer()
    candidates = _fsdp_wrap_candidates(layer)
    assert candidates == [layer.mlp.experts]


def test_fsdp_wrap_candidates_no_experts() -> None:
    """A dense MLP layer (no grouped experts) yields no candidates."""
    assert _fsdp_wrap_candidates(_FakeLayer(with_experts=False)) == []
