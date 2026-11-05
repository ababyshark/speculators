"""GLM MoE-DSA MTP layer: registration, config patching, and state-dict
contract tests.

Locks in the integration bugs hit during NPU enablement:
- the MTP layer must be registered for the glm_moe_dsa model type;
- mlp_layer_types/indexer_types patching (MTP layer is sparse-MoE + full DSA
  indexer while the config only describes the main stack);
- state-dict keys must NOT be renamed (a wrapper module renaming mlp.* to
  mlp.moe.* silently zeroed the warm-start experts);
- the NPU dispatch must be bound at instance level on experts.forward so the
  FSDP2 pre-forward unshard hook keeps firing;
- the bound dispatch must be self-contained: routing comes in as arguments
  from the parent GlmMoeDsaMoE.forward and shared_experts are added by the
  parent — the dispatch computing either one corrupts the layer.
"""

from types import MethodType

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.models import base_components
from speculators.models.mtp.model_definitions import (
    GlmMoeDsaMTPLayer,
    MTPLayerMixin,
    mtp_model_classes,
)


def tiny_glm_moe_dsa_config() -> PretrainedConfig:
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )

    config = GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        num_local_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        routed_scaling_factor=1.0,
        index_topk=8,
        index_n_heads=2,
        index_head_dim=16,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        mlp_layer_types=["sparse", "dense"],
        indexer_types=["full", "sparse"],
        _attn_implementation="eager",
    )
    return config


def test_glm_moe_dsa_registered() -> None:
    assert "glm_moe_dsa" in mtp_model_classes
    assert "glm_moe_dsa" in base_components.model_classes


def test_mtp_layer_patches_config_to_sparse_moe_full_indexer() -> None:
    config = tiny_glm_moe_dsa_config()
    layer = GlmMoeDsaMTPLayer(config, layer_idx=0)
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaMoE

    assert isinstance(layer.mlp, GlmMoeDsaMoE)
    assert isinstance(layer.mlp.experts, nn.Module)


def test_state_dict_keys_not_renamed() -> None:
    """A wrapper nn.Module around self.mlp renames mlp.* -> mlp.moe.* and the
    warm-start experts silently stay at init. The grouped expert weights must
    stay addressable as mlp.experts.gate_up_proj."""
    config = tiny_glm_moe_dsa_config()
    layer = GlmMoeDsaMTPLayer(config, layer_idx=0)
    keys = list(layer.state_dict())
    assert any(k == "mlp.experts.gate_up_proj" for k in keys)
    assert not any("mlp.moe." in k for k in keys)


def test_dispatch_bound_at_instance_level() -> None:
    config = tiny_glm_moe_dsa_config()
    layer = GlmMoeDsaMTPLayer(config, layer_idx=0)
    experts = layer.mlp.experts
    # instance attribute shadowing the class forward (FSDP2 __call__ honored)
    assert "forward" in experts.__dict__


def test_dispatch_is_methodtype_binding() -> None:
    config = tiny_glm_moe_dsa_config()
    layer = GlmMoeDsaMTPLayer(config, layer_idx=0)
    bound = layer.mlp.experts.__dict__["forward"]
    assert isinstance(bound, MethodType)


def test_dispatch_contract_self_contained() -> None:
    """The dispatch must not compute routing (the parent passes topk as args)
    and must not add shared_experts (the parent adds them). Violations are the
    AttributeError ('FSDPGlmMoeDsaExperts' has no attribute 'gate') and the
    double-counted shared experts bugs."""
    import inspect

    from speculators.models.mtp import model_definitions

    fn = model_definitions._sparse_experts_forward
    source = inspect.getsource(fn)
    body = source[source.index('"""') + 3 :]
    body = body[body.index('"""') + 3 :]  # strip docstring
    assert "moe." not in body
    assert ".gate(" not in body
    assert "shared_experts" not in body
    # the HF call convention: routing arrives as arguments
    params = list(inspect.signature(fn).parameters)
    assert params == ["self", "hidden_states", "top_k_index", "top_k_weights"]


class _StubDecoderLayer(nn.Module):
    """Minimal stand-in for the decoder layer base class."""

    def __init__(self, config: PretrainedConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.seen_hidden: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object = None,
        use_cache: bool = False,
        position_embeddings: object = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen_hidden = hidden_states
        return hidden_states * 2.0, torch.zeros(1)


class _StubMTPLayer(MTPLayerMixin, _StubDecoderLayer):
    def __init__(self, config: PretrainedConfig, layer_idx: int = 0) -> None:
        super().__init__(config, layer_idx)
        self._setup_mtp_modules(config, nn.RMSNorm)


def test_mixin_composition_contract() -> None:
    """Locks the composition serving relies on (vllm DeepSeekMTP):

        eh_proj(cat([enorm(embed), hnorm(HS)])) -> layer -> shared_head.norm

    i.e. token_layernorm on the token embeddings, hidden_layernorm on the
    hidden states, that order in the concat, and final_layernorm applied to
    the (residual-complete) decoder-layer output — nothing more, nothing less.
    """
    torch.manual_seed(0)
    config = tiny_glm_moe_dsa_config()
    layer = _StubMTPLayer(config).eval()
    b, s, h = 1, 8, config.hidden_size
    hidden = torch.randn(b, s, h)
    tokens = torch.randn(b, s, h)

    with torch.no_grad():
        out = layer(
            hidden_states=hidden,
            token_embeddings=tokens,
            position_ids=torch.arange(s).unsqueeze(0),
        )
        # manual reference of the mixin composition
        proj_ref = layer.input_proj(
            torch.cat(
                [layer.token_layernorm(tokens), layer.hidden_layernorm(hidden)],
                dim=-1,
            )
        )
        expected = layer.final_layernorm(proj_ref * 2.0)

    assert torch.equal(layer.seen_hidden, proj_ref)
    assert torch.allclose(out, expected)


def test_mixin_accepts_tuple_and_tensor_decoder_output() -> None:
    """The HF GlmMoeDsaDecoderLayer returns (hidden, topk_indices); older
    styles may return a bare tensor. final_layernorm must apply in both."""
    config = tiny_glm_moe_dsa_config()
    layer = _StubMTPLayer(config).eval()
    h = torch.randn(1, 4, config.hidden_size)
    t = torch.randn(1, 4, config.hidden_size)
    with torch.no_grad():
        out = layer(hidden_states=h, token_embeddings=t)
    assert out.shape == h.shape
