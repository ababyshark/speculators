"""MTP layer mixin and concrete layer classes.

MTP projects hidden states DOWN to standard hidden_size BEFORE attention via
input_proj, so q/k/v weights remain standard width. The mixin interface takes
two separate tensors (hidden_states, token_embeddings).
"""

import copy
import logging
from importlib.metadata import version as pkg_version
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
)

from speculators.models import base_components

logger = logging.getLogger(__name__)

__all__ = ["MTPLayerMixin", "mtp_model_classes", "resolve_model_type"]

_MIN_TRANSFORMERS_VERSION: dict[str, str] = {
    "qwen3_next": "4.57.0",
    "qwen3_5_text": "5.2.0",
    "qwen3_5_moe_text": "5.2.0",
}


def resolve_model_type(model_type: str) -> str:
    """Resolve a model_type to a canonical type in mtp_model_classes."""
    if model_type in mtp_model_classes:
        return model_type
    raise _unsupported_model_type_error(model_type)


def _unsupported_model_type_error(model_type: str) -> ValueError:
    """Build a descriptive error for an unresolvable model type."""
    if model_type in _MIN_TRANSFORMERS_VERSION:
        min_ver = _MIN_TRANSFORMERS_VERSION[model_type]
        installed = pkg_version("transformers")
        return ValueError(
            f"Model type '{model_type}' requires transformers>={min_ver} "
            f"(installed: {installed}). "
            f"Upgrade with: pip install 'transformers>={min_ver}'"
        )
    supported = sorted(mtp_model_classes.keys())
    return ValueError(
        f"Unsupported MTP model type '{model_type}'. Supported types: {supported}"
    )


def _last_full_attention_idx(config: PretrainedConfig) -> int:
    """Return the last layer index whose type is ``full_attention``.

    Hybrid models (Qwen3-Next, Qwen3.5-MoE) alternate
    linear_attention and full_attention layers. The MTP head always
    uses full_attention, so we must pick a matching index.
    """
    layer_types: list[str] = getattr(config, "layer_types", [])
    if not layer_types:
        return 0
    for i in reversed(range(len(layer_types))):
        if layer_types[i] == "full_attention":
            return i
    raise ValueError(
        "Hybrid MTP layer requires at least one 'full_attention' entry in "
        f"config.layer_types, got: {layer_types}"
    )


class MTPLayerMixin:
    """MTP-specific modifications for any decoder layer.

    Projects hidden states to standard hidden_size BEFORE attention
    via input_proj; q/k/v weights remain standard width.
    """

    # Provided by the decoder layer base class
    self_attn: Any
    mlp: Any
    input_layernorm: Any
    post_attention_layernorm: Any

    def _setup_mtp_modules(
        self, config: PretrainedConfig, norm_class: type[nn.Module]
    ) -> None:
        hidden_size, eps = config.hidden_size, config.rms_norm_eps
        self.hidden_layernorm = norm_class(hidden_size, eps=eps)
        self.token_layernorm = norm_class(hidden_size, eps=eps)
        self.input_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.final_layernorm = norm_class(hidden_size, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings: (tuple[torch.Tensor, torch.Tensor] | None) = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_normed = self.hidden_layernorm(hidden_states)
        embed_normed = self.token_layernorm(token_embeddings)
        proj = self.input_proj(torch.cat([embed_normed, hidden_normed], dim=-1))

        output = super().forward(  # type: ignore[misc]
            hidden_states=proj,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output
        return self.final_layernorm(hidden_states)


def _create_mtp_layer_class(
    name: str,
    decoder_class: type,
    norm_class: type[nn.Module],
    *,
    hybrid: bool = False,
) -> type:
    """Create an MTP layer class for a specific decoder architecture.

    :param name: Class name for the generated type (used in repr/debugging).
    :param decoder_class: Base decoder layer class to mix with MTPLayerMixin.
    :param norm_class: RMSNorm class for the architecture.
    :param hybrid: If True, select the last full_attention layer index
        (for models that alternate linear/full attention layers).
    """

    class _MTPLayer(MTPLayerMixin, decoder_class):  # type: ignore[misc]
        def __init__(self, config: PretrainedConfig, layer_idx: int = 0) -> None:
            modified = copy.copy(config)
            if hybrid:
                layer_idx = _last_full_attention_idx(modified)
            super().__init__(modified, layer_idx)  # type: ignore[arg-type]
            self._setup_mtp_modules(modified, norm_class)

    _MTPLayer.__name__ = name
    _MTPLayer.__qualname__ = name
    return _MTPLayer


Qwen3MTPLayer = _create_mtp_layer_class(
    "Qwen3MTPLayer", Qwen3DecoderLayer, Qwen3RMSNorm
)

mtp_model_classes: dict[str, base_components.ModelComponents] = {
    "qwen3": base_components.override_components(
        "qwen3", first_layer_class=Qwen3MTPLayer
    ),
}


if "qwen3_next" in base_components.model_classes:
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextDecoderLayer,
        Qwen3NextRMSNorm,
    )

    Qwen3NextMTPLayer = _create_mtp_layer_class(
        "Qwen3NextMTPLayer",
        Qwen3NextDecoderLayer,
        Qwen3NextRMSNorm,
        hybrid=True,
    )
    mtp_model_classes["qwen3_next"] = base_components.override_components(
        "qwen3_next", first_layer_class=Qwen3NextMTPLayer
    )

if "qwen3_5_text" in base_components.model_classes:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5RMSNorm,
    )

    Qwen35MTPLayer = _create_mtp_layer_class(
        "Qwen35MTPLayer",
        Qwen3_5DecoderLayer,
        Qwen3_5RMSNorm,
        hybrid=True,
    )
    mtp_model_classes["qwen3_5_text"] = base_components.override_components(
        "qwen3_5_text", first_layer_class=Qwen35MTPLayer
    )

if "qwen3_5_moe_text" in base_components.model_classes:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeRMSNorm,
    )

    Qwen35MoeMTPLayer = _create_mtp_layer_class(
        "Qwen35MoeMTPLayer",
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeRMSNorm,
        hybrid=True,
    )
    mtp_model_classes["qwen3_5_moe_text"] = base_components.override_components(
        "qwen3_5_moe_text", first_layer_class=Qwen35MoeMTPLayer
    )

if "glm_moe_dsa" in base_components.model_classes:
    import types

    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaDecoderLayer,
        GlmMoeDsaRMSNorm,
    )

    _SPARSE_SHAPE_LOGGED = {"done": False}

    def _sparse_experts_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Sorted per-expert sliced-GEMM dispatch for NPU training.

        Bound to ``experts.forward`` (instance level) so the FSDP2 pre-forward
        unshard hook keeps firing and the mlp.* state-dict keys stay
        unchanged (a wrapper module would rename them and silently zero the
        warm-start).

        The parent ``GlmMoeDsaMoE.forward`` computes the routing and passes
        it in as arguments, and adds ``shared_experts`` itself afterwards --
        this function must do neither.

        Replaces the HF implementation (256-iteration python loop with
        one_hot/nonzero boolean indexing = one D2H sync per layer, minutes
        per step on NPU). Jobs are globally sorted by expert; per-expert
        GEMMs run on contiguous slices after ONE device->host sync for the
        offsets. Mathematically identical to the HF dispatch; compute is
        K/E of the dense form. Measured: fwd 7.5s -> 312ms, bwd 128s ->
        856ms per batch.
        """

        def _local(t):
            # FSDP2 unshard: the DTensor's local tensor is the full param.
            if isinstance(t, torch.distributed.tensor.DTensor):
                return t.to_local()
            return t

        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_states.shape[-1])  # [N, H]
        w1 = _local(self.gate_up_proj)  # [E, 2I, H]
        w2 = _local(self.down_proj)  # [E, H, I]
        num_experts, hidden = w1.shape[0], x.shape[1]
        if not _SPARSE_SHAPE_LOGGED["done"]:
            logger.info("[sparse] w1 %s w2 %s", tuple(w1.shape), tuple(w2.shape))
            _SPARSE_SHAPE_LOGGED["done"] = True

        # K from numel (not shape[1]): callers may pass [N, K] or [B, S, K]
        topk = top_k_index.numel() // x.shape[0]
        flat_e = top_k_index.reshape(-1)
        flat_w = top_k_weights.reshape(-1).to(x.dtype)
        # Jobs are globally sorted by expert (stable). y_jobs is written in
        # SORTED job order, so the routing weights must be re-ordered to
        # match -- else each token's expert outputs get scaled by OTHER
        # tokens' weights (caught by the dispatch equivalence test).
        order = torch.argsort(flat_e)
        flat_w = flat_w[order]
        token_ids = order // topk  # job -> token
        counts = torch.bincount(flat_e, minlength=num_experts)
        offsets = counts.cumsum(0) - counts

        # ONE device->host sync for the expert offsets; everything else stays
        # on-device. Compute is ~K/E of the dense form.
        offs = offsets.tolist()
        cnts = counts.tolist()
        y_jobs = torch.empty(
            top_k_index.numel(), hidden, dtype=x.dtype, device=x.device
        )
        act = self.act_fn
        for expert in range(num_experts):
            start, count = offs[expert], cnts[expert]
            if count == 0:
                continue
            xe = x.index_select(0, token_ids[start : start + count])  # [c, H]
            g = xe @ w1[expert].t()  # [c, 2I]
            gate, up = g.chunk(2, dim=-1)
            y_jobs[start : start + count] = act(gate) * up @ w2[expert].t()  # [c, H]

        y_jobs = y_jobs * flat_w.unsqueeze(1)
        out = torch.zeros_like(x).index_add_(0, token_ids, y_jobs)
        return out.reshape(orig_shape)

    class GlmMoeDsaMTPLayer(MTPLayerMixin, GlmMoeDsaDecoderLayer):
        """MTP layer for GLM-5.x MoE-DSA verifiers (DeepSeek-style layout).

        The native MTP layer (checkpoint ``model.layers.<N>`` with
        ``shared_head``/``eh_proj``/``enorm``/``hnorm``) is a sparse-MoE layer
        with a full DSA indexer. ``mlp_layer_types`` and ``indexer_types`` only
        cover the main stack, so the copied config is patched so the layer
        constructed at ``layer_idx=0`` matches the checkpoint's MTP layer.
        """

        def __init__(self, config: PretrainedConfig, layer_idx: int = 0) -> None:
            modified = copy.copy(config)
            mlp_types = list(getattr(modified, "mlp_layer_types", None) or [])
            if mlp_types:
                mlp_types[layer_idx % len(mlp_types)] = "sparse"
                modified.mlp_layer_types = mlp_types
            indexer_types = list(getattr(modified, "indexer_types", None) or [])
            if indexer_types:
                indexer_types[layer_idx % len(indexer_types)] = "full"
                modified.indexer_types = indexer_types
            super().__init__(modified, layer_idx)
            self._setup_mtp_modules(modified, GlmMoeDsaRMSNorm)
            # Sorted per-expert dispatch via an INSTANCE-level forward patch on
            # the experts module. A wrapper nn.Module would rename the mlp.*
            # state keys (warm-start experts silently stay zero) and bypass the
            # FSDP2 pre-forward unshard hook; binding experts.forward keeps the
            # hook firing (module __call__ honors instance attributes). The
            # parent MoE computes routing and adds shared_experts itself.
            experts = self.mlp.experts
            experts.forward = types.MethodType(_sparse_experts_forward, experts)

    mtp_model_classes["glm_moe_dsa"] = base_components.override_components(
        "glm_moe_dsa", first_layer_class=GlmMoeDsaMTPLayer
    )
