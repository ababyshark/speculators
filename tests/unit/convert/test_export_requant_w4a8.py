"""CPU tests for the W4A8 requant write-back export (S1 review fix).

The v0.28 serving requires a W4A8-quantized MTP layer; the float write-back
was measured at acceptance 0.000.  These tests pin the requant_writeback
contract on a synthetic mini package: signed int4 double-packed codes +
per-channel weight_scale siblings for quantized weights, verbatim copies for
float tensors, and untouched shards outside the MTP layer.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "glm-mtp" / "code_mtp" / "export_glm_mtp_for_vllm.py"

spec = importlib.util.spec_from_file_location("export_glm_mtp_for_vllm", SCRIPT)
assert spec is not None and spec.loader is not None
export = importlib.util.module_from_spec(spec)
sys.modules["export_glm_mtp_for_vllm"] = export
spec.loader.exec_module(export)

from speculators.convert.w4a8_layout import (  # noqa: E402
    requant_int4_packed,
    requant_int8,
)


def _build_fixtures(tmp_path: Path):
    layer = 2
    target = tmp_path / "target"
    target.mkdir()
    trained = tmp_path / "trained"
    trained.mkdir()

    # Layer keys: one int4-packed weight + scale/offset/bias siblings, one
    # int8 weight + scale sibling, one float norm, one float router bias.
    gate_logical = torch.randn(4, 6)  # [out=4, in=6]
    attn_logical = torch.randn(4, 6)
    norm_float = torch.randn(4)
    router_bias = torch.randn(4)

    gate_packed, gate_scale, _, _ = requant_int4_packed(gate_logical)
    attn_q, attn_scale, _ = requant_int8(attn_logical)

    tensors = {
        f"model.layers.{layer}.mlp.experts.0.gate_proj.weight": gate_packed.to(torch.int8),
        f"model.layers.{layer}.mlp.experts.0.gate_proj.weight_scale": gate_scale,
        f"model.layers.{layer}.mlp.experts.0.gate_proj.weight_offset": torch.ones(4, 1),
        f"model.layers.{layer}.mlp.experts.0.gate_proj.scale_bias": torch.ones(4, 1),
        f"model.layers.{layer}.self_attn.q_a_proj.weight": attn_q,
        f"model.layers.{layer}.self_attn.q_a_proj.weight_scale": attn_scale,
        f"model.layers.{layer}.input_layernorm.weight": norm_float,
        f"model.layers.{layer}.mlp.gate.e_score_correction_bias": router_bias,
        # a tensor OUTSIDE the MTP layer that must stay untouched
        "model.layers.0.self_attn.q_a_proj.weight": torch.full((4, 6), 3.0),
    }
    weight_map = {k: "shard-a.safetensors" for k in tensors}
    save_file(tensors, str(target / "shard-a.safetensors"))
    (target / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (target / "config.json").write_text(json.dumps({"num_hidden_layers": layer}))
    (target / "quant_model_description.json").write_text("{}")

    # Trained (logical, float) tensors under the drafter key naming.
    save_file(
        {
            f"mtp_layers.0.mlp.experts.0.gate_proj.weight": gate_logical,
            f"mtp_layers.0.self_attn.q_a_proj.weight": attn_logical * 0.5,
            f"mtp_layers.0.input_layernorm.weight": norm_float + 0.1,
            f"mtp_layers.0.mlp.gate.e_score_correction_bias": router_bias + 0.1,
        },
        str(trained / "model.safetensors"),
    )
    return trained, target, layer, gate_logical, attn_logical * 0.5


def test_requant_writeback_produces_stored_layout(tmp_path):
    trained, target, layer, gate_logical, attn_trained = _build_fixtures(tmp_path)
    out = tmp_path / "out"

    export.requant_writeback(trained, target, out, layer)

    with safe_open(str(out / "shard-a.safetensors"), framework="pt") as handle:
        keys = set(handle.keys())  # noqa: SIM118
        gate_codes = handle.get_tensor(f"model.layers.{layer}.mlp.experts.0.gate_proj.weight")
        gate_scale = handle.get_tensor(f"model.layers.{layer}.mlp.experts.0.gate_proj.weight_scale")
        attn_codes = handle.get_tensor(f"model.layers.{layer}.self_attn.q_a_proj.weight")
        attn_scale = handle.get_tensor(f"model.layers.{layer}.self_attn.q_a_proj.weight_scale")
        norm = handle.get_tensor(f"model.layers.{layer}.input_layernorm.weight")
        bias = handle.get_tensor(f"model.layers.{layer}.mlp.gate.e_score_correction_bias")
        untouched = handle.get_tensor("model.layers.0.self_attn.q_a_proj.weight")

    # All keys preserved (codes + scale/offset/bias siblings + floats).
    assert keys == {
        f"model.layers.{layer}.mlp.experts.0.gate_proj.weight",
        f"model.layers.{layer}.mlp.experts.0.gate_proj.weight_scale",
        f"model.layers.{layer}.mlp.experts.0.gate_proj.weight_offset",
        f"model.layers.{layer}.mlp.experts.0.gate_proj.scale_bias",
        f"model.layers.{layer}.self_attn.q_a_proj.weight",
        f"model.layers.{layer}.self_attn.q_a_proj.weight_scale",
        f"model.layers.{layer}.input_layernorm.weight",
        f"model.layers.{layer}.mlp.gate.e_score_correction_bias",
        "model.layers.0.self_attn.q_a_proj.weight",
    }

    # int4: stored layout [out/2, in] signed codes, amax/8 scale.
    expect_packed, expect_scale, _, _ = requant_int4_packed(gate_logical)
    assert gate_codes.shape == (2, 6)
    assert torch.equal(gate_codes, expect_packed)
    assert torch.allclose(gate_scale, expect_scale)
    # dequantize round-trip stays close to the trained logical weights.
    from speculators.convert.w4a8_layout import unpack_int4_dim0, to_signed_int4

    dequant = unpack_int4_dim0(gate_codes) * gate_scale
    cos = torch.nn.functional.cosine_similarity(
        dequant.flatten(), gate_logical.flatten(), dim=0
    )
    # 4x6 random rows quantize coarsely; the real-weights round-trip gate
    # (cos 0.99998) lives in test_w4a8_layout.py.
    assert cos > 0.99

    # int8: same shape as logical, amax/127 scale.
    expect_q, expect_scale8, _ = requant_int8(attn_trained)
    assert attn_codes.shape == (4, 6)
    assert torch.equal(attn_codes, expect_q)
    assert torch.allclose(attn_scale, expect_scale8)

    # float tensors copied verbatim from the trained checkpoint.
    assert norm.shape == (4,)
    assert bias.shape == (4,)

    # Non-MTP tensor untouched.
    assert torch.equal(untouched, torch.full((4, 6), 3.0))


def test_requant_writeback_report_counts(tmp_path):
    trained, target, layer, *_ = _build_fixtures(tmp_path)
    out = tmp_path / "out"

    report = export.requant_writeback(trained, target, out, layer)

    assert report["stats"] == {"int4": 1, "int8": 1, "float": 2}
    assert report["trained_tensor_count"] == 4
    assert report["rewritten_tensors"] == 8  # 2 codes + 2 scales + 2 zeroed + 2 floats
    assert (out / "export_report.json").exists()
    assert (out / "config.json").exists()


def test_requant_writeback_rejects_unmapped_layer_tensors(tmp_path):
    trained, target, layer, *_ = _build_fixtures(tmp_path)
    # Drop one trained tensor: the layer tensor no longer has a source.
    save_file(
        {"mtp_layers.0.mlp.experts.0.gate_proj.weight": torch.randn(4, 6)},
        str(trained / "model.safetensors"),
    )
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="without a trained source"):
        export.requant_writeback(trained, target, out, layer)
