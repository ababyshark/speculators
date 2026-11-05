"""Vendor checkpoint loading fallback: non-standard sharded index names.

Vendor ModelSlim checkpoints ship `quant_model_weights.safetensors.index.json`
instead of HF's `model.safetensors.index.json`; before the fallback, layer
loading failed with FileNotFoundError. The fallback must pick up any
*.safetensors.index.json in the directory.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from speculators.utils.loading import load_model_layers


def test_load_model_layers_fallback_to_vendor_index(tmp_path: Path) -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    shard = tmp_path / "quant_model_weights-00001.safetensors"
    save_file({"model.layers.78.mlp.gate_proj.weight": tensor}, str(shard))
    index = tmp_path / "quant_model_weights.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {"model.layers.78.mlp.gate_proj.weight": shard.name},
            }
        )
    )

    loaded = load_model_layers(["model.layers.78.mlp.gate_proj.weight"], str(tmp_path))
    assert "model.layers.78.mlp.gate_proj.weight" in loaded
    assert torch.equal(loaded["model.layers.78.mlp.gate_proj.weight"], tensor)


def test_load_model_layers_suffix_matching(tmp_path: Path) -> None:
    tensor = torch.randn(2, 3)
    shard = tmp_path / "quant_model_weights-00001.safetensors"
    save_file({"model.layers.77.self_attn.o_proj.weight": tensor}, str(shard))
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"model.layers.77.self_attn.o_proj.weight": shard.name}}
        )
    )
    loaded = load_model_layers(["layers.77.self_attn.o_proj.weight"], str(tmp_path))
    assert torch.equal(loaded["layers.77.self_attn.o_proj.weight"], tensor)
