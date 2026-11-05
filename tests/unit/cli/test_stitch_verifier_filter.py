"""Tests for the stitch verifier-reference-tensor filter.

Training-side verifier reference tensors (``verifier_norm.weight``, a residual
``verifier_lm_head.weight``, ...) travel with speculators checkpoints for the
training loss but are not part of the native MTP layer; stitch must never
write them back into the verifier package.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from speculators.cli.stitch import stitch


@contextmanager
def read_tensors(path: Path):
    with safe_open(str(path), framework="pt") as handle:
        yield {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118


def _make_single_file_fixtures(root: Path) -> tuple[Path, Path]:
    verifier = root / "verifier"
    verifier.mkdir()
    finetuned = root / "finetuned"
    finetuned.mkdir()

    # Single-file verifier with one native MTP tensor (native key name).
    save_file(
        {"mtp.norm.weight": torch.zeros(2, 2)},
        str(verifier / "model.safetensors"),
    )

    # Finetuned checkpoint: trained MTP tensor + verifier reference tensors.
    save_file(
        {
            "mtp_layers.0.final_layernorm.weight": torch.ones(2, 2),
            "verifier_norm.weight": torch.full((2,), 1.25),
            "verifier_lm_head.weight": torch.full((3, 2), 2.5),
        },
        str(finetuned / "model.safetensors"),
    )
    return finetuned, verifier


def test_stitch_skips_verifier_reference_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    finetuned, verifier = _make_single_file_fixtures(tmp_path)

    output = stitch(finetuned, verifier, tmp_path / "stitched")

    with read_tensors(output / "model.safetensors") as tensors:
        # The trained MTP tensor replaced the verifier value...
        assert torch.equal(tensors["mtp.norm.weight"], torch.ones(2, 2))
        # ...and no verifier reference tensor leaked into the package.
        assert not [key for key in tensors if key.startswith("verifier_")]


def test_stitch_sharded_output_has_no_verifier_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    finetuned = tmp_path / "finetuned"
    finetuned.mkdir()

    save_file(
        {"mtp.norm.weight": torch.zeros(2, 2)},
        str(verifier / "shard-a.safetensors"),
    )
    (verifier / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.norm.weight": "shard-a.safetensors"}})
    )
    save_file(
        {
            "mtp_layers.0.final_layernorm.weight": torch.ones(2, 2),
            "verifier_norm.weight": torch.ones(2),
        },
        str(finetuned / "model.safetensors"),
    )

    output = stitch(finetuned, verifier, tmp_path / "stitched")

    with read_tensors(output / "shard-a.safetensors") as tensors:
        assert set(tensors) == {"mtp.norm.weight"}
        assert torch.equal(tensors["mtp.norm.weight"], torch.ones(2, 2))
