"""Convert the verified native extraction to a loadable Speculators checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.mtp import MTPDraftModel, MTPSpeculatorConfig
from speculators.models.mtp.glm_mtp import native_key_to_training
from speculators.proposals.greedy import GreedyTokenProposalConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src, dst = Path(args.source), Path(args.output)
    if dst.exists():
        raise FileExistsError(dst)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    config = MTPSpeculatorConfig(
        transformer_layer_config=AutoConfig.from_pretrained(src),
        mtp_shared_weights_path=str(src.resolve()),
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            default_proposal_method="greedy",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            verifier=VerifierConfig(
                name_or_path=args.teacher, architectures=["GlmMoeDsaForCausalLM"]
            ),
        ),
    )
    with torch.device("meta"):
        model = MTPDraftModel(config)
    expected = {
        k: tuple(v.shape)
        for k, v in model.state_dict().items()
        if k not in ("verifier_lm_head.weight",)
    }
    # the PR#1226 verifier_norm (the frozen teacher final-norm): supply the
    # weight from the verifier checkpoint's model.norm
    import glob as _glob

    _norm_w = None
    for _f in sorted(_glob.glob(args.teacher + "/quant_model_weights-*.safetensors")):
        with safe_open(_f, framework="pt", device="cpu") as _sf:
            if "model.norm.weight" in _sf.keys():
                _norm_w = _sf.get_tensor("model.norm.weight").to(torch.float32)
                break
    assert _norm_w is not None, "model.norm.weight not found in the teacher"

    mapping = {
        key: native_key_to_training(
            key, config.transformer_layer_config.num_hidden_layers
        )
        for key in index
    }
    # the PR#1226 verifier_norm (the frozen teacher final-norm): its weight
    # travels with the speculators checkpoint for the training loss
    mapping["model.norm.weight"] = "verifier_norm.weight"
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Conversion mapping is not bijective")
    if set(mapping.values()) != set(expected):
        raise ValueError(
            {
                "missing": sorted(set(expected) - set(mapping.values())),
                "unexpected": sorted(set(mapping.values()) - set(expected)),
            }
        )
    dst.mkdir(parents=True)
    config.save_pretrained(dst)
    output_index, records = {}, {}
    for number, shard in enumerate(sorted(set(index.values())), 1):
        tensors = {}
        with safe_open(str(src / shard), framework="pt") as f:
            for key in sorted(k for k, value in index.items() if value == shard):
                tensor = f.get_tensor(key)
                newkey = mapping[key]
                if tuple(tensor.shape) != expected[newkey]:
                    raise ValueError(
                        f"Shape mismatch: {key}: {tuple(tensor.shape)} != {expected[newkey]}"
                    )
                digest = hashlib.sha256(
                    tensor.view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                records[newkey] = dict(
                    source_key=key,
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    sha256=digest,
                )
                if newkey not in ("embed_tokens.weight", "lm_head.weight"):
                    tensors[newkey] = tensor.contiguous()
        if not tensors:
            continue
        filename = f"model-{number:05d}.safetensors"
        save_file(tensors, str(dst / filename))
        with safe_open(str(dst / filename), framework="pt") as f:
            for key in tensors:
                actual = hashlib.sha256(
                    f.get_tensor(key).view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                if actual != records[key]["sha256"]:
                    raise RuntimeError(f"Round-trip mismatch: {key}")
                output_index[key] = filename
        print(f"verified {filename}: {len(tensors)} tensors", flush=True)
    norm_shape = expected["verifier_norm.weight"]
    if tuple(_norm_w.shape) != norm_shape:
        raise ValueError(
            "verifier_norm.weight shape mismatch: "
            f"{tuple(_norm_w.shape)} != {norm_shape}"
        )
    norm_digest = hashlib.sha256(
        _norm_w.view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    save_file(
        {"verifier_norm.weight": _norm_w.contiguous()},
        str(dst / "verifier_norm.safetensors"),
    )
    with safe_open(str(dst / "verifier_norm.safetensors"), framework="pt") as f:
        actual = hashlib.sha256(
            f.get_tensor("verifier_norm.weight")
            .view(torch.uint8)
            .numpy()
            .tobytes()
        ).hexdigest()
    if actual != norm_digest:
        raise RuntimeError("Round-trip mismatch: verifier_norm.weight")
    output_index["verifier_norm.weight"] = "verifier_norm.safetensors"
    records["verifier_norm.weight"] = dict(
        source_key="model.norm.weight",
        shape=list(_norm_w.shape),
        dtype=str(_norm_w.dtype),
        sha256=norm_digest,
    )
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": output_index}, indent=2)
    )
    (dst / "conversion_report.json").write_text(
        json.dumps(
            dict(
                status="passed",
                source=str(src),
                teacher=args.teacher,
                tensor_count=len(records),
                saved_tensor_count=len(output_index),
                shared_weights_path=str(src),
                mapping=records,
                semantics_parity="NOT_YET_VALIDATED",
            ),
            indent=2,
        )
    )
    print(
        f"CONVERSION_PASSED {len(records)} tensors, independent shared head retained",
        flush=True,
    )


if __name__ == "__main__":
    main()
