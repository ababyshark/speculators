"""Export a trained GLM-5.3 MTP checkpoint to the native vLLM weight layout.

The training checkpoint omits MTP embed_tokens/shared_head because they are stored
in mtp_shared_weights_path. This exporter adds those two tensors and writes an
index that maps the 791 trained tensors and the shared tensors to separate files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def digest(tensor) -> str:
    import torch
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def reverse_key(key: str, layer: int) -> str:
    prefix = "mtp_layers.0."
    if not key.startswith(prefix):
        raise ValueError(f"Unexpected trained key: {key}")
    body = key[len(prefix):]
    replacements = (
        ("token_layernorm.", "enorm."),
        ("hidden_layernorm.", "hnorm."),
        ("input_proj.", "eh_proj."),
        ("final_layernorm.", "shared_head.norm."),
    )
    for src, dst in replacements:
        if body.startswith(src):
            body = dst + body[len(src):]
            break
    return f"model.layers.{layer}.{body}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--native-source", default="/data1/mtp_training/weights/glm53_mtp_native_step0")
    ap.add_argument("--target-model", default="/data1/weight/GLM-5.3-w4a8-spec")
    args = ap.parse_args()

    cp = Path(args.checkpoint)
    out = Path(args.output)
    native = Path(args.native_source)
    target = Path(args.target_model)
    if out.exists():
        raise FileExistsError(out)
    train_file = cp / "model.safetensors"
    if not train_file.exists():
        raise FileNotFoundError(train_file)

    target_cfg = json.loads((target / "config.json").read_text())
    layer = int(target_cfg["num_hidden_layers"])
    native_index = json.loads((native / "model.safetensors.index.json").read_text())["weight_map"]

    with safe_open(str(train_file), framework="pt", device="cpu") as f:
        train_keys = sorted(f.keys())
        mapped = {reverse_key(k, layer): k for k in train_keys}
    if len(mapped) != len(train_keys):
        raise RuntimeError("Reverse mapping is not bijective")
    shared = {
        f"model.layers.{layer}.embed_tokens.weight",
        f"model.layers.{layer}.shared_head.head.weight",
    }
    missing = set(native_index) - set(mapped)
    extra = set(mapped) - set(native_index)
    if missing != shared or extra:
        raise RuntimeError({"missing": sorted(missing), "extra": sorted(extra)})
    with safe_open(str(train_file), framework="pt", device="cpu") as tf:
        for nk, tk in mapped.items():
            train_shape = tuple(tf.get_tensor(tk).shape)
            with safe_open(str(native / native_index[nk]), framework="pt", device="cpu") as nf:
                native_shape = tuple(nf.get_tensor(nk).shape)
            if native_shape != train_shape:
                raise RuntimeError(f"Shape mismatch for {nk}: {train_shape} vs {native_shape}")

    out.mkdir(parents=True)
    os.symlink(str(train_file.resolve()), out / "model.safetensors")
    shared_tensors = {}
    shared_records = {}
    for nk in sorted(shared):
        shard = native / native_index[nk]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(nk).contiguous()
        shared_tensors[nk] = tensor
        shared_records[nk] = {
            "source": str(shard),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": digest(tensor),
        }
    shared_name = "mtp_shared.safetensors"
    save_file(shared_tensors, str(out / shared_name))
    weight_map = dict.fromkeys(sorted(mapped), "model.safetensors")
    weight_map.update(dict.fromkeys(sorted(shared), shared_name))
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2) + "\n"
    )
    shutil.copy2(target / "config.json", out / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        src = target / name
        if src.exists():
            shutil.copy2(src, out / name)
    records = []
    with safe_open(str(train_file), framework="pt", device="cpu") as f:
        for nk, tk in sorted(mapped.items()):
            t = f.get_tensor(tk)
            records.append({"name": nk, "source_key": tk, "shape": list(t.shape), "dtype": str(t.dtype), "sha256": digest(t)})
    (out / "export_report.json").write_text(json.dumps({
        "passed": True,
        "checkpoint": str(cp),
        "native_source": str(native),
        "target_model": str(target),
        "layer": layer,
        "trained_tensor_count": len(mapped),
        "shared_tensor_count": len(shared),
        "weight_map_count": len(weight_map),
        "weight_storage": "model.safetensors symlink to checkpoint; mtp_shared.safetensors contains omitted embedding/head",
        "shared_records": shared_records,
        "trained_records": records,
    }, indent=2) + "\n")
    print(f"EXPORT_PASSED trained={len(mapped)} shared={len(shared)} output={out}", flush=True)


if __name__ == "__main__":
    main()


