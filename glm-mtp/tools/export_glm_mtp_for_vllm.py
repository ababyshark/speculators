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

    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def reverse_key(key: str, layer: int) -> str:
    prefix = "mtp_layers.0."
    if not key.startswith(prefix):
        raise ValueError(f"Unexpected trained key: {key}")
    body = key[len(prefix) :]
    replacements = (
        ("token_layernorm.", "enorm."),
        ("hidden_layernorm.", "hnorm."),
        ("input_proj.", "eh_proj."),
        ("final_layernorm.", "shared_head.norm."),
    )
    for src, dst in replacements:
        if body.startswith(src):
            body = dst + body[len(src) :]
            break
    return f"model.layers.{layer}.{body}"


def requant_writeback(cp: Path, target: Path, out: Path, layer: int) -> dict:
    """Write trained float MTP tensors back as W4A8 quantized tensors.

    S1 review fix: our v0.28 serving requires a W4A8-quantized MTP layer;
    writing the float training weights back was measured at acceptance 0.000.
    This mode copies the target package and replaces every layer-<layer>
    tensor with the trained weight requantized into the stored layout
    (signed int4 double-packed along dim0 or int8 per-channel, validated in
    speculators.convert.w4a8_layout).  Float tensors (norms, eh_proj, router
    bias) are copied verbatim; embed_tokens/shared_head.head keep the
    verifier's stored values (the drafter shares them).
    """
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from speculators.convert.w4a8_layout import requant_int4_packed, requant_int8

    meta_suffixes = (".weight_scale", ".weight_offset", ".scale_bias")
    target_index_path = next(
        (
            target / name
            for name in (
                "quant_model_weights.safetensors.index.json",
                "model.safetensors.index.json",
            )
            if (target / name).exists()
        ),
        None,
    )
    if target_index_path is None:
        raise FileNotFoundError(f"No safetensors index under {target}")
    weight_map = json.loads(target_index_path.read_text())["weight_map"]
    prefix = f"model.layers.{layer}."

    train_file = cp / "model.safetensors"
    mapped: dict[str, str] = {}
    with safe_open(str(train_file), framework="pt", device="cpu") as f:
        all_train_keys = sorted(f.keys())
        skipped_reference = [k for k in all_train_keys if k.startswith("verifier_")]
        if skipped_reference:
            print(
                f"skipped {len(skipped_reference)} verifier reference key(s): {skipped_reference[:3]}"
            )
        train_keys = [k for k in all_train_keys if not k.startswith("verifier_")]
        mapped = {reverse_key(k, layer): k for k in train_keys}
    if len(mapped) != len(train_keys):
        raise RuntimeError("Reverse mapping is not bijective")
    native_keys = {k for k in weight_map if k.startswith(prefix)}
    # a30-era training checkpoints omit the shared embed/head (stored in
    # mtp_shared_weights_path); keep the target's original quantized bytes for
    # them via the existing-tensor carry-over in the shard rewrite below.
    shared_exempt = {
        f"{prefix}embed_tokens.weight",
        f"{prefix}shared_head.head.weight",
    }
    unmapped = (
        native_keys
        - set(mapped)
        - {k for k in native_keys if k.endswith(meta_suffixes)}
        - shared_exempt
    )
    if unmapped:
        raise RuntimeError({"layer tensors without a trained source": sorted(unmapped)})

    out.mkdir(parents=True)
    for name in ("config.json", "quant_model_description.json"):
        src = target / name
        if src.exists():
            shutil.copy2(src, out / name)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        src = target / name
        if src.exists():
            shutil.copy2(src, out / name)

    # Group the keys to rewrite by their stored shard; stream one shard at a
    # time so a 48GB package never materializes fully in host memory.
    rewrite: dict[str, dict[str, object]] = {}
    stats = {"int4": 0, "int8": 0, "float": 0}
    records: dict[str, dict] = {}
    with safe_open(str(train_file), framework="pt", device="cpu") as tf:
        for nk, tk in sorted(mapped.items()):
            w = tf.get_tensor(tk).float()
            shard = weight_map.get(nk)
            if shard is None:
                raise RuntimeError(f"Native key absent from target package: {nk}")
            base = nk.rsplit(".", 1)[0]
            scale_key = f"{base}.weight_scale"
            quantized = scale_key in weight_map
            with safe_open(
                str(target / weight_map[nk]), framework="pt", device="cpu"
            ) as sf:
                stored_shape = tuple(sf.get_tensor(nk).shape)
                stored_dtype = sf.get_tensor(nk).dtype
            if quantized:
                if stored_shape[0] * 2 == w.shape[0] and stored_shape[1] == w.shape[1]:
                    codes, scale, _offset, _bias = requant_int4_packed(w)
                    stats["int4"] += 1
                elif tuple(w.shape) == stored_shape:
                    codes, scale, _offset = requant_int8(w)
                    stats["int8"] += 1
                else:
                    raise RuntimeError(
                        f"stored {stored_shape} incompatible with logical {tuple(w.shape)} at {nk}"
                    )
                rewrite.setdefault(shard, {})[nk] = codes
                records[nk] = {
                    "mode": "requant",
                    "trained_key": tk,
                    "shape": list(codes.shape),
                    "sha256": digest(codes),
                }
                rewrite.setdefault(weight_map[scale_key], {})[scale_key] = scale
                records[scale_key] = {"mode": "scale", "trained_key": tk}
                for meta in (f"{base}.weight_offset", f"{base}.scale_bias"):
                    if meta in weight_map:
                        with safe_open(
                            str(target / weight_map[meta]), framework="pt", device="cpu"
                        ) as sf:
                            ref = sf.get_tensor(meta)
                        rewrite.setdefault(weight_map[meta], {})[meta] = (
                            torch.zeros_like(ref)
                        )
                        records[meta] = {"mode": "zeroed", "trained_key": tk}
            else:
                if stored_shape != tuple(w.shape):
                    raise RuntimeError(
                        f"float tensor shape mismatch at {nk}: stored {stored_shape} vs trained {tuple(w.shape)}"
                    )
                rewrite.setdefault(shard, {})[nk] = w.to(stored_dtype)
                stats["float"] += 1
                records[nk] = {
                    "mode": "float_copy",
                    "trained_key": tk,
                    "sha256": digest(w),
                }

    # Untouched shards are copied verbatim; touched shards are rewritten with
    # the trained substitutions applied.
    for shard in sorted({shard for shard in weight_map.values()}):
        if shard in rewrite:
            existing = {}
            metadata = None
            with safe_open(str(target / shard), framework="pt", device="cpu") as sf:
                metadata = sf.metadata()
                for key in sf.keys():  # noqa: SIM118
                    existing[key] = sf.get_tensor(key)
            existing.update(rewrite[shard])
            save_file(existing, str(out / shard), metadata=metadata)
        else:
            shutil.copy2(target / shard, out / shard)

    report = {
        "checkpoint": str(cp),
        "target": str(target),
        "layer": layer,
        "requant_mode": "w4a8 (signed int4 packed / int8 per-channel, offset=0 scale_bias=0)",
        "stats": stats,
        "rewritten_tensors": len(records),
        "trained_tensor_count": len(mapped),
        "records": records,
    }
    (out / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--native-source", default="/data1/mtp_training/weights/glm53_mtp_native_step0"
    )
    ap.add_argument("--target-model", default="/data1/weight/GLM-5.3-w4a8-spec")
    ap.add_argument(
        "--requant-w4a8",
        action="store_true",
        help="Write the trained MTP tensors back as W4A8 quantized tensors "
        "(codes + weight_scale siblings, offset/scale_bias zeroed) instead of "
        "exporting float weights. Required for the v0.28 W4A8-quantized MTP "
        "serving; float write-back was measured at acceptance 0.000.",
    )
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

    if args.requant_w4a8:
        report = requant_writeback(cp, target, out, layer)
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in ("stats", "rewritten_tensors", "trained_tensor_count")
                },
                indent=2,
            )
        )
        return

    native_index = json.loads((native / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    with safe_open(str(train_file), framework="pt", device="cpu") as f:
        all_train_keys = sorted(f.keys())
        # Training-side verifier reference tensors (verifier_norm.weight, ...)
        # are not part of the native MTP layer (same contract as stitch).
        skipped_reference = [k for k in all_train_keys if k.startswith("verifier_")]
        if skipped_reference:
            print(
                f"skipped {len(skipped_reference)} verifier reference key(s): {skipped_reference[:3]}"
            )
        train_keys = [k for k in all_train_keys if not k.startswith("verifier_")]
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
            with safe_open(
                str(native / native_index[nk]), framework="pt", device="cpu"
            ) as nf:
                native_shape = tuple(nf.get_tensor(nk).shape)
            if native_shape != train_shape:
                raise RuntimeError(
                    f"Shape mismatch for {nk}: {train_shape} vs {native_shape}"
                )

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
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        src = target / name
        if src.exists():
            shutil.copy2(src, out / name)
    records = []
    with safe_open(str(train_file), framework="pt", device="cpu") as f:
        for nk, tk in sorted(mapped.items()):
            t = f.get_tensor(tk)
            records.append(
                {
                    "name": nk,
                    "source_key": tk,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "sha256": digest(t),
                }
            )
    (out / "export_report.json").write_text(
        json.dumps(
            {
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
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"EXPORT_PASSED trained={len(mapped)} shared={len(shared)} output={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
