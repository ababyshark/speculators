"""Extract the native MTP layer from the W4A8-quantized GLM-5.3 checkpoint.

Adaptation of .23's extract_native_mtp.py for a checkpoint whose MTP layer is
W4A8-quantized (the .23's GLM-5.3-w4a8-spec had float MTP tensors and could
assert is_floating_point; OUR biasfix checkpoint quantized the experts int4
and the attention int8 — the dequantization runs here, once, at extraction).

Dequantization conventions (validated by serving: the signed warm-start
measured acceptance 0.38 vs the unsigned 0.26 — see test_w4a8_layout.py):
- FLOAT tensors: copied verbatim (embed_tokens, shared_head.head, eh_proj,
  norms, router bias)
- int8 per-channel: W = q * scale
- int4 double-packed along dim0 (LOW nibble = even row), SIGNED nibbles:
  W = signed_unpack(q) * scale
- scale_bias: the vendor's per-channel table; semantics not fully pinned down
  (measured ±23 magnitude vs ±0.08 weights). Use --zero-point-mode bias to
  build the (signed_nibble - scale_bias) * scale variant for the A/B gate.

Output: the .23 extraction layout (mtp-NNN.safetensors shards + index +
config.json + extraction_report.json) — consumed by convert_glm_mtp.py.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from speculators.convert.w4a8_layout import (  # noqa: E402
    dequant_tensor,
    unpack_int4_dim0,
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dequant_expert_weight(w, scale, zero_point_mode, scale_bias):
    """The int4 expert weights: the signed unpack, optionally minus the
    scale_bias zero-point (the A/B gate for the vendor semantics)."""
    p16 = w.to(torch.int16)
    lo = torch.where((p16 & 0x0F) > 7, (p16 & 0x0F) - 16, (p16 & 0x0F)).float()
    hi = torch.where(((p16 >> 4) & 0x0F) > 7, ((p16 >> 4) & 0x0F) - 16, ((p16 >> 4) & 0x0F)).float()
    nib = torch.stack([lo, hi], dim=1).reshape(scale.shape[0], w.shape[1])
    if zero_point_mode == "bias":
        return ((nib - scale_bias) * scale).bfloat16()
    return (nib * scale).bfloat16()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the W4A8 verifier checkpoint dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--layer-index", type=int, default=78)
    ap.add_argument(
        "--zero-point-mode",
        choices=["none", "bias"],
        default="none",
        help="int4 expert dequant zero-point: none = signed_nibble*scale "
        "(the serving-validated baseline); bias = (signed_nibble-scale_bias)*scale",
    )
    ap.add_argument("--shard-bytes", type=int, default=2 << 30, help="~2GB shards")
    args = ap.parse_args()

    prefix = f"model.layers.{args.layer_index}."
    src_dir = Path(args.model)
    index_file = src_dir / "quant_model_weights.safetensors.index.json"
    weight_map = json.loads(index_file.read_text())["weight_map"]
    keys = sorted(k for k in weight_map if k.startswith(prefix))
    # the W4A8 checkpoint carries per-weight quant metadata (scale/offset/
    # scale_bias); the extraction targets = the tensors themselves
    meta_suffixes = (".weight_scale", ".weight_offset", ".scale_bias")
    target_keys = [k for k in keys if not k.endswith(meta_suffixes)]
    # the .23 float-source checkpoint had 793 (no metadata); ours has 800
    # (791 weights + 8 norm biases + the router bias)
    assert len(target_keys) == 800, (
        f"expected 800 extractable tensors, got {len(target_keys)}"
    )
    keys = target_keys

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    stats = {"float": 0, "int8": 0, "int4": 0, "other": 0}
    for k in keys:
        shard = src_dir / weight_map[k]
        with safe_open(shard, framework="pt", device="cpu") as sf:
            t = sf.get_tensor(k)
        suffix = k.rsplit(".", 1)[-1]
        base = k[len(prefix):]
        if suffix in ("weight_scale", "weight_offset", "scale_bias") or base in (
            "mlp.gate.e_score_correction_bias",
        ):
            continue  # consumed by the weight dequant or dropped
        if t.dtype.is_floating_point:
            tensors[base] = t.contiguous()
            stats["float"] += 1
        elif t.dtype == torch.int8:
            scale_k = k.rsplit(".", 1)[0] + ".weight_scale"
            scale_key = next(
                (c for c in (scale_k, scale_k.replace("weight_scale", "scale"))),
                None,
            )
            with safe_open(src_dir / weight_map[scale_key], framework="pt", device="cpu") as sf:
                scale = sf.get_tensor(scale_key).float()
            bias_k = k.rsplit(".", 1)[0] + ".scale_bias"
            bias = None
            if bias_k in weight_map and args.zero_point_mode == "bias":
                with safe_open(src_dir / weight_map[bias_k], framework="pt", device="cpu") as sf:
                    bias = sf.get_tensor(bias_k).float()
            if (
                t.shape[0] * 2 == scale.shape[0]
            ):  # int4 packed
                if bias is None:
                    bias = torch.ones_like(scale)  # no-op subtraction
                tensors[base] = dequant_expert_weight(t, scale, args.zero_point_mode, bias).contiguous()
                stats["int4"] += 1
            else:
                w = dequant_tensor(t, scale)
                tensors[base] = w.contiguous()
                stats["int8"] += 1
        else:
            tensors[base] = t.contiguous()
            stats["other"] += 1

    print(f"extracted stats: {stats} | total {len(tensors)}")

    # the finite check (the .23 convention), chunked
    bad = 0
    for k, t in tensors.items():
        if t.dtype.is_floating_point and not torch.isfinite(t.float()).all():
            bad += 1
            print(f"NON-FINITE: {k}")
    assert bad == 0, f"{bad} non-finite tensors"

    # shard ~2GB
    shards, current, size = [], {}, 0
    for k in sorted(tensors):
        t = tensors[k]
        nbytes = t.numel() * t.element_size()
        if size + nbytes > args.shard_bytes and current:
            shards.append(current)
            current, size = {}, 0
        current[k] = t
        size += nbytes
    if current:
        shards.append(current)

    records = {}
    for i, shard in enumerate(shards):
        name = f"mtp-{i:03d}.safetensors"
        save_file(shard, str(out / name))
        digest = sha256_file(out / name)
        for k in shard:
            records[k] = {"shard": name, "sha256_file": digest}
        print(f"wrote {name}: {len(shard)} tensors")

    (out / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
                "weight_map": {k: r["shard"] for k, r in records.items()},
            }
        )
    )
    # the config for the convert step
    import shutil

    shutil.copy(src_dir / "config.json", out / "config.json")

    report = {
        "source": str(src_dir),
        "layer_index": args.layer_index,
        "zero_point_mode": args.zero_point_mode,
        "stats": stats,
        "tensor_count": len(tensors),
        "shards": [f"mtp-{i:03d}.safetensors" for i in range(len(shards))],
        "note": "W4A8 MTP layer dequantized at extraction (the .23 float-source flow adapted)",
    }
    (out / "extraction_report.json").write_text(json.dumps(report, indent=2))
    print(f"extraction complete -> {out}")


if __name__ == "__main__":
    main()
