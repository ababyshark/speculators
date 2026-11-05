"""Re-quantize a trained MTP drafter back into the W4A8 checkpoint layout.

Reads the ORIGINAL checkpoint's layer-78 quantized tensors as the LAYOUT
TEMPLATE (shapes/dtypes/scale placement) and re-fills them from the drafter's
BF16 weights, mirroring the observed quant conventions:
- int8 per-channel: scale = amax(out_ch)/127, offset = 0
- int4 packed 2-per-byte along dim0 (low nibble = even row):
  scale = amax(row)/8, offset = 0, scale_bias = 0
- FLOAT tensors (norms/gate/indexer-fp/etc.): copied directly from the drafter.

The serving path is then byte-identical to the working A-卷 configuration.
"""

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

QUANT_SUFFIXES = (".weight_scale", ".weight_offset", ".scale_bias")
REVERSE = {
    "input_proj.weight": "eh_proj.weight",
    "token_layernorm.weight": "enorm.weight",
    "hidden_layernorm.weight": "hnorm.weight",
    "final_layernorm.weight": "shared_head.norm.weight",
}


def drafter_layer78(drafter_dir):
    """yield (suffix, bf16 tensor) for the trained MTP layer, wrapper-normalized."""
    files = sorted(glob.glob(str(Path(drafter_dir) / "model-*.safetensors")))
    files = [f for f in files if "index" not in f] or [
        str(Path(drafter_dir) / "model.safetensors")
    ]
    for fp in files:
        with safe_open(fp, framework="pt") as f:
            for k in f.keys():
                if not k.startswith("mtp_layers.0."):
                    continue
                suffix = k[len("mtp_layers.0.") :]
                if suffix.startswith("mlp.moe."):
                    suffix = "mlp." + suffix[len("mlp.moe.") :]
                yield suffix, f.get_tensor(k).to(torch.bfloat16)


def build_trained_layer78(drafter_dir, n_routed, inter, hidden):
    tensors: dict[str, torch.Tensor] = {}
    for suffix, t in drafter_layer78(drafter_dir):
        if suffix == "mlp.experts.gate_up_proj":
            for e in range(t.shape[0]):
                tensors[f"mlp.experts.{e}.gate_proj.weight"] = t[e, :inter]
                tensors[f"mlp.experts.{e}.up_proj.weight"] = t[e, inter:]
        elif suffix == "mlp.experts.down_proj":
            for e in range(t.shape[0]):
                tensors[f"mlp.experts.{e}.down_proj.weight"] = t[e]
        elif suffix in REVERSE:
            tensors[REVERSE[suffix]] = t
        else:
            tensors[suffix] = t
    return tensors


def requant_weight(w: torch.Tensor, stored_shape, stored_dtype, scale_shape):
    """produce (q_packed, scale, offset, scale_bias) mirroring the stored layout.

    Structural rules (from the original stored tensor):
    - stored_shape == logical shape          -> int8 per-channel along dim0
    - stored_shape[0]*2 == logical shape[0]  -> int4 packed along dim0 (low=even)
    """
    W = w.to(torch.float32)
    if tuple(stored_shape) == tuple(W.shape):
        amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = amax / 127.0
        q = torch.clamp(torch.round(W / scale), -128, 127).to(torch.int8)
        offset = torch.zeros_like(scale)
        return q, scale, offset, None
    if stored_shape[0] * 2 == W.shape[0] and stored_shape[1] == W.shape[1]:
        amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = amax / 8.0
        q = torch.clamp(torch.round(W / scale), -8, 7).to(torch.int16)
        low = q[0::2].to(torch.uint8) & 0x0F
        high = (q[1::2].to(torch.uint8) & 0x0F) << 4
        packed = (low | high).to(torch.int8)
        offset = torch.zeros(scale_shape, dtype=torch.float32)
        bias = torch.zeros(scale_shape, dtype=torch.float32)
        return packed, scale, offset, bias
    raise ValueError(
        f"stored {tuple(stored_shape)} incompatible with logical {tuple(W.shape)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src = Path(args.model)
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    out.mkdir(parents=True)

    vcfg = json.load(open(src / "config.json"))
    n_routed, inter, hidden = (
        vcfg["n_routed_experts"],
        vcfg["moe_intermediate_size"],
        vcfg["hidden_size"],
    )
    trained = build_trained_layer78(args.drafter, n_routed, inter, hidden)
    print(f"trained layer-78 tensors: {len(trained)}")

    src_index = json.load(open(src / "quant_model_weights.safetensors.index.json"))
    src_weight_map = src_index["weight_map"]
    prefix = f"model.layers.{vcfg['num_hidden_layers']}."

    # group original layer-78 quant entries by base weight name
    orig_entries: dict[str, dict[str, str]] = {}
    for k in src_weight_map:
        if not k.startswith(prefix):
            continue
        suffix = k[len(prefix) :]
        if suffix.endswith(".weight"):
            base = suffix[: -len(".weight")]
        elif suffix.endswith(QUANT_SUFFIXES):
            base = suffix[: -len(suffix.split(".")[-1]) - 1]
        else:
            base = suffix
        if suffix.endswith(".weight"):
            orig_entries.setdefault(base, {})["weight"] = k
        elif suffix.endswith(QUANT_SUFFIXES):
            orig_entries.setdefault(base, {})[suffix.split(".")[-1]] = k

    replaced_keys: set[str] = set()
    new_weight_map = dict(src_weight_map)

    # group bases BY SHARD so each shard is read+written exactly once
    shard_jobs: dict[str, list[tuple[str, dict]]] = {}
    for base, parts in sorted(orig_entries.items()):
        if "weight" not in parts:
            continue
        tkey = base + ".weight"
        if tkey not in trained:
            continue
        shard = src_weight_map[parts["weight"]]
        shard_jobs.setdefault(shard, []).append((base, parts))

    for shard, jobs in sorted(shard_jobs.items()):
        tensors = {}
        with safe_open(src / shard, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        for base, parts in jobs:
            wkey = parts["weight"]
            out_key = prefix + base + ".weight"
            tkey = base + ".weight"
            replaced_keys.add(wkey)
            new_weight_map[out_key] = wkey  # same shard placement

            with safe_open(src / src_weight_map[wkey], framework="pt") as f:
                sl = f.get_slice(wkey)
                stored_shape = tuple(sl.get_shape())
                stored_dtype = sl.get_dtype()

            W = trained[tkey]
            if "weight_scale" in parts:
                with safe_open(
                    src / src_weight_map[parts["weight_scale"]], framework="pt"
                ) as f:
                    scale_shape = tuple(f.get_slice(parts["weight_scale"]).get_shape())
                q, scale, offset, bias = requant_weight(
                    W, stored_shape, stored_dtype, scale_shape
                )
                tensors[wkey] = q
                tensors[parts["weight_scale"]] = scale
                if "weight_offset" in parts:
                    tensors[parts["weight_offset"]] = offset
                if "scale_bias" in parts:
                    tensors[parts["scale_bias"]] = bias
                print(f"requantized {base}: stored {stored_shape}")
            else:
                tensors[wkey] = W
                print(f"float-copied {base}")
        save_file(tensors, str(out / shard), metadata={"format": "pt"})

    # copy metadata + everything else
    for name in os.listdir(src):
        p = src / name
        if (
            p.is_file()
            and not name.endswith(".safetensors")
            and name != "quant_model_weights.safetensors.index.json"
        ):
            shutil.copy(p, out / name)
    n_link = 0
    for name in os.listdir(src):
        sp = src / name
        if name.endswith(".safetensors") and not (out / name).exists():
            os.symlink(sp, out / name)
            n_link += 1
        elif sp.is_dir() and not (out / name).exists():
            os.symlink(sp, out / name)
    json.dump(src_index, open(out / "quant_model_weights.safetensors.index.json", "w"))
    print(f"requant write-back at {out} (symlinked {n_link})")


if __name__ == "__main__":
    main()
