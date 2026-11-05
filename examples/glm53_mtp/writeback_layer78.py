"""Write the trained MTP drafter (speculators checkpoint) back into a copy of
the GLM-5.3 W4A8 checkpoint as a BF16 (FLOAT) layer 78.

- Only shards containing layer-78 tensors are materialized; unchanged shards
  are symlinked (saves ~350 GB).
- quant_model_description.json: replaced tensors -> FLOAT; their scale/offset/
  scale_bias entries are removed (quantization is no longer applicable).
- The weight index is rewritten accordingly.

Usage:
    python writeback_layer78.py \
        --drafter checkpoints/glm53_mtp_t5fp32/3 \
        --model /mnt/weight2/l00840580-glm53-w4a8-biasfix \
        --output /mnt/weight2/l00840580-glm53-w4a8-mtp5-trained
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REVERSE = {
    "mtp_layers.0.input_proj.weight": "eh_proj.weight",
    "mtp_layers.0.token_layernorm.weight": "enorm.weight",
    "mtp_layers.0.hidden_layernorm.weight": "hnorm.weight",
    "mtp_layers.0.final_layernorm.weight": "shared_head.norm.weight",
}


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

    drafter_index = json.load(open(Path(args.drafter) / "model.safetensors.index.json"))
    drafter_map = drafter_index["weight_map"]

    # build layer-78 replacement map: model.layers.78.<x> -> bf16 tensor
    prefix = "model.layers.78."
    replacements: dict[str, torch.Tensor] = {}
    with safe_open(Path(args.drafter) / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            if not k.startswith("mtp_layers.0."):
                continue
            suffix = k[len("mtp_layers.0.") :]
            if (
                suffix == "input_layernorm.weight"
                or suffix == "post_attention_layernorm.weight"
                or suffix.startswith("self_attn.")
                or suffix.startswith("mlp.")
            ):
                new_k = prefix + suffix
            elif suffix in REVERSE:
                new_k = prefix + REVERSE[suffix]
            else:
                continue  # embed_tokens / shared_head.head: not written back
            replacements[new_k] = f.get_tensor(k).to(torch.bfloat16)
    print(f"replacements: {len(replacements)} layer-78 tensors")
    assert len(replacements) > 700, "unexpectedly few replacements"

    src_index = json.load(open(src / "quant_model_weights.safetensors.index.json"))
    src_weight_map = src_index["weight_map"]

    # copy/patch metadata files
    for name in os.listdir(src):
        p = src / name
        if (
            p.is_file()
            and not name.endswith(".safetensors")
            and name != "quant_model_weights.safetensors.index.json"
        ):
            shutil.copy(p, out / name)

    # group layer-78 tensors by source shard
    touched_shards: dict[str, set[str]] = {}
    for k in replacements:
        shard = src_weight_map[k]
        touched_shards.setdefault(shard, set()).add(k)

    new_weight_map = dict(src_weight_map)
    desc_changed = 0
    for shard, keys in sorted(touched_shards.items()):
        dst_name = shard  # keep the same shard filename
        dst_path = out / dst_name
        tensors = {}
        with safe_open(src / shard, framework="pt") as f:
            src_keys = list(f.keys())
            for k in src_keys:
                if k in keys:
                    tensors[k] = replacements[k]
                else:
                    tensors[k] = f.get_tensor(k)
        # drop quant sidecar tensors for replaced weights
        for k in keys:
            for suf in (".weight_scale", ".weight_offset", ".scale_bias"):
                side = k[: -len(".weight")] + suf if k.endswith(".weight") else k + suf
                tensors.pop(side, None)
                new_weight_map.pop(side, None)
                desc_changed += 1
        save_file(tensors, str(dst_path), metadata={"format": "pt"})
        print(f"patched shard {dst_name}: {len(keys)} replaced")

    # patch description
    desc = json.load(open(src / "quant_model_description.json"))
    for k in replacements:
        if k in desc:
            desc[k] = "FLOAT"
        for suf in ("weight_scale", "weight_offset", "scale_bias"):
            base = k[: -len(".weight")] if k.endswith(".weight") else k
            desc.pop(f"{base}.{suf}", None)
    json.dump(desc, open(out / "quant_model_description.json", "w"))
    print(
        f"description patched ({desc_changed} sidecars dropped, {len(replacements)} -> FLOAT)"
    )

    json.dump(
        src_index | {"weight_map": new_weight_map},
        open(out / "quant_model_weights.safetensors.index.json", "w"),
    )

    # symlink untouched shards
    n_link = 0
    for name in os.listdir(src):
        if name.endswith(".safetensors") and not (out / name).exists():
            os.symlink(src / name, out / name)
            n_link += 1
    print(f"symlinked {n_link} untouched shards")
    print(f"write-back checkpoint at {out}")


if __name__ == "__main__":
    main()
