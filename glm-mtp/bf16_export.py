"""Export a speculators drafter checkpoint as a native MTP checkpoint dir.

Writes the drafter's BF16 tensors under native layer-78 key names with the
verifier's config, so vllm's own MTP draft-model loading path
(--speculative-config '{"method":"mtp","model":<dir>}' ) consumes it directly
with BF16 kernels — bypassing both the W4A8 requant and any speculators
serving integration. Used for the quantization-vs-kernel-semantics
discrimination experiment.

Usage: bf16_export.py --drafter <ckpt_dir> --model <biasfix> --output <dir>
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch

# speculators drafter key -> native layer-78 key
REMAP = {
    "input_proj.weight": "eh_proj.weight",
    "token_layernorm.weight": "enorm.weight",
    "hidden_layernorm.weight": "hnorm.weight",
    "final_layernorm.weight": "shared_head.norm.weight",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    sd = {}
    for f in sorted(glob.glob(f"{args.drafter}/model*.safetensors")):
        if "index" in f:
            continue
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                sd[k] = sf.get_tensor(k)

    native: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k in ("embed_tokens.weight", "lm_head.weight"):
            continue  # the drafter uses the verifier's embed/lm_head
        if k.startswith("mtp_layers.0."):
            base = k[len("mtp_layers.0.") :]
            if base in REMAP:
                native[f"model.layers.78.{REMAP[base]}"] = v.contiguous()
            else:
                native[f"model.layers.78.{base}"] = v.contiguous()
        else:
            native[k] = v.contiguous()

    save_file(native, str(out / "model.safetensors"))
    print(f"exported {len(native)} tensors -> {out}/model.safetensors")

    # config: the verifier's config (the MTP loader derives the drafter from it)
    src_cfg = Path(args.model) / "config.json"
    shutil.copy(src_cfg, out / "config.json")
    for extra in (
        "config.py",
        "generation_config.json",
        "rot.safetensors",
        "mtp_norm_bias.safetensors",
    ):
        src = Path(args.model) / extra
        if src.exists():
            shutil.copy(src, out / extra)
    print("config + rot + aux files copied")


if __name__ == "__main__":
    main()
