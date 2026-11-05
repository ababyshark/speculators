"""Transform extracted hidden states into the serving rot-space (NPU-accelerated):
H' = H @ rot.weight^T  (identical to AscendDeepSeekMTP's self.rot(hidden)).

Reusable: works on any hs_*.safetensors dir produced by
`speculators generate-offline-data` (extract_hidden_states mode).
Idempotent: existing outputs are skipped unless --overwrite.
"""

import argparse
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hs-dir",
        required=True,
        help="dir with hs_*.safetensors (hidden_states, token_ids)",
    )
    ap.add_argument(
        "--rot", required=True, help="path to rot.safetensors (rot.weight [H, H])"
    )
    ap.add_argument("--rot-key", default="rot.weight")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="npu:0", help="npu:N / cuda:N / cpu")
    ap.add_argument(
        "--fp32-matmul", action="store_true", help="matmul in fp32 (default: fp32)"
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dev = args.device
    with safe_open(args.rot, framework="pt") as f:
        R = f.get_tensor(args.rot_key).to(torch.float32)
    Rt = R.t().contiguous().to(dev)  # y = H @ R^T

    src = Path(args.hs_dir)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(
        f for f in os.listdir(src) if f.startswith("hs_") and f.endswith(".safetensors")
    )
    todo = [f for f in files if args.overwrite or not (dst / f).exists()]
    print(f"{len(files)} files, {len(todo)} to transform -> {dst} on {dev}", flush=True)

    import time

    t0 = time.time()
    for i, name in enumerate(todo):
        with safe_open(src / name, framework="pt") as f:
            hs = f.get_tensor("hidden_states")  # [seq, 1, H] bf16
            toks = f.get_tensor("token_ids")
        seq = hs.shape[0]
        h2 = hs.reshape(seq, -1).to(dev, torch.float32)
        out = (h2 @ Rt).to(torch.bfloat16).cpu().reshape(hs.shape)
        save_file(
            {"hidden_states": out.contiguous(), "token_ids": toks}, str(dst / name)
        )
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"{i + 1}/{len(todo)}  {el / (i + 1) * 1000:.0f}ms/file", flush=True)
    print(f"done: {len(todo)} transformed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
