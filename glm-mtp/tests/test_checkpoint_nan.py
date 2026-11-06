# ruff: noqa
#!/usr/bin/env python3
"""Test 3: drafter checkpoint has no NaN/Inf in float tensors.

Regression for: suspecting trained weights contained Inf when serving died
(turned out to be environment, but the check must stay as a gate).

Usage: python3 test_checkpoint_nan.py <checkpoint_dir>
Exits 0 clean, 1 if bad tensors found.
"""

import glob
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    d = Path(sys.argv[1])
    files = sorted(glob.glob(str(d / "*.safetensors")))
    if not files:
        print(f"FAIL no safetensors in {d}")
        sys.exit(2)
    bad = []
    for f in files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                t = sf.get_tensor(k)
                if t.is_floating_point():
                    n = int((~torch.isfinite(t)).sum())
                    if n:
                        bad.append((Path(f).name, k, n, t.numel()))
    if bad:
        print(f"FAIL {len(bad)} non-finite tensors")
        for name, k, n, tot in bad[:20]:
            print(f"    {name}:{k} {n}/{tot}")
        sys.exit(1)
    print(f"PASS: {len(files)} shards clean (no NaN/Inf)")
    sys.exit(0)


if __name__ == "__main__":
    main()
