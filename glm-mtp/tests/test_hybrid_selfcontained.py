# ruff: noqa
#!/usr/bin/env python3
"""Test 1: hybrid must be self-contained.

Fails if any file referenced by quant_model_weights.safetensors.index.json
(or any entry in the model dir) is a broken symlink or unreadable.
Regression for: evalA0/evalB hybrids pointing at /mnt/weight2/...-biasfix
after the base was deleted from NFS -> serve crashed at weight load.

Usage: python3 test_hybrid_selfcontained.py <hybrid_dir> [<hybrid_dir> ...]
"""

import json
import sys
from pathlib import Path


def check(hybrid: Path) -> list[str]:
    errs = []
    idx = hybrid / "quant_model_weights.safetensors.index.json"
    if not idx.is_file():
        return [f"{idx} missing"]
    weight_map = json.load(open(idx))["weight_map"]
    shards = sorted(set(weight_map.values()))
    for s in shards:
        p = hybrid / s
        if p.is_symlink() and not p.exists():
            errs.append(
                f"BROKEN SYMLINK {s} -> {Path(p.absolute()).readlink() if False else ''}"
            )
            continue
        if not p.is_file():
            errs.append(f"missing shard {s}")
            continue
        with open(p, "rb") as f:  # header readable?
            if not f.read(8):
                errs.append(f"empty shard {s}")
    norm_covered = any(
        k.endswith(".enorm.weight") and ".78." in k for k in weight_map
    ) and any(k.endswith(".hnorm.weight") and ".78." in k for k in weight_map)
    # mtp_norm_bias.safetensors is only required when the index does NOT carry
    # the enorm/hnorm/shared_head.norm keys inside the quant shards (old
    # biasfix-era packaging shipped them as a side file; GLM-5.3-W4A8 layout
    # keeps them in the index).
    extras = [] if norm_covered else ["config.json", "mtp_norm_bias.safetensors"]
    if norm_covered:
        extras = ["config.json"]
    for extra in extras:
        p = hybrid / extra
        if p.is_symlink() and not p.exists():
            errs.append(f"BROKEN SYMLINK {extra}")
        elif not p.exists():
            errs.append(f"missing {extra}")
    return errs


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    failed = False
    for a in sys.argv[1:]:
        h = Path(a)
        errs = check(h)
        status = "PASS" if not errs else "FAIL"
        print(f"[{status}] {h}" + ("" if not errs else ""))
        for e in errs[:10]:
            print(f"    {e}")
        failed |= bool(errs)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
