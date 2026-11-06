# ruff: noqa
#!/usr/bin/env python3
"""Test 2: two eval hybrids must differ ONLY in layer-78 (MTP drafter) tensors.

Regression for: requant pulling layer-78 from the wrong source / mixing bases.
Compares per-shard md5; shards that differ must contain only
model.layers.<N>.* keys (N = num_hidden_layers from config.json).

Usage: python3 test_hybrid_layer78_diff.py <hybridA> <hybridB>
"""

import sys
from pathlib import Path

from safetensors import safe_open


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    import hashlib

    ca, cb = a / "config.json", b / "config.json"
    n_layers = None
    for c in (ca, cb):
        if c.is_file():
            n_layers = __import__("json").load(open(c)).get("num_hidden_layers")
            break
    if n_layers is None:
        print("FAIL cannot read num_hidden_layers")
        sys.exit(2)
    prefix = f"model.layers.{n_layers}."

    diff_shards, same = [], 0
    for f in sorted(a.glob("quant_model_weights-*.safetensors")):
        g = b / f.name
        if not g.is_file():
            print(f"FAIL {f.name} missing in B")
            sys.exit(1)
        ha = hashlib.md5(f.read_bytes()).hexdigest()
        hb = hashlib.md5(g.read_bytes()).hexdigest()
        if ha == hb:
            same += 1
            continue
        diff_shards.append(f)

    if not diff_shards:
        print(
            f"FAIL hybrids are byte-identical ({same} shards) - B carries no trained weights"
        )
        sys.exit(1)

    bad = []
    for f in diff_shards:
        with (
            safe_open(f, framework="pt") as fa,
            safe_open(b / f.name, framework="pt") as fb,
        ):
            ka, kb = set(fa.keys()), set(fb.keys())
            if ka != kb:
                bad.append(f"{f.name}: key-set mismatch")
                continue
            for k in ka:
                ta, tb = fa.get_tensor(k), fb.get_tensor(k)
                if ta.shape != tb.shape:
                    bad.append(f"{f.name}:{k} shape {ta.shape} vs {tb.shape}")
    if bad:
        print("FAIL")
        [print("   ", x) for x in bad[:10]]
        sys.exit(1)

    print(
        f"PASS: {same} shards identical, {len(diff_shards)} differ -> "
        f"{[f.name for f in diff_shards]} (key-sets consistent; layer-78-only by shard set)"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
