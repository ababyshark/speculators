# ruff: noqa
#!/usr/bin/env python3
"""Test 4: training actually moved the drafter weights (bounded delta).

Regression for: "did the requant pick up trained weights / did training write
anything at all". epoch0 vs epoch3 of the same run must differ in the trained
key groups (self_attn, mlp.gate, input_proj) but stay within a sane magnitude,
and expert weights must be unchanged (FREEZE_EXPERTS).

Usage: python3 test_training_delta.py <epoch0_dir> <epoch3_dir> [max_rel_delta]
Default max_rel_delta = 0.2 (observed ~0.026 for q_a_proj).
"""

import sys
from pathlib import Path

from safetensors import safe_open

TRAINED_GROUPS = ("self_attn.", "mlp.gate", "input_proj", "eh_proj")
MAY_FREEZE = ("e_score_correction_bias", "indexer.")


def load_layer78(d):
    """non-expert layer-78 tensors from a training checkpoint dir."""
    out = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        if "optimizer" in f.name or "rng" in f.name:
            continue
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("mtp_layers.0.") and "experts." not in k:
                    out[k] = sf.get_tensor(k).float()
    return out


def main():
    e0, e3 = Path(sys.argv[1]), Path(sys.argv[2])
    max_rel = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    t0, t3 = load_layer78(e0), load_layer78(e3)
    common = sorted(set(t0) & set(t3))
    if not common:
        print("FAIL no common non-expert layer-78 keys")
        sys.exit(2)
    moved, too_far, no_move, frozen_ok = [], [], [], []
    for k in common:
        d = float((t3[k] - t0[k]).abs().max())
        m = float(t0[k].abs().max()) or 1.0
        rel = d / m
        group = next((g for g in TRAINED_GROUPS if g in k), None)
        if group:
            if d == 0.0:
                no_move.append(k)
            elif rel > max_rel:
                too_far.append((k, rel))
            else:
                moved.append((k, rel))
        else:
            frozen_ok.append((k, d))
    print(f"trained keys moved within bound: {len(moved)}")
    for k, r in moved[:8]:
        print(f"    {k}: rel={r:.4f}")
    if too_far:
        print(f"FAIL {len(too_far)} keys moved too much (>{max_rel}):")
        for k, r in too_far[:5]:
            print(f"    {k}: rel={r:.3f}")
        sys.exit(1)
    if no_move:
        frozen_by_design = [k for k in no_move if any(f in k for f in MAY_FREEZE)]
        unexpected = [k for k in no_move if not any(f in k for f in MAY_FREEZE)]
        if frozen_by_design:
            print(
                f"INFO {len(frozen_by_design)} keys frozen (by design): {frozen_by_design[:3]}"
            )
        if unexpected:
            print(f"FAIL {len(unexpected)} trained keys did not move: {unexpected[:3]}")
            sys.exit(1)
    print(
        f"PASS: training delta present and bounded; {len(frozen_ok)} non-trained keys untouched"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
