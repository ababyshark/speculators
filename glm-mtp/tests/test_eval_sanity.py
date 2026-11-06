# ruff: noqa
#!/usr/bin/env python3
"""Test 6: eval result sanity (eval_ab.py output).

Regression for: negative accept_length / malformed JSON that burned hours.

Checks: valid single JSON; n_prompts == 32; all counter deltas >= 0;
per-position rates within [0, 1]; mean_accept_len in [1, 6].

Usage: python3 test_eval_sanity.py <eval_result.json>
"""

import json
import sys


def main():
    try:
        r = json.load(open(sys.argv[1]))
    except Exception as e:
        print(f"FAIL not valid JSON: {e}")
        sys.exit(1)
    errs = []
    if r.get("n_prompts") != 32:
        errs.append(f"n_prompts={r.get('n_prompts')} != 32")
    dacc = r.get("d_accepted_by_pos", {})
    for p in map(str, range(5)):
        v = dacc.get(p)
        if v is None:
            errs.append(f"missing position {p}")
        elif v < 0:
            errs.append(f"negative accepted delta at pos {p}: {v}")
    if r.get("d_draft_steps", 0) <= 0:
        errs.append(f"d_draft_steps={r.get('d_draft_steps')} must be > 0")
    rates = r.get("pos_accept_rates", {})
    for p, v in rates.items():
        if v is not None and not (0.0 <= v <= 1.0):
            errs.append(f"rate[{p}]={v} out of [0,1]")
    mal = r.get("mean_accept_len")
    if mal is None or not (1.0 <= mal <= 6.0):
        errs.append(f"mean_accept_len={mal} out of [1,6]")
    if errs:
        print("FAIL")
        [print("   ", e) for e in errs]
        sys.exit(1)
    print(
        f"PASS: mean_accept_len={mal:.3f}, "
        f"rates={ {k: (round(v, 3) if v else v) for k, v in rates.items()} }"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
