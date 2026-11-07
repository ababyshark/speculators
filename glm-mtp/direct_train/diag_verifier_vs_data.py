#!/usr/bin/env python3
"""Step-by-step diagnostic: verifier-argmax vs training-data tokens.

Answers the core question of "why does training make acceptance worse":
for every supervised position of the training samples, does the VERIFIER's
own argmax agree with the data token that CE trains the drafter to predict?
If agreement is low, CE-on-data provably pulls the drafter away from
verifier-argmax alignment (which is what acceptance measures).

Method: vLLM completions with echo+logprobs returns, for every prompt
position, the actual token's rank in the verifier's distribution.
rank==1  <=>  verifier argmax == data token at that position.

Per-sample logs + aggregate JSON. Usage:
  python3 diag_verifier_vs_data.py <data_path> <verifier> <endpoint> <out_json> [n_rows]
"""
# ruff: noqa
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

DATA, VERIFIER, ENDPOINT, OUT = sys.argv[1:5]
N_ROWS = int(sys.argv[5]) if len(sys.argv) > 5 else 132  # 32 eval + 100 more
BASE = ENDPOINT.rstrip("/")


def post(payload, timeout=600):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + "/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def one_sample(idx, ids, mask):
    p = {"model": VERIFIER, "prompt": ids, "max_tokens": 1,
         "echo": True, "logprobs": 5, "temperature": 0.0}
    try:
        resp = post(p)
    except Exception as e:  # noqa: BLE001
        return {"idx": idx, "error": str(e)[:120]}
    pl = resp["choices"][0].get("prompt_logprobs")
    if not pl or len(pl) < len(ids):
        return {"idx": idx, "error": f"prompt_logprobs len {len(pl) if pl else 0}"}
    agree = []  # rank==1 per position (position i predicts token i given prefix)
    lp = []
    for pos, entry in enumerate(pl):
        if entry is None:
            continue
        actual = str(ids[pos])
        d = entry.get(actual)
        if d is None:
            agree.append(0)  # actual token outside top-5: argmax definitely differs
            lp.append(-20.0)
        else:
            agree.append(1 if d.get("rank") == 1 else 0)
            lp.append(d["logprob"])
    sup = [a for a, m in zip(agree, mask) if m]
    return {
        "idx": idx, "seq_len": len(ids),
        "agree_all": float(np.mean(agree)) if agree else None,
        "agree_supervised": float(np.mean(sup)) if sup else None,
        "supervised_positions": len(sup),
        "mean_data_logprob": float(np.mean(lp)) if lp else None,
        "n_argmax_disagree": int((1 - np.array(agree)).sum()),
        "first_disagree_positions": [i for i, a in enumerate(agree) if a == 0][:8],
    }


def main():
    d = load_from_disk(DATA)
    n = min(N_ROWS, len(d))
    rows = []
    t0 = time.time()
    for i in range(n):
        row = d[i]
        ids = [int(x) for x in row["input_ids"]]
        mask = [bool(x) for x in row["loss_mask"]]
        r = one_sample(i, ids, mask)
        rows.append(r)
        status = r.get("error") or f"agree_sup={r.get('agree_supervised')}"
        print(f"[{i + 1}/{n}] {status}", flush=True)
        if i == 0:  # smoke the response shape before burning through all rows
            pl_probe = None
    el = time.time() - t0

    ok = [r for r in rows if "agree_supervised" in r]
    agg = {
        "n_samples": len(ok),
        "elapsed_min": round(el / 60, 1),
        "verifier_argmax_equals_data_token": {
            "all_positions": float(np.mean([r["agree_all"] for r in ok])),
            "supervised_positions": float(np.mean([r["agree_supervised"] for r in ok])),
        },
        "mean_verifier_logprob_of_data_tokens": float(
            np.mean([r["mean_data_logprob"] for r in ok])),
        "per_sample": rows,
    }
    Path(OUT).write_text(json.dumps(agg, indent=1))
    print(json.dumps({k: v for k, v in agg.items() if k != "per_sample"}, indent=1))


if __name__ == "__main__":
    main()
