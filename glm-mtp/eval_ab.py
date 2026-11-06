# ruff: noqa
#!/usr/bin/env python3
"""A/B acceptance eval: whole-run counter deltas from /metrics.

Usage: python3 eval_ab.py <prompts_jsonl> <out_json>
"""

import json
import re
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8100"
POS_RE = re.compile(
    r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d)"[^}]*\}\s+([\d.e+]+)'
)
DRAFT_RE = re.compile(r"vllm:spec_decode_num_draft_tokens_total\{[^}]*\}\s+([\d.e+]+)")
DRAFTS_RE = re.compile(r"vllm:spec_decode_num_drafts_total\{[^}]*\}\s+([\d.e+]+)")
ACC_RE = re.compile(r"vllm:spec_decode_num_accepted_tokens_total\{[^}]*\}\s+([\d.e+]+)")
GEN_RE = re.compile(r"vllm:generation_num_tokens_total\{[^}]*\}\s+([\d.e+]+)")


def scrape():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    acc = {int(m.group(1)): float(m.group(2)) for m in POS_RE.finditer(txt)}
    val = lambda rex: float(m.group(1)) if (m := rex.search(txt)) else 0.0
    return {
        "accepted": acc,
        "draft": val(DRAFT_RE),
        "drafts": val(DRAFTS_RE),
        "accepted_total": val(ACC_RE),
        "gen": val(GEN_RE),
    }


def send(prompt_ids, max_tokens=64, tok=None):
    if tok is not None:
        payload = {"prompt": tok.decode(prompt_ids, skip_special_tokens=True)}
    else:
        payload = {"prompt": prompt_ids}  # v0.28: token-id list goes in prompt
    body = json.dumps(
        {**payload, "max_tokens": max_tokens, "temperature": 0.0, "seed": 1024}
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        json.loads(r.read())
    return time.time() - t0


def main():
    prompts, out_path = sys.argv[1], sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else None
    tok = None
    if model_path:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rows = [json.loads(l) for l in open(prompts) if l.strip()]
    ids = [
        r["prompt_token_ids"]
        if "prompt_token_ids" in r
        else r["prompt"]["prompt_token_ids"]
        for r in rows
    ]
    print(f"{len(ids)} prompts, len range {min(map(len, ids))}-{max(map(len, ids))}")

    before = scrape()
    print("before:", before)
    times = []
    for i, p in enumerate(ids):
        times.append(send(p, tok=tok))
        print(f"  [{i + 1}/{len(ids)}] {times[-1]:.1f}s", flush=True)

    # metrics may lag final request; scrape until stable
    after = scrape()
    for _ in range(5):
        time.sleep(3)
        s2 = scrape()
        if s2 == after:
            break
        after = s2
    print("after:", after)

    dacc = {
        p: after["accepted"].get(p, 0) - before["accepted"].get(p, 0) for p in range(5)
    }
    ddraft = after["draft"] - before["draft"]
    ddrafts = after["drafts"] - before["drafts"]
    dacc_total = after["accepted_total"] - before["accepted_total"]
    dgen = after["gen"] - before["gen"]
    result = {
        "n_prompts": len(ids),
        "d_accepted_by_pos": dacc,
        "d_draft_steps": ddrafts,
        "d_draft_total": ddraft,
        "d_accepted_total": dacc_total,
        "d_gen_total": dgen,
        "wall_times": times,
    }
    # per-position accept rate: each draft step proposes 1 token per position
    rates = {k: (dacc[k] / ddrafts if ddrafts > 0 else None) for k in range(5)}
    result["pos_accept_rates"] = rates
    result["mean_accept_len"] = (1 + dacc_total / ddrafts) if ddrafts else None
    json.dump(result, open(out_path, "w"), indent=1)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "d_accepted_by_pos",
                    "d_draft_steps",
                    "d_draft_total",
                    "d_accepted_total",
                    "pos_accept_rates",
                    "mean_accept_len",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
