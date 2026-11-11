# ruff: noqa
"""Per-prompt self-consistency + acceptance test for the 32K longctx experiment.
For each prompt: two identical greedy requests (temp=0, max_tokens=512);
captures output text, /metrics acceptance deltas per request, and R1-vs-R2
output consistency. Sequential (no concurrency) for clean metric attribution.
Usage: python3 selfcons_test.py <prompts_jsonl> <out_json> [max_tokens]
"""
import json, os, sys, time, urllib.request
BASE = os.environ.get("EVAL_BASE", "http://127.0.0.1:8455")
MODEL = os.environ.get("SERVED_MODEL", "glm53")
MAXT = int(sys.argv[3]) if len(sys.argv) > 3 else 512
IN, OUT = sys.argv[1], sys.argv[2]

def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["acc"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_drafted_tokens_total"):
            out["draft"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            out["drafts"] = float(line.split()[-1])
    return out

def gen(prompt_ids, maxt):
    body = json.dumps({"model": MODEL, "prompt": prompt_ids,
                       "max_tokens": maxt, "temperature": 0.0,
                       "ignore_eos": True, "stream": False}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    m0 = metrics()
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    m1 = metrics()
    ch = out["choices"][0]
    acc = m1.get("acc", 0) - m0.get("acc", 0)
    dr = m1.get("draft", 0) - m0.get("draft", 0)
    df = m1.get("drafts", 0) - m0.get("drafts", 0)
    return {"text": ch.get("text", ""), "finish": ch.get("finish_reason"),
            "accepted": acc, "drafted": dr, "drafts": df,
            "accept_len": (acc / df) if df else 0.0, "latency_s": round(dt, 1),
            "prompt_tokens": out["usage"]["prompt_tokens"]}

rows = [json.loads(l) for l in open(IN) if l.strip()]
print(f"{len(rows)} prompts, max_tokens={MAXT}", flush=True)
results = []
for i, r in enumerate(rows):
    pids = r["prompt_token_ids"]
    m = metrics()
    r1 = gen(pids, MAXT)
    r2 = gen(pids, MAXT)
    same = r1["text"] == r2["text"]
    n_same = sum(1 for a, b in zip(r1["text"], r2["text"]) if a == b)
    denom = max(len(r1["text"]), len(r2["text"]), 1)
    rec = {"idx": r.get("idx"), "split": r.get("split"),
           "prompt_tokens": r1["prompt_tokens"],
           "self_consistent": same, "char_consistency": round(n_same / denom, 3),
           "r1_accept_len": round(r1["accept_len"], 3),
           "r2_accept_len": round(r2["accept_len"], 3),
           "r1_text_head": r1["text"][:80], "r2_text_head": r2["text"][:80],
           "latency_s": r1["latency_s"]}
    results.append(rec)
    print(f"[{i}] pt={r1['prompt_tokens']} consistent={same} "
          f"cons={rec['char_consistency']} al={rec['r1_accept_len']}", flush=True)

agg = {}
for sp in ("long32k_in", "long32k_held", "short_ctrl"):
    sub = [r for r in results if r.get("split") == sp]
    if sub:
        agg[sp] = {
            "n": len(sub),
            "self_consistency_rate": round(sum(r["self_consistent"] for r in sub) / len(sub), 3),
            "mean_accept_len_r1": round(sum(r["r1_accept_len"] for r in sub) / len(sub), 3),
            "mean_accept_len_r2": round(sum(r["r2_accept_len"] for r in sub) / len(sub), 3),
        }
json.dump({"config": {"prefix_caching": os.environ.get("PC_MODE", "?")},
           "per_prompt": results, "aggregate_by_split": agg},
          open(OUT, "w"), indent=1, ensure_ascii=False)
print("AGG:", json.dumps(agg, indent=1), flush=True)
print("DONE", flush=True)
