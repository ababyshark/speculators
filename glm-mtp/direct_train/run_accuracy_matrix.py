# ruff: noqa
"""Run the accuracy x consistency matrix: 24 prompts x 3 runs, sequential.
Per prompt: extract answer from output (substring/code eval), compare vs
expected (accuracy) + across runs (consistency). Output table JSON."""
import json, os, re, sys, time, urllib.request
BASE = os.environ.get("EVAL_BASE", "http://127.0.0.1:8455")
MODEL = "glm53"
IN = "/mnt/weight2/l00840580-direct-v13/eval_accuracy_matrix.jsonl"
OUT = "/mnt/weight2/l00840580-direct-v13/accuracy_matrix_results.json"
N_RUNS = 3

def gen(prompt_ids, maxt=256):
    body = json.dumps({"model": MODEL, "prompt": prompt_ids,
                       "max_tokens": maxt, "temperature": 0.0, "seed": 1024,
                       "stream": False}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        out = json.loads(r.read())
    ch = out["choices"][0]
    return ch.get("text", ""), round(time.time() - t0, 1), out["usage"]["prompt_tokens"]

rows = [json.loads(l) for l in open(IN) if l.strip()]
print(f"{len(rows)} prompts x {N_RUNS} runs", flush=True)
results = []
for i, r in enumerate(rows):
    outs = []
    for run in range(N_RUNS):
        text, lat, pt = gen(r["prompt_token_ids"])
        outs.append(text)
        print(f"[{i}] {r['scenario']}/{r['length']} run{run}: lat={lat}s out={text[:60]!r}", flush=True)
    expect = r["expect"]
    match = [str(expect) in o for o in outs]
    n_match = sum(match)
    consistent = len(set(o.strip() for o in outs)) == 1
    # numeric extraction for math/code: first number in output
    acc = n_match / N_RUNS
    results.append({**{k: r[k] for k in ("scenario", "length", "expect")},
                    "outputs": outs, "match": match, "n_match": n_match,
                    "accuracy": round(acc, 3), "consistent": consistent})
    print(f"    accuracy={acc:.2f} consistent={consistent}", flush=True)

agg = {}
for sc in ("english", "chinese", "math", "code"):
    for ln in ("1K", "2K", "4K", "8K", "16K", "32K"):
        sub = [r for r in results if r["scenario"] == sc and r["length"] == ln]
        if sub:
            agg[f"{sc}/{ln}"] = {"accuracy": round(sum(r["accuracy"] for r in sub) / len(sub), 3),
                                 "consistent": all(r["consistent"] for r in sub)}
json.dump({"per_prompt": results, "aggregate": agg}, open(OUT, "w"), indent=1, ensure_ascii=False)
print("AGG:", json.dumps(agg, indent=1, ensure_ascii=False), flush=True)
print("DONE", flush=True)
