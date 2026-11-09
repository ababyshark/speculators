# ruff: noqa
"""Generate on-policy answers for xql-cases train prompts (400 records)."""
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
for k in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"): os.environ.pop(k, None)
ENDPOINT = "http://127.0.0.1:8000/v1/completions"
MODEL = "/mnt/weight2/l00840580-direct-v13/hybrid_B_v18_float"
OUT = "/mnt/weight2/l00840580-direct-v13/xql_gen_text.jsonl"
MAX_TOTAL = 2304
WORKERS = 8
recs = [json.loads(l) for l in open("/mnt/share_space/l00840580/tmp_v15_eval/xql_cases_prompts.jsonl")]
train = [r for r in recs if r["split"] == "train"]
done = set()
if Path(OUT).exists():
    done = {json.loads(l)["idx"] for l in open(OUT)}
todo = [r for r in train if r["idx"] not in done]
n_done = 0
print(f"shard xql: to generate {len(todo)} (done {len(done)})", flush=True)

def gen_one(r):
    prompt = r["prompt_token_ids"]
    budget = MAX_TOTAL - len(prompt) - 4
    if budget < 32:
        return None
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": budget,
                       "temperature": 0.0, "seed": 1024, "stream": False}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={"Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                out = json.loads(resp.read())
            return (r["idx"], out["choices"][0].get("text") or "")
        except Exception as e:
            if attempt == 5:
                print(f"[{r['idx']}] FAILED: {e}", flush=True)
                return None
            time.sleep(20 * (attempt + 1))

t0 = time.time()
with open(OUT, "a") as fout, ThreadPoolExecutor(WORKERS) as ex:
    for res in ex.map(gen_one, todo):
        if res is None:
            continue
        idx, text = res
        fout.write(json.dumps({"idx": idx, "text": text}) + "\n")
        fout.flush()
        n_done += 1
        if n_done % 25 == 0:
            print(f"{n_done}/{len(todo)} done, {time.time()-t0:.0f}s", flush=True)
print(f"XQL GEN COMPLETE: {n_done}, {time.time()-t0:.0f}s", flush=True)
