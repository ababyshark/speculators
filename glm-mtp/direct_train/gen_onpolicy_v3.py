# ruff: noqa
"""Route-B data generation v2: 300-record subset across TWO endpoints.

Usage (inside container):
  python3 gen_onpolicy_v2.py a   # records sampled[:150]   -> endpoint 8000 -> gen_a.jsonl
  python3 gen_onpolicy_v2.py b   # records sampled[150:300] -> endpoint 8101 -> gen_b.jsonl
Deterministic sample (seed 42) so both shards cover the same 300.
"""

import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(k, None)

sys.path.insert(0, "/vllm-workspace/speculators/src")
from datasets import load_from_disk  # noqa: E402

SHARD = sys.argv[1].lower()
assert SHARD in ("a", "b")
ENDPOINT = (
    "http://127.0.0.1:8000/v1/completions"
    if SHARD == "a"
    else "http://127.0.0.1:8101/v1/completions"
)
OUT_JSONL = f"/mnt/weight2/l00840580-direct-v13/onpolicy_gen2_{SHARD}.jsonl"
MODEL = "/mnt/weight2/l00840580-direct-v13/hybrid_B_v18_float"
DATA = "/mnt/share_space/l00840580/dspark-glm53/data_prep_rot"
MAX_TOTAL = 2304
WORKERS = 12

rng = random.Random(42)
sample = sorted(rng.sample(range(2052), 300))
lo, hi = (0, 150) if SHARD == "a" else (150, 300)
todo_base = sample[lo:hi]

d = load_from_disk(DATA)
print(f"shard {SHARD}: {len(todo_base)} records, endpoint {ENDPOINT}", flush=True)

done = set()
if Path(OUT_JSONL).exists():
    with open(OUT_JSONL) as f:
        for line in f:
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass
todo = [i for i in todo_base if i not in done]
n_done = 0
print(f"to generate: {len(todo)} (done {len(done)})", flush=True)


def gen_one(idx: int):
    row = d[int(idx)]
    ids = [int(x) for x in row["input_ids"]]
    lm = [int(x) for x in row["loss_mask"]]
    try:
        m1 = lm.index(1)
    except ValueError:
        return None
    prompt = ids[: m1 + 5]
    budget = MAX_TOTAL - len(prompt) - 4
    if budget < 32:
        return None
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": budget,
            "temperature": 0.0,
            "seed": 1024,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json"}
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                out = json.loads(r.read())
            return (idx, prompt, out["choices"][0].get("text") or "")
        except Exception as e:  # noqa: BLE001
            if attempt == 5:
                print(f"[{idx}] FAILED: {e}", flush=True)
                return None
            time.sleep(20 * (attempt + 1))


t0 = time.time()
with open(OUT_JSONL, "a") as fout, ThreadPoolExecutor(WORKERS) as ex:
    for res in ex.map(gen_one, todo):
        if res is None:
            continue
        idx, prompt, toks = res
        fout.write(json.dumps({"idx": idx, "prompt": prompt, "text": toks}) + "\n")
        fout.flush()
        n_done += 1
        if n_done % 10 == 0:
            print(f"{n_done}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)
print(
    f"SHARD {SHARD} COMPLETE: {n_done} generated, {time.time() - t0:.0f}s", flush=True
)
