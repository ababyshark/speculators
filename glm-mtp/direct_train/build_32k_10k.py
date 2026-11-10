"""Build 10,000 long-context (~30.5K tokens) records from mixed sources:
  - open-perfectblend regen texts (public, model MAY have seen similar)
  - xql internal case docs (internal, unseen)
  - cmb on-policy texts (internal-generated, unseen)
Each record: concatenated doc units to ~30.5K tokens + format-locked
cross-document question. Streamed via datasets.Dataset.from_generator."""
import glob, json, os, random, sys
os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
sys.path.insert(0, "/vllm-workspace/speculators/src")
import datasets
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/mnt/weight2/GLM-5.3-W4A8", trust_remote_code=True)
PREFIX = "<gMASK><sop><|system|>Reasoning Effort: Max<|user|>"
ASST = "<|assistant|><think></think>"
OUT = "/mnt/weight2/l00840580-direct-v13/data_32k_10k"
TARGET = 30500
N_RECORDS = 10000

rng = random.Random(777)

# --- collect source UNITS (text chunks of ~800-2000 chars each) ---
units = []   # (bucket, text)
def add_units(text, bucket, minlen=400):
    for chunk in text.split("\n\n"):
        c = chunk.strip()
        if minlen < len(c) < 6000:
            units.append((bucket, c))

n_pb = 0
import glob as _g
for p in _g.glob("/mnt/share_space/l00840580/speculators-work/data/open-perfectblend/*.jsonl"):
    try:
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    t = d.get("text") or (d.get("conversations") and
                        " ".join(x.get("value", "") for x in d["conversations"])) or ""
                    if t:
                        add_units(t, "pb")
                        n_pb += 1
                except Exception:
                    continue
                if n_pb > 40000:
                    break
    except Exception:
        continue
print(f"pb rows scanned: {n_pb}", flush=True)

n_xql = 0
for p in glob.glob("/mnt/share_space/l00840580/xql-cases/zone_*/*/*.md"):
    try:
        t = open(p, encoding="utf-8", errors="ignore").read()
        add_units(t, "xql")
        n_xql += 1
    except Exception:
        continue
print(f"xql docs: {n_xql}", flush=True)

n_cmb = 0
for name in ("cmb_gen_a.jsonl", "cmb_gen_b.jsonl"):
    p = f"/mnt/weight2/l00840580-direct-v13/{name}"
    if not os.path.exists(p):
        continue
    for line in open(p, encoding="utf-8", errors="ignore"):
        try:
            r = json.loads(line)
            if r.get("text") and len(r["text"]) > 300:
                add_units(r["text"], "cmb")
                n_cmb += 1
        except Exception:
            continue
print(f"cmb texts: {n_cmb}", flush=True)

# pre-encode all units once (fast tokenizer)
enc_units = []
for bucket, text in units:
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > 1200:
        ids = ids[:1200]
    if len(ids) >= 200:
        enc_units.append((bucket, ids))
print(f"encoded units: {len(enc_units)} total tokens: {sum(len(i) for _, i in enc_units)}", flush=True)

# buckets: pb=may-have-seen(public), xql/cmb=unseen(internal)
by_bucket = {"pb": [], "xql": [], "cmb": []}
for b, i in enc_units:
    by_bucket[b].append(i)
rng.shuffle(by_bucket["pb"]); rng.shuffle(by_bucket["xql"]); rng.shuffle(by_bucket["cmb"])

QUESTION = ("\n\n以上是若干份公司内部与外部的技术文档。请完成以下任务，严格按要求格式输出：\n"
    "1) 统计文档中出现的模型/硬件名称，按出现频次列出前 5 个（格式：名称 - 次数）。\n"
    "2) 归纳文档集合的共性主题，列出 3 类（格式：主题 - 一句话说明）。\n"
    "3) 给出一份跨文档的要点汇总清单，格式：1)... 2)... 3)... 4)... 5)...\n"
    "不要输出清单以外的任何内容。")
pre = tok.encode(PREFIX, add_special_tokens=False)
asst = tok.encode(ASST, add_special_tokens=False)
q_ids = tok.encode(QUESTION, add_special_tokens=False)

mix_plan = (["pb"] * 4000) + (["xql"] * 3000) + (["cmb"] * 3000)
rng.shuffle(mix_plan)

def gen():
    idx = 0
    cursor = {"pb": 0, "xql": 0, "cmb": 0}
    for bucket in mix_plan:
        pool = by_bucket[bucket]
        ids, di = [], 0
        while len(ids) < TARGET and cursor[bucket] + di < len(pool):
            u = pool[(cursor[bucket] + di) % len(pool)]
            take = TARGET - len(ids)
            if len(u) > take:
                u = u[:take]
            ids += u
            di += 1
            if di >= len(pool):
                cursor[bucket] = 0   # recycle pool if exhausted
        pids = pre + ids + q_ids + asst
        idx += 1
        yield {"idx": idx, "bucket": bucket, "ctx_tokens": len(ids),
               "prompt_token_ids": pids}
        if idx % 500 == 0:
            print(f"built {idx}/{N_RECORDS}", flush=True)

ds = datasets.Dataset.from_generator(gen)
ds.save_to_disk(OUT)
print(f"DONE -> {OUT}", flush=True)
