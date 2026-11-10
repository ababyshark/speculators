"""Build 32K long-context eval prompts from internal cases (OOD content).
Each 32K prompt = 40-70 case docs concatenated with document markers + a
format-locked cross-document question. Plus 8 short-context controls using
the SAME content style for the length-effect comparison."""
import glob, json, os, random
os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
import sys
sys.path.insert(0, "/vllm-workspace/speculators/src")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/mnt/weight2/GLM-5.3-W4A8", trust_remote_code=True)
PREFIX = "<gMASK><sop><|system|>Reasoning Effort: Max<|user|>"
ASST = "<|assistant|><think></think>"
pre = tok.encode(PREFIX, add_special_tokens=False)
asst = tok.encode(ASST, add_special_tokens=False)

mds = sorted(glob.glob("/mnt/share_space/l00840580/xql-cases/zone_*/*/*.md"))
docs = []
for p in mds:
    try:
        t = open(p, encoding="utf-8", errors="ignore").read().strip()
    except Exception:
        continue
    if len(t) > 400:
        docs.append(t)
rng = random.Random(321)
rng.shuffle(docs)
print(f"docs available: {len(docs)}", flush=True)

def build_long(target_tokens, seed):
    r = random.Random(seed)
    pool = docs[:]
    r.shuffle(pool)
    ids, used, di = [], [], 0
    q_head = None
    while len(ids) < target_tokens and di < len(pool):
        d = pool[di]; di += 1
        seg = f"\n\n===== 内部案例文档 {di} =====\n" + d
        sids = tok.encode(seg, add_special_tokens=False)
        if len(ids) + len(sids) > target_tokens:
            take = target_tokens - len(ids)
            sids = sids[:take]
        ids += sids
        used.append(d[:60].replace("\n", " "))
    return ids, di

QUESTION = ("\n\n以上是多个公司内部昇腾推理部署案例文档。请完成以下任务，严格按要求格式输出：\n"
    "1) 统计所有案例中出现的模型名称，按出现次数从高到低列出前 5 个（格式：模型名 - 次数）。\n"
    "2) 归纳案例中反复出现的共性问题类型，列出 3 类（格式：类型名 - 一句话说明）。\n"
    "3) 给出一份汇总修复步骤清单，格式：1)... 2)... 3)... 4)... 5)...\n"
    "不要输出清单以外的任何内容。")

OUT = "/mnt/weight2/l00840580-direct-v13/eval_32k_prompts.jsonl"
recs = []
TARGET = 30500
for i in range(8):
    ids, nd = build_long(TARGET, seed=100 + i)
    pids = pre + ids + tok.encode(QUESTION, add_special_tokens=False) + asst
    recs.append({"idx": 1000 + i, "split": "long32k_in", "n_docs": nd,
                 "prompt_token_ids": pids})
    print(f"long_in {i}: ctx={len(ids)} tokens, docs={nd}", flush=True)
for i in range(8):
    ids, nd = build_long(TARGET, seed=500 + i)
    pids = pre + ids + tok.encode(QUESTION, add_special_tokens=False) + asst
    recs.append({"idx": 2000 + i, "split": "long32k_held", "n_docs": nd,
                 "prompt_token_ids": pids})
    print(f"long_held {i}: ctx={len(ids)} tokens, docs={nd}", flush=True)
# short controls: single-doc QA, same style
for i in range(8):
    d = docs[3000 + i]
    seg = f"===== 内部案例文档 =====\n{d[:4000]}"
    sids = tok.encode(seg, add_special_tokens=False)
    pids = pre + sids + tok.encode(QUESTION, add_special_tokens=False) + asst
    recs.append({"idx": 3000 + i, "split": "short_ctrl", "n_docs": 1,
                 "prompt_token_ids": pids})
    print(f"short_ctrl {i}: ctx={len(sids)} tokens", flush=True)

with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print("splits:", Counter(r["split"] for r in recs), flush=True)
print(f"-> {OUT}", flush=True)
