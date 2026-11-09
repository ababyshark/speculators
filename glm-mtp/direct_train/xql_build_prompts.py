# ruff: noqa
"""Build MTP-training prompts from internal xql-cases (Ascend deployment cases).

Styles: RAG-QA / summarization / extraction / raw continuation over case .md
content + CSV-structured Q&A. Output jsonl: idx, prompt_token_ids, split,
source, style. Split (seed 42): 400 train / 32 in_eval / 32 held_eval.
"""
import glob
import json
import os
import random
import re
import sys

os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)

from transformers import AutoTokenizer

ROOT = "/mnt/share_space/l00840580/xql-cases"
OUT = "/mnt/share_space/l00840580/tmp_v15_eval/xql_cases_prompts.jsonl"
TOK = "/mnt/weight2/GLM-5.3-W4A8"
PREFIX = "<gMASK><sop><|system|>Reasoning Effort: Max<|user|>"
ASST = "<|assistant|><think></think>"
CONTENT_BUDGET = 1150  # tokens for the case body inside the user turn

tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
pre_ids = tok.encode(PREFIX, add_special_tokens=False)

def enc(t):
    return tok.encode(t, add_special_tokens=False)

def clean_md(text):
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # images
    text = re.sub(r"\[链接\]\([^)]*\)", "", text)               # link shortforms
    text = re.sub(r"https?://\S+", "", text)                    # urls
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def budget_content(body, limit):
    ids = enc(body)
    if len(ids) <= limit:
        return body, len(ids)
    cut = body
    while len(enc(cut)) > limit and len(cut) > 200:
        cut = cut[: int(len(cut) * 0.9)].rsplit("\n", 1)[0]
    return cut + "\n...(案例内容因长度截断)", len(enc(cut))

records = []
mds = sorted(glob.glob(f"{ROOT}/zone_*/*/*.md")) + sorted(glob.glob(f"{ROOT}/series_*/*.md"))
print(f"md files: {len(mds)}", flush=True)
styles = ["rag_qa", "summarize", "extract"]

for i, path in enumerate(mds):
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    text = clean_md(text)
    title_m = re.match(r"#\s*(.+)", text)
    title = title_m.group(1).strip() if title_m else Path(path).stem
    body = text
    if len(enc(body)) < 150:            # skip near-empty stubs
        continue
    content, clen = budget_content(body, CONTENT_BUDGET)
    style = styles[i % 3]
    if style == "rag_qa":
        q = (f"以下是公司内部的一篇昇腾推理部署案例文档：\n\n{content}\n\n"
             f"请根据案例内容回答：1) 该案例的核心问题/挑战是什么？2) 最终采取了什么解决方案？3) 效果如何？")
    elif style == "summarize":
        q = f"请用三句话总结以下内部技术案例的背景、解决方案与最终效果：\n\n{content}"
    else:
        q = (f"从以下内部案例中提取所有出现的模型名称、硬件型号和关键性能数字，"
             f"以列表形式输出：\n\n{content}")
    prompt_ids = pre_ids + enc(q) + enc(ASST)
    if len(prompt_ids) > 2000:          # hard cap
        continue
    records.append({"idx": len(records), "prompt_token_ids": prompt_ids,
                    "source": path.replace(ROOT + "/", ""), "style": style,
                    "title": title})

# CSV-structured Q&A (scene/problem/solution fields)
import csv as _csv
csv_p = f"{ROOT}/新案例结构化_20261022.csv"
n_csv = 0
try:
    with open(csv_p, encoding="utf-8-sig", errors="ignore") as f:
        for row in _csv.DictReader(f):
            scene, prob, sol = row.get("场景", ""), row.get("问题", ""), row.get("方案", "")
            if len(scene) + len(prob) + len(sol) < 60:
                continue
            q = (f"在{scene.strip()}场景下遇到了这样的问题：{prob.strip()}\n\n"
                 f"请结合你的知识给出：可能的排查思路和推荐解决方案。")
            prompt_ids = pre_ids + enc(q) + enc(ASST)
            if len(prompt_ids) > 2000:
                continue
            records.append({"idx": len(records), "prompt_token_ids": prompt_ids,
                            "source": "csv", "style": "csv_qa",
                            "title": row.get("标题", "")[:60]})
            n_csv += 1
            if n_csv >= 150:
                break
except Exception as e:
    print(f"csv skipped: {e}", flush=True)

rng = random.Random(42)
rng.shuffle(records)
splits = {"train": records[:400], "in_eval": records[400:432], "held_eval": records[432:464]}
with open(OUT, "w") as f:
    for split, recs in splits.items():
        for r in recs:
            r["split"] = split
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"total {len(records)}: train={len(splits['train'])} "
      f"in_eval={len(splits['in_eval'])} held_eval={len(splits['held_eval'])}", flush=True)
lens = [len(r["prompt_token_ids"]) for r in records]
print(f"prompt len range {min(lens)}-{max(lens)}", flush=True)
