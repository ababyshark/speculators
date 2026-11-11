# ruff: noqa
"""Build the accuracy x consistency eval matrix:
4 scenarios (english/chinese/math/code) x 6 lengths (1K/2K/4K/8K/16K/32K)
x 3 runs. Every prompt has a PLANTED verifiable answer (constructed here).
Filler = internal case corpus (OOD) + cmb English texts."""
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

# fillers
zh_fill, en_fill, code_fill = [], [], []
for p in sorted(glob.glob("/mnt/share_space/l00840580/xql-cases/zone_*/*/*.md")):
    try:
        t = open(p, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    for chunk in t.split("\n\n"):
        c = chunk.strip()
        if 300 < len(c) < 2500:
            (code_fill if ("def " in c or "```" in c or "import " in c) else zh_fill).append(c)
for name in ("cmb_gen_a.jsonl", "cmb_gen_b.jsonl"):
    p = f"/mnt/weight2/l00840580-direct-v13/{name}"
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="ignore"):
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            for chunk in t.split("\n\n"):
                c = chunk.strip()
                if 300 < len(c) < 2500:
                    en_fill.append(c)
print(f"fillers: zh={len(zh_fill)} en={len(en_fill)} code={len(code_fill)}", flush=True)
rng = random.Random(20261110)
rng.shuffle(zh_fill); rng.shuffle(en_fill); rng.shuffle(code_fill)

LENGTHS = [("1K", 900), ("2K", 1900), ("4K", 3900), ("8K", 7900), ("16K", 15800), ("32K", 30500)]

def enc(t): return tok.encode(t, add_special_tokens=False)

def build(scenario, L_name, L_target, seed):
    r = random.Random(seed)
    ids, q = [], ""
    expect = ""
    if scenario == "math":
        a, b, c = r.randint(1000, 9999), r.randint(100, 999), r.randint(1000, 9999)
        expect = str(a * b + c)
        q = f"\n\n请直接计算以下算式的结果，只输出数值：{a} × {b} + {c} = ？"
    elif scenario == "code":
        seed_v = r.randint(100, 9999)
        mult = r.choice([7919, 104729, 65537])
        expect = str((seed_v * mult) % 65536)
        fn = f"def verify_token(seed):\n    return (seed * {mult}) % 65536\n"
        ids += enc(f"===== 内部工具代码片段 =====\n{fn}")
        q = f"\n\n以上代码中 verify_token({seed_v}) 的返回值是多少？只输出数值。"
    elif scenario == "english":
        code_name = f"PRJ-{r.randint(10000, 99999)}-{r.choice(['ALPHA','BRAVO','DELTA'])}"
        team = r.choice(["Atlas", "Sirius", "Vega", "Orion"])
        fact = f"The internal project code for the {team} deployment is {code_name}."
        ids += enc("\n\n" + fact)
        q = f"\n\nAccording to the documents above, what is the internal project code for the {team} deployment? Answer with the code only."
        expect = code_name
    else:  # chinese
        code_name = f"内部代号-{r.randint(10000, 99999)}-{r.choice(['天权','天枢','瑶光'])}"
        team = r.choice(["天权团队", "紫微团队", "开阳团队"])
        fact = f"内部代号 {code_name} 的项目由 {team} 负责交付。"
        ids += enc("\n\n" + fact)
        q = f"\n\n以上文档中提到 {team} 负责的项目的内部代号是什么？只回答代号本身。"
        expect = code_name
    # filler to length
    budget = L_target - len(pre) - len(asst) - len(enc(q)) - 30
    fill_pool = en_fill if scenario == "english" else (code_fill if scenario == "code" else zh_fill)
    fi = r.randint(0, max(0, len(fill_pool) - 1))
    while len(ids) < budget and fill_pool:
        chunk = fill_pool[fi % len(fill_pool)]; fi += 1
        cids = enc("\n\n" + chunk)
        if len(ids) + len(cids) > budget:
            cids = cids[:budget - len(ids)]
        ids += cids
        if not fill_pool:
            break
    # plant the fact INSIDE the filler (not at the very end for retrieval difficulty)
    if scenario in ("english", "chinese"):
        pos = int(len(ids) * r.choice([0.25, 0.5, 0.75]))
        fact_ids = enc("\n\n" + fact)
        ids = ids[:pos] + fact_ids + ids[pos:]
    pids = pre + ids + enc(q) + asst
    return {"prompt_token_ids": pids, "expect": expect, "ctx_tokens": len(ids)}

out = []
for sc in ("english", "chinese", "math", "code"):
    for L_name, L_target in LENGTHS:
        rec = build(sc, L_name, L_target, seed=hash((sc, L_name)) % 2**31)
        rec.update({"scenario": sc, "length": L_name})
        out.append(rec)
        print(f"{sc}/{L_name}: ctx={rec['ctx_tokens']} expect={expect if (expect:=rec['expect']) else ''}", flush=True)

with open("/mnt/weight2/l00840580-direct-v13/eval_accuracy_matrix.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"matrix: {len(out)} prompts -> eval_accuracy_matrix.jsonl", flush=True)
