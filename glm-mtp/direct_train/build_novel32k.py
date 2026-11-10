"""Build NOVEL-generation 32K eval prompts from swezero agent trajectories.

Same-repo trajectory groups concatenated to ~28-30K context (system prompt +
task + bash tool interactions over real repos), then a continuation prompt:
"continue the agent session with the next THOUGHT + bash command" — the
continuation is novel (not copyable from context). 8 in_eval + 8 held_eval.
"""
import glob
import json, os, random
from collections import defaultdict
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

rows_by_repo = defaultdict(list)
files = sorted(glob.glob("/mnt/weight2/train-dataset/raw/agentic-sft/swezero_train-*.jsonl"))[:8]
n_scanned = 0
for fp in files:
    with open(fp, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            repo = d.get("repo", "")
            msgs = d.get("messages", [])
            full = "\n\n".join(
                (f"[{m.get('role')}]\n{m.get('content','')}" if m.get("role") != "user"
                 else f"[user]\n{m.get('content','')}") for m in msgs)
            if len(full) > 3000:
                rows_by_repo[repo].append(full)
            n_scanned += 1
print(f"scanned {n_scanned} rows, repos with content: {len(rows_by_repo)}", flush=True)

# build 30K contexts from same-repo trajectory groups
cands = []
for repo, trs in rows_by_repo.items():
    if len(trs) < 6:
        continue
    rng = random.Random(hash(repo) % 2**31)
    for _ in range(3):
        sel = rng.sample(trs, min(len(trs), 8))
        ids = []
        for t in sel:
            tids = tok.encode(t, add_special_tokens=False)
            if len(ids) + len(tids) > 29500:
                tids = tids[:29500 - len(ids)]
            ids += tids
            if len(ids) >= 29500:
                break
        if 24000 <= len(ids) <= 30000:
            cands.append((repo, ids))
    if len(cands) >= 40:
        break
print(f"novel-32K contexts: {len(cands)}", flush=True)

rng = random.Random(55)
rng.shuffle(cands)
CONT = ("\n\n[以上为本 agent 会话的完整历史。请继续该会话：输出下一个 THOUGHT 段落"
        "（简述你当前的判断与计划），随后输出恰好一个 ```bash 代码块（一条命令），"
        "不要输出任何其他内容。]")
recs = []
for i, (repo, ids) in enumerate(cands[:16]):
    pids = pre + ids + tok.encode(CONT, add_special_tokens=False) + asst
    split = "novel32k_in" if i < 8 else "novel32k_held"
    recs.append({"idx": 5000 + i, "repo": repo, "split": split,
                 "prompt_token_ids": pids})

OUT = "/mnt/share_space/l00840580/tmp_v15_eval/eval_novel32k_ids.jsonl"
with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print("splits:", Counter(r["split"] for r in recs), flush=True)
print(f"-> {OUT}", flush=True)
