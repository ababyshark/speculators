# ruff: noqa
"""Assemble on-policy dataset v2 from TEXT responses: re-encode with the
GLM tokenizer, verify decode roundtrip, build Arrow + 32-prompt eval set."""

import json
import random
from pathlib import Path

RUN = Path("/mnt/weight2/l00840580-direct-v13")
MAX_TOTAL = 2304

import datasets
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/mnt/weight2/GLM-5.3-W4A8", trust_remote_code=True)

rng = random.Random(42)
sample = sorted(rng.sample(range(2052), 300))
ss = set(sample)

rows = {}
for name in ("onpolicy_gen2_a.jsonl", "onpolicy_gen2_b.jsonl"):
    for line in open(RUN / name):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["idx"] in ss and r.get("prompt") and r.get("text") is not None:
            rows[r["idx"]] = r
avail = sorted(ss & set(rows))
print(f"available: {len(avail)}/300")

data_rows, rt_bad = [], 0
for idx in avail:
    r = rows[idx]
    dec_prompt = tok.decode(r["prompt"])
    full_text = dec_prompt + r["text"]
    ids = tok.encode(full_text, add_special_tokens=False)
    if len(ids) > MAX_TOTAL:
        ids = ids[:MAX_TOTAL]
    if tok.decode(ids) != full_text[: len(tok.decode(ids))]:
        rt_bad += 1
    boundary = len(tok.encode(dec_prompt, add_special_tokens=False))
    boundary = min(boundary, len(ids))
    lm = [0] * boundary + [1] * (len(ids) - boundary)
    data_rows.append({"input_ids": ids, "loss_mask": lm, "seq_len": len(ids)})
print(f"rows: {len(data_rows)}, roundtrip mismatch: {rt_bad}")

out = RUN / "data_prep_rot_onpolicy300"
datasets.Dataset.from_list(data_rows).save_to_disk(str(out))
print(f"dataset -> {out}")

rng2 = random.Random(7)
eval_idx = sorted(rng2.sample(avail, 32))
with open(RUN / "eval_prompts_onpolicy32_ids.jsonl", "w") as f:
    for idx in eval_idx:
        f.write(
            json.dumps({"prompt_token_ids": rows[idx]["prompt"], "idx": idx}) + "\n"
        )
print("eval set -> eval_prompts_onpolicy32_ids.jsonl (32 prompts)")
