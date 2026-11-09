# ruff: noqa
"""Assemble the xql on-policy dataset: merge gen shards, re-encode text with
roundtrip verification, build Arrow for v20 training."""
import json
from pathlib import Path

RUN = Path("/mnt/weight2/l00840580-direct-v13")
MAX_TOTAL = 2304
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/mnt/weight2/GLM-5.3-W4A8", trust_remote_code=True)

recs = [json.loads(l) for l in open("/mnt/share_space/l00840580/tmp_v15_eval/xql_cases_prompts.jsonl")]
train = {r["idx"]: r for r in recs if r["split"] == "train"}

rows = {}
for name in ("xql_gen_text.jsonl", "xql_gen_a.jsonl", "xql_gen_b.jsonl"):
    p = RUN / name
    if not p.exists():
        continue
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["idx"] in train and r.get("text") is not None:
            rows[r["idx"]] = r["text"]
print(f"covered: {len(rows)}/400")

data_rows, rt_bad = [], 0
for idx in sorted(train):
    r = train[idx]
    text = rows.get(idx)
    if text is None:
        print(f"missing idx {idx} — skipped")
        continue
    dec_prompt = tok.decode(r["prompt_token_ids"])
    full_text = dec_prompt + text
    ids = tok.encode(full_text, add_special_tokens=False)
    if len(ids) > MAX_TOTAL:
        ids = ids[:MAX_TOTAL]
    if tok.decode(ids) != full_text[:len(tok.decode(ids))]:
        rt_bad += 1
    boundary = min(len(tok.encode(dec_prompt, add_special_tokens=False)), len(ids))
    lm = [0] * boundary + [1] * (len(ids) - boundary)
    data_rows.append({"input_ids": ids, "loss_mask": lm, "seq_len": len(ids)})
print(f"rows: {len(data_rows)}, roundtrip mismatch: {rt_bad}")

import datasets
out = RUN / "data_prep_rot_xql"
datasets.Dataset.from_list(data_rows).save_to_disk(str(out))
print(f"dataset -> {out}")
