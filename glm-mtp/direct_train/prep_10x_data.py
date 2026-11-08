#!/usr/bin/env python3
"""Build a 10x training dataset from the regen jsonl source, replicating the
data_prep_rot format exactly.

Template (reverse-engineered from data_prep_rot row 0):
  [gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{q}<|assistant|><think></think>{a}
loss_mask: 0 through <|assistant|>, 1 from <think> to the end (answer tokens).
Source rows are Qwen-format regen text; question/response are extracted and
re-wrapped in the GLM template. Rows already present in data_prep_rot are
skipped (question-hash dedup). Filters: seq_len <= max_len, non-empty answer.

Usage: python3 prep_10x_data.py <regen_jsonl> <existing_data_dir> <out_dir> <target_rows>
"""
# ruff: noqa
import json
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

SRC, EXISTING, OUT, TARGET = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
MAX_LEN = 2304
PREFIX = "<gMASK><sop><|system|>Reasoning Effort: Max<|user|>"

tok = AutoTokenizer.from_pretrained(
    "/mnt/weight2/GLM-5.3-W4A8", trust_remote_code=True)
prefix_ids = tok.encode(PREFIX, add_special_tokens=False)
asst_ids = tok.encode("<|assistant|>", add_special_tokens=False)
think_ids = tok.encode("<think></think>", add_special_tokens=False)
n_pre = len(prefix_ids) + len(asst_ids)  # mask=0 through <|assistant|>

# dedup: questions already in the existing dataset
sys.path.insert(0, "/vllm-workspace/speculators/src")
existing_qs = set()
d_old = Dataset.load_from_disk(EXISTING)
for r in d_old:
    ids = [int(x) for x in r["input_ids"]]
    text = tok.decode(ids, skip_special_tokens=False)
    if "<|user|>" in text and "<|assistant|>" in text:
        q = text.split("<|user|>", 1)[1].split("<|assistant|>", 1)[0]
        existing_qs.add(q.strip())
print(f"existing dataset: {len(d_old)} rows, {len(existing_qs)} unique questions",
      flush=True)


def parse_qwen(text):
    """extract (question, response) from a Qwen-format regen text."""
    if "<|im_start|>user" not in text or "<|im_start|>assistant" not in text:
        return None, None
    q = text.split("<|im_start|>user", 1)[1]
    q = q.split("<|im_end|>", 1)[0].strip("\n")
    a = text.split("<|im_start|>assistant", 1)[1]
    a = a.split("<|im_end|>", 1)[0].strip("\n")
    if a.startswith("<think>"):
        a = a.split("</think>", 1)[-1]
    return q.strip(), a.strip()


rows, skipped_dup, skipped_len, scanned = [], 0, 0, 0
with open(SRC) as f:
    for line in f:
        if len(rows) >= TARGET:
            break
        scanned += 1
        try:
            r = json.loads(line)
        except Exception:
            continue
        q, a = parse_qwen(r.get("text", ""))
        if not q or not a:
            continue
        if q in existing_qs:
            skipped_dup += 1
            continue
        ids = prefix_ids + tok.encode(q, add_special_tokens=False) + asst_ids \
            + think_ids + tok.encode(a, add_special_tokens=False)
        if not (8 < len(ids) <= MAX_LEN):
            skipped_len += 1
            continue
        mask = [0] * n_pre + [1] * (len(ids) - n_pre)
        rows.append({"input_ids": ids, "loss_mask": mask, "seq_len": len(ids)})
        if len(rows) % 1000 == 0:
            print(f"{len(rows)}/{TARGET} (scanned {scanned}, dup {skipped_dup}, "
                  f"len-skip {skipped_len})", flush=True)

print(f"final: {len(rows)} rows (scanned {scanned}, dup-skip {skipped_dup}, "
      f"len-skip {skipped_len})", flush=True)
ds = Dataset.from_list(rows)
ds.save_to_disk(OUT)
print(f"saved -> {OUT}", flush=True)
