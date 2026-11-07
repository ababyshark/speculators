#!/usr/bin/env python3
"""CPU-only data contract validation for the direct-training pipeline.

Validates data_prep_rot against the speculators ArrowDataset contract:
- load_from_disk works; columns == {input_ids, loss_mask, seq_len}
- every row: len(input_ids) == len(loss_mask) == seq_len
- token ids within verifier vocab; loss_mask has >=1 supervised token
- train/val split at TRAIN_RATIO yields non-empty complementary splits

Writes a JSON report; exit 0 iff all checks pass.
"""
# ruff: noqa
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import AutoConfig

DATA = Path(sys.argv[1])
VERIFIER = Path(sys.argv[2])
OUT = Path(sys.argv[3])
TRAIN_RATIO = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9006

report = {"data": str(DATA), "checks": {}}
d = load_from_disk(str(DATA))
report["checks"]["rows"] = len(d)
report["checks"]["columns_ok"] = set(d.column_names) >= {"input_ids", "loss_mask", "seq_len"}

vocab = AutoConfig.from_pretrained(str(VERIFIER), trust_remote_code=True).vocab_size
sl = np.array(d["seq_len"])
bad_len, bad_mask, bad_tok, empty_mask = 0, 0, 0, 0
for i in range(len(d)):
    row = d[i]
    ids, mask = row["input_ids"], row["loss_mask"]
    if not (len(ids) == len(mask) == row["seq_len"]):
        bad_len += 1
    if sum(1 for m in mask if m) == 0:
        empty_mask += 1
    if max(ids) >= vocab or min(ids) < 0:
        bad_tok += 1
report["checks"]["len_mismatch_rows"] = bad_len
report["checks"]["empty_loss_mask_rows"] = empty_mask
report["checks"]["out_of_vocab_rows"] = bad_tok
report["checks"]["vocab_size"] = vocab
report["checks"]["seq_len_min"] = int(sl.min())
report["checks"]["seq_len_max"] = int(sl.max())
report["checks"]["total_tokens_m"] = round(float(sl.sum()) / 1e6, 2)
split_idx = int(len(d) * TRAIN_RATIO)
report["checks"]["train_rows"] = split_idx
report["checks"]["val_rows"] = len(d) - split_idx
report["checks"]["hs_storage_gb"] = round(float(sl.sum()) * 6144 * 2 * 3 / 1e9, 1)

ok = (
    report["checks"]["columns_ok"]
    and bad_len == 0
    and empty_mask == 0
    and bad_tok == 0
    and report["checks"]["val_rows"] > 0
)
report["status"] = "PASS" if ok else "FAIL"
OUT.write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
sys.exit(0 if ok else 1)
