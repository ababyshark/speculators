"""Dump N held-out eval prompts (parquet order continues past the training set)."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

TRAIN_FILE = "/mnt/share_space/l00840580/dspark-glm53/data/onpolicy_2500.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default="/models/datasets/spec_dataset/ultrachat_200k/data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
    )
    ap.add_argument(
        "--output",
        default="/mnt/share_space/l00840580/dspark-glm53/data/eval_prompts.jsonl",
    )
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--skip", type=int, default=2052)
    args = ap.parse_args()

    seen = set()
    with open(TRAIN_FILE) as f:
        for line in f:
            seen.add(json.loads(line)["conversations"][0]["content"])
    print(f"train prompts: {len(seen)}")

    table = pq.read_table(args.parquet, columns=["prompt"])
    out = []
    for p in table.column("prompt").to_pylist():
        if p and len(p) < 4096 and p not in seen:
            seen.add(p)
            out.append(p)
            if len(out) >= args.num + args.skip:
                break
    held = out[args.skip : args.skip + args.num]
    with open(args.output, "w") as f:
        for p in held:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
    print(f"held-out eval prompts written: {len(held)} -> {args.output}")


if __name__ == "__main__":
    main()
