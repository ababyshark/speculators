"""Tokenize on-policy conversations into speculator-format passthrough rows:
{"input_ids": [...], "loss_mask": [...], "seq_len": n}

Assistant tokens (incl. EOS) get loss_mask=1; prompt tokens 0.
"""

import argparse
import json

from transformers import AutoTokenizer

MODEL = "/mnt/weight2/l00840580-glm53-w4a8-biasfix"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seq-length", type=int, default=8192)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    def as_ids(enc):
        if hasattr(enc, "input_ids"):
            enc = enc.input_ids
        if hasattr(enc, "ids"):
            return list(enc.ids)
        if isinstance(enc[0], (list, tuple)):
            return list(enc[0])
        return list(enc)

    n_ok, n_skip = 0, 0
    with open(args.input) as f, open(args.output, "w") as out:
        for line in f:
            row = json.loads(line)
            convs = row["conversations"]
            if not convs or convs[-1]["role"] != "assistant":
                n_skip += 1
                continue
            full = as_ids(
                tok.apply_chat_template(
                    convs, tokenize=True, add_generation_prompt=False
                )
            )
            prefix = as_ids(
                tok.apply_chat_template(
                    convs[:-1], tokenize=True, add_generation_prompt=True
                )
            )
            if len(full) > args.seq_length or len(full) <= len(prefix) + 1:
                n_skip += 1
                continue
            mask = [0] * len(prefix) + [1] * (len(full) - len(prefix))
            out.write(
                json.dumps({"input_ids": full, "loss_mask": mask, "seq_len": len(full)})
                + "\n"
            )
            n_ok += 1
    print(f"tokenized {n_ok}, skipped {n_skip}")


if __name__ == "__main__":
    main()
