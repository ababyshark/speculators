"""Offline A/B MTP acceptance evaluation on the training data.

No serving involved: rolls the MTP drafter forward greedily on the captured
verifier hidden states and compares each proposal against the true next token
from the training data. Acceptance per position 1..5 = the fraction of
positions where the drafter's argmax matches the data's next token.

A卷 = the pre-training native drafter (drafter_init/glm53-mtp-speculators-v9)
B卷 = the post-training checkpoint  (checkpoints/glm53_mtp5_v12/epoch3_end)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def load_drafter(ckpt: str, device):
    """Load a speculators-format MTP drafter into a plain module."""
    from transformers import AutoConfig, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval().to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--hs-dir", default="/mnt/share_space/l00840580/dspark-glm53/data_prep_full/hidden_states")
    ap.add_argument("--norm-weight", default="/mnt/weight2/l00840580-glm53-w4a8-biasfix")
    ap.add_argument("--output", required=True)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--device", default="npu:0")
    args = ap.parse_args()

    device = args.device
    drafter = load_drafter(args.drafter, device)
    core = drafter.model  # GlmMTPCore-like; forward = (hidden, input_ids, ...) -> (logits, new_hidden)

    # the verifier's final norm (all-ones weight = the pure RMS) for the pre-norm captures
    import glob as g

    norm_w = None
    for f in sorted(g.glob(args.norm_weight + "/quant_model_weights-*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as sf:
            if "model.norm.weight" in sf.keys():
                norm_w = sf.get_tensor("model.norm.weight").float()
                break
    eps = 1e-5

    files = sorted(g.glob(args.hs_dir + "/hs_*.safetensors"))[: args.rows]
    accepts = [[] for _ in range(args.steps)]

    for ri, fp in enumerate(files):
        with safe_open(fp, framework="pt", device="cpu") as sf:
            hs = sf.get_tensor("hidden_states").float()[0]  # [L, 6144] pre-norm
            toks = sf.get_tensor("token_ids")
        L = min(hs.shape[0], toks.shape[0]) - args.steps
        x = hs[:L]
        post = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm_w
        h = post.to(torch.bfloat16).to(device)
        ids = toks[:L].to(device)
        future = toks[L : L + args.steps].to(device)  # the oracle next tokens

        for step in range(args.steps):
            with torch.no_grad():
                logits, h = core(h.unsqueeze(0), ids.unsqueeze(0), spec_step_index=step)
            pred = logits[0, -1].argmax()
            accepts[step].append(int(pred.item() == future[step].item()))
            # teacher-force: feed the TRUE next token + the drafter's new hidden
            ids = torch.cat([ids, future[step : step + 1]])
            h = h[:, -1:]
        if ri % 8 == 0:
            print(f"row {ri}/{len(files)}", flush=True)

    rates = {f"pos{i+1}": sum(a) / max(len(a), 1) for i, a in enumerate(accepts)}
    # the overall accepted-length = sum of the per-pos accept rates (the greedy MTP metric)
    rates["expected_accepted_len"] = round(sum(rates.values()), 4)
    report = {"label": args.label, "drafter": args.drafter, "rows": len(files), **rates}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.output, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
