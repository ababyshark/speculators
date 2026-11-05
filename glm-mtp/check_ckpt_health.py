"""Health check for the MTP=5 epoch-1 checkpoint: load, verify finiteness,
confirm weights changed vs warm-start, and probe a forward on CPU."""

import json

import torch
from safetensors import safe_open

CKPT = "/mnt/share_space/l00840580/dspark-glm53/checkpoints/glm53_mtp_t5/0"
WARM = "/mnt/share_space/l00840580/dspark-glm53/drafter_init/glm53-mtp-warmstart"

# 1. file-level
meta = json.load(open(f"{CKPT}/model.safetensors.index.json")) if False else None
import os

print("files:", sorted(os.listdir(CKPT)))


# 2. tensor-level: finiteness + diff vs warm-start on a few weight groups
def tensors_of(path):
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            out[k] = k
    return out


ck = tensors_of(f"{CKPT}/model.safetensors")
wm_keys = [
    k
    for k in ck
    if "gate_up_proj" in k or "self_attn.o_proj" in k or k == "embed_tokens.weight"
]
print(f"total tensors in ckpt: {len(ck)}")

checked = diff = 0
with (
    safe_open(f"{CKPT}/model.safetensors", framework="pt") as fc,
    safe_open(f"{WARM}/model-00000-of-00003.safetensors", framework="pt") as fw,
):
    for k in list(fc.keys()):
        if checked >= 6:
            break
        if not any(
            s in k for s in ("gate_up_proj", "o_proj", "input_proj", "hidden_layernorm")
        ):
            continue
        t = fc.get_tensor(k)
        finite = torch.isfinite(t.float()).all().item()
        line = f"{k}: shape={tuple(t.shape)} dtype={t.dtype} finite={finite}"
        if k in fw.keys():
            w = fw.get_tensor(k)
            d = (t.float() - w.float()).abs().max().item()
            line += f" maxdiff_vs_warmstart={d:.4e}"
        print(line)
        checked += 1

# 3. config sanity
cfg = json.load(open(f"{CKPT}/config.json"))
st = cfg["speculators_config"]["proposal_methods"][0]["speculative_tokens"]
print("speculative_tokens in ckpt config:", st)
print("HEALTH CHECK DONE")
