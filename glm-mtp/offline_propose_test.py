"""Offline: compare vendor vs trained MTP layer proposals on real hidden states.

For each input variant (raw H, rot(H) with W^T, rot(H) with W), run the vendor
layer (warmstart) and the trained layers (t5fp32 raw-H-trained, t5rot
rot-H-trained) through enorm/hnorm/eh_proj -> layer -> shared_head.head and
report argmax agreement with the actual next token (from token_ids) and with
each other.
"""

import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaRMSNorm,
)

sys.path.insert(0, "/mnt/share_space/l00840580/my-contributions/speculators/src")

MODEL = "/mnt/weight2/l00840580-glm53-w4a8-biasfix"

from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaRotaryEmbedding,
)


def load_layer(path):
    import glob

    sd = {}
    files = sorted(glob.glob(f"{path}/model*.safetensors"))
    files = [f for f in files if "index" not in f]
    for fp in files:
        with safe_open(fp, framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
    return sd


def build_mtp(sd, config, dtype=torch.bfloat16, device="npu:0"):
    """Minimal MTP module from the drafter state dict (HF GlmMoeDsa form)."""
    from torch import nn

    class Attn(nn.Module):
        pass

    # assemble via GlmMoeDsa modeling: use the decoder layer class directly

    import copy

    cfg = copy.copy(config)
    mlp_types = list(cfg.mlp_layer_types)
    mlp_types[0] = "sparse"
    cfg.mlp_layer_types = mlp_types
    indexer_types = list(cfg.indexer_types)
    indexer_types[0] = "full"
    cfg.indexer_types = indexer_types
    layer = GlmMoeDsaDecoderLayer(cfg, layer_idx=0).to(dtype).to(device)
    layer.eval()
    return layer


def main():
    torch.manual_seed(0)
    dev = "npu:0"
    vcfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg = vcfg
    cfg._attn_implementation = "sdpa"

    warm = load_layer(
        "/mnt/share_space/l00840580/dspark-glm53/drafter_init/glm53-mtp-warmstart"
    )
    raw_trained = load_layer(
        "/mnt/share_space/l00840580/dspark-glm53/checkpoints/glm53_mtp_t5fp32/epoch3_end"
    )
    rot_trained = load_layer(
        "/mnt/share_space/l00840580/dspark-glm53/checkpoints/glm53_mtp_t5rot/epoch3_end"
    )

    with safe_open(
        "/mnt/weight2/l00840580-glm53-w4a8-biasfix/rot.safetensors", framework="pt"
    ) as f:
        R = f.get_tensor("rot.weight").to(torch.float32).to(dev)

    # input: a real hidden state
    with safe_open(
        "/mnt/share_space/l00840580/dspark-glm53/data_prep_full/hidden_states/hs_0.safetensors",
        framework="pt",
    ) as f:
        H = (
            f.get_tensor("hidden_states")
            .reshape(-1, cfg.hidden_size)
            .to(dev, torch.bfloat16)[:256]
        )
        toks = f.get_tensor("token_ids").to(dev)[: len(H)]
        H = H[: len(toks) - 2]  # leave room for the +1/+2 alignment

    H_rotT = (H.float() @ R.t()).to(torch.bfloat16)
    H_rot = (H.float() @ R).to(torch.bfloat16)

    embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size).to(dev, torch.bfloat16)
    embed.weight.data = warm["embed_tokens.weight"].to(dev, torch.bfloat16)

    enorm = GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(dev, torch.bfloat16)
    hnorm = GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(dev, torch.bfloat16)
    head_norm = GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(
        dev, torch.bfloat16
    )
    enorm.weight.data = warm["mtp_layers.0.token_layernorm.weight"].to(
        dev, torch.bfloat16
    )
    hnorm.weight.data = warm["mtp_layers.0.hidden_layernorm.weight"].to(
        dev, torch.bfloat16
    )
    head_norm.weight.data = warm["mtp_layers.0.final_layernorm.weight"].to(
        dev, torch.bfloat16
    )
    eh_w = warm["mtp_layers.0.input_proj.weight"].to(dev, torch.bfloat16)
    head_w = warm["lm_head.weight"].to(dev, torch.bfloat16)

    layer = build_mtp(warm, cfg).eval()
    # load warm weights into the layer
    ws = {}
    for k, v in warm.items():
        if k.startswith("mtp_layers.0."):
            ws[k[len("mtp_layers.0.") :]] = v
    # strip the wrapper level for mlp
    ws = {
        (k.replace("mlp.moe.", "mlp.", 1) if k.startswith("mlp.moe.") else k): v
        for k, v in ws.items()
    }
    missing, unexpected = layer.load_state_dict(ws, strict=False)
    print("missing:", len(missing), "unexpected:", len(unexpected))

    layer_rot = build_mtp(rot_trained, cfg).eval()
    ws_r = {}
    for k, v in rot_trained.items():
        if k.startswith("mtp_layers.0."):
            ws_r[k[len("mtp_layers.0.") :]] = v
    ws_r = {
        (k.replace("mlp.moe.", "mlp.", 1) if k.startswith("mlp.moe.") else k): v
        for k, v in ws_r.items()
    }
    layer_rot.load_state_dict(ws_r, strict=False)

    layer_raw = build_mtp(raw_trained, cfg).eval()
    ws_p = {}
    for k, v in raw_trained.items():
        if k.startswith("mtp_layers.0."):
            ws_p[k[len("mtp_layers.0.") :]] = v
    ws_p = {
        (k.replace("mlp.moe.", "mlp.", 1) if k.startswith("mlp.moe.") else k): v
        for k, v in ws_p.items()
    }
    layer_raw.load_state_dict(ws_p, strict=False)

    rotary = GlmMoeDsaRotaryEmbedding(cfg).to(dev, torch.bfloat16)
    position_ids = torch.arange(len(H), device=dev).unsqueeze(0)

    @torch.no_grad()
    def propose(layer, h_input):
        # MTP step-0 alignment: embed(t+1) + HS(t) -> predict t+2;
        # rotary positions continue the sequence (t+1 onward).
        e = embed(toks[1 : len(h_input)])
        hn = hnorm(h_input[:-1])
        T = e.shape[0]
        proj = (torch.cat([enorm(e), hn], dim=-1) @ eh_w.t()).unsqueeze(0)  # [1, T, H]
        pos = torch.arange(1, T + 1, device=dev).unsqueeze(0)
        pos_emb = rotary(proj, pos)
        out, _ = layer(
            hidden_states=proj,
            position_ids=pos,
            position_embeddings=pos_emb,
        )
        logits = head_w @ head_norm(out[0]).t()  # [V, T]
        return logits.t().argmax(-1)  # [T] predicts toks[2:]

    # MTP step-0 target: token t+2 for input (embed t+1, HS t)
    actual = toks[2 : len(H) + 1]

    for name, h_in in [("raw", H), ("rot(W^T)", H_rotT), ("rot(W)", H_rot)]:
        p_warm = propose(layer, h_in)
        agree_warm = (p_warm == actual).float().mean().item()
        p_rot = propose(layer_rot, h_in)
        agree_rot = (p_rot == actual).float().mean().item()
        p_raw = propose(layer_raw, h_in)
        agree_raw = (p_raw == actual).float().mean().item()
        same_wr = (p_warm == p_rot).float().mean().item()
        same_ww = (p_warm == p_raw).float().mean().item()
        print(
            f"{name:9s}: agree_with_next  warm={agree_warm:.3f} rot_trained={agree_rot:.3f} "
            f"raw_trained={agree_raw:.3f} | warm==rot_trained {same_wr:.3f} warm==raw_trained {same_ww:.3f}"
        )


if __name__ == "__main__":
    main()
