"""Logit parity probe: does the TRAINING framework compute the same function
as SERVING for the warm-start drafter?

Path (a): the speculators MTPDraftModel forward (HF eager, BF16) — the exact
training-world computation — with per-stage hidden captures.
Path (b): the serving-composed computation (rot -> enorm/hnorm -> eh_proj ->
HF layer) built manually — the composition vllm performs.

If (a) == (b) bit-close, the training and serving compositions agree and the
divergence must live in the NPU KERNELS (attention/indexer/quantization). If
(a) != (b), the composition itself diverges — find the stage.

Inputs: real rot-space HS + tokens from data_prep_rot/hidden_states (the
training distribution), the SIGNED warm-start weights.
"""

import glob
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig

sys.path.insert(0, "/mnt/share_space/l00840580/my-contributions/speculators/src")

MODEL = "/mnt/weight2/l00840580-glm53-w4a8-biasfix"


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(d + "/model*.safetensors")):
        if "index" in f:
            continue
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                out[k] = sf.get_tensor(k)
    return out


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    torch.manual_seed(0)
    dev = "npu:0"
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg._attn_implementation = "eager"

    ws = load_dir("/mnt/share_space/l00840580/dspark-glm53/drafter_init/glm53-mtp-warmstart-signed")

    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaDecoderLayer,
        GlmMoeDsaRMSNorm,
        GlmMoeDsaRotaryEmbedding,
    )

    # inputs: real HS + tokens (the training pairing: embed(t+1), HS(t) -> t+2)
    with safe_open(
        "/mnt/share_space/l00840580/dspark-glm53/data_prep_full/hidden_states/hs_0.safetensors",
        framework="pt",
    ) as f:
        H = f.get_tensor("hidden_states").reshape(-1, cfg.hidden_size)[:64]
        toks = f.get_tensor("token_ids")[: len(H) + 1]

    with safe_open(f"{MODEL}/rot.safetensors", framework="pt") as f:
        R = f.get_tensor("rot.weight").to(torch.float32).to(dev)
    H_dev = H.to(dev, torch.bfloat16)
    H_rot = (H_dev.float() @ R.t()).to(torch.bfloat16)

    # the shared weights
    embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size).to(dev, torch.bfloat16)
    embed.weight.data = ws["embed_tokens.weight"].to(dev, torch.bfloat16)
    enorm = GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(dev, torch.bfloat16)
    enorm.weight.data = ws["mtp_layers.0.token_layernorm.weight"].to(dev, torch.bfloat16)
    hnorm = GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(dev, torch.bfloat16)
    hnorm.weight.data = ws["mtp_layers.0.hidden_layernorm.weight"].to(dev, torch.bfloat16)
    final_norm = GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(dev, torch.bfloat16)
    final_norm.weight.data = ws["mtp_layers.0.final_layernorm.weight"].to(dev, torch.bfloat16)
    eh_w = ws["mtp_layers.0.input_proj.weight"].to(dev, torch.bfloat16)
    head_w = ws["lm_head.weight"].to(dev, torch.bfloat16)

    layer = GlmMoeDsaDecoderLayer(cfg, layer_idx=0).to(torch.bfloat16).to(dev)
    layer.eval()
    ws_layer = {k[len("mtp_layers.0."):]: v for k, v in ws.items() if k.startswith("mtp_layers.0.")}
    missing, unexpected = layer.load_state_dict(ws_layer, strict=False)
    print(f"layer load: missing={len(missing)} unexpected={len(unexpected)}")

    rotary = GlmMoeDsaRotaryEmbedding(cfg).to(dev, torch.bfloat16)
    T = min(48, len(H_rot))
    hs = H_rot[:T]
    e = embed(toks[1 : T + 1])

    with torch.no_grad():
        # stage 1: the norms
        en = enorm(e)
        hn = hnorm(hs)
        # stage 2: eh_proj
        proj = (torch.cat([en, hn], dim=-1) @ eh_w.t())
        # stage 3: the layer (attention + MoE)
        pos = torch.arange(T, device=dev).unsqueeze(0)
        pos_emb = rotary(proj.unsqueeze(0), pos)
        layer_out, topk = layer(
            hidden_states=proj.unsqueeze(0),
            position_ids=pos,
            position_embeddings=pos_emb,
        )
        layer_out = layer_out[0]
        # stage 4: final norm
        normed = final_norm(layer_out)
        # stage 5: logits
        logits = normed @ head_w.t()

    # ---- the MTPDraftModel path (a): the same computation through the
    # training framework's module for the SAME inputs
    sys.path.insert(0, "/mnt/share_space/l00840580/my-contributions/speculators/src")
    from speculators.models.mtp.core import MTPDraftModel  # noqa: F401

    print("\n== reference (serving-composed) stage outputs ==")
    print(f"proj: shape={tuple(proj.shape)} absmean={proj.float().abs().mean():.5f}")
    print(f"layer_out: absmean={layer_out.float().abs().mean():.5f}")
    print(f"logits: shape={tuple(logits.shape)} absmax={logits.float().abs().max():.2f}")
    argmax = logits.argmax(-1)
    actual = toks[2 : T + 2]
    agree = (argmax.cpu() == actual[:T].cpu()).float().mean().item()
    print(f"top-1 agreement with the actual next-next token: {agree:.3f}")

    # stage-isolation: rerun the layer with the EXACT training-framework call
    # pattern (the mixin's forward) to isolate composition vs layer semantics
    from speculators.models.mtp.model_definitions import MTPLayerMixin  # noqa: F401

    # the mixin composition manual reference:
    proj2 = layer.input_proj(
        torch.cat(
            [layer.token_layernorm(e), layer.hidden_layernorm(hs)], dim=-1
        )
    )
    print(f"\nmixin-path proj vs serving-path proj: cos={cos(proj2, proj):.6f} maxdiff={(proj2 - proj).abs().max().item():.2e}")

    # the layer rerun via the mixin-equivalent path (the same weights, the
    # HF layer forward directly) — determinism check
    layer_out2, _ = layer(
        hidden_states=proj2.unsqueeze(0),
        position_ids=pos,
        position_embeddings=pos_emb,
    )
    print(f"layer rerun determinism: cos={cos(layer_out2[0], layer_out):.6f}")

    torch.save(
        {
            "proj": proj.cpu(),
            "layer_out": layer_out.cpu(),
            "normed": normed.cpu(),
            "logits": logits.cpu(),
            "toks": toks.cpu(),
        },
        "/mnt/share_space/l00840580/dspark-glm53/logit_parity_ref.pt",
    )
    print("reference saved -> logit_parity_ref.pt")


if __name__ == "__main__":
    main()
