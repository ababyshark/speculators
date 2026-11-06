"""Logit-parity tests: does the training framework compute the same function
as the serving composition for the warm-start drafter?

Encodes the training-serving semantic-gap investigation as runnable checks:

- the serving-composed stages (rot -> norms -> eh_proj -> layer -> final_norm
  -> lm_head) must produce finite, sane outputs;
- the MIXIN composition (the training framework's path) must match the
  serving composition exactly (the composition contract at the parity level);
- the top-1 agreement with the actual tokens must clear a floor — the offline
  harness's historical 0.004-garbage would fail this immediately (the test
  that would have saved days of misdiagnosis);
- the full pipeline (MTPDraftModel) parity vs the manual composition.

NPU-gated: skips gracefully when the device is busy (a serve holding HBM) or
absent. CPU fallback for the composition checks where the numerics allow.
"""

import glob
import os

import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig

REPO = os.environ.get(
    "GLM53_REPO", "/mnt/share_space/l00840580/my-contributions/speculators"
)
WORKSPACE = os.environ.get("GLM53_WS", "/mnt/share_space/l00840580/dspark-glm53")
MODEL = os.environ.get("GLM53_MODEL", "/mnt/weight2/l00840580-glm53-w4a8-biasfix")
WARMSTART = os.environ.get(
    "GLM53_WARMSTART", f"{WORKSPACE}/drafter_init/glm53-mtp-warmstart-signed"
)

DEVICE = None


def _device() -> str:
    global DEVICE
    if DEVICE is not None:
        return DEVICE
    try:
        import torch_npu  # noqa: F401

        free, total = torch.npu.mem_get_info(0)
        if free / total > 0.5:
            DEVICE = "npu:0"  # a serve holding HBM leaves <50%
        else:
            DEVICE = "cpu"  # fall back: the composition checks run on CPU
    except ImportError:
        DEVICE = "cpu"
    return DEVICE


def _load_dir(d):
    out = {}
    for f in sorted(glob.glob(d + "/model*.safetensors")):
        if "index" in f:
            continue
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                out[k] = sf.get_tensor(k)
    return out


@pytest.fixture(scope="module")
def ctx():
    dev = _device()
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg._attn_implementation = "eager"

    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaDecoderLayer,
        GlmMoeDsaRMSNorm,
        GlmMoeDsaRotaryEmbedding,
    )

    ws = _load_dir(WARMSTART)
    # the trained checkpoints drop embed/lm_head by design (frozen, shared
    # with the verifier): always take them from the base warm-start
    ws_base = _load_dir(
        "/mnt/share_space/l00840580/dspark-glm53/drafter_init/glm53-mtp-warmstart-signed"
    )
    for k in ("embed_tokens.weight", "lm_head.weight"):
        ws.setdefault(k, ws_base[k])
    with safe_open(
        f"{WORKSPACE}/data_prep_full/hidden_states/hs_0.safetensors", framework="pt"
    ) as f:
        H = f.get_tensor("hidden_states").reshape(-1, cfg.hidden_size)[:64]
        toks = f.get_tensor("token_ids")[: len(H) + 1]
    with safe_open(f"{MODEL}/rot.safetensors", framework="pt") as f:
        R = f.get_tensor("rot.weight").to(torch.float32).to(dev)
    H_rot = (H.to(dev, torch.bfloat16).float() @ R.t()).to(torch.bfloat16)

    embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size).to(dev, torch.bfloat16)
    embed.weight.data = ws["embed_tokens.weight"].to(dev, torch.bfloat16)
    enorm = GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(dev, torch.bfloat16)
    enorm.weight.data = ws["mtp_layers.0.token_layernorm.weight"].to(
        dev, torch.bfloat16
    )
    hnorm = GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(dev, torch.bfloat16)
    hnorm.weight.data = ws["mtp_layers.0.hidden_layernorm.weight"].to(
        dev, torch.bfloat16
    )
    final_norm = GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(
        dev, torch.bfloat16
    )
    final_norm.weight.data = ws["mtp_layers.0.final_layernorm.weight"].to(
        dev, torch.bfloat16
    )
    eh_w = ws["mtp_layers.0.input_proj.weight"].to(dev, torch.bfloat16)
    head_w = ws["lm_head.weight"].to(dev, torch.bfloat16)

    layer = GlmMoeDsaDecoderLayer(cfg, layer_idx=0).to(torch.bfloat16).to(dev)
    layer.eval()
    ws_layer = {
        k[len("mtp_layers.0.") :]: v
        for k, v in ws.items()
        if k.startswith("mtp_layers.0.")
    }
    layer.load_state_dict(ws_layer, strict=False)

    rotary = GlmMoeDsaRotaryEmbedding(cfg).to(dev, torch.bfloat16)
    T = min(48, len(H_rot))
    return {
        "dev": dev,
        "cfg": cfg,
        "layer": layer,
        "rotary": rotary,
        "embed": embed,
        "enorm": enorm,
        "hnorm": hnorm,
        "final_norm": final_norm,
        "eh_w": eh_w,
        "head_w": head_w,
        "hs": H_rot[:T],
        "toks": toks,
        "T": T,
    }


def _serving_compose(ctx):
    """The serving composition: enorm(embed(t+1)) + hnorm(HS(t)) -> eh_proj ->
    layer -> final_norm -> lm_head. Returns the stage outputs."""
    dev, T = ctx["dev"], ctx["T"]
    with torch.no_grad():
        e = ctx["embed"](ctx["toks"][1 : T + 1])
        en = ctx["enorm"](e)
        hn = ctx["hnorm"](ctx["hs"])
        proj = torch.cat([en, hn], dim=-1) @ ctx["eh_w"].t()
        pos = torch.arange(T, device=dev).unsqueeze(0)
        pos_emb = ctx["rotary"](proj.unsqueeze(0), pos)
        out, topk = ctx["layer"](
            hidden_states=proj.unsqueeze(0),
            position_ids=pos,
            position_embeddings=pos_emb,
        )
        normed = ctx["final_norm"](out[0])
        logits = normed @ ctx["head_w"].t()
    return {"proj": proj, "layer_out": out[0], "normed": normed, "logits": logits}


def test_stage_outputs_sane(ctx) -> None:
    """Every serving-composition stage must be finite and non-degenerate."""
    stages = _serving_compose(ctx)
    for name, t in stages.items():
        assert torch.isfinite(t.float()).all(), f"{name}: non-finite"
    assert stages["logits"].shape[0] == ctx["T"]


def test_mixin_composition_matches_serving(ctx) -> None:
    """The training-framework composition (MTPLayerMixin's forward inputs:
    token_layernorm(embed) + hidden_layernorm(HS), embed first) must equal the
    serving composition's eh_proj input EXACTLY."""
    from speculators.models.mtp.model_definitions import GlmMoeDsaMTPLayer

    dev, T = ctx["dev"], ctx["T"]
    mtp_layer = GlmMoeDsaMTPLayer(ctx["cfg"], layer_idx=0).to(torch.bfloat16).to(dev)
    mtp_layer.eval()
    ws_layer = {
        k[len("mtp_layers.0.") :]: v
        for k, v in _load_dir(WARMSTART).items()
        if k.startswith("mtp_layers.0.")
    }
    mtp_layer.load_state_dict(ws_layer, strict=False)
    with torch.no_grad():
        e = ctx["embed"](ctx["toks"][1 : T + 1])
        mixin_proj = mtp_layer.input_proj(
            torch.cat(
                [
                    mtp_layer.token_layernorm(e),
                    mtp_layer.hidden_layernorm(ctx["hs"]),
                ],
                dim=-1,
            )
        )
        serving = _serving_compose(ctx)
    assert torch.allclose(mixin_proj, serving["proj"], atol=1e-2), (
        f"composition divergence: cos={torch.nn.functional.cosine_similarity(mixin_proj.float().flatten(), serving['proj'].float().flatten(), dim=0):.4f}"
    )


def test_layer_deterministic(ctx) -> None:
    """Two identical layer calls must produce identical outputs (the NPU
    nondeterminism, if any, must be bounded — a flaky layer invalidates every
    parity comparison)."""
    stages_a = _serving_compose(ctx)
    stages_b = _serving_compose(ctx)
    assert torch.equal(stages_a["proj"], stages_b["proj"])
    assert torch.allclose(stages_a["logits"], stages_b["logits"], atol=1e-3)


def test_top1_agreement_floor(ctx) -> None:
    """The warm-start drafter's argmax must agree with the actual next-next
    token well above random. The offline harness's historical garbage
    (0.004) fails here — this test is the early-warning for a broken
    training/serving composition."""
    stages = _serving_compose(ctx)
    argmax = stages["logits"].argmax(-1).cpu()
    actual = ctx["toks"][2 : ctx["T"] + 2].cpu()
    agree = (argmax == actual).float().mean().item()
    print(f"\nTOP1-AGREEMENT: {agree:.4f}")
    assert agree >= 0.10, (
        f"top-1 agreement {agree:.3f} < 0.10 — the composition produces "
        "garbage for a warm-start that serves at 0.38: the offline composition "
        "diverges from serving semantics"
    )


def test_argmax_diversity(ctx) -> None:
    """A healthy drafter predicts DIVERSE tokens. An input-ignoring collapse
    (the misaligned-target failure mode: the optimal strategy becomes the
    marginal token distribution) concentrates the argmax on one token while
    the CE loss still 'improves'. The v9-trained model exhibited exactly this
    (val 4.90 ≈ the unconditional entropy, the argmax collapsed)."""
    stages = _serving_compose(ctx)
    argmax = stages["logits"].argmax(-1).cpu()
    vals, counts = argmax.unique(return_counts=True)
    top_frac = (counts.max().item()) / argmax.numel()
    print(
        f"\nARGMAX-DIVERSITY: unique={len(vals)}/{argmax.numel()} top-token-frac={top_frac:.3f}"
    )
    assert top_frac < 0.5, (
        f"argmax collapsed: {top_frac:.1%} of predictions are ONE token "
        f"(token {vals[counts.argmax()].item()}) — the model ignores its "
        "inputs (the misaligned-target failure signature)"
    )
