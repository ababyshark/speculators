"""Extract GLM-5.3's native MTP layer (model.layers.78) from a ModelSlim W4A8
checkpoint into a speculators MTP drafter warm-start checkpoint (BF16).

Dequant rules (validated empirically on 2026-11-02, see dspark-glm53/README.md):
- No weight_scale sibling  -> FLOAT tensor, copy as-is.
- weight int8, weight.shape[0] == scale.shape[0]
    -> per-channel int8: W = q * scale (+ offset; observed all-zero).
- weight int8, weight.shape[0] * 2 == scale.shape[0]
    -> int4 packed 2-per-byte along dim0 (low nibble = even row): W = unpack(q) * scale.
- scale_bias is NOT applied (fused-kernel-only term; a per-channel additive of
  its magnitude is functionally impossible in the residual stream).

Usage:
    python extract_mtp_layer.py --model /mnt/weight2/l00840580-glm53-w4a8-biasfix \
        --output /mnt/share_space/l00840580/dspark-glm53/drafter_init/glm53-mtp-warmstart
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.mtp import MTPSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

SHARD_SIZE = 4 * 1024**3
QUANT_SUFFIXES = (".weight_scale", ".weight_offset", ".scale_bias")


class ShardReader:
    def __init__(self, base: Path, weight_map: dict[str, str]):
        self.base = base
        self.weight_map = weight_map
        self._open_path: str | None = None
        self._f: safe_open | None = None

    def get(self, name: str) -> torch.Tensor:
        path = self.base / self.weight_map[name]
        if self._open_path != str(path):
            if self._f is not None:
                self._f.__exit__(None, None, None)
            self._f = safe_open(path, framework="pt")
            self._open_path = str(path)
        return self._f.get_tensor(name)

    def close(self):
        if self._f is not None:
            self._f.__exit__(None, None, None)
            self._f = None
            self._open_path = None


def to_signed_int4(nibbles: torch.Tensor) -> torch.Tensor:
    # int16 arithmetic: uint8 inputs wrap (13-16=253) and silently produce
    # all-positive garbage expert weights (the unsigned-interpretation bug
    # that corrupted the warm-start and collapsed serving acceptance).
    signed = nibbles.to(torch.int16)
    return torch.where(signed > 7, signed - 16, signed).to(torch.float32)


def dequant(
    weight: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor | None
) -> torch.Tensor:
    if weight.shape[0] * 2 == scale.shape[0]:
        packed = weight.to(torch.int16)
        low = (packed & 0x0F).to(torch.uint8)
        high = ((packed >> 4) & 0x0F).to(torch.uint8)
        unpacked = torch.stack([to_signed_int4(low), to_signed_int4(high)], dim=1)
        unpacked = unpacked.reshape(scale.shape[0], weight.shape[1])
        w = unpacked * scale.to(torch.float32)
    elif weight.shape[0] == scale.shape[0]:
        w = weight.to(torch.float32) * scale.to(torch.float32)
    else:
        raise ValueError(
            f"Cannot match weight {tuple(weight.shape)} to scale {tuple(scale.shape)}"
        )
    if offset is not None and offset.abs().max() > 0:
        w = w + offset.to(torch.float32)
    return w


def remap_key(k: str, prefix: str) -> str | None:
    suffix = k[len(prefix) :]
    if suffix == "eh_proj.weight":
        return "mtp_layers.0.input_proj.weight"
    if suffix == "enorm.weight":
        return "mtp_layers.0.token_layernorm.weight"
    if suffix == "hnorm.weight":
        return "mtp_layers.0.hidden_layernorm.weight"
    if suffix == "shared_head.norm.weight":
        return "mtp_layers.0.final_layernorm.weight"
    if suffix in ("embed_tokens.weight", "shared_head.head.weight"):
        return None  # drafter uses the verifier's main embed / lm_head
    return f"mtp_layers.0.{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-speculative-tokens", type=int, default=3)
    args = ap.parse_args()

    base = Path(args.model)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    vcfg = AutoConfig.from_pretrained(base, trust_remote_code=True)
    prefix = f"model.layers.{vcfg.num_hidden_layers}."

    index = json.load(open(base / "quant_model_weights.safetensors.index.json"))
    weight_map = index["weight_map"]
    reader = ShardReader(base, weight_map)

    state: dict[str, torch.Tensor] = {}
    weight_out: dict[str, str] = {}
    stats = {"float": 0, "int8": 0, "int4": 0, "skipped": 0}
    shard_idx = 0
    shard_bytes = 0

    def flush():
        nonlocal shard_idx, shard_bytes
        if not state:
            return
        name = f"model-PLACEHOLDER-{shard_idx:05d}.safetensors"
        save_file(state, str(out / name), metadata={"format": "pt"})
        for k in state:
            weight_out[k] = name
        print(
            f"  shard {shard_idx:05d}: {len(state)} tensors, {shard_bytes / 1e9:.2f} GB"
        )
        state.clear()
        shard_bytes = 0
        shard_idx += 1

    def emit(name: str, tensor: torch.Tensor):
        nonlocal shard_bytes
        state[name] = tensor.contiguous()
        shard_bytes += tensor.numel() * tensor.element_size()
        if shard_bytes >= SHARD_SIZE:
            flush()

    mtp_keys = sorted(k for k in weight_map if k.startswith(prefix))
    print(f"MTP layer ({prefix}*) tensors: {len(mtp_keys)}")
    state_all: dict[str, torch.Tensor] = {}
    for k in mtp_keys:
        if k.endswith(QUANT_SUFFIXES):
            continue  # consumed by their base tensor
        new_name = remap_key(k, prefix)
        if new_name is None:
            stats["skipped"] += 1
            continue
        t = reader.get(k)
        if k.endswith(".weight"):
            scale_name = k[: -len(".weight")] + ".weight_scale"
            offset_name = k[: -len(".weight")] + ".weight_offset"
        else:
            scale_name, offset_name = k + "_scale", k + "_offset"
        if scale_name in weight_map:
            scale = reader.get(scale_name)
            offset = reader.get(offset_name) if offset_name in weight_map else None
            kind = "int4" if t.shape[0] * 2 == scale.shape[0] else "int8"
            t = dequant(t, scale, offset).to(torch.bfloat16)
            stats[kind] += 1
        else:
            stats["float"] += 1
            if not k.endswith("e_score_correction_bias") and t.dtype in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                t = t.to(torch.bfloat16)
        state_all[new_name] = t
    reader.close()

    # Post-process 1: fuse per-expert HF-linear weights into the grouped
    # parameters used by transformers' GlmMoeDsaExperts:
    #   experts.{e}.gate_proj / up_proj -> experts.gate_up_proj  [E, 2*I, H]
    #   experts.{e}.down_proj           -> experts.down_proj     [E, H, I]
    n_experts = vcfg.n_routed_experts
    inter = vcfg.moe_intermediate_size
    hidden = vcfg.hidden_size
    has_grouped = any(
        f"mtp_layers.0.mlp.experts.{e}.gate_proj.weight" in state_all for e in range(1)
    )
    if has_grouped:
        gate_up = torch.empty(n_experts, 2 * inter, hidden, dtype=torch.bfloat16)
        down = torch.empty(n_experts, hidden, inter, dtype=torch.bfloat16)
        for e in range(n_experts):
            ebase = f"mtp_layers.0.mlp.experts.{e}."
            gate = state_all.pop(ebase + "gate_proj.weight")
            up = state_all.pop(ebase + "up_proj.weight")
            down[e] = state_all.pop(ebase + "down_proj.weight")
            gate_up[e] = torch.cat([gate, up], dim=0)
        state_all["mtp_layers.0.mlp.experts.gate_up_proj"] = gate_up
        state_all["mtp_layers.0.mlp.experts.down_proj"] = down
        print(f"fused {n_experts} experts into grouped gate_up_proj/down_proj")

    # Post-process 2: drop norm *.bias tensors (vendor bias-fix artifacts;
    # RMSNorm in the HF implementation has no bias).
    biases = [k for k in state_all if k.endswith(".bias") and "k_norm" not in k]
    for k in biases:
        v = state_all.pop(k)
        print(f"dropped {k} (absmax {v.abs().max().item():.3e})")

    for name, t in state_all.items():
        emit(name, t)
    flush()

    for src, dst in (
        ("model.embed_tokens.weight", "embed_tokens.weight"),
        ("lm_head.weight", "lm_head.weight"),
    ):
        t = reader.get(src).to(torch.bfloat16)
        emit(dst, t)
    flush()
    reader.close()

    total = shard_idx
    renamed: dict[str, str] = {}
    for i in range(total):
        old = out / f"model-PLACEHOLDER-{i:05d}.safetensors"
        new = out / f"model-{i:05d}-of-{total:05d}.safetensors"
        old.rename(new)
        renamed[old.name] = new.name
    weight_out = {k: renamed[v] for k, v in weight_out.items()}

    index_out = {
        "metadata": {
            "total_size": sum((out / f).stat().st_size for f in weight_out.values())
        },
        "weight_map": weight_out,
    }
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump(index_out, f, indent=2)

    spec_cfg = MTPSpeculatorConfig(
        transformer_layer_config=vcfg,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[
                GreedyTokenProposalConfig(
                    speculative_tokens=args.num_speculative_tokens
                )
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig.from_pretrained(str(base)),
        ),
    )
    spec_cfg.save_pretrained(str(out))

    print("stats:", stats)
    print(f"drafter warm-start written to {out}")


if __name__ == "__main__":
    main()
