"""Canonical W4A8 (ModelSlim) quantization layout primitives.

The bridge between BF16 drafter training and W4A8 serving. These conventions
were validated against the real GLM-5.3-W4A8 checkpoint (round-trip cos
0.99998 measured) and are the single source of truth for both the extraction
(examples/glm53_mtp/extract_mtp_layer.py) and the write-back
(requant_layer78.py).

Layout contract (validated against the vendor checkpoint):
- int8 per-channel: stored [out, in] int8, scale [out, 1] float32,
  ``W = q * scale``; requant scale = amax(out_ch)/127.
- int4 double-packed along dim0 (low nibble = EVEN row): stored [out/2, in]
  int8 with scale [out, 1]; requant scale = amax(row)/8, codes clamp [-8, 7].
- weight_offset exists but measured all-zero; scale_bias is a fused-kernel
  side table (NOT part of the weight reconstruction — any additive combination
  with the ±20-magnitude bias destroys the ±0.08-scale weights).
- FLOAT tensors (eh_proj, norms, router bias) are copied, never requantized.
- Key remap: eh_proj->input_proj, enorm->token_layernorm,
  hnorm->hidden_layernorm, shared_head.norm->final_layernorm;
  embed_tokens/shared_head.head dropped (the drafter shares the verifier's).
"""

import torch

# native (verifier layer-78) suffix -> drafter (speculators) suffix
KEY_REMAP_NATIVE_TO_DRAFTER = {
    "eh_proj.weight": "input_proj.weight",
    "enorm.weight": "token_layernorm.weight",
    "hnorm.weight": "hidden_layernorm.weight",
    "shared_head.norm.weight": "final_layernorm.weight",
}
KEY_REMAP_DRAFTER_TO_NATIVE = {v: k for k, v in KEY_REMAP_NATIVE_TO_DRAFTER.items()}
# dropped on extraction: the drafter shares the verifier's embed/lm_head
DROPPED_NATIVE_KEYS = {"embed_tokens.weight", "shared_head.head.weight"}

INT4_MIN, INT4_MAX = -8, 7


INT4_SIGN_THRESHOLD = 7  # nibbles above this are negative (two's complement)
INT4_PACK_RATIO = 2  # two int4 values per byte along dim0


def to_signed_int4(nibbles: torch.Tensor) -> torch.Tensor:
    """Nibbles in [0, 15] -> signed int4 in [-8, 7].

    The arithmetic MUST run in a signed dtype: uint8 inputs wrap (13-16=253)
    and silently produce all-positive garbage — the exact bug that shifted
    every warm-start expert weight by +8*scale (the unsigned interpretation).
    """
    signed = nibbles.to(torch.int16)
    return torch.where(signed > INT4_SIGN_THRESHOLD, signed - 16, signed).to(
        torch.float32
    )


def unpack_int4_dim0(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int4 double-packed along dim0 (LOW nibble = even row).

    [n, c] int8 -> [2n, c] float32.
    """
    packed16 = packed.to(torch.int16)
    low = (packed16 & 0x0F).to(torch.uint8)
    high = ((packed16 >> 4) & 0x0F).to(torch.uint8)
    unpacked = torch.stack([to_signed_int4(low), to_signed_int4(high)], dim=1)
    return unpacked.reshape(packed.shape[0] * 2, packed.shape[1])


def pack_int4_dim0(values: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values [2n, c] along dim0, LOW nibble = even row.

    Inverse of :func:`unpack_int4_dim0` for values in [-8, 7].
    """
    q = values.to(torch.int16).clamp(INT4_MIN, INT4_MAX)
    low = (q[0::2] & 0x0F).to(torch.uint8)
    high = ((q[1::2] & 0x0F) & 0x0F).to(torch.uint8) << 4
    return (low | high).to(torch.int8)


def dequant_tensor(
    weight: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor | None = None
) -> torch.Tensor:
    """Dequantize a stored W4A8 weight tensor to float32.

    Dispatch: weight.shape[0]*2 == scale.shape[0] -> int4 packed;
    weight.shape[0] == scale.shape[0] -> int8 per-channel; else error.
    The offset is added only when it is non-zero (the vendor convention).
    """
    if weight.shape[0] * INT4_PACK_RATIO == scale.shape[0]:
        w = unpack_int4_dim0(weight) * scale.to(torch.float32)
    elif weight.shape[0] == scale.shape[0]:
        w = weight.to(torch.float32) * scale.to(torch.float32)
    else:
        raise ValueError(
            f"Cannot match weight {tuple(weight.shape)} to scale {tuple(scale.shape)}"
        )
    if offset is not None and offset.abs().max() > 0:
        w = w + offset.to(torch.float32)
    return w


def requant_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Requantize [out, in] float32 to int8 per-channel: amax/127, offset 0."""
    amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
    offset = torch.zeros_like(scale)
    return q, scale, offset


def requant_int4_packed(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Requantize [out, in] float32 to int4 packed along dim0: amax/8.

    Returns (packed [out/2, in] int8, scale [out, 1], offset, scale_bias)
    with offset and scale_bias zeroed (the vendor layout contract).
    """
    amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / 8.0
    q = torch.clamp(torch.round(w / scale), INT4_MIN, INT4_MAX).to(torch.int16)
    packed = pack_int4_dim0(q.to(torch.float32))
    offset = torch.zeros_like(scale)
    bias = torch.zeros_like(scale)
    return packed, scale, offset, bias


def remap_key_native_to_drafter(key: str) -> str | None:
    """Native layer-78 key -> drafter key; None = drop (embed/head)."""
    if key in DROPPED_NATIVE_KEYS:
        return None
    return KEY_REMAP_NATIVE_TO_DRAFTER.get(key, key)


def requant_like_stored(w: torch.Tensor, stored_shape: tuple[int, int]):
    """Requantize to match a stored tensor's layout (the write-back contract).

    Chooses int8 vs int4-packed by the stored shape; returns the tensors in
    the requant_layer78.py convention (offset=0, scale_bias=0 for int4).
    """
    if w.dim() != 2:  # noqa: PLR2004 -- 2 = matrix dimensionality
        raise ValueError(
            f"requant expects a 2-D [out, in] tensor, got {tuple(w.shape)}"
        )
    if stored_shape[0] == w.shape[0] and stored_shape[1] == w.shape[1]:
        return requant_int8(w)
    if (
        stored_shape[0] * INT4_PACK_RATIO == w.shape[0]
        and stored_shape[1] == w.shape[1]
    ):
        return requant_int4_packed(w)
    raise ValueError(
        f"stored {stored_shape} incompatible with logical {tuple(w.shape)}"
    )
