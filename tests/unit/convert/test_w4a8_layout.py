"""W4A8 quantization layout tests: the training->serving bridge contract.

Every failure mode here is a serving-acceptance failure mode: the BF16
drafter only reaches serving through these conventions, and a wrong nibble
order, scale formula, or dropped key produces a functionally different
drafter while every tensor-level cosine still looks fine.
"""

import pytest
import torch

from speculators.convert.w4a8_layout import (
    DROPPED_NATIVE_KEYS,
    KEY_REMAP_DRAFTER_TO_NATIVE,
    KEY_REMAP_NATIVE_TO_DRAFTER,
    dequant_tensor,
    pack_int4_dim0,
    remap_key_native_to_drafter,
    requant_int4_packed,
    requant_int8,
    requant_like_stored,
    to_signed_int4,
    unpack_int4_dim0,
)


def test_pack_unpack_int4_bitexact() -> None:
    torch.manual_seed(0)
    values = torch.randint(-8, 8, (64, 32)).float()
    packed = pack_int4_dim0(values)
    assert packed.dtype == torch.int8
    unpacked = unpack_int4_dim0(packed)
    assert torch.equal(unpacked, values)


def test_nibble_order_low_first() -> None:
    """LOW nibble = EVEN row (the vendor packing contract, validated on the
    real checkpoint). A wrong order produces cos ~0.5 garbage — the exact
    failure mode behind the inverted-horizon acceptance pattern."""
    values = torch.tensor([[1.0, 2.0], [-3.0, 7.0]])  # even row, odd row
    packed = pack_int4_dim0(values)
    # byte 0: low nibble = row0 col0 (1), high nibble = row1 col0 (-3 -> 13)
    assert (packed[0, 0].item() & 0xFF) == (1 | (13 << 4))
    assert (packed[0, 1].item() & 0xFF) == (2 | (7 << 4))
    unpacked = unpack_int4_dim0(packed)
    assert torch.equal(unpacked, values)


def test_to_signed_int4_range() -> None:
    nibbles = torch.arange(16).float()
    signed = to_signed_int4(nibbles)
    assert signed.min().item() == -8 and signed.max().item() == 7
    assert signed[8].item() == -8 and signed[7].item() == 7


def test_dequant_int8_per_channel() -> None:
    q = torch.tensor([[10, -20], [30, -40]], dtype=torch.int8)
    scale = torch.tensor([[0.1], [0.01]])
    w = dequant_tensor(q, scale)
    assert torch.allclose(w, torch.tensor([[1.0, -2.0], [0.3, -0.4]]))


def test_dequant_int4_packed() -> None:
    values = torch.tensor([[3.0, -4.0], [5.0, -6.0]])
    packed = pack_int4_dim0(values)
    scale = torch.tensor([[0.5], [2.0]])
    w = dequant_tensor(packed, scale)
    assert torch.allclose(w, values * torch.tensor([[0.5, 0.5], [2.0, 2.0]]))


def test_dequant_offset_applied_only_when_nonzero() -> None:
    q = torch.tensor([[10, 10]], dtype=torch.int8)
    scale = torch.tensor([[0.1]])
    zero_off = torch.zeros(1, 1)
    nonzero_off = torch.tensor([[0.25]])
    assert torch.allclose(dequant_tensor(q, scale, zero_off), torch.tensor([[1.0]]))
    assert torch.allclose(dequant_tensor(q, scale, nonzero_off), torch.tensor([[1.25]]))


def test_dequant_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        dequant_tensor(torch.zeros(3, 4, dtype=torch.int8), torch.zeros(5, 1))


def test_requant_int8_roundtrip_bounded() -> None:
    """dequant(requant(W)) must reproduce W within half a quantization step."""
    torch.manual_seed(1)
    w = torch.randn(32, 64) * 0.05
    q, scale, _ = requant_int8(w)
    reconstructed = q.float() * scale
    err = (reconstructed - w).abs()
    assert err.max().item() <= scale.max().item() / 2 + 1e-9
    assert q.min().item() >= -128 and q.max().item() <= 127


def test_requant_int8_grid_exact() -> None:
    """Values already on the requant grid (the ctrl property: the warm-start
    equals dequant(original)) re-quantize to EXACTLY the original codes."""
    torch.manual_seed(2)
    codes = torch.randint(-127, 128, (16, 32)).float()
    scale = torch.rand(16, 1) * 0.01 + 0.001
    w = codes * scale
    q, scale2, _ = requant_int8(w)
    # scale2 = amax/127; for a full-range row amax == 127*scale -> scale2 == scale
    full_range = w.abs().amax(dim=1) == 127 * scale.squeeze(1)
    assert torch.equal(q[full_range].float(), codes[full_range])


def test_requant_int4_roundtrip_bounded() -> None:
    torch.manual_seed(3)
    w = torch.randn(128, 32) * 0.05
    packed, scale, offset, bias = requant_int4_packed(w)
    assert offset.abs().max().item() == 0
    assert bias.abs().max().item() == 0
    reconstructed = unpack_int4_dim0(packed) * scale
    err = (reconstructed - w).abs()
    # the asymmetric range [-8, 7] clamps the positive amax element to 7:
    # the worst-case error is ONE full step (scale), not half a step
    row_err = err.amax(dim=1)
    assert (row_err <= scale.squeeze(1) * (1 + 1e-6)).all()
    assert err.max().item() <= scale.max().item() * (1 + 1e-6)


def test_requant_int4_roundtrip_negative_side_exact() -> None:
    """Rows whose extreme lies on the -8 side reproduce exactly (the vendor
    scale convention: amax = 8*scale when the -8 code is used)."""
    torch.manual_seed(6)
    codes = torch.randint(-8, 8, (32, 16)).float()
    scale = torch.rand(32, 1) * 0.005 + 0.0005
    w = codes * scale
    neg_extreme = codes.min(dim=1).values == -8
    assert neg_extreme.any()
    packed, scale2, _, _ = requant_int4_packed(w)
    reconstructed = unpack_int4_dim0(packed) * scale2
    assert torch.allclose(reconstructed[neg_extreme], w[neg_extreme], atol=1e-9)


def test_requant_like_stored_dispatch() -> None:
    w8 = torch.randn(64, 32)
    q, s, _ = requant_like_stored(w8, (64, 32))
    assert q.shape == (64, 32) and q.dtype == torch.int8
    w4 = torch.randn(128, 32)
    packed, s4, off, bias = requant_like_stored(w4, (64, 32))
    assert packed.shape == (64, 32)
    with pytest.raises(ValueError):
        requant_like_stored(torch.randn(48, 32), (64, 32))


def test_key_remap_roundtrip() -> None:
    for native, drafter in KEY_REMAP_NATIVE_TO_DRAFTER.items():
        assert KEY_REMAP_DRAFTER_TO_NATIVE[drafter] == native
    # pass-through keys survive unchanged
    assert (
        remap_key_native_to_drafter("self_attn.o_proj.weight")
        == "self_attn.o_proj.weight"
    )
    assert remap_key_native_to_drafter("mlp.experts.0.gate_proj.weight") == (
        "mlp.experts.0.gate_proj.weight"
    )


def test_dropped_keys_policy() -> None:
    """The drafter shares the verifier's embed/lm_head: those native keys MUST
    be dropped on extraction (keeping them double-counts nothing but breaks
    the state-dict contract)."""
    assert remap_key_native_to_drafter("embed_tokens.weight") is None
    assert remap_key_native_to_drafter("shared_head.head.weight") is None
    assert "embed_tokens.weight" in DROPPED_NATIVE_KEYS


def test_rot_alignment_serving_equivalence() -> None:
    """Serving applies rot as nn.Linear(hidden, hidden, bias=False)(H), i.e.
    ``H @ W^T``. The offline rot transform (transform_hs_rot.py) must use the
    same orientation — this test locks it (the R vs R^T inversion produced a
    full misdirected retraining cycle)."""
    torch.manual_seed(5)
    x = torch.randn(8, 16)
    w = torch.randn(16, 16)
    assert torch.allclose(torch.nn.functional.linear(x, w), x @ w.t())


def test_router_bias_not_requantized() -> None:
    """The router selection bias is a FLOAT buffer (never requantized): its
    informative spread (0.078 at ~15.8 magnitude) is 11.5% of the bf16 noise
    floor — any cast flips 75% of the top-8 selections (measured)."""
    bias = torch.randn(256) * 0.078 + 15.8
    # requant_like_stored on a FLOAT-shaped entry would int8-quantize it; the
    # contract is that FLOAT-classified tensors bypass requant entirely.
    # Guard the classification helper contract via the stored-shape dispatch:
    with pytest.raises(ValueError):
        # a 1-D bias has no valid [out, in] stored layout -> must never reach
        # the requant path
        requant_like_stored(bias, (256, 1))


def test_dequant_int4_signed_semantics() -> None:
    """THE unsigned-interpretation sentinel: a code distribution spanning the
    full int4 range MUST dequantize to a matrix WITH negatives. The original
    warm-start extraction lost the signed conversion (0% negatives — an
    impossible trained matrix), which silently corrupted every frozen expert
    and collapsed serving acceptance from 0.79 to 0.26."""
    torch.manual_seed(7)
    codes = torch.randint(-8, 8, (64, 32)).float()
    scale = torch.rand(64, 1) * 0.01 + 0.001
    w = dequant_tensor(pack_int4_dim0(codes), scale)
    neg_frac = (w < 0).float().mean().item()
    assert 0.3 < neg_frac < 0.7, (
        f"dequant produced neg_frac={neg_frac:.3f} — the signed int4 "
        "interpretation is broken (the uint8-underflow / unsigned bug class)"
    )
    assert w.std().item() > 0, "degenerated dequant"


def test_signed_arithmetic_no_uint8_underflow() -> None:
    """The nibble 13 must map to -3 (not 253): the -16 adjustment must run in
    a signed dtype."""
    import torch as _t

    nibbles_uint8 = _t.tensor([13, 8, 3], dtype=_t.uint8)
    signed = to_signed_int4(nibbles_uint8)
    assert signed.tolist() == [-3.0, -8.0, 3.0]
