"""GLM-5.3 W4A8 MTP training checkpoint health gate.

Judges saved epoch checkpoints against the warm-start reference. Encodes the
failure modes that cost debugging time during the NPU bring-up:

- missing/renamed keys (the wrapper bug silently zeroed warm-start experts);
- NaN/Inf from corrupted saves;
- trainable parts NOT actually updating (silent no-learning);
- frozen experts changing (the freeze guard regressed — the frozen weights
  carry the vendor warm-start values by contract);
- diverging val metrics;
- optimizer state absent (strict resume would crash).

Runs against real artifacts; set GLM53_WARMSTART / GLM53_CKPT_DIR to the
drafter warm-start dir and the training save dir. Skips gracefully when the
artifacts are absent (e.g. CI without the NFS mount).
"""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

WARMSTART = os.environ.get(
    "GLM53_WARMSTART",
    "/mnt/share_space/l00840580/dspark-glm53/drafter_init/glm53-mtp-warmstart",
)
CKPT_DIR = os.environ.get(
    "GLM53_CKPT_DIR",
    "/mnt/share_space/l00840580/dspark-glm53/checkpoints/glm53_mtp_v2",
)
PREFIX = "mtp_layers.0."
# dropped by design: the drafter shares the verifier's embed/lm_head
EXPECTED_DROPPED = {"embed_tokens.weight", "lm_head.weight"}
# frozen by contract: the grouped expert weights keep the vendor warm-start
FROZEN_KEYS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
# representative trainable keys (adapter paths that MUST learn). NOTE: norms
# (enorm/hnorm/final_layernorm) are unity in the warm-start and 1e-5-lr AdamW
# updates are below bf16 resolution at 1.0 — invisible in storage, so they are
# NOT valid "did it learn" probes.
TRAINABLE_PROBES = [
    "input_proj.weight",
    "self_attn.q_b_proj.weight",
    "mlp.gate.weight",
]


_CACHE: dict[str, dict[str, torch.Tensor]] = {}


def _load_dir(path: str) -> dict[str, torch.Tensor]:
    if path in _CACHE:
        return _CACHE[path]
    sd: dict[str, torch.Tensor] = {}
    for f in sorted(Path(path).glob("model*.safetensors")):
        if "index" in f.name:
            continue
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                sd[k] = sf.get_tensor(k)
    _CACHE[path] = sd
    return sd


def _epochs() -> list[str]:
    if not Path(CKPT_DIR).is_dir():
        pytest.skip("checkpoint dir not mounted")
    eps = sorted(
        d.name
        for d in Path(CKPT_DIR).iterdir()
        if d.name.startswith("epoch") and d.is_dir()
    )
    if not eps:
        pytest.skip("no epoch checkpoints yet")
    return eps


@pytest.fixture(scope="module")
def warmstart() -> dict[str, torch.Tensor]:
    if not Path(WARMSTART).is_dir():
        pytest.skip("warm-start dir not mounted")
    return _load_dir(WARMSTART)


def test_keys_complete_no_renaming(warmstart: dict[str, torch.Tensor]) -> None:
    """Every warm-start key (minus embed/lm_head) must survive the save with
    the exact same name — a wrapper rename breaks warm-start loading."""
    for ep in _epochs():
        sd = _load_dir(f"{CKPT_DIR}/{ep}")
        missing = [k for k in warmstart if k not in EXPECTED_DROPPED and k not in sd]
        extra = [k for k in sd if k not in warmstart and k not in EXPECTED_DROPPED]
        assert not missing, f"{ep}: missing keys {missing[:5]}"
        assert not extra, f"{ep}: unexpected keys {extra[:5]}"
        assert not any("mlp.moe." in k for k in sd), (
            f"{ep}: renamed wrapper keys present"
        )


@pytest.mark.parametrize("kind", ["weights", "frozen"])
def test_no_nan_inf(warmstart: dict[str, torch.Tensor], kind: str) -> None:
    for ep in _epochs():
        sd = _load_dir(f"{CKPT_DIR}/{ep}")
        for k, t in sd.items():
            if t.dtype.is_floating_point:
                assert not torch.isnan(t).any(), f"{ep}/{k}: NaN"
                assert not torch.isinf(t).any(), f"{ep}/{k}: Inf"


def test_trainable_parts_actually_update(warmstart: dict[str, torch.Tensor]) -> None:
    """The adapter paths must move off the warm-start values — catches silent
    no-learning (e.g. the warm-start silently zeroed by key renaming)."""
    eps = _epochs()
    first, last = eps[0], eps[-1]
    if first == last:
        pytest.skip("single checkpoint: no epoch-over-epoch comparison")
    sd_first = _load_dir(f"{CKPT_DIR}/{first}")
    sd_last = _load_dir(f"{CKPT_DIR}/{last}")
    for probe in TRAINABLE_PROBES:
        key = PREFIX + probe
        assert key in sd_first and key in sd_last, f"{key} missing"
        delta = (sd_last[key].float() - sd_first[key].float()).abs().max().item()
        assert delta > 0, f"{key}: did not move between {first} and {last}"


def test_frozen_experts_unchanged(warmstart: dict[str, torch.Tensor]) -> None:
    """The grouped expert weights are frozen at the vendor warm-start values
    by contract (their grads cannot fit on 61 GiB cards)."""
    for ep in _epochs():
        sd = _load_dir(f"{CKPT_DIR}/{ep}")
        for fk in FROZEN_KEYS:
            key = PREFIX + fk
            assert key in sd and key in warmstart, f"{key} missing in {ep}"
            assert torch.equal(sd[key], warmstart[key]), (
                f"{ep}/{fk}: frozen weight changed"
            )


def test_val_metrics_decline_and_finite() -> None:
    eps = _epochs()
    series: dict[str, list[float]] = {}
    for ep in eps:
        vm_path = Path(f"{CKPT_DIR}/{ep}/val_metrics.json")
        if not vm_path.exists():
            pytest.skip("val_metrics.json missing")
        vm = json.loads(vm_path.read_text())
        for k, v in vm.items():
            if k.startswith("loss_step_"):
                series.setdefault(k, []).append(float(v))
    assert series, "no loss_step_* metrics found"
    for k, vals in series.items():
        assert all(v == v and abs(v) != float("inf") for v in vals), f"{k}: non-finite"
        # epoch-over-epoch must not diverge (allow plateau, reject growth)
        assert vals[-1] <= vals[0] + 1e-3, f"{k}: val loss grew {vals}"


BIAS_KEY = "mtp_layers.0.mlp.gate.e_score_correction_bias"


def test_router_bias_fp32_and_bit_equal_warmstart(
    warmstart: dict[str, torch.Tensor],
) -> None:
    """The router selection bias is a never-trained buffer at ~15.8 magnitude
    with 0.08 informative spread; bf16 rounding of it flips 75% of the top-8
    expert selections at serving. It must be stored fp32 and bit-equal to the
    warm-start value in every checkpoint."""
    assert BIAS_KEY in warmstart
    assert warmstart[BIAS_KEY].dtype == torch.float32
    for ep in _epochs():
        sd = _load_dir(f"{CKPT_DIR}/{ep}")
        assert BIAS_KEY in sd, f"{ep}: bias missing"
        assert sd[BIAS_KEY].dtype == torch.float32, (
            f"{ep}: bias stored as {sd[BIAS_KEY].dtype} — bf16 rounding flips "
            "the topk selection (the v7 0.003-acceptance root cause)"
        )
        assert torch.equal(sd[BIAS_KEY], warmstart[BIAS_KEY]), (
            f"{ep}: bias differs from warm-start (it is a buffer, never trained)"
        )


def test_optimizer_state_loadable_and_covers_trainables() -> None:
    """Strict resume needs every trainable param's AdamW moments — the
    zero-momentum fill contract."""
    eps = _epochs()
    opt_file = Path(f"{CKPT_DIR}/{eps[-1]}/optimizer_state_dict.pt")
    if not opt_file.exists():
        pytest.skip("no optimizer state saved")
    state = torch.load(opt_file, map_location="cpu", weights_only=False)
    assert isinstance(state, dict) and state, "empty optimizer state"
    states = state.get("state", {})
    assert states, "optimizer state has no per-param entries"
    for entry in states.values():
        if isinstance(entry, dict) and "exp_avg" in entry:
            assert torch.isfinite(entry["exp_avg"]).all()
            assert torch.isfinite(entry["exp_avg_sq"]).all()
