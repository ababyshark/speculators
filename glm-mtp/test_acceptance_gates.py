"""Acceptance-eval gate tests for the GLM-5.3 MTP pipeline.

Encodes the acceptance-measurement lessons as runnable checks:

- the eval JSON must be WELL-FORMED (ok == num_prompts: the 404 bug once made
  a whole eval report drafts=0 while every request had failed);
- per-position rates must lie in [0, 1];
- the recorded baselines must hold (regression floors);
- the SERVE health (endpoint up) before any eval is trusted;
- an on-demand eval run (env-gated) with the acceptance floors.

Run: GLM53_EVAL_DIR=... pytest glm-mtp/test_acceptance_gates.py -v
On-demand eval: GLM53_EVAL_MODEL=<dir> GLM53_EVAL_LABEL=<label> pytest ... -k run_eval
"""

import json
import os
import urllib.request
from pathlib import Path

import pytest

EVAL_DIR = Path(
    os.environ.get("GLM53_EVAL_DIR", "/mnt/share_space/l00840580/dspark-glm53/eval_results")
)
SERVE_URL = os.environ.get("GLM53_SERVE_URL", "http://127.0.0.1:8100")

# recorded baselines (from the project's measurement history)
FLOORS = {
    "eval_ctrl_warmstart.json": 0.1,  # the untrained write-back measured 0.264
}
EVAL_FLOOR = 0.02  # an eval below this = effectively broken (v7 measured 0.003)


def _load(name: str) -> dict:
    p = EVAL_DIR / name
    if not p.exists():
        pytest.skip(f"{name} not present")
    return json.loads(p.read_text())


def test_serve_endpoint_alive() -> None:
    """The eval numbers are only trustworthy when the serve is actually up
    (the 404-incident produced a plausible-looking all-zero report)."""
    try:
        with urllib.request.urlopen(f"{SERVE_URL}/v1/models", timeout=10) as r:
            body = r.read(200)
    except OSError:
        pytest.skip("serving endpoint not running")
    assert b"object" in body, f"unexpected endpoint body: {body[:80]}"


@pytest.mark.parametrize("name", sorted(FLOORS))
def test_eval_json_wellformed(name: str) -> None:
    data = _load(name)
    assert data["ok"] == data["num_prompts"], (
        f"{name}: {data['num_prompts'] - data['ok']} requests FAILED — the "
        "acceptance numbers are meaningless (the 404-bug class)"
    )
    rates = data["acceptance_rates"]
    assert isinstance(rates, list), f"{name}: rates must be a list"
    for pos, rate in enumerate(rates):
        assert 0.0 <= rate <= 1.0, f"{name}: pos{pos} rate out of range: {rate}"
    assert 1.0 <= data["mean_accepted_length"] <= 6.0


@pytest.mark.parametrize("name,floor", sorted(FLOORS.items()))
def test_acceptance_floor(name: str, floor: float) -> None:
    data = _load(name)
    pos0 = data["acceptance_rates"][0]
    assert pos0 >= floor, f"{name}: pos0={pos0:.4f} below the floor {floor}"


def test_trained_acceptance_regression() -> None:
    """The trained-model eval must beat the broken-training floor. The v7
    (corrupted routing weights) measured 0.0030; any new training run below
    EVAL_FLOOR means the training-serving gap is back."""
    data = _load("eval_v9_signed.json")
    pos0 = data["acceptance_rates"][0]
    assert pos0 >= EVAL_FLOOR, f"trained pos0={pos0:.4f} below {EVAL_FLOOR}"


@pytest.mark.skipif(
    not os.environ.get("GLM53_EVAL_MODEL"), reason="GLM53_EVAL_MODEL not set"
)
def test_run_eval_on_demand() -> None:
    """Run a live acceptance eval against the running serve and assert the
    floor. GLM53_EVAL_MODEL=<served model dir>, GLM53_EVAL_LABEL=<label>."""
    label = os.environ.get("GLM53_EVAL_LABEL", "gate")
    model = os.environ["GLM53_EVAL_MODEL"]
    out = EVAL_DIR / f"eval_{label}.json"
    import subprocess

    script = Path(
        os.environ.get(
            "GLM53_MEASURE",
            "/mnt/share_space/l00840580/dspark-glm53/measure_acceptance.py",
        )
    )
    r = subprocess.run(
        [
            os.environ.get("GLM53_PYTHON", "/mnt/share_space/l00840580/dspark-glm53/venv/bin/python"),
            str(script),
            "--label",
            label,
            "--model",
            model,
            "--save-json",
            str(out),
            "--concurrency",
            "32",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert r.returncode == 0, f"eval failed: {r.stdout[-500:]} {r.stderr[-500:]}"
    data = json.loads(out.read_text())
    assert data["ok"] == data["num_prompts"]
    assert data["acceptance_rates"]["0"] >= EVAL_FLOOR
