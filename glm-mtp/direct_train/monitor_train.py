# ruff: noqa
#!/usr/bin/env python3
"""Training-run monitor & post-mortem for the GLM-5.3 MTP distill pipeline.

Encodes every failure mode hit during the v13/v14/v15 sessions:

  F1 (v15 bug)  steps-resolution: --num-speculative-steps K requested but the
                run trained single-step (cli.py used to overwrite the flag
                with draft-config speculative_tokens=1). Detected via missing
                align_step_1..K-1 in the train log.
  F2 (v15)      data-distribution mismatch: align_step_0 at training start far
                below the reference (v15: 0.536 on Qwen-regen 10x data vs
                0.737 on 1x data) -> drafter does not agree with the verifier
                on this data; training will drag it off the serving dist.
  F3 (v13/v15)  data-fit drift: token_accuracy_step_0 (CE on data tokens)
                climbs well above align_step_0 -> drafter is fitting data
                tokens the verifier would not emit (86.7% lesson).
  F4            step_ms growth: fetch/generation bottleneck (HS endpoint
                on_missing=generate steals throughput mid-run).
  F5            Tracebacks in the log (count + location).
  F6            no checkpoint materialized despite checkpoint_freq steps.
  F7            HCCL/OOM signatures (EI0015, hcclCommInitRootInfoConfig,
                "GiB allocated") -> zombie-worker/device-release playbook.
  F8            HS endpoint liveness (optional curl /v1/models).
  F9            optional diag json (diag_verifier_vs_data.py) acceptance:
                supervised-position agreement should be >= ~0.86 (1x data
                reference); lower -> expect F2-style degradation.

Logs are appended across runs (v13/v14/v15 share one file); the monitor
segments runs by global_step resets and analyzes the LAST segment only.

Usage:
  python3 monitor_train.py --log train.log --expected-steps 5 \
      [--max-steps 2310] [--align-ref 0.737] [--acc-ref 0.86] \
      [--checkpoints DIR] [--endpoint URL] [--diag-json FILE] [--watch S]

Exit codes: 0 ok/warn, 1 fail (F1/F5/F6/F7 hard), 2 usage error.
--watch S re-runs every S seconds (Ctrl-C to stop); without it: one shot.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

PAT_GSTEP = re.compile(r"global_step=(\d+)")
PAT_STEP_MS = re.compile(r"step_ms=([0-9.eE+-]+)")
PAT_ALIGN = re.compile(r"align_step_(\d+)=([0-9.]+)")
PAT_ACC = re.compile(r"token_accuracy_step_(\d+)=([0-9.]+)")
PAT_LOSS = re.compile(r"train/loss=([0-9.eE+-]+)")
PAT_TB = re.compile(r"Traceback \(most recent call last\)")
F7_SIGS = (
    "EI0015",
    "hcclCommInitRootInfoConfig",
    "GiB allocated",  # OOM freeze frame
    "HCCL error",
)
HARD_F7 = ("EI0015", "hcclCommInitRootInfoConfig")


def parse_log(path: str) -> dict:
    """Parse the log; every series entry is (line_no, value) so runs can be
    segmented by the last global_step reset."""
    gsteps: list[tuple[int, int]] = []
    step_ms: list[tuple[int, float]] = []
    loss: list[tuple[int, float]] = []
    trace_lines: list[int] = []
    f7_hits: dict[str, int] = {}
    series: dict[str, list[tuple[int, float]]] = {}
    run_start_lines: list[int] = [1]
    prev_gstep: int | None = None
    with open(path, errors="replace") as f:
        for i, line in enumerate(f, 1):
            if m := PAT_GSTEP.search(line):
                g = int(m.group(1))
                if prev_gstep is not None and g <= prev_gstep:
                    run_start_lines.append(i)
                prev_gstep = g
                gsteps.append((i, g))
            for pat, key in ((PAT_ALIGN, "align"), (PAT_ACC, "acc")):
                for m in pat.finditer(line):
                    series.setdefault(f"{key}_{m.group(1)}", []).append(
                        (i, float(m.group(2)))
                    )
            if m := PAT_STEP_MS.search(line):
                try:
                    step_ms.append((i, float(m.group(1))))
                except ValueError:
                    pass
            if m := PAT_LOSS.search(line):
                loss.append((i, float(m.group(1))))
            if PAT_TB.search(line):
                trace_lines.append(i)
            for sig in F7_SIGS:
                if sig in line:
                    f7_hits[sig] = f7_hits.get(sig, 0) + 1
    return {
        "gsteps": gsteps,
        "step_ms": step_ms,
        "loss": loss,
        "series": series,
        "trace_lines": trace_lines,
        "f7": f7_hits,
        "run_start_lines": run_start_lines,
    }


def last_run(d: dict, key: str) -> list:
    """Values of a series inside the last run segment."""
    vals = d["series"].get(key, d.get(key)) or []
    if not vals:
        return []
    cut = d["run_start_lines"][-1]
    tail = [v for ln, v in vals if ln >= cut]
    return tail if tail else [v for _, v in vals]


def trend(vals: list[float], frac: float = 0.1) -> tuple[float, float, float, float]:
    """(first raw value, tail median, min, max)."""
    if not vals:
        return (float("nan"),) * 4
    k = max(1, int(len(vals) * frac))
    return vals[0], statistics.median(vals[-k:]), min(vals), max(vals)


def check_endpoint(url: str, timeout: float = 5.0) -> str:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=timeout) as r:
            return f"UP ({r.status})"
    except Exception as e:  # noqa: BLE001
        return f"DOWN ({type(e).__name__}: {e})"


def report(args: argparse.Namespace, d: dict) -> tuple[list[str], list[str]]:
    fails: list[str] = []
    warns: list[str] = []
    s = {k: last_run(d, k) for k in d["series"]}
    gsteps = last_run(d, "gsteps")
    step_ms = last_run(d, "step_ms")
    loss = last_run(d, "loss")
    n_runs = len(d["run_start_lines"])
    n_gstep = len(gsteps)
    last_gstep = gsteps[-1] if gsteps else 0
    print("=" * 72)
    print(f"train log : {args.log}  ({n_runs} run segment(s); analyzing the last)")
    print(
        f"logged it : {n_gstep}  last global_step={last_gstep}"
        + (f" / {args.max_steps}" if args.max_steps else "")
    )
    if args.max_steps and step_ms and last_gstep < args.max_steps:
        med = statistics.median(step_ms[-50:])
        eta_s = (args.max_steps - last_gstep) * med / 1000
        print(f"step_ms   : median(last50)={med / 1000:.2f}s  ETA~{eta_s / 3600:.2f}h")
    if loss:
        f, l, lo, hi = trend(loss)
        print(f"loss      : {f:.3f} -> {l:.3f}  (min {lo:.3f} max {hi:.3f})")

    # F1 steps resolution ----------------------------------------------------
    obs_steps = sorted(int(k.split("_")[1]) for k in s if k.startswith("align_"))
    exp = args.expected_steps
    if exp and exp > 1:
        missing = [k for k in range(1, exp) if f"align_{k}" not in s]
        if missing and n_gstep >= 50:
            fails.append(
                f"F1 STEPS-RESOLUTION: requested steps={exp} but align_step "
                f"{missing} absent after {n_gstep} logged steps -> run is "
                "training single-step (cli.py override bug / draft config "
                "speculative_tokens). Aborting now saves GPU-hours."
            )
        else:
            print(
                f"steps     : align_step_{obs_steps} present (expect 0..{exp - 1}) OK"
            )
    else:
        print(f"steps     : align_step_{obs_steps} present")

    # F2 data-distribution mismatch ------------------------------------------
    align0 = s.get("align_0") or []
    if align0:
        f, l, lo, hi = trend(align0)
        print(
            f"align_0   : first {f:.3f} -> tail-median {l:.3f}"
            f"  (min {lo:.3f} max {hi:.3f})"
        )
        if args.align_ref is not None and f < args.align_ref - 0.10:
            warns.append(
                f"F2 DATA-MISMATCH: align_0 first {f:.3f} vs reference "
                f"{args.align_ref:.3f} (gap>0.10). v15 signature: drafter and "
                "verifier disagree on this dataset (e.g. off-policy answers); "
                "training will pull the drafter off the serving distribution."
            )

    # F3 data-fit drift -------------------------------------------------------
    acc0 = s.get("acc_0") or []
    if acc0 and align0:
        acc_l = statistics.median(acc0[-50:])
        al_l = statistics.median(align0[-50:])
        print(f"acc_0     : tail-median {acc_l:.3f}  (align_0 tail-median {al_l:.3f})")
        if acc_l - al_l > 0.08:
            warns.append(
                f"F3 DATA-FIT DRIFT: token_accuracy_0 {acc_l:.3f} exceeds "
                f"align_0 {al_l:.3f} by >0.08 -> CE term is fitting data "
                "tokens the verifier would not emit (v13/v15 signature)."
            )

    # deeper-step trends when present ----------------------------------------
    for k in obs_steps[1:]:
        f, l, lo, hi = trend(s[f"align_{k}"])
        print(
            f"align_{k}   : first {f:.3f} -> tail-median {l:.3f}"
            f"  (min {lo:.3f} max {hi:.3f})"
        )

    # F4 step_ms growth -------------------------------------------------------
    if len(step_ms) >= 200:
        f, l, lo, hi = trend(step_ms)
        print(f"step_ms   : head-median {f / 1000:.2f}s -> tail-median {l / 1000:.2f}s")
        if f > 0 and l / f > 1.5:
            warns.append(
                f"F4 THROUGHPUT: step_ms grew {l / f:.1f}x "
                f"({f / 1000:.2f}s->{l / 1000:.2f}s). Check HS endpoint "
                "(on_missing=generate) and host load."
            )

    # F5 tracebacks -----------------------------------------------------------
    if d["trace_lines"]:
        fails.append(
            f"F5 TRACEBACK: {len(d['trace_lines'])} in log "
            f"(first at line {d['trace_lines'][0]})."
        )

    # F7 HCCL/OOM signatures --------------------------------------------------
    for sig, cnt in d["f7"].items():
        lvl = fails if sig in HARD_F7 else warns
        lvl.append(
            f"F7 {sig}: {cnt} hit(s). HANDOFF section 4 playbook: kill zombie "
            "VLLMWorker on host by npu-smi PID, sleep 120, restart."
        )

    # F6 checkpoints -----------------------------------------------------------
    if args.checkpoints and n_gstep and last_gstep >= 100:
        ck = Path(args.checkpoints)
        subs = [p for p in ck.iterdir() if p.is_dir()] if ck.is_dir() else []
        if not subs:
            fails.append(
                f"F6 NO CHECKPOINT: {ck} has no checkpoint dirs after "
                f"{last_gstep} steps (checkpoint_freq=20). v15 lesson: "
                "verify save path/flags before burning the full run."
            )
        else:
            best = ck / "checkpoint_best"
            print(
                f"checkpoints: {len(subs)} dir(s), best="
                f"{'link ok' if best.exists() else 'MISSING'}"
            )

    # F8 endpoint --------------------------------------------------------------
    if args.endpoint:
        st = check_endpoint(args.endpoint)
        print(f"endpoint  : {args.endpoint} {st}")
        if st.startswith("DOWN"):
            warns.append(f"F8 HS ENDPOINT DOWN: {args.endpoint} ({st}).")

    # F9 diag acceptance -------------------------------------------------------
    if args.diag_json:
        try:
            dj = json.loads(Path(args.diag_json).read_text())
            rate = None
            for key in ("rate", "agreement", "verifier_argmax_agreement"):
                v = dj.get(key)
                if isinstance(v, (int, float)):
                    rate = float(v)
                    break
                if isinstance(v, dict) and isinstance(v.get("rate"), (int, float)):
                    rate = float(v["rate"])
                    break
            if rate is not None:
                print(f"diag      : verifier-vs-data agreement {rate:.3f}")
                if args.acc_ref is not None and rate < args.acc_ref - 0.02:
                    warns.append(
                        f"F9 DATA AGREEMENT LOW: {rate:.3f} < {args.acc_ref:.3f}. "
                        "CE-on-data and teacher-forcing contexts diverge from "
                        "serving; expect acceptance degradation (v13 lesson)."
                    )
            else:
                print(f"diag      : {args.diag_json} loaded (no agreement field)")
        except Exception as e:  # noqa: BLE001
            warns.append(f"F9 diag json unreadable: {e}")

    print("-" * 72)
    for w in warns:
        print(f"WARN  {w}")
    for f_ in fails:
        print(f"FAIL  {f_}")
    if not warns and not fails:
        print("ALL CHECKS PASS")
    print("=" * 72)
    return warns, fails


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--log", required=True)
    ap.add_argument(
        "--expected-steps",
        type=int,
        default=None,
        help="num_speculative_steps requested for this run",
    )
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument(
        "--align-ref",
        type=float,
        default=None,
        help="expected align_step_0 at init (0.737 on 1x data)",
    )
    ap.add_argument(
        "--acc-ref",
        type=float,
        default=0.86,
        help="diag verifier-vs-data agreement reference",
    )
    ap.add_argument("--checkpoints", default=None)
    ap.add_argument("--endpoint", default=None, help="HS endpoint base URL")
    ap.add_argument("--diag-json", default=None)
    ap.add_argument("--watch", type=int, default=0, help="re-run every N s")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"log not found: {args.log}", file=sys.stderr)
        return 2

    while True:
        d = parse_log(args.log)
        _, fails = report(args, d)
        if not args.watch:
            return 1 if fails else 0
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
