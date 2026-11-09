# Speculators — Agent Guide

## GLM MTP5: RFC-first workflow

[glm-mtp/docs/RFC_glm_mtp5.md](./glm-mtp/docs/RFC_glm_mtp5.md) is the **normative specification and single source of truth** for all GLM MTP5 work. It exists so that an agent can (re)implement the system from the RFC alone, without reading the current implementation.

- **Spec first**: new GLM MTP5 features start as a numbered requirement (`REQ-*`) or decision record (`ADR-*`) in the RFC; the implementation follows it. Do not reverse-engineer the RFC from code — the RFC leads.
- **Spec, not code**: the RFC carries at most pseudocode. Keep semantics precise (index alignment, normalization, error table) instead of pasting implementations. The RFC is written forward-design: motivation → principles (§2) → alternatives (§3) → end-to-end (§4) → normative spec (§8-10) → acceptance (§6, §15-16).
- **Keep it current**: any implementation deviation, new decision, or resolved open item (`OPEN-*`) MUST be written back into the RFC in the same change. Accepted gates (`G-A*`) and the test matrix (`T-*`) are the acceptance contract.
- Known legacy defects are recorded as fixed target semantics in RFC §12 (`DFX-*`) — re-implementations MUST use the fixed behavior, not the historical one. Field-incident lessons are distilled into RFC §13 engineering rules (`E-*`) — code review checks against that list.

## Pull requests

Read [CONTRIBUTING.md](./CONTRIBUTING.md) before contributing. Install the git hooks once per clone, after installing the `[dev]` extras:

```bash
pre-commit install
```

Fill in the PR template's **Purpose** and **Tests** sections (see [PULL_REQUEST_TEMPLATE.md](./.github/PULL_REQUEST_TEMPLATE.md)). A reviewer should get the point in under 30 seconds:

- **Purpose**: 1–3 sentences or a few bullets on *why*, plus the issue link if available. Not a restatement of the diff.
- **Tests**: the command a reviewer can run, and whether it passed. Do not paste the output — CI prints it, and a wall of captured log pushes the point of the PR off the screen. Never claim a run you did not do, or numbers you did not measure.
- Keep the description proportional to the diff. A few changed lines do not need a paragraph each; if the description is longer than what it describes, cut it.
- Do not state unverified hypotheses as facts.
- In a stacked PR, do not restate where it sits in the stack ("first of N", "stacked on #1234", "top of the stack") or list the other PRs. GitHub renders the stack already. Reference another PR only for something the stack view does not carry, such as which commit a measurement was taken against.
- No emoji headings, no "comprehensive"/"robust"/"production-ready", no per-file summary tables.
- Do not hard-wrap the description at a fixed column. GitHub reflows it, so manual line breaks only make the diff of a later edit noisy.
- Leave the "I (a human) have written or reviewed the code" checkbox unchecked.

## Provenance

Training, eval, and vLLM-launch scripts write reproducibility artifacts so runs can be recreated later.

- **Training** (`src/speculators/train/utils.py`): `save_train_command()` writes `train_command.txt` (timestamp, git SHA, world size, package versions, full argv) and `speculators.patch` (uncommitted diff) into the checkpoint directory.
- **Eval** (`scripts/evaluate/evaluate.py`): `save_eval_provenance()` writes `eval_command.txt` (timestamp, git SHA, package versions, full argv) into the output directory.
- **vLLM launch** (`scripts/launch_vllm.py`): `_save_vllm_provenance()` writes `vllm_command.txt`, `vllm.patch`, and `checkpoint_sha256.txt` (plus `drafter_checkpoint_sha256.txt` in eval mode) — but **only when `--provenance-dir` is passed**. Two subcommands:
  - `launch_vllm.py train MODEL` (default): hidden-states extraction for training data generation.
  - `launch_vllm.py eval MODEL --spec-model DRAFTER`: speculative decoding serving for evaluation.

**Always pass `--provenance-dir` when running `scripts/launch_vllm.py`** — point it at the associated run's output/checkpoint directory (e.g. `--provenance-dir <output_dir>`) so the vLLM provenance is co-located with the training/eval it belongs to.

When publishing a model or eval results (HuggingFace, GitHub comments, etc.), always include the provenance artifacts (`train_command.txt`, `eval_command.txt`, `vllm_command.txt`, `checkpoint_sha256.txt`, `drafter_checkpoint_sha256.txt`, patches) so the run can be reproduced.
