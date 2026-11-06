# ruff: noqa
#!/usr/bin/env python3
"""Test 5: serving contract for the eval stack (vllm-ascend v0.28 style).

Regression for: 400 on prompt_token_ids (must use prompt-as-int-list),
missing per-position spec-decode metrics, server not up.

Usage: python3 test_serving_contract.py [base_url]   (default http://127.0.0.1:8100)
Exits 0 if all pass.
"""

import json
import re
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8100"
POS_RE = re.compile(
    r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d)"[^}]*\}'
)


def get(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read().decode()


def main():
    # 1. models endpoint
    try:
        models = json.loads(get("/v1/models"))
        assert models["data"], "empty model list"
        print(f"PASS /v1/models -> {models['data'][0]['id'][:60]}")
    except Exception as e:
        print(f"FAIL /v1/models: {e}")
        sys.exit(1)

    # 2. token-id prompt via prompt-as-list (v0.28 contract)
    body = json.dumps(
        {"prompt": [151644, 872, 198], "max_tokens": 4, "temperature": 0.0}
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
        assert out["choices"][0]["finish_reason"] in ("length", "stop")
        print("PASS /v1/completions token-id prompt (prompt=[ints])")
    except Exception as e:
        print(f"FAIL token-id prompt: {e}")
        sys.exit(1)

    # 3. spec-decode per-position metrics exposed
    try:
        metrics = get("/metrics")
        poses = sorted(int(m.group(1)) for m in POS_RE.finditer(metrics))
        assert poses, "no per-position accepted-token metrics found"
        assert 0 in poses, f"position 0 missing, got {poses}"
        print(f"PASS /metrics spec_decode per-pos counters at positions {poses}")
    except Exception as e:
        print(f"FAIL metrics: {e}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
