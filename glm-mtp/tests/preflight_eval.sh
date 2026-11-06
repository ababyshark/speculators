#!/bin/bash
# Fast preflight for an eval hybrid BEFORE touching NPUs (<30s, no NPU needed).
# Usage: bash preflight_eval.sh <hybrid_dir>
set -u
H=${1:?usage: preflight_eval.sh <hybrid_dir>}
T=/mnt/share_space/l00840580/my-contributions/speculators/glm-mtp/tests
FAIL=0

echo "[preflight] hybrid: $H"

python3 $T/test_hybrid_selfcontained.py "$H" || FAIL=1

python3 - "$H" <<'EOF' || FAIL=1
import json, sys
from pathlib import Path
h = Path(sys.argv[1])
c = json.load(open(h / "config.json"))
for k in ("num_hidden_layers", "hidden_size", "n_routed_experts", "moe_intermediate_size"):
    assert k in c, f"config missing {k}"
assert (h / "quant_model_weights.safetensors.index.json").is_file()
print(f"[preflight] config ok: layers={c['num_hidden_layers']} hidden={c['hidden_size']}")
EOF

# tokenizer present? (needed by local-stack eval path)
ls "$H"/tokenizer.json >/dev/null 2>&1 && echo "[preflight] tokenizer ok" \
    || echo "[preflight] WARN no tokenizer.json (use token-id prompts)"

if [ $FAIL -eq 0 ]; then echo "[preflight] READY"; else echo "[preflight] NOT READY"; fi
exit $FAIL
