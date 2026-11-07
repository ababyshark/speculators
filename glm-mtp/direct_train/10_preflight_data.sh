#!/bin/bash
# Stage 0+1 (no NPU): preflight + data validation + output dirs.
# Usage: bash 10_preflight_data.sh
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh

echo "== [0] asset checks =="
for p in "$VERIFIER/config.json" "$DRAFTER_INIT/config.json" "$DATA_PATH/dataset_info.json" \
         "$WORK/src/speculators/train/__main__.py" "$WORK/scripts/launch_vllm.py"; do
    [ -e "$p" ] || { echo "MISSING: $p"; exit 1; }
done
echo "assets OK"

echo "== [1] data contract validation (CPU-only) =="
mkdir -p "$RUN_ROOT" "$LOG_DIR"
$PYTHON_BIN "$DT/validate_data.py" "$DATA_PATH" "$VERIFIER" "$RUN_ROOT/data_validation.json" "$TRAIN_RATIO"

echo "== [2] HS dir init =="
mkdir -p "$HS_PATH"
N=$(ls "$HS_PATH" 2>/dev/null | grep -c "^hs_.*safetensors$" || true)
echo "HS dir: $HS_PATH (existing: $N files; target 2052)"

echo "PREFLIGHT+DATA OK — next: 30_generate_hs.sh (needs 8-card window)"
