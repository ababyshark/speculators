#!/bin/bash
set -euo pipefail

# Run inside the reference container after copying this workspace's remote_scripts
# files to /workspace/adapter. No VS Code dependency.

ROOT=/workspace/adapter
SPEC=/workspace/speculators

install -Dm0644 "$ROOT/remote_scripts/glm_mtp.py" \
  "$SPEC/src/speculators/models/mtp/glm_mtp.py"
python "$ROOT/remote_scripts/install_glm_mtp_adapter.py"

python -m compileall -q \
  "$SPEC/src/speculators/models/mtp/glm_mtp.py" \
  "$SPEC/src/speculators/models/mtp/core.py" \
  "$SPEC/src/speculators/models/mtp/data.py" \
  "$SPEC/src/speculators/train"

echo GLM_MTP_V2_REMOTE_PATCH_PASSED
