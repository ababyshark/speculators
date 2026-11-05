#!/bin/bash
set -euo pipefail

# Reference-style launch_vllm.py teacher service for online MTP training.
# Hidden states are extracted online from target layer 78 and returned through
# the training endpoint; no full hidden-state dataset is precomputed.
MODEL=${MODEL:-/data1/weight/GLM-5.3-w4a8-spec}
PORT=${PORT:-8000}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export PATH=/data1/mtp_training/venv/bin:$PATH
export PYTHONPATH=/workspace/adapter:/workspace/speculators/src:/workspace/speculators/hs_connectors/src:${PYTHONPATH:-}

cd /workspace/speculators
exec python scripts/launch_vllm.py "$MODEL" \
  --target-layer-ids 78 \
  -- \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --quantization ascend --kv-cache-dtype auto \
  --no-enable-prefix-caching
