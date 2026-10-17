#!/bin/bash
# Single-node baseline: hidden-state extraction vLLM service (4 NPUs, DP4, port 9019).
# NPUs 8-11 serve the extractor; launch_train.sh uses 12-15 (physically disjoint).
# Preconditions: port 9019 free; /dev/shm/hidden_states_eagle3 absent or empty.
# Usage: bash launch_extractor.sh   (backgrounds itself; keep running during training)
set -euo pipefail
source "$(dirname "$0")/env.sh"

VLLM_NPUS="${VLLM_NPUS:-8,9,10,11}"
VLLM_DP_SIZE=4
VLLM_PORT=$EAGLE_VLLM_PORT
mkdir -p "$EAGLE_OUTPUT_DIR"

LOG=$LOG_DIR/extractor_single_node.log
if [ "${1:-}" != "--bg" ]; then
    nohup bash "$0" --bg > "$LOG" 2>&1 &
    PID=$!
    echo "$PID" > "${LOG%.log}.pid"
    echo "Started in background: PID=$PID"
    echo "Follow: tail -f $LOG"
    echo "Stop:   kill \$(cat ${LOG%.log}.pid)"
    exit 0
fi

# ---- background exec ----
# Kill stale api servers first: multiple vLLM api servers share a port via
# SO_REUSEPORT, and an orphaned one keeps serving (intermittent 404s on
# extraction) even after the parent is killed.
pkill -9 -f "vllm.entrypoints.cli.main" 2>/dev/null || true
sleep 3
if ss -tlnp | grep -q ":$VLLM_PORT "; then
    echo "Error: port $VLLM_PORT still in use; clean it up and retry" >&2
    exit 1
fi

exec env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" $PY $SCRIPTS/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    -- --data-parallel-size $VLLM_DP_SIZE \
       --no-enable-chunked-prefill \
       --served-model-name "$MODEL" \
       --port $VLLM_PORT
