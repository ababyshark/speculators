#!/bin/bash
# Two-node Eagle3 training -- vLLM hidden-state extractor (one instance PER NODE).
#
# Topology: on each node, NPUs 8-11 run this extractor (vLLM DP4, port 9019);
# NPUs 12-15 run the training ranks (launch_train_node.sh). Training endpoints
# are pinned per node:
#   node0 (ip-a) -> http://ip-a:9019/v1
#   node1 (ip-b) -> http://ip-b:9019/v1
# Each node's training ranks only talk to their LOCAL extractor, which keeps the
# generation load symmetric. (An earlier round-robin over both endpoints sent
# requests across nodes, whose responses carry /dev/shm paths the other node
# cannot read, and piled all traffic on one machine while the other idled.)
#
# Key difference vs single-node extraction:
#   1. Hidden states go to NODE-LOCAL /dev/shm (HS_SHM_DIR), never to shared
#      storage. With endpoints pinned per node, a training rank only ever
#      consumes its local extractor's output, so the transfer is a RAM read
#      (~15ms). A shared-storage round trip measured ~500ms fetch per step and
#      halved aggregate throughput (see report.md).
#   2. target-layer-ids fixed to Eagle3's "2 18 33" (launch_vllm.py appends the
#      last layer 36 automatically, extracting [2, 18, 33, 36]).
#   3. --provenance-dir drops vllm_command.txt / vllm.patch / weight sha256
#      next to the training run for reproducibility.
#
# Usage:  bash launch_extractor_node.sh     (backgrounds itself; keep running during training)
# Preconditions: port 9019 free on this node (stop any old extractor first).
# Restart order: stop both trainers -> stop old extractors -> start this on both
#                nodes -> start launch_train_node.sh on both nodes.
set -euo pipefail
source "$(dirname "$0")/env.sh"

VLLM_NPUS="8,9,10,11"
VLLM_DP_SIZE=4
VLLM_PORT=9019
EAGLE3_TARGET_LAYER_IDS="2 18 33"

HS_SHM_DIR=${HS_SHM_DIR:-/dev/shm/hidden_states_eagle3_multi}

export EAGLE3_MULTI_DIR=${EAGLE3_MULTI_DIR:-$WORK_ROOT/eagle3-output-multi}
mkdir -p "$EAGLE3_MULTI_DIR"

if command -v df >/dev/null 2>&1; then
    SHM_AVAIL_MB=$(df -Pm /dev/shm 2>/dev/null | awk 'NR==2 {print $4}')
    echo "/dev/shm available: ${SHM_AVAIL_MB:-?}MB"
    if [ -n "${SHM_AVAIL_MB:-}" ] && [ "$SHM_AVAIL_MB" -lt 65536 ]; then
        echo "Warning: /dev/shm < 64GB; in-flight samples may not fit (override HS_SHM_DIR)" >&2
    fi
fi

# Port pre-check: refuse to start if 9019 is already serving, otherwise the new
# instance would fail to bind silently while traffic still hits the old process.
if curl -sf -m 2 "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "Error: port ${VLLM_PORT} already in use on this node (old extractor?)" >&2
    exit 1
fi

# Clear leftovers from the previous run (safe: trainers are stopped first, so
# nothing reads this directory concurrently)
rm -rf "$HS_SHM_DIR"
mkdir -p "$HS_SHM_DIR"

LOG=$LOG_DIR/extractor_node_$(hostname -s).log
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
# --served-model-name forces the served name to equal $MODEL, otherwise the
# trainer's model check 404s.
exec env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" \
    $PY $SCRIPTS/launch_vllm.py "$MODEL" \
    --target-layer-ids $EAGLE3_TARGET_LAYER_IDS \
    --hidden-states-path "$HS_SHM_DIR" \
    --provenance-dir "$EAGLE3_MULTI_DIR" \
    -- --data-parallel-size $VLLM_DP_SIZE \
       --no-enable-chunked-prefill \
       --served-model-name "$MODEL" \
       --port $VLLM_PORT
