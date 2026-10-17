#!/bin/bash
# Eagle3 two-node training (2 nodes x 4 ranks = 8 training NPUs, static torchrun).
#
# Nodes: ip-a = NODE_RANK 0 (head / MASTER_ADDR)
#        ip-b = NODE_RANK 1
# NPUs:  12,13,14,15 per node (extractor from launch_extractor_node.sh uses 8-11).
#
# Online data plane (mirrors the validated single-node topology):
#   - Endpoints pinned per node: node0 uses http://ip-a:9019/v1 only, node1 uses
#     http://ip-b:9019/v1 only. (A two-endpoint round robin routed requests to
#     the peer node, whose responses carried /dev/shm paths unreadable locally,
#     and once loaded one machine while the other's AI cores idled, with 17
#     failovers.)
#   - Hidden states go through node-local /dev/shm (launch_extractor_node.sh
#     writes, this script reads -- the paths must match exactly). Generated
#     files are deleted after being read (consume-once): generation and
#     training overlap with zero shared-storage round trips.
#   - With a single endpoint EndpointRouter degenerates to pass-through + retry
#     backoff, i.e. single-node behavior.
# Preconditions:
#   1) The extractor (launch_extractor_node.sh) is up and /health on the LOCAL
#      node answers (this script waits for it).
#   2) Both nodes see $EAGLE3_MULTI_DIR (shared storage for checkpoints/logs only).
#   3) env.sh resolves the same PY/TORCHRUN/paths on both machines.
#
# Usage: NODE_RANK=0 bash launch_train_node.sh   (on ip-a)
#        NODE_RANK=1 bash launch_train_node.sh   (on ip-b)
#        (NODE_RANK is inferred from the host IP when unset)
set -euo pipefail
source "$(dirname "$0")/env.sh"

# ---------- multi-node plumbing ----------
MASTER_ADDR=${MASTER_ADDR:-ip-a}
MASTER_PORT=${MASTER_PORT:-29600}
NNODES=${NNODES:-2}
TRAIN_NPUS="12,13,14,15"
NUM_TRAIN_NPUS=4

# Infer NODE_RANK from the host IP (can be overridden explicitly)
HOST_IPS=$(hostname -I 2>/dev/null || true)
if [ -z "${NODE_RANK:-}" ]; then
    case "$HOST_IPS" in
        *ip-a*) NODE_RANK=0 ;;
        *ip-b*) NODE_RANK=1 ;;
        *) echo "Error: cannot infer NODE_RANK (hostname -I: $HOST_IPS); set NODE_RANK=0/1"; exit 1 ;;
    esac
fi

# Force the HCCL cross-node NIC (critical fix): the host has several docker
# bridges that exist only on one node; HCCL's auto NIC pick can select such a
# bridge IP and the peer's connect() then hangs forever. Pin the management NIC
# both nodes can reach (the one torchrun's rendezvous already uses).
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-<mgmt-nic>}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-<mgmt-nic>}
if [ -z "${HCCL_IF_IP:-}" ]; then
    case "$HOST_IPS" in
        *ip-a*) export HCCL_IF_IP=ip-a ;;
        *ip-b*) export HCCL_IF_IP=ip-b ;;
    esac
fi
echo "HCCL: SOCKET_IFNAME=$HCCL_SOCKET_IFNAME IF_IP=${HCCL_IF_IP:-unset}"

# Endpoints pinned per node (core of the online data plane). FileTransfer-based
# handoff returns the extractor's LOCAL file path in the response and the
# trainer reads that path directly, so only the local extractor's output can be
# consumed; single-endpoint failover degrades to same-endpoint backoff.
LOCAL_ENDPOINT_BY_RANK=("http://ip-a:9019/v1" "http://ip-b:9019/v1")
VLLM_ENDPOINTS=${VLLM_ENDPOINTS:-${LOCAL_ENDPOINT_BY_RANK[$NODE_RANK]}}

# Multi-node run outputs (must match launch_extractor_node.sh; do NOT reuse the
# single-node output dir)
EAGLE3_MULTI_DIR=${EAGLE3_MULTI_DIR:-$WORK_ROOT/eagle3-output-multi}
HIDDEN_STATES_PATH_MULTI=${HS_SHM_DIR:-/dev/shm/hidden_states_eagle3_multi}
mkdir -p "$EAGLE3_MULTI_DIR/checkpoints" "$EAGLE3_MULTI_DIR/logs" "$HIDDEN_STATES_PATH_MULTI"

# Training hyperparameters
EPOCHS=2
LR=1.0e-4
SEQ_LENGTH=4096
NUM_WORKERS=${NUM_WORKERS:-8}

# Multi-node startup is slow (cross-node HCCL setup + model load); default 60 min
# (upstream default is 30).
export SPECULATORS_DIST_TIMEOUT_MINUTES=${SPECULATORS_DIST_TIMEOUT_MINUTES:-60}

# Force the multi-node branch code (overrides any older editable install)
export PYTHONPATH="$SPECULATORS_ROOT/src:$SPECULATORS_ROOT/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64

# torchrun rendezvous needs the local hostname resolvable (often missing in containers)
if ! getent hosts "$(hostname)" >/dev/null 2>&1; then
    echo "127.0.0.1 localhost $(hostname)" >> /etc/hosts 2>/dev/null \
        && echo "Added $(hostname) to /etc/hosts" \
        || echo "Warning: $(hostname) unresolvable and /etc/hosts read-only; rendezvous may hang" >&2
fi

# ---------- extractor health gate (all local endpoints must answer) ----------
for ep in $VLLM_ENDPOINTS; do
    base=${ep%/v1}
    echo "Waiting for extractor: $base/health ..."
    until curl -sf -m 3 "$base/health" > /dev/null 2>&1; do
        sleep 5
    done
    echo "  $ep ready"
done

# ---------- launch ----------
LOG=$LOG_DIR/train_multi_node${NODE_RANK}.log
if [ "${1:-}" != "--bg" ]; then
    nohup bash "$0" --bg > "$LOG" 2>&1 &
    PID=$!
    echo "$PID" > "${LOG%.log}.pid"
    echo "Started in background (node_rank=$NODE_RANK): PID=$PID"
    echo "Follow: tail -f $LOG"
    echo "Stop:   kill \$(cat ${LOG%.log}.pid)"
    exit 0
fi

# ---- background exec ----
exec env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" $TORCHRUN \
    --nnodes $NNODES \
    --node-rank $NODE_RANK \
    --nproc-per-node $NUM_TRAIN_NPUS \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    $SCRIPTS/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --train-data-ratio 0.9999 \
    --vllm-endpoints $VLLM_ENDPOINTS \
    --save-path "$EAGLE3_MULTI_DIR/checkpoints" \
    --log-dir "$EAGLE3_MULTI_DIR/logs" \
    --epochs $EPOCHS --lr $LR \
    --optimizer adamw \
    --loss-fn kl_div \
    --total-seq-len $SEQ_LENGTH \
    --speculator-type eagle3 \
    --draft-arch llama --num-layers 1 \
    --target-layer-ids 2 18 33 \
    --draft-vocab-size 32000 \
    --token-freq-path "$TOKEN_FREQ_PATH" \
    --ttt-steps 3 \
    --full-attention-indices 0 \
    --draft-attn-impl sdpa \
    --hidden-states-path "$HIDDEN_STATES_PATH_MULTI" \
    --on-missing generate \
    --scheduler-type cosine --scheduler-warmup-ratio 0.04 --weight-decay 0.0 \
    --num-workers $NUM_WORKERS --checkpoint-freq 0.1 --log-freq 1 \
    --logger tensorboard
