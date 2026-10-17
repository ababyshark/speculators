#!/bin/bash
# Single-node baseline: Eagle3 draft training, 4-NPU torchrun + DDP.
# NPUs 12,13,14,15 train; the extractor (launch_extractor.sh) uses 8-11.
# Preconditions:
#   1) build_token_freq.sh has produced token_freq.pt (vocabulary mapping input)
#   2) launch_extractor.sh is up (port 9019)
#   3) $DATA_PATH exists (pretokenized on-policy regenerated corpus)
# Usage: bash launch_train.sh   (backgrounds itself)
set -euo pipefail
source "$(dirname "$0")/env.sh"

TRAIN_NPUS="${TRAIN_NPUS:-12,13,14,15}"
NUM_TRAIN_NPUS=4
EPOCHS=${EPOCHS:-2}
LR=${LR:-1.0e-4}
SEQ_LENGTH=4096
TTT_STEPS=3          # draft 3 tokens per step (Eagle3 recipe; align num_speculative_tokens at serving time)
NUM_LAYERS=1         # Eagle3 draft is a single decoder layer (framework default for eagle3)
DRAFT_ARCH=llama     # Eagle3 uses a llama-style draft layer (as in the Qwen3-8B speculator.eagle3)
mkdir -p "$EAGLE_OUTPUT_DIR"

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: training data not found: $DATA_PATH (prepare it first, or check DATA_PATH)"
    exit 1
fi
if [ ! -f "$TOKEN_FREQ_PATH" ]; then
    echo "Error: token_freq.pt not found: $TOKEN_FREQ_PATH (run ../build_token_freq.sh first)"
    exit 1
fi

LOG=$LOG_DIR/train_single_node.log
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
# torchrun --standalone rendezvous resolves the local hostname; containers often
# miss that /etc/hosts entry and c10d init hangs with "Name or service not known".
if ! getent hosts "$(hostname)" >/dev/null 2>&1; then
    echo "127.0.0.1 localhost $(hostname)" >> /etc/hosts 2>/dev/null \
        && echo "Added $(hostname) to /etc/hosts" \
        || echo "Warning: $(hostname) unresolvable and /etc/hosts read-only; rendezvous may hang" >&2
fi
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64

exec env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" $TORCHRUN \
    --standalone --nproc_per_node $NUM_TRAIN_NPUS \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --train-data-ratio 0.9999 \
    --vllm-endpoint "http://localhost:${EAGLE_VLLM_PORT}/v1" \
    --save-path "$EAGLE_OUTPUT_DIR/checkpoints" \
    --log-dir "$EAGLE_OUTPUT_DIR/logs" \
    --epochs $EPOCHS --lr $LR \
    --optimizer adamw \
    --loss-fn kl_div \
    --total-seq-len $SEQ_LENGTH \
    --speculator-type eagle3 \
    --draft-arch $DRAFT_ARCH --num-layers $NUM_LAYERS \
    --target-layer-ids $TARGET_LAYER_IDS \
    --draft-vocab-size $DRAFT_VOCAB_SIZE \
    --token-freq-path "$TOKEN_FREQ_PATH" \
    --ttt-steps $TTT_STEPS \
    --full-attention-indices 0 \
    --draft-attn-impl sdpa \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --on-missing generate \
    --scheduler-type cosine --scheduler-warmup-ratio 0.04 --weight-decay 0.0 \
    --num-workers 8 --checkpoint-freq 0.1 --log-freq 1 \
    --logger tensorboard
