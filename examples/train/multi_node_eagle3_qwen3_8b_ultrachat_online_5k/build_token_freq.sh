#!/bin/bash
# Build token_freq.pt: token-frequency statistics feeding Eagle3's draft
# vocabulary mapping (d2t/t2d for the 32000-token draft vocab).
# Reuses the pretokenized corpus in $DATA_PATH (no re-preprocessing); writes
# token_freq.pt into that directory. Both the two-node and single-node launch
# scripts pass it via --token-freq-path.
# Precondition: $DATA_PATH exists (run `speculators prepare-data` first).
# Usage: bash build_token_freq.sh   (backgrounds itself)
set -euo pipefail
source "$(dirname "$0")/env.sh"

LOG=$LOG_DIR/build_token_freq.log
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
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: data directory not found: $DATA_PATH"
    echo "      Prepare it first:"
    echo "      $PY -m speculators.cli prepare-data --model $MODEL --data <regenerated.jsonl> --output $DATA_PATH --seq-length 4096"
    exit 1
fi

exec $PY "$(dirname "$0")/build_token_freq.py" \
    --data-path "$DATA_PATH" \
    --output "$TOKEN_FREQ_PATH" \
    --num-workers 8
