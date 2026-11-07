#!/bin/bash
# Orchestrator for the direct-training pipeline (v13).
# Usage: bash run_all.sh [from_stage]    e.g. bash run_all.sh 3
# Stages: 0/1 preflight+data (no NPU) | 2 HS gen (8 cards) | 3 train (8 cards)
#         4 export+eval (8 cards)
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh
FROM=${1:-0}

run() { echo; echo "########## $* ##########"; bash "$@"; }

[ "$FROM" -le 1 ] && run 10_preflight_data.sh
[ "$FROM" -le 2 ] && run 30_generate_hs.sh
[ "$FROM" -le 3 ] && run 40_train.sh
BEST=$(ls -d $SAVE_PATH/best 2>/dev/null || ls -dt $SAVE_PATH/*/ 2>/dev/null | head -1)
[ "$FROM" -le 4 ] && run 50_export_requant_eval.sh "$BEST"
echo; echo "PIPELINE COMPLETE — results under $RUN_ROOT"
