#!/bin/bash
# Stage 4 (NPU, 8 cards): export trained checkpoint -> W4A8 hybrid -> A/B eval.
# Uses the PROVEN formal100 exporter (float write-back measured acceptance 0.000,
# hence --requant-w4a8 is mandatory).
# Usage: bash 50_export_requant_eval.sh [CHECKPOINT_DIR]
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh
CKPT=${1:?usage: 50_export_requant_eval.sh <checkpoint_dir, e.g. $SAVE_PATH/best>}

echo "== [4.1] export + W4A8 write-back (formal100 tool, one step) =="
sudo docker exec $TRAIN_CONTAINER bash -c "cd /workspace/speculators && \
  $PYTHON_BIN glm-mtp/tools/export_glm_mtp_for_vllm.py \
    --checkpoint $CKPT \
    --output $HYBRID_OUT \
    --native-source $DRAFTER_INIT \
    --target-model $VERIFIER \
    --requant-w4a8 2>&1 | tail -8"

echo "== [4.2] wire clean symlinks =="
ssh-local-ln() { ln -sfn "$2" "/mnt/share_space/l00840580/glm53-eval/$1"; }
ssh-local-ln evalV13A $BASE/hybrids/evalA0_base-w4a8
ssh-local-ln evalV13B $HYBRID_OUT

echo "== [4.3] preflight the hybrid (no-NPU gate, <30s) =="
bash $WORK/glm-mtp/tests/preflight_eval.sh $HYBRID_OUT

echo "== [4.4] serve + A/B eval =="
echo "  A卷: /mnt/share_space/l00840580/glm53-eval/evalV13A  (epoch0 init, proven baseline 2.581)"
echo "  B卷: /mnt/share_space/l00840580/glm53-eval/evalV13B  (direct-trained v13)"
echo "For each side, run on the 8-card machine:"
echo "  MODEL=/mnt/share_space/l00840580/glm53-eval/evalV13X SERVE_LOG_SUFFIX=v13X \\"
echo "    bash $WORK/glm-mtp/tests/launch_and_validate.sh \\"
echo "    $BASE/eval_prompts_train32_ids.jsonl /mnt/share_space/l00840580/glm53-eval/results/eval_v13X.json"
echo "STAGE 4 DONE"
