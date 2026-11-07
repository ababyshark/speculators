#!/bin/bash
# Stage 3 (NPU, 8 cards FSDP): direct training, formal100 recipe.
# Runs INSIDE $TRAIN_CONTAINER (created by 30_generate_hs.sh).
# Usage: bash 40_train.sh
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh

echo "== [3] FSDP training (formal100 recipe) =="
sudo docker exec $TRAIN_CONTAINER bash -c "cd /workspace/speculators && \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
  source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null; \
  source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=\${ATB_CXX_ABI:-1} 2>/dev/null; \
  export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES OMP_NUM_THREADS=1 HCCL_WHITELIST_DISABLE=1 \
         PYTORCH_NPU_ALLOC_CONF=expandable_segments:True; \
  $PYTHON_BIN -m torch.distributed.run --nproc-per-node 8 \
    src/speculators/train/__main__.py \
    --speculator-type mtp \
    --verifier-name-or-path $VERIFIER \
    --trust-remote-code \
    --from-pretrained $DRAFTER_INIT \
    --data-path $DATA_PATH \
    --hidden-states-backend file \
    --hidden-states-path $HS_PATH \
    --on-missing raise \
    --total-seq-len $TOTAL_SEQ_LEN \
    --train-data-ratio $TRAIN_RATIO \
    --max-val-batches 1 \
    --hidden-states-dtype bfloat16 \
    --num-workers 0 --prefetch-factor 1 --max-transfer-workers 1 \
    --num-speculative-steps $NUM_SPEC_STEPS \
    --optimizer adamw --optimizer-implementation eager \
    --lr $LR --weight-decay $WEIGHT_DECAY \
    --scheduler-type cosine --scheduler-warmup-steps 1 \
    --scheduler-total-steps $MAX_STEPS \
    --epochs $EPOCHS --checkpoint-freq $CHECKPOINT_FREQ --save-best \
    --save-path $SAVE_PATH \
    --fsdp-shard --deterministic-algorithms \
    --max-steps $MAX_STEPS --log-freq 1 --logger '' \
    --log-dir $LOG_DIR --run-name glm53_mtp_direct_v13" \
  2>&1 | tee "$LOG_DIR/train_v13.log" | grep -E "loss|step|checkpoint|error|Error" || true

[ -d "$SAVE_PATH" ] && ls "$SAVE_PATH" | tail -3
echo "STAGE 3 DONE — next: 50_export_requant_eval.sh"
