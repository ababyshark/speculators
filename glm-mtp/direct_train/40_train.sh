#!/bin/bash
# Stage 3 (NPU): concurrent formal100 layout on 16 cards — RUN INSIDE THE CONTAINER:
#   cards 8-15 -> HS-generation vllm endpoint (already up, see 30_generate_hs.sh notes)
#   cards 0-7  -> FSDP training, --on-missing generate (HS streamed live, no persistence)
# Usage (host):  docker exec -d vllm_ascend_v0.30.0 bash -c 'nohup bash /vllm-workspace/speculators/glm-mtp/direct_train/40_train.sh > /mnt/weight2/l00840580-direct-v13/logs/train_wrapper.log 2>&1'
set -euo pipefail
cd /vllm-workspace/speculators && source glm-mtp/direct_train/env.sh

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=${ATB_CXX_ABI:-1} 2>/dev/null || true
set -u
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 HCCL_WHITELIST_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=180
export SPECULATORS_MTP_NO_PACK=1  # GLM K=1: one sample per batch (packing unsupported)

echo "== [3] FSDP training on cards 0-7; endpoint on cards 8-15 =="
python3 -m torch.distributed.run --nproc-per-node 8 \
  src/speculators/train/__main__.py \
  --speculator-type mtp \
  --verifier-name-or-path $VERIFIER \
  --trust-remote-code \
  --from-pretrained $DRAFTER_INIT \
  --data-path $DATA_PATH \
  --hidden-states-backend file \
  --hidden-states-path $HS_PATH \
  --vllm-endpoint http://127.0.0.1:$TRAIN_PORT/v1 \
  --on-missing generate \
  --request-timeout 600 --max-retries 2 \
  --generation-validation-retries 2 \
  --max-consecutive-generation-failures 20 \
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
  --log-dir $LOG_DIR --run-name glm53_mtp_direct_v15_distill5 \
  2>&1 | tee $LOG_DIR/train_v13.log | grep -E "loss|step|checkpoint|Error" || true

echo "== [3] done; latest checkpoints: =="
ls -t $SAVE_PATH 2>/dev/null | head -3
