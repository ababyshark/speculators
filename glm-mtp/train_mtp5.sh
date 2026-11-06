#!/bin/bash
# GLM-5.3 W4A8 MTP=5 continued training — the glm-mtp5 branch pipeline.
#
# Chain: extraction (done) -> convert (done, 793 tensors) -> THIS SCRIPT.
# Semantics: the reference adapter (glm_mtp.py) + MTP=5 teacher-forced
# recursion on ONE native MTP layer, the raw pre-norm captures with the
# frozen verifier_norm applied in-forward (the capture manifest declares
# stage=pre_norm), NO_PACK single-sample batches.
#
# Usage: RUN_ID=<id> bash train_mtp5.sh
set -x
cd /mnt/share_space/l00840580/dspark-glm53

BASE=/mnt/share_space/l00840580/dspark-glm53
REPO=/mnt/share_space/l00840580/my-contributions/speculators
VENV=$BASE/venv
MODEL=/mnt/weight2/l00840580-glm53-w4a8-biasfix
DRAFTER=$BASE/drafter_init/glm53-mtp-speculators-v9
DATA=$BASE/data_prep_full                      # the RAW pre-norm captures (no offline rot!)
MANIFEST=$BASE/drafter_init/glm53-mtp-native-zpnone_capture_manifest.json  # the reference-contract manifest (stage=pre_norm -> verifier_norm in-forward)
RUN_ID=${RUN_ID:-mtp5$(date +%m%d%H%M)}
SAVE_PATH=$BASE/checkpoints/glm53_mtp5_$RUN_ID
LOG_DIR=$BASE/logs
mkdir -p $LOG_DIR $SAVE_PATH

export OMP_NUM_THREADS=4
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=1800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SPECULATORS_MTP_NO_PACK=1               # the GLM forward requires single unpacked sequences
export PYTHONUNBUFFERED=1

# ---- the conversion gate (idempotent: skip when the checkpoint exists)
if [ ! -f "$DRAFTER/model.safetensors.index.json" ] && [ ! -f "$DRAFTER/model.safetensors" ]; then
  $VENV/bin/python $REPO/glm-mtp/extract_native_mtp_w4a8.py \
    --model $MODEL \
    --output $BASE/drafter_init/glm53-mtp-native-zpnone \
    --zero-point-mode none || exit 1
  $VENV/bin/python $REPO/glm-mtp/tools/convert_glm_mtp.py \
    --source $BASE/drafter_init/glm53-mtp-native-zpnone \
    --teacher $MODEL \
    --output $DRAFTER || exit 1
fi

# ---- the preflight
bash $BASE/preflight_check.sh "${1:-}" || { echo "PREFLIGHT FAILED"; exit 1; }

# ---- training: MTP=5 teacher-forced recursion, 16 cards FSDP
$VENV/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node 16 \
  -m speculators.train \
  --verifier-name-or-path $MODEL \
  --from-pretrained $DRAFTER \
  --speculator-type mtp \
  --data-path $DATA \
  --capture-manifest $MANIFEST \
  --num-speculative-steps 5 \
  --mtp-train-forward fastmtp \
  --save-path $SAVE_PATH \
  --total-seq-len 2048 \
  --epochs 4 \
  --lr 1e-5 \
  --fsdp-shard \
  --optimizer adamw \
  --optimizer-implementation eager \
  --on-missing skip \
  --max-val-batches 1 \
  --num-workers 2 \
  --trust-remote-code

[ -d $SAVE_PATH/epoch3_end ] || { echo "TRAIN FAILED — epoch3_end missing"; exit 1; }

echo "=== training done; requant -> serve -> eval (the W4A8 signed write-back)"
rm -rf $BASE/hybrids/mtp5_$RUN_ID-w4a8
$VENV/bin/python $BASE/requant_layer78.py \
  --drafter $SAVE_PATH/epoch3_end \
  --model $MODEL \
  --output $BASE/hybrids/mtp5_$RUN_ID-w4a8 || exit 1

SERVE_LOG_SUFFIX=$RUN_ID MODEL=$BASE/hybrids/mtp5_$RUN_ID-w4a8 bash $BASE/serve_eval_B.sh >> $LOG_DIR/serve_eval_B_$RUN_ID.log 2>&1
for i in $(seq 1 60); do
  sleep 30
  curl -s --noproxy '*' http://127.0.0.1:8100/v1/models >/dev/null 2>&1 && break
done
$VENV/bin/python $BASE/measure_acceptance.py --label mtp5_$RUN_ID \
  --model $BASE/hybrids/mtp5_$RUN_ID-w4a8 \
  --save-json $BASE/eval_results/eval_mtp5_$RUN_ID.json --concurrency 32
echo "=== pipeline $RUN_ID done"
