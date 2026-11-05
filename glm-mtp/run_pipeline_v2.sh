#!/bin/bash
# v2 pipeline: wrapper-key fix -> train (experts now actually load+train)
# -> requant trained -> serve B -> eval B (UltraChat 200 held-out)
set -x
cd /mnt/share_space/l00840580/dspark-glm53
bash preflight_check.sh "${1:-}" || { echo "PREFLIGHT FAILED — aborting"; exit 1; }
VENV=${VENV:-/mnt/share_space/l00840580/dspark-glm53/venv}
MODEL=${MODEL:-/mnt/weight2/l00840580-glm53-w4a8-biasfix}
BASE=${BASE:-/mnt/share_space/l00840580/dspark-glm53}
# Every run is isolated by RUN_ID: checkpoints, requant output, eval json and
# serve logs all derive from it. Concurrent runs (per-host or per-experiment)
# must use distinct RUN_IDs.
RUN_ID=${RUN_ID:-run$(date +%m%d%H%M)}
SAVE_PATH=${SAVE_PATH:-$BASE/checkpoints/glm53_mtp_v2_$RUN_ID}
LOG_DIR=$BASE/logs

export OMP_NUM_THREADS=4
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=1800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

echo "=== [1/4] train (MTP=5, fp32, raw-HS, 4 epochs) — experts now load"
$VENV/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node 16 \
  -m speculators.train \
  --verifier-name-or-path $MODEL \
  --from-pretrained $BASE/drafter_init/glm53-mtp-warmstart-signed \
  --speculator-type mtp \
  --data-path $BASE/data_prep_rot \
  --save-path $SAVE_PATH \
  --total-seq-len 2048 \
  --epochs 4 \
  --lr 1e-5 \
  --fsdp-shard \
  --optimizer adamw \
  --optimizer-implementation eager \
  --on-missing skip \
  --num-workers 2 \
  --trust-remote-code

[ -d $SAVE_PATH/epoch3_end ] || { echo "TRAIN FAILED — epoch3_end missing, not continuing"; exit 1; }

echo "=== [2/4] requant trained epoch3_end into W4A8 layout"
rm -rf $SAVE_PATH-w4a8
$VENV/bin/python requant_layer78.py \
  --drafter $SAVE_PATH/epoch3_end \
  --model $MODEL \
  --output $SAVE_PATH-w4a8

echo "=== [3/4] serve B (native mtp, 5 tokens)"
P1="VLLM:"; P2="EngineCor"; P3="entrypoints.cli.mai"
pkill -9 -f "${P1}:" 2>/dev/null; pkill -9 -f "${P2}e" 2>/dev/null; pkill -9 -f "${P3}n" 2>/dev/null
sleep 3
MODEL=$SAVE_PATH-w4a8 SERVE_LOG_SUFFIX=$RUN_ID bash serve_eval_B.sh >> $LOG_DIR/serve_eval_B_$RUN_ID.log 2>&1
for i in $(seq 1 40); do
  sleep 30
  if curl -s --noproxy '*' http://127.0.0.1:8100/v1/models >/dev/null 2>&1; then break; fi
done
curl -s --noproxy '*' http://127.0.0.1:8100/v1/models | head -c 60; echo

echo "=== [4/4] B eval on UltraChat 200 held-out"
$VENV/bin/python measure_acceptance.py --label B_trained_v2 \
  --model $SAVE_PATH-w4a8 \
  --save-json $BASE/eval_results/eval_B_$RUN_ID.json --concurrency 32
echo "=== pipeline v2 done"
