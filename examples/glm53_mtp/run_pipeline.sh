#!/bin/bash
# Full MTP continued-training pipeline (unattended):
# wait for data generation -> tokenize -> prepare-data -> extraction server ->
# offline hidden states -> training (2 epochs)
set -x
cd /mnt/share_space/l00840580/dspark-glm53
VENV=/mnt/share_space/l00840580/dspark-glm53/venv
MODEL=/mnt/weight2/l00840580-glm53-w4a8-biasfix
LOG_DIR=/mnt/share_space/l00840580/dspark-glm53/logs
DATA_JSONL=/mnt/share_space/l00840580/dspark-glm53/data/onpolicy_2500.jsonl
TARGET_LINES=2000

export OMP_NUM_THREADS=4
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=1800
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:1024

echo "=== [1/5] waiting for data generation (target $TARGET_LINES lines)"
while true; do
  n=$(wc -l < $DATA_JSONL 2>/dev/null || echo 0)
  gen_running=$(pgrep -fc "gen_onpolicy" || true)
  if [ "$n" -ge "$TARGET_LINES" ] || [ "$gen_running" = "0" ]; then
    break
  fi
  sleep 60
done
n=$(wc -l < $DATA_JSONL)
echo "generation done: $n lines"

echo "=== [2/5] tokenize"
$VENV/bin/python tokenize_onpolicy.py --input $DATA_JSONL \
  --output data/tok_full.jsonl --seq-length 4096

echo "=== [3/5] prepare-data"
rm -rf data_prep_full
$VENV/bin/python -m speculators prepare-data --model $MODEL \
  --data data/tok_full.jsonl --output data_prep_full

echo "=== [4/5] switch to extraction server + offline hidden states"
bash serve_extract.sh >> $LOG_DIR/serve_extract.log 2>&1
for i in $(seq 1 60); do
  sleep 30
  if curl -s --noproxy '*' http://127.0.0.1:8100/v1/models >/dev/null 2>&1; then break; fi
done
rm -rf data_prep_full/hidden_states
$VENV/bin/python -m speculators generate-offline-data --model $MODEL \
  --endpoint http://127.0.0.1:8100/v1 \
  --preprocessed-data data_prep_full --concurrency 48

echo "=== [5/5] stop extraction server, run training"
P1="VLLM:"; P2="EngineCor"; P3="entrypoints.cli.mai"
pkill -9 -f "${P1}:" 2>/dev/null; pkill -9 -f "${P2}e" 2>/dev/null; pkill -9 -f "${P3}n" 2>/dev/null
sleep 5

$VENV/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node 16 \
  -m speculators.train \
  --verifier-name-or-path $MODEL \
  --from-pretrained /mnt/share_space/l00840580/dspark-glm53/drafter_init/glm53-mtp-warmstart \
  --speculator-type mtp \
  --data-path /mnt/share_space/l00840580/dspark-glm53/data_prep_full \
  --save-path /mnt/share_space/l00840580/dspark-glm53/checkpoints/glm53_mtp_v1 \
  --total-seq-len 2048 \
  --epochs 2 \
  --lr 1e-5 \
  --fsdp-shard \
  --optimizer adamw \
  --optimizer-implementation eager \
  --on-missing skip \
  --trust-remote-code

echo "=== pipeline done"
