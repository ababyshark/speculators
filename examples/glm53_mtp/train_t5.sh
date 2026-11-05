#!/bin/bash
# MTP=5 continued training: 2052 on-policy samples, 16-rank FSDP, 2 epochs
set -x
cd /mnt/share_space/l00840580/dspark-glm53

export OMP_NUM_THREADS=4
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=1800
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:1024

VENV=/mnt/share_space/l00840580/dspark-glm53/venv
MODEL=/mnt/weight2/l00840580-glm53-w4a8-biasfix

$VENV/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node 16 \
  -m speculators.train \
  --verifier-name-or-path $MODEL \
  --from-pretrained /mnt/share_space/l00840580/dspark-glm53/drafter_init/glm53-mtp-warmstart \
  --speculator-type mtp \
  --data-path /mnt/share_space/l00840580/dspark-glm53/data_prep_full \
  --save-path /mnt/share_space/l00840580/dspark-glm53/checkpoints/glm53_mtp_t5fp32 \
  --total-seq-len 2048 \
  --epochs 4 \
  --lr 1e-5 \
  --fsdp-shard \
  --optimizer adamw \
  --optimizer-implementation eager \
  --on-missing skip \
  --num-workers 2 \
  --trust-remote-code
