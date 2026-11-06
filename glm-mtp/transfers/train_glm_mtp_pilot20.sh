#!/bin/bash
set -euo pipefail
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export HCCL_CONNECT_TIMEOUT=3600 HCCL_BUFFSIZE=200
export LCCL_DETERMINISTIC=1 HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0 ATB_LLM_LCOC_ENABLE=0 HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false SPECULATORS_MTP_NO_PACK=1
export PATH=/data1/mtp_training/venv/bin:$PATH
export PYTHONPATH=/workspace/speculators/src:/workspace/speculators/hs_connectors/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/data1/mtp_training
MODEL=/data1/weight/GLM-5.3-w4a8-spec
DRAFT=$ROOT/weights/glm53_mtp_speculators_step0
DATA=$ROOT/data/prepared_8k_dflash2_official
RUN=$ROOT/runs/glm53_mtp_pilot20
mkdir -p "$RUN" "$ROOT/online_hidden_states"
cd /workspace/speculators
torchrun --standalone --nproc_per_node=8 -m speculators.train \
  --speculator-type mtp \
  --verifier-name-or-path "$MODEL" --trust-remote-code \
  --from-pretrained "$DRAFT" \
  --data-path "$DATA" --hidden-states-backend file \
  --hidden-states-path "$ROOT/online_hidden_states" \
  --vllm-endpoint http://127.0.0.1:8000/v1 --on-missing generate \
  --request-timeout 600 --max-retries 2 --generation-validation-retries 2 \
  --max-consecutive-generation-failures 20 --hidden-states-lock-timeout 180 \
  --total-seq-len 8192 --train-data-ratio 0.9005663797009516 \
  --hidden-states-dtype bfloat16 --num-workers 0 --prefetch-factor 1 \
  --max-transfer-workers 1 --num-speculative-steps 1 \
  --optimizer adamw --optimizer-implementation eager --lr 1e-5 --weight-decay 0.1 \
  --scheduler-type cosine --scheduler-warmup-steps 2 --scheduler-total-steps 20 \
  --epochs 1 --checkpoint-freq 10 --save-best --save-path "$RUN/checkpoints" \
  --fsdp-shard --gradient-checkpointing --deterministic-algorithms \
  --max-steps 20 --log-freq 1 --logger json --log-dir "$RUN/logs" \
  --run-name glm53_mtp_pilot20
