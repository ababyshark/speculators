#!/bin/bash
set -euo pipefail

# GLM-5.3 MTP five-step training entrypoint.
# Defaults to one optimizer update for the first launch gate. Increase
# MAX_STEPS only after the five-step NPU smoke test passes.
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export LCCL_DETERMINISTIC=${LCCL_DETERMINISTIC:-1}
export HCCL_DETERMINISTIC=${HCCL_DETERMINISTIC:-true}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export ATB_MATMUL_SHUFFLE_K_ENABLE=${ATB_MATMUL_SHUFFLE_K_ENABLE:-0}
export ATB_LLM_LCOC_ENABLE=${ATB_LLM_LCOC_ENABLE:-0}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export SPECULATORS_MTP_NO_PACK=1
export PATH=/data1/mtp_training/venv/bin:$PATH
export PYTHONPATH=/workspace/speculators/src:/workspace/speculators/hs_connectors/src:${PYTHONPATH:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

ROOT=${ROOT:-/data1/mtp_training}
MODEL=${MODEL:-/data1/weight/GLM-5.3-w4a8-spec}
DATA=${DATA:-$ROOT/data/prepared_8k_dflash2_official}
RUN=${RUN:-$ROOT/runs/glm53_mtp_v2_smoke}
DRAFT=${DRAFT:-$RUN/native_init}
if [ ! -f "$DRAFT/config.json" ]; then
  mkdir -p "$RUN"
  rm -rf "$DRAFT"
  python /workspace/adapter/convert_glm_mtp.py \
    --source "$MODEL" --teacher "$MODEL" --output "$DRAFT"
fi
SERVICE=${SERVICE:-http://127.0.0.1:8000/v1}
HIDDEN_STATES=${HIDDEN_STATES:-$ROOT/online_hidden_states}

mkdir -p "$RUN" "$HIDDEN_STATES" "$RUN/logs" "$RUN/checkpoints"
cd /workspace/speculators
exec torchrun --standalone --nproc_per_node=8 -m speculators.train \
  --speculator-type mtp \
  --verifier-name-or-path "$MODEL" --trust-remote-code \
  --from-pretrained "$DRAFT" \
  --data-path "$DATA" --hidden-states-backend file \
  --hidden-states-path "$HIDDEN_STATES" \
  --vllm-endpoint "$SERVICE" --on-missing generate \
  --request-timeout 600 --max-retries 2 \
  --generation-validation-retries 2 --max-consecutive-generation-failures 20 \
  --hidden-states-lock-timeout 180 --total-seq-len 8192 \
  --train-data-ratio 0.9005663797009516 --hidden-states-dtype bfloat16 \
  --num-workers 0 --prefetch-factor 1 --max-transfer-workers 1 \
  --num-speculative-steps 5 --step-weight-beta 0.6 --max-anchors 512 \
  --loss-implementation fused --optimizer adamw \
  --optimizer-implementation eager --lr 1e-5 --weight-decay 0.1 \
  --scheduler-type cosine --scheduler-warmup-steps 0 --scheduler-total-steps 100 \
  --epochs 1 --checkpoint-freq 1 --save-best --save-path "$RUN/checkpoints" \
  --fsdp-shard --deterministic-algorithms --max-steps "${MAX_STEPS:-1}" \
  --log-freq 1 --logger "" --log-dir "$RUN/logs" \
  --run-name glm53_mtp_formal100_8k
