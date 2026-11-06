#!/bin/bash
set -euo pipefail

# REVIEW DRAFT: native-init, reference-style online MTP training.
# Requires the five-step GLM forward implementation described in the code
# delivery document. It is intentionally not marked as production-ready until
# the remote tree receives this implementation and the NPU smoke test passes.
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export SPECULATORS_MTP_NO_PACK=1
export PATH=/data1/mtp_training/venv/bin:$PATH
export PYTHONPATH=/workspace/speculators/src:/workspace/speculators/hs_connectors/src:${PYTHONPATH:-}

MODEL=${MODEL:-/data1/weight/GLM-5.3-w4a8-spec}
DATA=${DATA:-/data1/spec_train_data/prepared_8k_dflash2_official}
RUN=${RUN:-/data1/mtp_training/runs/glm53_mtp_v2_smoke}
DRAFT=${DRAFT:-$RUN/native_init}
HIDDEN_STATES=${HIDDEN_STATES:-$RUN/online_hidden_states}

mkdir -p "$RUN" "$RUN/logs" "$HIDDEN_STATES"
if [ ! -f "$DRAFT/config.json" ]; then
  echo "Preparing a fresh Speculators MTP init from the original verifier: $MODEL"
  rm -rf "$DRAFT"
  python /workspace/adapter/convert_glm_mtp.py \
    --source "$MODEL" \
    --teacher "$MODEL" \
    --output "$DRAFT"
fi

cd /workspace/speculators
exec torchrun --standalone --nproc_per_node=8 -m speculators.train \
  --verifier-name-or-path "$MODEL" \
  --speculator-type mtp \
  --from-pretrained "$DRAFT" \
  --num-speculative-steps 5 \
  --step-weight-beta 0.6 \
  --data-path "$DATA" \
  --vllm-endpoint http://127.0.0.1:8000/v1 \
  --on-missing generate \
  --hidden-states-path "$HIDDEN_STATES" \
  --hidden-states-lock-timeout 180 \
  --total-seq-len 8192 \
  --optimizer adamw --optimizer-implementation eager \
  --lr 1e-5 --weight-decay 0.1 \
  --fsdp-shard --gradient-checkpointing --deterministic-algorithms \
  --max-steps 1 --checkpoint-freq 1 \
  --save-path "$RUN/checkpoints" \
  --log-dir "$RUN/logs" --log-freq 1 --logger json
