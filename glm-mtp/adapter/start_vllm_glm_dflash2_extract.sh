#!/bin/bash
set -euo pipefail
# Checkpoint decoder layers: 5,19,33,47,61,75 (zero-based).
# Extractor captures post-layer boundary idx+1: 6,20,34,48,62,76.
# launch_vllm.py appends 78 for teacher supervision, separate from draft features.
MODEL=/data1/weight/GLM-5.3-w4a8-spec
RUN_DIR=/workspace/logs/glm_dflash2_extract_20261102
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="/workspace/adapter:${PYTHONPATH:-}"
export HCCL_BUFFSIZE=200
export LCCL_DETERMINISTIC=1 HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0 ATB_LLM_LCOC_ENABLE=0
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN_DIR/hidden_states"
cd /workspace/speculators
exec python scripts/launch_vllm.py train "$MODEL" \
  --target-layer-ids 6 20 34 48 62 76 \
  --hidden-states-path "$RUN_DIR/hidden_states" \
  --provenance-dir "${PROVENANCE_DIR:-$RUN_DIR/provenance_sorted_sfa_$(date +%Y%m%d_%H%M%S)}" \
  --no-hash-checkpoints \
  -- \
  --host 0.0.0.0 --port 8000 \
  --api-server-count 1 --renderer-num-workers 1 \
  --tensor-parallel-size 8 --data-parallel-size 1 \
  --worker-extension-cls glm_teacher_sorted_sfa_extension.GLMTeacherSortedSFA \
  --enable-expert-parallel --seed 1024 \
  --served-model-name "$MODEL" \
  --max-num-seqs 2 --max-model-len 16385 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code --gpu-memory-utilization 0.90 \
  --quantization ascend --kv-cache-dtype auto \
  --attention_config.indexer_kv_dtype int8 \
  --no-enable-prefix-caching --enforce-eager
