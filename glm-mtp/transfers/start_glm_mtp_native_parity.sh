#!/bin/bash
set -euo pipefail
MODEL=/data1/weight/GLM-5.3-w4a8-spec
RUN_DIR=/data1/mtp_training/native_parity_service
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTHONPATH="/workspace/tools:/workspace/adapter:${PYTHONPATH:-}"
export PATH="/data1/mtp_training/venv/bin:$PATH"
export HCCL_BUFFSIZE=200 HCCL_CONNECT_TIMEOUT=3600
export LCCL_DETERMINISTIC=1 HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0 ATB_LLM_LCOC_ENABLE=0
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN_DIR"
cd /workspace/speculators
exec python scripts/launch_vllm.py eval "$MODEL" \
  --spec-model "$MODEL" --spec-tokens 1 --spec-method mtp \
  --provenance-dir "$RUN_DIR/provenance_$(date +%Y%m%d_%H%M%S)" \
  --no-hash-checkpoints -- \
  --host 127.0.0.1 --port 8001 \
  --api-server-count 1 --renderer-num-workers 1 \
  --tensor-parallel-size 8 --data-parallel-size 1 \
  --worker-extension-cls glm_mtp_native_capture_extension.GLMMTPCapture \
  --enable-expert-parallel --seed 1024 \
  --served-model-name "$MODEL" \
  --max-num-seqs 1 --max-model-len 8193 --max-num-batched-tokens 4096 \
  --trust-remote-code --gpu-memory-utilization 0.92 \
  --quantization ascend --kv-cache-dtype auto \
  --attention_config.indexer_kv_dtype int8 \
  --no-enable-prefix-caching --enforce-eager
