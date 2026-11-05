#!/bin/bash
# Hidden-states extraction server (vllm 0.28 + ExampleHiddenStatesConnector)
# Dumps last-layer hidden states to a shared dir; speculators generate-offline-data consumes.
set -x

MODEL=/mnt/weight2/l00840580-glm53-w4a8-biasfix
PORT=8100
HS_DIR=/mnt/share_space/l00840580/dspark-glm53/hs_dump
LOG_DIR=/mnt/share_space/l00840580/dspark-glm53/logs
mkdir -p $HS_DIR $LOG_DIR

PAT1="VLLM:"
PAT2="EngineCor"
PAT3="entrypoints.cli.mai"
pkill -9 -f "${PAT1}:" 2>/dev/null
pkill -9 -f "${PAT2}e" 2>/dev/null
pkill -9 -f "${PAT3}n" 2>/dev/null
sleep 2

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_BUFFSIZE=256
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

nohup vllm serve $MODEL \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 8 \
  --seed 1024 \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --trust-remote-code \
  --quantization ascend \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.92 \
  --kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"'"$HS_DIR"'"}}' \
  --speculative-config '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[78]}}}' \
  > $LOG_DIR/serve_extract.log 2>&1 &

echo "PID: $!"
echo "log: $LOG_DIR/serve_extract.log"
