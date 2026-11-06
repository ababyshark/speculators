#!/bin/bash
# Serve a hybrid for A/B acceptance eval.
# Usage: MODEL=<path> SERVE_LOG_SUFFIX=<tag> bash serve_eval.sh
set -x
BASE=/mnt/share_space/l00840580/glm53-eval
MODEL=${MODEL:?set MODEL}
LOG_DIR=$BASE/logs
TAG=${SERVE_LOG_SUFFIX:-manual}

pkill -9 -f "[v]llm.serv" 2>/dev/null
pkill -9 -f "[V]LLM::" 2>/dev/null
sleep 2

export OMP_NUM_THREADS=1 HCCL_OP_EXPANSION_MODE=AIV HCCL_WHITELIST_DISABLE=1
export HCCL_BUFFSIZE=256 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# /vllm-workspace/vllm is a dev checkout (post-v0.33.0) reporting __version__="dev";
# vllm-ascend's version gate rejects it, so pin the main-lane version explicitly.
export VLLM_VERSION=0.33.0

nohup vllm serve $MODEL \
  --host 0.0.0.0 --port 8100 \
  --tensor-parallel-size 8 \
  --seed 1024 --max-model-len 4096 --max-num-batched-tokens 8192 \
  --max-num-seqs 32 --trust-remote-code --quantization ascend --enforce-eager \
  --enable-expert-parallel --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"enforce_eager":true}' \
  > $LOG_DIR/serve_$TAG.log 2>&1 &
echo "PID: $!  log: $LOG_DIR/serve_$TAG.log"
