#!/bin/bash
# Eval serving A: baseline — native MTP from the W4A8 checkpoint itself, 5 tokens
set -x
cd /mnt/share_space/l00840580/dspark-glm53
MODEL=${MODEL:-/mnt/share_space/l00840580/dspark-glm53/hybrids/v8-w4a8}
LOG_DIR=/mnt/share_space/l00840580/dspark-glm53/logs
P1="VLLM:"; P2="EngineCor"; P3="entrypoints.cli.mai"
pkill -9 -f "${P1}:" 2>/dev/null; pkill -9 -f "${P2}e" 2>/dev/null; pkill -9 -f "${P3}n" 2>/dev/null
sleep 2

export OMP_NUM_THREADS=1 HCCL_OP_EXPANSION_MODE=AIV HCCL_WHITELIST_DISABLE=1
export HCCL_BUFFSIZE=256 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

nohup vllm serve $MODEL \
  --host 0.0.0.0 --port 8100 \
  --tensor-parallel-size 8 --data-parallel-size 2 \
  --seed 1024 --max-model-len 4096 --max-num-batched-tokens 8192 \
  --max-num-seqs 32 --trust-remote-code --quantization ascend \
  --enable-expert-parallel --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"enforce_eager":true}' \
  > $LOG_DIR/serve_eval_B_trained.log 2>&1 &
echo "PID: $!"
