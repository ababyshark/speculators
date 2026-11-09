# shellcheck disable=SC2016
#!/bin/bash
# Generation endpoint: v18 float hybrid + MTPx5 speculative decoding.
# Usage: PORT=8000 DEVICES=8,9,10,11,12,13,14,15 TAG=A bash serve_gen.sh
set -x
MODEL=/mnt/weight2/l00840580-direct-v13/hybrid_B_v18_float
PORT=${PORT:?}
DEVICES=${DEVICES:?}
TAG=${TAG:-gen}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=${ATB_CXX_ABI:-1} 2>/dev/null || true
export ASCEND_RT_VISIBLE_DEVICES=$DEVICES
export OMP_NUM_THREADS=1 HCCL_WHITELIST_DISABLE=1 HCCL_BUFFSIZE=256
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
nohup python3 -m vllm.entrypoints.cli.main serve $MODEL \
  --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 8 --seed 1024 --max-model-len 4096 \
  --max-num-batched-tokens 4096 --max-num-seqs 16 --trust-remote-code \
  --quantization ascend --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  > /mnt/weight2/l00840580-direct-v13/logs/serve_gen_$TAG.log 2>&1 &
echo "PID $!"
