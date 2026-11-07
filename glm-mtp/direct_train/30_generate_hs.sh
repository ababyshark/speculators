#!/bin/bash
# Stage 2 (NPU, 8 cards): create training container, install speculators+adapter,
# launch HS-generation vllm endpoint, fill all 2052 HS files, stop endpoint.
# Usage: bash 30_generate_hs.sh
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh

echo "== [2.0] container =="
sudo docker rm -f $TRAIN_CONTAINER 2>/dev/null || true
sudo docker run -d --name $TRAIN_CONTAINER --shm-size 64g --privileged \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /mnt/share_space:/mnt/share_space -v /mnt/weight2:/mnt/weight2 -v /home:/home \
  -p $TRAIN_PORT:$TRAIN_PORT \
  $TRAIN_IMAGE sleep infinity

sudo docker exec $TRAIN_CONTAINER bash -c "
  git clone --depth 1 file:///mnt/share_space/l00840580/my-contributions/speculators /workspace/speculators 2>/dev/null || \
  cp -r /mnt/share_space/l00840580/my-contributions/speculators /workspace/speculators
  cd /workspace/speculators
  $PYTHON_BIN -m pip install -e . --no-deps -q
  $PYTHON_BIN glm-mtp/adapter/install_glm_mtp_adapter.py --provenance-dir $LOG_DIR/provenance || true
"

echo "== [2.1] HS endpoint (TP8, ascend quant, HS connector producer) =="
sudo docker exec -d $TRAIN_CONTAINER bash -c "cd /workspace/speculators && \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
  source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null; \
  source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=\${ATB_CXX_ABI:-1} 2>/dev/null; \
  export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES OMP_NUM_THREADS=1 HCCL_WHITELIST_DISABLE=1 \
         PYTORCH_NPU_ALLOC_CONF=expandable_segments:True; \
  $PYTHON_BIN scripts/launch_vllm.py train \
    --verifier-name-or-path $VERIFIER \
    --hidden-states-path $HS_PATH \
    --host 0.0.0.0 --port $TRAIN_PORT --tensor-parallel-size 8 \
    --max-num-seqs 2 --max-model-len 4096 --max-num-batched-tokens 4096 \
    --trust-remote-code --gpu-memory-utilization 0.90 \
    --quantization ascend --no-enable-prefix-caching --enforce-eager \
    > $LOG_DIR/hs_endpoint.log 2>&1"

echo "waiting for endpoint..."
for i in $(seq 1 40); do
  sleep 30
  curl -s -m 5 http://127.0.0.1:$TRAIN_PORT/v1/models >/dev/null 2>&1 && break
  grep -qE "EngineCore failed|died unexpectedly" $LOG_DIR/hs_endpoint.log 2>/dev/null && {
    echo "ENDPOINT DIED"; tail -30 $LOG_DIR/hs_endpoint.log; exit 1; }
done
curl -s -m 5 http://127.0.0.1:$TRAIN_PORT/v1/models >/dev/null || { echo "ENDPOINT TIMEOUT"; exit 1; }
echo "endpoint up"

echo "== [2.2] fill all HS (resume-safe) =="
sudo docker exec $TRAIN_CONTAINER bash -c "cd /workspace/speculators && \
  $PYTHON_BIN glm-mtp/direct_train/fill_hidden_states.py \
    $DATA_PATH $HS_PATH $VERIFIER http://127.0.0.1:$TRAIN_PORT/v1 16"

echo "== [2.3] stop endpoint, free cards for training =="
sudo docker exec $TRAIN_CONTAINER pkill -9 -f "[v]llm serve" || true
sleep 10
echo "STAGE 2 DONE: $HS_PATH"
