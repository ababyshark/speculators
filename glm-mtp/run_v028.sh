#!/bin/bash
# GLM-5.3-W4A8 serving + MTP baseline (nightly vllm-ascend v0.30.0-a3)
export IMAGE=quay.io/ascend/vllm-ascend:v0.28.0.rc1-a3
export NAME=l00840580-glm53-serve3

docker run -it -d \
--name $NAME \
--network host \
--shm-size=64g \
--privileged \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /mnt/weight:/models \
-v /mnt/weight2:/mnt/weight2 \
-v /home:/home \
-v /mnt/share_space/sact-platform/debug-data/l00840580:/mnt/share_space \
-e HCCL_WHITELIST_DISABLE=1 \
-e HCCL_BUFFSIZE=256 \
-e HCCL_CONNECT_TIMEOUT=1800 \
$IMAGE bash
