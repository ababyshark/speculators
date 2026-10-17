#!/bin/bash
# Shared environment for the two-node Eagle3 online training example.
#
# Docker exec does not run the image ENTRYPOINT, so CANN/ATB runtime env must be
# sourced manually before launching anything. Values below come from a validated
# 2-node Ascend 910 A3 deployment (16 NPUs per node, vllm-ascend 0.27.1rc1);
# adapt them instead of configuring them.
#
# Node layout (same on both machines):
#   NPUs  8-11 : hidden-state extractor (vLLM DP4, port 9019)  <- launch_extractor_node.sh
#   NPUs 12-15 : draft training ranks (4 per node, dp_size=8)  <- launch_train_node.sh
#
# Network: hosts are referenced as ip-a (node rank 0 / master) and ip-b (node rank 1).
# Replace them with your hosts' management-plane IPs (the NIC torchrun/HCCL use).

export PY=${PY:-/usr/local/python3.12.13/bin/python3}
export TORCHRUN=${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}

# speculators source tree with the multi-node branch (editable install or checkout).
# launch_train_node.sh prepends $SPECULATORS_ROOT/src and
# $SPECULATORS_ROOT/hs_connectors/src to PYTHONPATH to override any older
# editable install.
export SPECULATORS_ROOT=${SPECULATORS_ROOT:-/path/to/speculators}
export SCRIPTS=$SPECULATORS_ROOT/scripts

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=${ATB_CXX_ABI:-1}
set -u

# Paths
export MODEL=${MODEL:-/models/Qwen3-8B}
# Shared (NFS or equivalent) workspace visible on BOTH nodes; only checkpoints
# and logs go here. Hidden states never touch it (see HS_SHM_DIR below).
export WORK_ROOT=${WORK_ROOT:-/mnt/share_space/speculators-work}
# Pretokenized, on-policy regenerated corpus (input_ids + loss_mask), produced by
# `speculators prepare-data`. Must be visible on both nodes.
export DATA_PATH=${DATA_PATH:-$WORK_ROOT/regen-data}
export TOKEN_FREQ_PATH=$DATA_PATH/token_freq.pt

export LOG_DIR=$WORK_ROOT/logs
mkdir -p "$LOG_DIR" 2>/dev/null || true
export PYTHONUNBUFFERED=1

# Multi-node run outputs (checkpoints/logs on shared storage; both nodes write here)
export EAGLE3_MULTI_DIR=${EAGLE3_MULTI_DIR:-$WORK_ROOT/eagle3-output-multi}
# Hidden states land on NODE-LOCAL shared memory, never on shared storage:
# the extractor writes files there and returns the local path in its response,
# and the trainer reads that path directly, so it must be local to whichever
# node runs the training ranks (consume-once; only in-flight samples live there).
export HS_SHM_DIR=${HS_SHM_DIR:-/dev/shm/hidden_states_eagle3_multi}
