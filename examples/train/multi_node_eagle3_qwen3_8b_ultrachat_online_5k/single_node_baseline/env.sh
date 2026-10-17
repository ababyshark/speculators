#!/bin/bash
# Single-node baseline environment (Eagle3 online training, 4 training ranks).
# Kept as the comparison baseline for the two-node example; same model, data
# and hyperparameters, just one machine.

export PY=${PY:-/usr/local/python3.12.13/bin/python3}
export TORCHRUN=${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}

# speculators source tree (editable install or checkout)
export SPECULATORS_ROOT=${SPECULATORS_ROOT:-/path/to/speculators}
export SCRIPTS=$SPECULATORS_ROOT/scripts

# CANN / ATB runtime (docker exec does not run the image ENTRYPOINT)
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=${ATB_CXX_ABI:-1}
set -u

export MODEL=${MODEL:-/models/Qwen3-8B}
export WORK_ROOT=${WORK_ROOT:-/mnt/share_space/speculators-work}
export DATA_PATH=${DATA_PATH:-$WORK_ROOT/regen-data}

export LOG_DIR=$WORK_ROOT/logs
mkdir -p "$LOG_DIR" 2>/dev/null || true
export PYTHONUNBUFFERED=1

# ---- Eagle3 pipeline config ----
export EAGLE_OUTPUT_DIR=$WORK_ROOT/eagle3-output

# Extractor layers (Eagle3 recipe: low/mid/high of the 36-layer Qwen3-8B ->
# 2/18/33; launch_vllm.py appends the last layer 36, extracting [2, 18, 33, 36].
# The trainer passes target-layer-ids 2 18 33, without the last layer).
export TARGET_LAYER_IDS="2 18 33"

# Hidden states directory (tmpfs; extractor and trainer share this container)
export HIDDEN_STATES_PATH=/dev/shm/hidden_states_eagle3

# Ports (offset from any other pipeline sharing the machine)
export EAGLE_VLLM_PORT=${EAGLE_VLLM_PORT:-9019}   # hidden-state extraction service
export EAGLE_SERVE_PORT=${EAGLE_SERVE_PORT:-9091} # speculative serving (later)

# Eagle3 vocabulary mapping needs token frequencies
export TOKEN_FREQ_PATH=$DATA_PATH/token_freq.pt
export DRAFT_VOCAB_SIZE=${DRAFT_VOCAB_SIZE:-32000}
