#!/bin/bash
# Central config for the direct-training pipeline (v13). Source this file.
# All paths follow the surviving-asset layout documented in PLAN.md.

export WORK=/mnt/share_space/l00840580/my-contributions/speculators
export DT=$WORK/glm-mtp/direct_train
export BASE=/mnt/share_space/l00840580/dspark-glm53

# --- assets ---
export VERIFIER=/mnt/weight2/GLM-5.3-W4A8
export DRAFTER_INIT=$BASE/drafter_init/glm53-mtp-speculators-v9
export DATA_PATH=$BASE/data_prep_rot_10x        # Arrow: input_ids/loss_mask/seq_len, 2052 rows

# --- outputs ---
export RUN_ROOT=/mnt/weight2/l00840580-direct-v13
export HS_PATH=$RUN_ROOT/hidden_states        # ~84GB (3 layers x 2052 records)
export SAVE_PATH=$RUN_ROOT/checkpoints_v15
export LOG_DIR=$RUN_ROOT/logs
export HYBRID_OUT=$RUN_ROOT/hybrid

# --- training recipe (formal100, proven 2026-11-02) ---
export NUM_SPEC_STEPS=5
export LR=1e-5
export WEIGHT_DECAY=0.1
export EPOCHS=1
export MAX_STEPS=2310  # full epoch on 10x: 18470 train samples / 8 ranks
export SPECULATORS_MTP_DISTILL_ALPHA=0.9
export SPECULATORS_MTP_DISTILL_TEMP=2.0
export CHECKPOINT_FREQ=20
export TRAIN_RATIO=0.9006
export TOTAL_SEQ_LEN=2304                     # >= max seq_len 2281 (padding-only, semantic-neutral)

# --- machine window ---
export VISIBLE_DEVICES=0,1,2,3,4,5,6,7        # 35 first 8 cards, or local 8
export TRAIN_IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:nightly-releases-v0.30.0-a3
export TRAIN_CONTAINER=glm-mtp-direct-v13
export TRAIN_PORT=8000                        # HS-generation vllm endpoint (formal100 convention)

export PYTHON_BIN=/usr/local/python3.12.13/bin/python3
