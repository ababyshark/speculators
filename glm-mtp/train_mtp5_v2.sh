#!/bin/bash
# GLM-5.3 W4A8 MTP=5 continued training — glm-mtp5 branch, v2 data contract.
#
# Data chain (the serving-alignment fix):
#   raw capture (pre-norm, layer-77 output)
#     -> final_norm (the verifier's model.norm, all-ones weight = pure RMS)
#     -> rot (H @ R^T, the serving's rot.weight sidecar, is_rot_used=true)
#   = the exact tensor AscendDeepSeekMTP feeds its MTP layer at serving.
# The training forward runs FastMTP WITHOUT an extra norm (the data carries
# the norm+rot); the capabilities post_norm warning is expected and correct.
#
# Usage: RUN_ID=<id> bash train_mtp5_v2.sh
set -x
cd /mnt/share_space/l00840580/dspark-glm53

BASE=/mnt/share_space/l00840580/dspark-glm53
REPO=/mnt/share_space/l00840580/my-contributions/speculators
VENV=$BASE/venv
MODEL=/mnt/weight2/l00840580-glm53-w4a8-biasfix
DRAFTER=$BASE/drafter_init/glm53-mtp-speculators-v9
# the reference contract: the training consumes the TOKEN dataset (the online
# teacher computes the hidden states in-loop; the norm/rot = the model's own
# layers); the offline HS capture only feeds the parity gates via the manifest.
DATA=$BASE/data_prep_rot
MANIFEST=$BASE/data_prep_full/capture_manifest.json
RUN_ID=${RUN_ID:-mtp5v2$(date +%m%d%H%M)}
SAVE_PATH=$BASE/checkpoints/glm53_mtp5_$RUN_ID
LOG_DIR=$BASE/logs
mkdir -p $LOG_DIR $SAVE_PATH

export OMP_NUM_THREADS=4
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=1800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SPECULATORS_MTP_NO_PACK=1              # the GLM forward: single unpacked sequence
export SPECULATORS_MTP_FREEZE_EXPERTS=1       # the experts stay at the signed warm-start
export PYTHONUNBUFFERED=1

# ---- step 1: the data prep (idempotent; skip when the marker exists)
if [ ! -f "$DATA_RAW/.done" ]; then
  mkdir -p $DATA_RAW/hidden_states
  $VENV/bin/python - <<'PYEOF'
import glob, os
from safetensors import safe_open
from safetensors.torch import save_file
import torch

with safe_open("/mnt/weight2/l00840580-glm53-w4a8-biasfix/rot.safetensors", framework="pt") as f:
    R = f.get_tensor("rot.weight").to(torch.float32)
norm_w = None
for f in sorted(glob.glob("/mnt/weight2/l00840580-glm53-w4a8-biasfix/quant_model_weights-*.safetensors")):
    with safe_open(f, framework="pt", device="cpu") as sf:
        if "model.norm.weight" in sf.keys():
            norm_w = sf.get_tensor("model.norm.weight").float()
            break
assert norm_w is not None, "model.norm.weight not found"
eps = 1e-5
files = sorted(glob.glob("data_prep_full/hidden_states/hs_*.safetensors"))
assert files, "no raw HS files"
for i, fp in enumerate(files):
    out = "data_prep_rot_v2/hidden_states/" + os.path.basename(fp)
    if os.path.exists(out):
        continue
    with safe_open(fp, framework="pt", device="cpu") as sf:
        hs = sf.get_tensor("hidden_states")
        toks = sf.get_tensor("token_ids")
    x = hs.float()
    post = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm_w
    rotated = (post @ R.t()).to(torch.bfloat16)
    save_file({"hidden_states": rotated.reshape(hs.shape), "token_ids": toks}, out)
    if i % 500 == 0:
        print(f"{i}/{len(files)}", flush=True)
print("data_prep_rot_v2 complete:", len(files), "files", flush=True)
PYEOF
  touch $DATA_RAW/.done
fi
[ -f "$DATA_RAW/.done" ] || { echo "DATA PREP FAILED"; exit 1; }

# ---- step 1b: the token dataset (data_prep_rot) + the manifest checks only —
#      the training consumes the tokens (the online teacher); the v2 HS files
#      are for the parity-gate provenance, not a training input.
[ -f "$DATA/state.json" ] || { echo "TOKEN DATASET MISSING (data_prep_rot)"; exit 1; }
[ -f "$MANIFEST" ] || { echo "CAPTURE MANIFEST MISSING"; exit 1; }

# ---- step 2: the conversion gate (idempotent)
if [ ! -f "$DRAFTER/model.safetensors.index.json" ] && [ ! -f "$DRAFTER/model.safetensors" ]; then
  $VENV/bin/python $REPO/glm-mtp/extract_native_mtp_w4a8.py \
    --model $MODEL \
    --output $BASE/drafter_init/glm53-mtp-native-zpnone \
    --zero-point-mode none || exit 1
  $VENV/bin/python $REPO/glm-mtp/tools/convert_glm_mtp.py \
    --source $BASE/drafter_init/glm53-mtp-native-zpnone \
    --teacher $MODEL \
    --output $DRAFTER || exit 1
fi

# ---- step 3: the preflight
bash $BASE/preflight_check.sh "${1:-}" || { echo "PREFLIGHT FAILED"; exit 1; }

# ---- step 4: training (MTP=5, 16 cards FSDP)
$VENV/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node 16 \
  -m speculators.train \
  --verifier-name-or-path $MODEL \
  --from-pretrained $DRAFTER \
  --speculator-type mtp \
  --data-path $DATA \
  --capture-manifest $MANIFEST \
  --num-speculative-steps 5 \
  --mtp-train-forward fastmtp \
  --save-path $SAVE_PATH \
  --total-seq-len 2048 \
  --epochs 4 \
  --lr 1e-5 \
  --fsdp-shard \
  --optimizer adamw \
  --optimizer-implementation eager \
  --on-missing skip \
  --max-val-batches 1 \
  --num-workers 2 \
  --trust-remote-code

[ -d $SAVE_PATH/epoch3_end ] || { echo "TRAIN FAILED — epoch3_end missing"; exit 1; }

# ---- step 5: requant -> serve -> eval (the signed W4A8 write-back)
rm -rf $BASE/hybrids/mtp5_$RUN_ID-w4a8
$VENV/bin/python $BASE/requant_layer78.py \
  --drafter $SAVE_PATH/epoch3_end \
  --model $MODEL \
  --output $BASE/hybrids/mtp5_$RUN_ID-w4a8 || exit 1

SERVE_LOG_SUFFIX=$RUN_ID MODEL=$BASE/hybrids/mtp5_$RUN_ID-w4a8 bash $BASE/serve_eval_B.sh >> $LOG_DIR/serve_eval_B_$RUN_ID.log 2>&1
for i in $(seq 1 60); do
  sleep 30
  curl -s --noproxy '*' http://127.0.0.1:8100/v1/models >/dev/null 2>&1 && break
done
$VENV/bin/python $BASE/measure_acceptance.py --label mtp5_$RUN_ID \
  --model $BASE/hybrids/mtp5_$RUN_ID-w4a8 \
  --save-json $BASE/eval_results/eval_mtp5_$RUN_ID.json --concurrency 32
echo "=== pipeline $RUN_ID done"
