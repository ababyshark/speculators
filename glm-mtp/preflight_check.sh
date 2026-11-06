#!/bin/bash
# Preflight check for the GLM-5.3 MTP training/serving pipeline.
# Validates every pitfall hit during this project. Exit 1 on any FAIL.
# Usage (MUST run inside the container on the 35 host, where npu-smi/CANN live):
#   ssh <host> "docker exec <container> bash $BASE/preflight_check.sh [--fix]"
#   (--fix: kill orphans / rm stale ckpt)

BASE=/mnt/share_space/l00840580/dspark-glm53
MODEL=/mnt/weight2/l00840580-glm53-w4a8-biasfix
RUN_ID=${RUN_ID:-}
CKPT=${SAVE_PATH:-$BASE/checkpoints/glm53_mtp_v2_${RUN_ID:+$RUN_ID}}
FAIL=0

pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; FAIL=1; }

echo "=== [1] NPU state"
n_orphan=$(ps aux | grep -E "speculators.train|vllm serve|torch.distributed" | grep -v grep | wc -l)
n_orphan=$((n_orphan - 0))
if [ "$n_orphan" -gt 0 ]; then
  if [ "$1" = "--fix" ]; then
    P1="speculators.trai"; P2="torch.distribu"; P3="VLLM:"; P4="EngineCor"; P5="entrypoints.cli.mai"
    pkill -9 -f "${P1}n" 2>/dev/null; pkill -9 -f "${P2}ed.run" 2>/dev/null
    pkill -9 -f "${P3}:" 2>/dev/null; pkill -9 -f "${P4}e" 2>/dev/null; pkill -9 -f "${P5}n" 2>/dev/null
    sleep 3; echo "  FIX: killed $n_orphan orphan processes"
    n_orphan=$(ps aux | grep -E "speculators.train|vllm serve|torch.distributed" | grep -v grep | wc -l)
  fi
fi
[ "$n_orphan" -eq 0 ] && pass "no orphan training/serving processes" || fail "$n_orphan orphan processes hold NPU memory (rerun with --fix)"
hbm=$(npu-smi info | grep "0000:9D" | sed "s/|/ /g;s/  */ /g" | awk '{print $8}' | cut -d/ -f1)
[ "${hbm:-0}" -lt 10000 ] && pass "NPU0 HBM near baseline ($hbm MB)" || fail "NPU0 HBM $hbm MB — leftover memory not freed"
health=$(npu-smi info | grep -c "Ascend910")
[ "$health" -ge 16 ] && pass "16 NPUs healthy" || fail "NPU health count $health < 16"

echo "=== [2] disk"
avail=$(df --output=avail /mnt/share_space | tail -1 | tr -d ' ')
[ "$avail" -gt 419430400 ] && pass "share_space avail ${avail}KB" || fail "share_space free ${avail}KB < 400GB (checkpoints need ~400G for 4 epochs)"

echo "=== [3] training data (rot-space)"
n_rot=$(ls $BASE/data_prep_rot/hidden_states/hs_*.safetensors 2>/dev/null | wc -l)
[ "$n_rot" -eq 2052 ] && pass "rot hidden_states: $n_rot/2052" || fail "rot hidden_states $n_rot != 2052"
[ -f "$BASE/data_prep_rot/dataset_info.json" ] && pass "arrow dataset present" || fail "arrow dataset missing in data_prep_rot"

echo "=== [4] code state (glm-mtp5: reference adapter semantics)"
MD=$BASE/../my-contributions/speculators/src/speculators/models/mtp/glm_mtp.py
CORE=$BASE/../my-contributions/speculators/src/speculators/models/mtp/core.py
grep -q "glm_moe_dsa" $BASE/../my-contributions/speculators/src/speculators/models/mtp/model_definitions.py 2>/dev/null \
  && pass "glm_moe_dsa registered in model_definitions" || fail "glm_moe_dsa NOT registered"
grep -q "def forward_glm_mtp" $MD 2>/dev/null \
  && pass "5-step teacher-forced recursion present" || fail "forward_glm_mtp missing"
grep -q "forward_glm_mtp" $CORE 2>/dev/null \
  && pass "core dispatches to forward_glm_mtp" || fail "core does not dispatch to forward_glm_mtp"
grep -q "num_speculative_steps=self.config.num_speculative_steps" $CORE 2>/dev/null \
  && pass "dispatch honors config num_speculative_steps (MTP=5)" || fail "dispatch does NOT honor config steps"
grep -qE "position_ids\[..., ?None\] ?== ?0, ?0, ?token_embeddings" $MD 2>/dev/null \
  && pass "position-0 token-embedding zeroing present" || fail "position-0 zeroing MISSING"
grep -q "SPECULATORS_MTP_NO_PACK" $BASE/../my-contributions/speculators/src/speculators/train/dataloader.py 2>/dev/null \
  && pass "NO_PACK single-sample sampler present" || fail "NO_PACK sampler missing"

echo "=== [5] model + warm-start"
[ -f "$MODEL/config.json" ] && [ -f "$MODEL/rot.safetensors" ] && pass "model dir + rot.safetensors present" || fail "model dir incomplete"
[ -f "$BASE/drafter_init/glm53-mtp-warmstart-signed/model.safetensors.index.json" ] || [ -f "$BASE/drafter_init/glm53-mtp-warmstart-signed/model.safetensors" ] \
  && pass "warm-start present" || fail "warm-start missing"
n_ws=$(ls $BASE/drafter_init/glm53-mtp-warmstart-signed/model*.safetensors 2>/dev/null | grep -vc index)
[ "$n_ws" -ge 1 ] && pass "warm-start shards: $n_ws" || fail "warm-start has no weight files"

echo "=== [6] stale checkpoints (must NOT exist at launch — bogus resume)"
if [ -d "$CKPT" ]; then
  # shared-NFS multi-host: a foreign host may legitimately be training into
  # its own SAVE_PATH — only flag if OUR launch would resume into it
  if ps aux | grep -v grep | grep -q "save-path $CKPT"; then
    pass "checkpoint dir in use by an active run on this host"
  else
    fail "STALE $CKPT exists — use a per-host SAVE_PATH (rm it or pick another)"
  fi
else
  pass "no stale checkpoint at $CKPT"
fi

echo "=== [7] serving port"
ss -tln 2>/dev/null | grep -q ":8100 " && fail "port 8100 occupied" || pass "port 8100 free"

echo "=== [8] venv + speculators importable"
cd $BASE && $BASE/venv/bin/python -c "import speculators; import torch_npu" 2>/dev/null \
  && pass "speculators + torch_npu importable in venv" || fail "venv import broken"

if [ "$FAIL" -ne 0 ]; then
  echo "PREFLIGHT: FAIL — fix the above before launching"
  exit 1
fi
echo "PREFLIGHT: ALL PASS"
