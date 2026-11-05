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

echo "=== [4] code state (glm_moe_dsa fixes present)"
grep -q "glm_moe_dsa" $BASE/../my-contributions/speculators/src/speculators/models/base_components.py 2>/dev/null \
  && pass "base_components: glm_moe_dsa registered" || fail "glm_moe_dsa NOT registered in base_components"
MD=$BASE/../my-contributions/speculators/src/speculators/models/mtp/model_definitions.py
SPARSE_FN=$(sed -n '/def _sparse_experts_forward/,/class GlmMoeDsaMTPLayer/p' $MD 2>/dev/null)
grep -q "_sparse_experts_forward" $MD 2>/dev/null \
  && pass "sparse expert dispatch present" || fail "sparse dispatch missing"
grep -q "types.MethodType(_sparse_experts_forward, experts)" $MD 2>/dev/null \
  && pass "MethodType binding on experts.forward" || fail "MethodType binding missing"
grep -q "_ChunkedTokenDimMoE" $MD 2>/dev/null \
  && fail "dead wrapper class still present" || pass "no wrapper class"
echo "$SPARSE_FN" | grep -q "moe\." \
  && fail "sparse fn references moe.* (must be self-contained: parent passes routing, adds shared)" || pass "sparse fn self-contained"
echo "$SPARSE_FN" | grep -qE "\.gate\(" \
  && fail "sparse fn must not compute routing (parent passes topk as args)" || pass "no routing recompute in sparse fn"
grep -q "_VocabCE" $BASE/../my-contributions/speculators/src/speculators/models/mtp/core.py 2>/dev/null \
  && pass "_VocabCE present" || fail "_VocabCE missing"
grep -q 'down_proj.requires_grad_(False)' $BASE/../my-contributions/speculators/src/speculators/train/cli.py 2>/dev/null \
  && pass "experts-freeze present in cli" || fail "experts freeze missing in cli.py"
grep -q 'data_prep_rot' $BASE/run_pipeline_v2.sh && pass "pipeline uses rot data" || fail "pipeline --data-path does not point at data_prep_rot"
grep -qE 'vllm serve \$M|--model \$SAVE_PATH-w4a8' $BASE/run_pipeline_v2.sh && pass "eval model matches requant output" || fail "eval --model mismatch (404 bug)"

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
