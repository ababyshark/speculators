#!/bin/bash
# Launch eval serving + fast validation + acceptance eval, all-in-one.
# Usage: MODEL=<hybrid_dir> TAG=<label> bash launch_and_validate.sh [prompts_jsonl] [out_json]
# - preflight (no NPU, ~30s) -> abort before serving if broken
# - serve on :8100 (local stack: TP8, eager, VLLM_VERSION pin)
# - poll startup; on ready run serving-contract test (fast fail)
# - run eval_ab.py and sanity-check the result
set -u
T=/mnt/share_space/l00840580/my-contributions/speculators/glm-mtp/tests
S=/mnt/share_space/l00840580/glm53-eval/scripts
MODEL=${MODEL:?set MODEL}
TAG=${TAG:?set TAG}
PROMPTS=${1:-/mnt/share_space/l00840580/dspark-glm53/eval_prompts_train32_ids.jsonl}
OUT=${2:-/mnt/share_space/l00840580/glm53-eval/results/eval_$TAG.json}

bash $T/preflight_eval.sh "$MODEL" || { echo "PREFLIGHT FAILED - not serving"; exit 1; }

echo "[launch] starting serve ($TAG)"
MODEL=$MODEL SERVE_LOG_SUFFIX=$TAG nohup bash $S/serve_eval.sh > /dev/null 2>&1

LOG=/mnt/share_space/l00840580/glm53-eval/logs/serve_$TAG.log
for i in $(seq 1 40); do
    sleep 30
    if grep -q "Application startup complete" $LOG 2>/dev/null; then
        echo "[launch] startup complete after ~$((i*30))s"
        break
    fi
    if grep -qE "EngineCore failed|died unexpectedly" $LOG 2>/dev/null; then
        echo "SERVE DIED - see $LOG"; exit 1
    fi
done
grep -q "Application startup complete" $LOG || { echo "TIMEOUT waiting startup"; exit 1; }

echo "[validate] serving contract"
python3 $T/test_serving_contract.py || { echo "CONTRACT FAILED"; exit 1; }

echo "[eval] running 32-prompt acceptance eval"
TOKARG=""
ls "$MODEL"/tokenizer.json >/dev/null 2>&1 && TOKARG="$MODEL"
python3 $S/eval_ab.py "$PROMPTS" "$OUT" $TOKARG 2>&1 | tail -20
python3 $T/test_eval_sanity.py "$OUT" || { echo "EVAL SANITY FAILED"; exit 1; }
echo "[done] result: $OUT"
