#!/bin/bash
# Run all hybrid/checkpoint tests for the A/B eval artifacts.
# Usage: bash run_tests.sh
set -u
T=/mnt/share_space/l00840580/my-contributions/speculators/glm-mtp/tests
B=/mnt/share_space/l00840580/dspark-glm53
FAIL=0

echo "=== hybrid self-contained ==="
python3 $T/test_hybrid_selfcontained.py \
    $B/hybrids/evalA_init-v12 $B/hybrids/evalB_mtp5v12 \
    $B/hybrids/evalA0_base-w4a8 $B/hybrids/evalB3_base-w4a8 || FAIL=1

echo "=== checkpoint NaN scan ==="
python3 $T/test_checkpoint_nan.py $B/checkpoints/glm53_mtp5_v12/epoch3_end || FAIL=1

echo "=== training delta (epoch0 vs epoch3) ==="
python3 $T/test_training_delta.py \
    $B/checkpoints/glm53_mtp5_v12/epoch0_end \
    $B/checkpoints/glm53_mtp5_v12/epoch3_end || FAIL=1

echo "=== eval result sanity ==="
for f in /mnt/share_space/l00840580/glm53-eval/results/eval_A_35.json \
         /mnt/share_space/l00840580/glm53-eval/results/eval_B_35.json; do
    [ -f "$f" ] && python3 $T/test_eval_sanity.py "$f" || FAIL=1
done

echo "=== serving contract (needs a live server on :8100) ==="
curl -s -m 3 http://127.0.0.1:8100/v1/models >/dev/null 2>&1 && \
    python3 $T/test_serving_contract.py || echo "SKIP no server on :8100"

echo; [ $FAIL -eq 0 ] && echo "ALL RAN: no failures" || echo "SOME TESTS FAILED"
exit $FAIL
