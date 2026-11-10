#!/bin/bash
# Staged-probe novel-32K eval: tiny probe -> 32K probe -> chunked eval.
LOGD=/mnt/weight2/l00840580-direct-v13/logs
D=/mnt/weight2/l00840580-direct-v13
REPO=/vllm-workspace/speculators
MODEL=/mnt/weight2/GLM-5.3-W4A8

probe_tiny() {
  curl -s -m 30 http://127.0.0.1:8100/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"'$MODEL'","prompt":"hi","max_tokens":5}' 2>/dev/null | grep -q '"text"'
}
probe_32k() {
  # 30K-token real-size probe: a synthetic long context + tiny question
  python3 - <<'PY'
import json, urllib.request, random
rng = random.Random(9)
body_units = ["内部部署案例片段 %d：模型 GLM-5.3 在 Atlas 800T A3 上使用 vllm-ascend 完成投机解码调优，包含 npu-smi 检查、W4A8 量化与 PD 分离配置。" % i for i in range(300)]
full = "\n".join(body_units)
payload = {"model": "/mnt/weight2/GLM-5.3-W4A8", "prompt": full,
           "max_tokens": 16, "temperature": 0.0, "stream": False}
req = urllib.request.Request("http://127.0.0.1:8100/v1/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
r = urllib.request.urlopen(req, timeout=300)
print("32K_PROBE_OK", len(json.loads(r.read())["choices"][0].get("text") or ""))
PY
}
ensure_serve() {
  if probe_tiny; then return 0; fi
  echo "$(date '+%T') tiny probe failed, relaunching" >> $LOGD/novel32k_v3.log
  for p in $(ps aux | grep -E 'vllm.entrypoints|VLLM::|ApiServer' | grep -v grep | awk '{print $2}'); do kill -9 $p 2>/dev/null; done
  sleep 300
  MODEL=$MODEL SERVE_LOG_SUFFIX=32KV3 bash /tmp/serve_40.sh > /dev/null 2>&1
  for i in $(seq 1 80); do
    probe_tiny && return 0
    sleep 30
  done
  return 1
}

echo "$(date '+%T') staged probes start" >> $LOGD/novel32k_v3.log
for round in 1 2 3; do
  ensure_serve || { echo "$(date '+%T') round $round serve failed" >> $LOGD/novel32k_v3.log; continue; }
  echo "$(date '+%T') tiny probe OK" >> $LOGD/novel32k_v3.log
  if probe_32k 2>&1 | grep -q 32K_PROBE_OK; then
    echo "$(date '+%T') 32K probe OK -> starting chunked eval" >> $LOGD/novel32k_v3.log
    break
  fi
  echo "$(date '+%T') 32K probe CRASHED the engine" >> $LOGD/novel32k_v3.log
done

python3 - <<'PY'
import json
rows = [json.loads(l) for l in open('/mnt/weight2/l00840580-direct-v13/eval_novel32k_ids.jsonl')]
for ci in range(4):
    with open(f'/mnt/weight2/l00840580-direct-v13/eval_novel32k_chunk{ci}.jsonl', 'w') as f:
        for r in rows[ci*4:(ci+1)*4]:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('chunks ok')
PY

for ci in 0 1 2 3; do
  for try in 1 2 3; do
    ensure_serve && probe_32k > /dev/null 2>&1 || { echo "$(date '+%T') pre-chunk $ci serve dead" >> $LOGD/novel32k_v3.log; sleep 30; continue; }
    echo "$(date '+%T') chunk $ci try $try" >> $LOGD/novel32k_v3.log
    cd $REPO && python3 glm-mtp/eval_ab.py $D/eval_novel32k_chunk$ci.jsonl $LOGD/eval_novel32k_chunk$ci.json >> $LOGD/novel32k_v3.log 2>&1
    if [ -s $LOGD/eval_novel32k_chunk$ci.json ]; then
      echo "$(date '+%T') chunk $ci OK" >> $LOGD/novel32k_v3.log; break
    fi
    echo "$(date '+%T') chunk $ci failed" >> $LOGD/novel32k_v3.log
    sleep 30
  done
done
echo "ALL_DONE" >> $LOGD/novel32k_v3.log
