# shellcheck disable=SC2016
#!/bin/bash
# Self-healing loop: restart the MTP generation endpoint when it dies.
WD_LOG=/mnt/weight2/l00840580-direct-v13/logs/watchdog_xql.log
while true; do
  sleep 30
  for PORT in 8000 8102; do
  if ! docker exec vllm_ascend_v0.30.0 curl -s -m 5 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q '"data"'; then
    echo "$(date '+%F %T') DOWN -> restart" >> $WD_LOG
    if [ "$PORT" = "8000" ]; then DEV=8,9,10,11,12,13,14,15; TAG=A2; else DEV=0,1,2,3,4,5,6,7; TAG=B2; fi
    docker exec vllm_ascend_v0.30.0 bash -c "for p in \$(ps aux | grep -E 'vllm.entrypoints.*$PORT|VLLM::' | grep -v grep | awk '{print \$2}'); do kill -9 \$p 2>/dev/null; done" 2>/dev/null
    sleep 120
    docker exec -d vllm_ascend_v0.30.0 bash -c "PORT=$PORT DEVICES=$DEV TAG=$TAG bash /mnt/share_space/l00840580/tmp_v15_eval/serve_gen.sh" 2>/dev/null
    echo "$(date '+%F %T') relaunched, waiting ready" >> $WD_LOG
    for i in $(seq 1 20); do
      sleep 45
      if docker exec vllm_ascend_v0.30.0 curl -s -m 5 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q '"data"'; then
        echo "$(date '+%F %T') PORT $PORT READY again" >> $WD_LOG; break
      fi
    done
  done
done
