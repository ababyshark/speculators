# ruff: noqa
#!/bin/bash
# usage: n3_loop.sh <model_tag> ; expects /root/eval_ab.py + eval jsonls
cd /root
mkdir -p /root/n3
MODEL=$1
for s in eval_xql_in32 eval_xql_held32 eval_prompts_onpolicy32_ids; do
  for r in 1 2 3; do
    echo "run ${MODEL} ${s} ${r}"
    python3 eval_ab.py /mnt/weight2/l00840580-direct-v13/${s}.jsonl \
      /root/n3/${MODEL}_${s}_r${r}.json 2>&1 | grep -E "mean_accept|Error" | tail -1
  done
done
echo "N3_${MODEL}_DONE"
