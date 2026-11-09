# GLM-5.3 W4A8 MTP 蒸馏训练与评测 HowTo

> 适用：vllm-ascend v0.30.0 容器（16 卡 A3）、speculators 分支 `glm-mtp5-v2`。 方案：冻结 verifier（GLM-5.3-W4A8 量化主模型），仅微调 1 层 MTP drafter（791 张量）； verifier 通过 HS 端点提供 final-norm hidden states（训练输入 + 蒸馏目标）， 不参与反传。交付用 float 写回（v0.30 已验证；W4A8 requant 会抹掉 ~1% 量级训练增量）。

## 0. 当前基线（32 条 train 样本、贪心、MTP×5、同导出路径对比）

| 配置                                      | mean_accept_len | 说明                                                     |
| ----------------------------------------- | --------------- | -------------------------------------------------------- |
| **A_float（init drafter float serving）** | **2.908**       | 零训练最优交付，推荐基线                                 |
| native（init W4A8 量化）                  | 2.763           | 量化损伤 ~0.145                                          |
| B_v16 float（TV 5 步递归训练）            | 2.716           | 旧目标（数据 token teacher forcing）为负增益             |
| A（init requant，旧基线）                 | 2.551           | requant 双重削弱产物，勿再作对照                         |
| v17（agree 过滤 + teacher-force 混搭）    | 2.606           | 输入/目标上下文错配，teacher-force 必须配 on-policy 数据 |

**判定标准：对照 A_float=2.908，高于它才算训练有效。**

## 1. 前置条件

- 容器 `vllm_ascend_v0.30.0`（privileged，挂载 /mnt），repo 在 `/vllm-workspace/speculators` （`pip install -e . --no-deps`）。
- 资产：verifier `/mnt/weight2/GLM-5.3-W4A8`；drafter init `/mnt/share_space/l00840580/dspark-glm53/drafter_init/glm53-mtp-speculators-v9`； 数据 `.../data_prep_rot`（1x, 2052 条）。
- HS 库可预存（`fill_hidden_states.py`）或完全走 `--on-missing generate` 实时生成 （当前默认：不提前保存，依赖端点在线）。

## 2. 训推一体布局（16 卡，先端点后训练）

- **卡 8-15**：HS 端点（`scripts/launch_vllm.py train`，TP8，端口 8000）， 就绪判据 = `curl -s http://127.0.0.1:8000/v1/models` 返回 `"data"`（权重加载 ~8-10 分钟， npu-smi 卡 8-15 出现 ~56GB/卡的 VLLMWorker_TP 才算真就绪）。**训练启动即校验端点， 端点未就绪会直接崩（APIConnectionError）。**
- **卡 0-7**：FSDP 训练（`40_train.sh`，torchrun 8 rank）。
- 启动顺序（宿主机）：

```bash
# 1) 端点（见 §2 第一条命令；日志 logs/hs_endpoint_*.log）
# 2) curl 验证就绪后：
docker exec -d vllm_ascend_v0.30.0 bash -c \
  'nohup bash /vllm-workspace/speculators/glm-mtp/direct_train/40_train.sh \
   > /mnt/weight2/l00840580-direct-v13/logs/train_wrapper.log 2>&1'
```

## 3. 训练配置（唯一入口 `env.sh`）

| 变量                                     | 当前值              | 说明                                                                                             |
| ---------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------ |
| `DATA_PATH`                              | data_prep_rot（1x） | on-policy 数据就绪后切 `data_prep_rot_onpolicy`                                                  |
| `NUM_SPEC_STEPS`                         | 5                   | 递归展开步数（CLI 显式传参现已生效，见 cli.py 修复）                                             |
| `MAX_STEPS`                              | 231                 | 1x 一个 epoch；10x 数据用 2310                                                                   |
| `SAVE_PATH`                              | checkpoints_v18     | 每 `CHECKPOINT_FREQ=20` 步存档                                                                   |
| `SPECULATORS_MTP_DISTILL_ALPHA`          | 0.9                 | loss = 0.9 TV + 0.1 CE（TEMP=2.0）                                                               |
| `SPECULATORS_MTP_ACTIVATION_CKPT`        | 0                   | **必须**：FSDP2 多步递归下层级 checkpoint recompute 会触发 mixed Tensor/DTensor                  |
| `SPECULATORS_MTP_AGREE_FILTER`           | 1                   | 仅在 data token == verifier argmax 的监督位施加 loss（去掉 ~13.9% OOD 梯度）                     |
| `SPECULATORS_MTP_TEACHER_FORCE_VERIFIER` | 0                   | 递归喂 verifier argmax token；**仅在 on-policy 数据上开启**（否则输入/目标上下文错配，v17 教训） |
| `SPECULATORS_MTP_NO_PACK`                | 1                   | 每批一样本（GLM MTP 不支持 packing）                                                             |

## 4. 训练监控

```bash
# 宿主机直接跑（NFS 读日志）；exit!=0 有 FAIL；--watch 300 持续监控
python3 /mnt/share_space/l00840580/speculators-work/glm-mtp-direct/monitor_train.py \
  --log /mnt/weight2/l00840580-direct-v13/logs/train_v13.log \
  --expected-steps 5 --max-steps 231 --align-ref 0.737 \
  --checkpoints /mnt/weight2/l00840580-direct-v13/checkpoints_v18
```

判读：`align_step_0..4` 必须全部出现（F1，否则步数未生效）；`align_step_0` 应上升； `agree_rate_step_k` = 过滤保留率（1x 数据 ~0.86，on-policy 数据应 ≈1.0）； `acc_0` 与 `align_0` 差值 >0.08 = CE 在拟合数据 token（漂移告警）。

## 5. 导出与交付（float 写回，v0.30 标准）

```bash
# trained checkpoint -> W4A8 包 + FLOAT MTP 层（791 张量 float 写回，
# quant_model_description.json 逐张量 FLOAT，scale 兄弟条目移除）
python3 /mnt/share_space/l00840580/tmp_v15_eval/build_float_hybrid.py \
  # 内部路径：checkpoint=checkpoints_vXX/0, base=hybrid_B_v16, out=hybrid_...
# 校验 + serving
bash glm-mtp/tests/preflight_eval.sh <hybrid_dir>
MODEL=<hybrid_dir> SERVE_LOG_SUFFIX=<tag> bash /tmp/serve_40.sh   # 卡 0-7 TP8, 端口 8100
# 就绪判据：serve 日志 "Application startup complete"
```

注意：`--requant-w4a8` 导出仅在必须 W4A8 MTP 层的旧栈（v0.28）使用—— 它会抹掉训练增量（float Δw \<< int4 量化步长），且 v0.28 的"float 写回 0.000"是加载失败误判。

## 6. 评测

```bash
docker exec vllm_ascend_v0.30.0 bash -c "cd /vllm-workspace/speculators && \
  python3 glm-mtp/eval_ab.py \
  /mnt/share_space/l00840580/dspark-glm53/eval_prompts_train32_ids.jsonl \
  /mnt/weight2/l00840580-direct-v13/logs/eval_<tag>.json"
python3 glm-mtp/tests/test_eval_sanity.py <out.json>   # 必跑
```

协议（勿改动，保证可比）：32 条 train 样本输入、`max_tokens=64, temperature=0.0, seed=1024`、 MTP×5、TP8、贪心；指标 = 请求前后 `/metrics` 计数器做差 （`pos_accept_rates[k]`、`mean_accept_len = 1 + accepted_total/draft_steps`）。 对照必须同导出路径：float 系（A_float/B_float）与 requant 系数字不可互比。

## 7. 常见故障（详见根 HANDOFF §2.4/§4）

- 端点未就绪启动训练 → APIConnectionError：先 curl 验证再启动
- OOM "only 4.6 GiB allocated"：卡上有僵尸 VLLMWorker，按 **npu-smi 进程表**宿主 PID kill -9（内存行会骗人），sleep 120 再启
- `pkill -f vllm.entrypoints` 会连 HS 端点一起杀：定点 PID 清理
- 训练日志只有 align_step_0：步数未生效（cli.py 已修，monitor F1 兜底）
- resume 报 Missing optimizer state：换 SAVE_PATH 或 `--no-resume-from-checkpoint`
- 磁盘清理曾删除 HS 库/hybrid/checkpoints：重要产物先确认再清
