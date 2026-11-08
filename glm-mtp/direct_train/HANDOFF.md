# GLM-5.3 W4A8 MTP 蒸馏增训 — 交接手册（HANDOFF）

> 日期：2026-11-08 ｜ 分支：`glm-mtp5-v2` @ `49011f5`（GitHub: ababyshark/speculators，已推送）
> 运行机器：**10.246.63.40**（A3-syn-30，root，16 卡，容器 `vllm_ascend_v0.30.0`）
> 本机（调试容器 10.246.63.12）释放后，**全部训练活动只依赖 40 机 + GitHub 仓库**。

---

## 1. 方案思路（为什么走到这里）

### 1.1 问题定义
GLM-5.3 W4A8（quantization ascend, TP8）的 native MTP drafter 在官方栈（vllm 0.28.0rc1/v0.30.0 容器）
上平均接受长度 **2.763**（32 条训练样本、贪心、MTP×5）。目标：通过增训提升接受率。

### 1.2 已证伪的路线（勿回头）
| 路线 | 结果 | 结论 |
|---|---|---|
| v12：.23 框架 + 离线 HS（pre_norm/layer78 捕获语义） | 2.581→2.573 | 无效；HS 语义错位 + 目标错位 |
| v13：直连 HS + CE 训练（formal100 配方，speculators trainer） | 2.551→2.426 | HS 语义对齐了但**目标仍错** |

### 1.3 根因诊断（32 样本逐 token logprobs，`diag_verifier_vs_data.py`）
- **verifier argmax 与训练数据 token 的重合率只有 86.7%**（监督位）
- CE-on-data 训练把 drafter 拉向"数据分布"，在 13.3% 分歧位上系统性背叛 verifier
- loss 降（1.10→0.66）与接受率降**同向发生** = 目标错位的铁证，非实现 bug

### 1.4 现行方案（v15，正在跑）
**TV 距离蒸馏**：`loss = 0.9 × TV(drafter‖verifier) + 0.1 × CE(data)`
- TV 距离直接优化投机接受率：`acceptance = Σ min(p,q) = 1 − TV`（speculators.losses.eager.tv_loss，
  DSpark 历史配方 `{"ce":0.1,"tv":0.9}` 同源）
- 蒸馏目标 = `lm_head(hidden_states 切片)`：hidden_states 是 verifier final-norm 输出、
  lm_head 与 verifier 头共享 → 目标就是 serving 时的真实分布
- **v15 新增**：`num_speculative_steps=5`（训练时递归展开，drafter 见自己的输出）+
  **10 倍数据 20520 条**（v14 单步版结果 2.480 仍低于基线，pos1-4 恶化指向递归训练缺失）
- v14 证据：训练中 `align_step_0` 0.737→0.758 上升（CE 训练是下降的）——TV 目标方向正确

### 1.5 关键参照数字（32 条训练样本、贪心、同导出路径）
| 配置 | mean_accept_len | pos0-4 |
|---|---|---|
| native（原始 W4A8，无 requant） | 2.763 | 0.756/0.469/0.268/0.163/0.106 |
| A：init 经 v13 导出路径（基线） | 2.551 | 0.734/0.419/0.226/0.117/0.055 |
| B_v13：CE 训练 | 2.426 | 0.740/0.437/0.228/0.107/0.060 |
| B_v14：TV 蒸馏单步 | 2.480 | 0.736/0.401/0.204/0.095/0.044 |
| **B_v15：TV 蒸馏 + steps=5 + 10x** | 训练中 | — |
| verifier-argmax == 数据 token（诊断） | 监督位 86.7% | — |

---

## 2. 40 机现状与资源地图

### 2.1 机器
- `ssh root@10.246.63.40`（密码同前）；hostname `A3-syn-30`；16 chip（8 卡×2 chip）
- 关键容器：**`vllm_ascend_v0.30.0`**（privileged，挂载 /mnt、/tmp，含 vllm 0.30.0 +
  speculators 0.10.0a30 editable @ `/vllm-workspace/speculators` + transformers 5.16.1）
- 其余容器（vllm-ascend、vllm-ascend-new 等）非本任务，勿动

### 2.2 路径
| 内容 | 位置（容器内 = 宿主机同路径） |
|---|---|
| 训练输出根 | `/mnt/weight2/l00840580-direct-v13/`（40 本地 100T nvme，可写） |
| ├ checkpoints（v13 CE） | `.../checkpoints/0/model.safetensors` |
| ├ checkpoints（v14 TV 单步） | `.../checkpoints_v14/0/model.safetensors` |
| ├ **checkpoints（v15 蒸馏 5 步，运行中）** | `.../checkpoints_v15/`（每 20 步存档） |
| ├ hybrid 导出 | `.../hybrid_B_v14`、`hybrid_B_trained`、`hybrid_A_init` |
| ├ 日志 | `.../logs/train_v13.log`、`hs_endpoint_*.log`、`serve_*.log`、`eval_*.json` |
| 代码（容器内运行副本） | `/vllm-workspace/speculators`（git checkout，已同步 49011f5 + 本地补丁） |
| 代码（Git 权威副本） | GitHub `ababyshark/speculators` 分支 `glm-mtp5-v2` |
| 训练数据 10x | `/mnt/share_space/l00840580/dspark-glm53/data_prep_rot_10x`（20520 行 Arrow） |
| 训练数据 1x（v13/v14 用） | `/mnt/share_space/l00840580/dspark-glm53/data_prep_rot`（2052 行） |
| verifier | `/mnt/weight2/GLM-5.3-W4A8`（TP8 量化加载器最低要求 TP8） |
| drafter init | `/mnt/share_space/l00840580/dspark-glm53/drafter_init/glm53-mtp-speculators-v9` |
| 评测集 | `/mnt/share_space/l00840580/dspark-glm53/eval_prompts_train32_ids.jsonl`（32 条训练样本） |
| 干净路径软链 | `/mnt/share_space/l00840580/glm53-eval/evalA0b|evalB3b|evalV13A|evalV13B`（评测用，路径含 `dspark` 会触发 vllm 主干方法误判，务必用干净名） |

### 2.3 卡位约定（16 卡分半，formal100 布局）
- **卡 0-7（ASCEND_RT_VISIBLE_DEVICES=0..7）**：FSDP 训练（`torch.distributed.run --nproc-per-node 8`）
- **卡 8-15（VISIBLE=8..15）**：HS 提取端点（`scripts/launch_vllm.py train`，端口 8000，
  30 个 API server，`ExampleHiddenStatesConnector` 写 `--hidden-states-path`）
- 两者物理隔离；**任何一个阶段启动前必须确认另一半的进程状态，僵尸 VLLMWorker 会泄漏 ~57GB/卡**

---

## 3. 脚本调用方法（全部在 40 机上执行）

### 3.0 进入容器
```bash
ssh root@10.246.63.40
docker exec -it vllm_ascend_v0.30.0 bash
cd /vllm-workspace/speculators
```

### 3.1 训练（v15 正在跑；重启用同一命令）
```bash
# 宿主机上：
docker exec -d vllm_ascend_v0.30.0 bash -c \
  'nohup bash /vllm-workspace/speculators/glm-mtp/direct_train/40_train.sh \
   > /mnt/weight2/l00840580-direct-v13/logs/train_wrapper.log 2>&1'
# 容器内等价：bash /vllm-workspace/speculators/glm-mtp/direct_train/40_train.sh
# 配置全部在 direct_train/env.sh：DATA_PATH / NUM_SPEC_STEPS=5 / MAX_STEPS=2310 /
#   SPECULATORS_MTP_DISTILL_ALPHA=0.9 / SAVE_PATH=.../checkpoints_v15
```

### 3.2 HS 端点（训练前必须就绪）
```bash
docker exec -d vllm_ascend_v0.30.0 bash -c 'cd /vllm-workspace/speculators && \
  ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 OMP_NUM_THREADS=1 \
  HCCL_WHITELIST_DISABLE=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  python3 scripts/launch_vllm.py train /mnt/weight2/GLM-5.3-W4A8 \
  --hidden-states-path /mnt/weight2/l00840580-direct-v13/hidden_states \
  --trust-remote-code -- --tensor-parallel-size 8 --port 8000 --max-num-seqs 4 \
  --max-model-len 4096 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.90 \
  --quantization ascend --no-enable-prefix-caching --enforce-eager \
  > /mnt/weight2/l00840580-direct-v13/logs/hs_endpoint_v15.log 2>&1'
# 就绪判据：curl -s http://127.0.0.1:8000/v1/models 返回模型列表（约 8-10 分钟）
```

### 3.3 进度监控
```bash
# 步数/速度/对齐度/错误
docker exec vllm_ascend_v0.30.0 bash -c \
  'grep -oE "global_step=[0-9]+" /mnt/weight2/l00840580-direct-v13/logs/train_v13.log | tail -1;
   grep -oE "profile/step_ms=[0-9.e+]+" /mnt/weight2/l00840580-direct-v13/logs/train_v13.log | tail -1;
   grep -oE "align_step_0=[0-9.]+" /mnt/weight2/l00840580-direct-v13/logs/train_v13.log | sed -n 1p;
   grep -oE "align_step_0=[0-9.]+" /mnt/weight2/l00840580-direct-v13/logs/train_v13.log | tail -1;
   grep -cE "Traceback" /mnt/weight2/l00840580-direct-v13/logs/train_v13.log'
# 判读：align_step_0 应当随训练上升（TV 蒸馏生效的直接证据）；v13(CE) 是下降的
```

### 3.4 导出 → serving → A/B 评测（训练出 checkpoint 后）
```bash
# 1) 导出（--requant-w4a8 必需；float 写回接受率≈0）
docker exec vllm_ascend_v0.30.0 bash -c 'cd /vllm-workspace/speculators && \
  python3 glm-mtp/tools/export_glm_mtp_for_vllm.py \
  --checkpoint /mnt/weight2/l00840580-direct-v13/checkpoints_v15/0 \
  --output /mnt/weight2/l00840580-direct-v13/hybrid_B_v15 \
  --native-source /mnt/share_space/l00840580/dspark-glm53/drafter_init/glm53-mtp-native-zpnone \
  --target-model /mnt/weight2/GLM-5.3-W4A8 --requant-w4a8'
cp /mnt/weight2/GLM-5.3-W4A8/quant_model_weights.safetensors.index.json \
   /mnt/weight2/l00840580-direct-v13/hybrid_B_v15/   # 导出器不拷 index，必须手补

# 2) preflight（<1 分钟，无卡校验）
bash /vllm-workspace/speculators/glm-mtp/tests/preflight_eval.sh \
  /mnt/weight2/l00840580-direct-v13/hybrid_B_v15

# 3) serving（先杀端点释放 8-15 或确认 0-7 干净；卡 0-7 TP8）
docker exec -d vllm_ascend_v0.30.0 bash -c \
  'MODEL=/mnt/weight2/l00840580-direct-v13/hybrid_B_v15 SERVE_LOG_SUFFIX=B15 \
   bash /tmp/serve_40.sh'     # /tmp/serve_40.sh 已在容器内；不在则见 §4 重建

# 4) 评测（就绪判据：serve 日志出现 "Application startup complete"）
docker exec vllm_ascend_v0.30.0 bash -c 'cd /vllm-workspace/speculators && \
  python3 glm-mtp/eval_ab.py \
  /mnt/share_space/l00840580/dspark-glm53/eval_prompts_train32_ids.jsonl \
  /mnt/weight2/l00840580-direct-v13/logs/eval_B_v15.json'
# 对照基线：A=2.551（hybrid_A_init 同法可复测）；> A 即蒸馏训练有效
```

### 3.5 测试与验证
```bash
# 全套回归（hybrid 自包含/NaN/训练增量/eval sanity/serving contract）
bash /vllm-workspace/speculators/glm-mtp/tests/run_tests.sh
# 诊断（verifier-argmax vs 数据 token 重合率；本机 8 卡即可跑）
python3 /vllm-workspace/speculators/glm-mtp/direct_train/diag_verifier_vs_data.py \
  <data_path> /mnt/weight2/GLM-5.3-W4A8 http://127.0.0.1:8100/v1 <out.json> 32
```

---

## 4. 已知故障与处置（踩过的坑，按概率排序）

| 症状 | 根因 | 处置 |
|---|---|---|
| 训练 rank 报 `hcclCommInitRootInfoConfig ... EI0015` | 上一个进程被杀后设备未释放完（竞态） | 等 2-3 分钟重启；预防：杀进程后 sleep 120 |
| `npu-smi` 显示卡占用 ~57GB 但进程表有 `VLLMWorker_TP`（宿主 PID） | 被杀 vllm 的 worker 僵尸泄漏 HBM | 按 npu-smi 进程表 PID 在宿主机 `kill -9`；卡 0-7 与端点 8-15 的 PID 表要分开看 |
| 训练 OOM "only 4.6 GiB allocated" | 同上（僵尸占卡） | 同上 |
| resume 报 `Missing optimizer state for parameter 'verifier_norm.weight'` | 换 loss/模型结构后从旧 checkpoint 恢复 | 换新 SAVE_PATH 或加 `--no-resume-from-checkpoint` |
| `--speculative-config` 引号报错 | ssh→docker→bash 多层引号被吃 | 用脚本文件（`/tmp/serve_40.sh`）传递，勿内联 |
| eval 报 Connection refused | 评测抢在端口监听前 | 确认 serve 日志 "Application startup complete" 后再跑 |
| 数据路径含 `dspark` 触发 vllm 方法误判（0.28/0.30 栈） | speculative.py 子串匹配 | 用干净软链（glm53-eval/...）；导出目录名避开 `dspark` |
| 40 容器内 python 路径 | 容器是 3.12.16（非 3.12.13） | `python3` 即可，env.sh 的 PYTHON_BIN 在 40 上不用 |

---

## 5. 换机启动清单（新机器从零复刻）

1. **代码**：`git clone -b glm-mtp5-v2 https://github.com/ababyshark/speculators && cd speculators`
   （或 `git pull` 已有 clone；训练机容器内跑 `pip install -e . --no-deps`）
2. **容器**：特权容器 + 完整 /mnt 挂载 + driver 三件套（参照 `mk_evalab.sh` 模板，repo 里
   `glm-mtp/tests/` 无需卡即可验证）；镜像用 `quay.io/ascend/vllm-ascend:v0.28.0.rc1-a3`
   （评测）或 `nightly-releases-v0.30.0-a3`（训练，40 在用）；transformers 需 ≥5.15
   （本机 5.16.1 可整套拷贝：site-packages 下 transformers + tokenizers 两个目录）
3. **资产**：verifier `/mnt/weight2/GLM-5.3-W4A8`（NFS /data3 全网可见）；数据
   `data_prep_rot_10x`（NFS）；drafter_init（NFS）
4. **顺序**：preflight(`10_preflight_data.sh`) → HS 端点(卡 8-15) → 训练(卡 0-7，
   `40_train.sh`) → 导出 → preflight → serving → 评测 → 与 §1.5 表格对照
5. **配置**：`direct_train/env.sh` 是唯一参数入口（数据路径/步数/α/卡位/保存路径）

---

## 6. 评测协议（保证可比性，勿改动）

- 数据：`eval_prompts_train32_ids.jsonl`（32 条训练样本输入，token 级一致）
- 参数：`max_tokens=64, temperature=0.0, seed=1024`；MTP×5；TP8；贪心
- 指标：整段请求前后抓 `/metrics` 计数器做差（`pos_accept_rates[k] = accepted[k]/draft_steps`，
  `mean_accept_len = 1 + accepted_total/draft_steps`）
- 对照：**必须用同一条导出路径的 A/B**（不同 requant 路径的数字不可直接比）
- sanity：`tests/test_eval_sanity.py`（JSON 合法性/正值/范围）

---

## 7. 结果解读指南（截至 v14 的规律）

- pos0 ≈ 单步对齐：TV 蒸馏训练中已见上升（0.737→0.758）
- pos1-4 = 递归接受：v14 单步训练下恶化 → v15 用 steps=5 递归训练修这一点
- 平均接受长度 vs 基线 A=2.551：**>A 即蒸馏训练有效**；追 native 2.763 需再修 requant
  路径损失（双重量化 ~0.2，独立工程问题）
- 若 v15 有效：下一步选项 = 数据再加倍 / 步数继续 / β 衰减调参 / 把 86.7% 挪高
  （verifier 侧量化优化，长期）
