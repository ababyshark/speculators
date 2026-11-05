# GLM-5.3 原生 MTP 在线增训方案 v2

状态：待审视，尚未按本方案启动训练。

日期：2026-11-03（Asia/Shanghai）

## 1. 方案基准

本版以 Speculators 官方 `examples/train/mtp_qwen3_5_9b_gsm8k_online.sh` 为流程基准：

1. 使用 `python scripts/launch_vllm.py` 启动 verifier 服务。
2. 服务按 `--target-layer-ids` 提供在线 hidden states。
3. 训练使用 `--on-missing generate`，按 batch 请求 hidden states，不预生成整个数据集的 hidden-state 文件。
4. 训练入口只指定 verifier，不指定独立 DRAFT 初始化目录。
5. 训练结束使用 `speculators stitch-mtp` 把训练后的 MTP 参数合回 verifier checkpoint。

官方脚本明确说明：MTP 微调从模型原生 MTP head 开始，训练初始化阶段自动从 verifier 提取 native MTP weights，不需要单独的转换步骤。

## 2. 路径与设备

| 项目 | 路径/配置 |
|---|---|
| verifier | `/data1/weight/GLM-5.3-w4a8-spec` |
| 原始 JSONL | `/data1/spec_train_data/train_10k.jsonl` |
| 8K prepared 数据 | `/data1/spec_train_data/prepared_8k_dflash2_official` |
| Speculators | `/workspace/speculators` |
| 训练脚本目录 | `/workspace/code2` |
| 容器 | `vllm_030_mtp_train` |
| 教师 NPU | 0–7，TP=8 |
| 训练 NPU | 8–15，FSDP 分片 |
| 教师端口 | 8000 |
| 推理回归端口 | 8010 |
| HCCL 超时 | `HCCL_CONNECT_TIMEOUT=3600` |

## 3. 教师服务和 hidden states

`--provenance-dir` 只用于启动审计和复现，保存启动配置、版本/权重索引摘要和相关 provenance 文件；它不参与 forward，也不改变 hidden states。官方参考脚本本身不要求该参数，因此当前 v2 服务脚本默认不传它；如果需要审计，可由外层启动器把命令、环境和日志保存到单独目录。

教师服务改为官方入口形式：

```bash
cd /workspace/speculators
python scripts/launch_vllm.py \
  /data1/weight/GLM-5.3-w4a8-spec \
  --target-layer-ids 78 \
  -- \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --quantization ascend \
  --kv-cache-dtype auto \
  --no-enable-prefix-caching
```

GLM 的 `target-layer-ids=78` 对应主模型最后一层输出，作为 MTP 的 teacher hidden state。普通 `launch_vllm.py` 在服务请求期间提取该层 hidden states，训练端通过 endpoint 在线消费；这里不是提前为整个数据集生成静态特征。若当前容器中的 `launch_vllm.py --help` 显示参数略有差异，应以该版本帮助为准。

服务启动后必须检查：

- `GET /health` 返回 200；
- 请求一次短序列后，训练端能够从 endpoint 收到 target layer 78 hidden states；
- 服务日志记录模型路径、target layer 78 和 vLLM 参数；
- 同一输入重复请求的 hidden states 在排序扩展启用后逐位一致。

## 4. 数据准备

原始回复已经由 GLM-5.3 生成，满足 MTP 数据与目标模型一致的前提。优先复用已经审计过的 8K 数据；如果需要从原始 JSONL 重建，使用模型自己的 render endpoint：

```bash
cd /workspace/speculators
speculators prepare-data \
  --model /data1/weight/GLM-5.3-w4a8-spec \
  --data /data1/spec_train_data/train_10k.jsonl \
  --render-endpoint http://127.0.0.1:8000 \
  --output /data1/spec_train_data/prepared_8k_glm_mtp \
  --seq-length 8192
```

转换后要复验样本数、token 上限、loss mask、会话边界和训练/验证隔离。当前已审计数据为 22,070 条，训练 19,875 条，验证 2,195 条。

## 5. DRAFT 初始权重的修正

旧脚本中的：

```bash
DRAFT=/data1/mtp_training/weights/glm53_mtp_speculators_step0
--from-pretrained "$DRAFT"
```

来自此前自定义 GLM 适配流程：先从原生 `model.layers.78.*` 提取并映射成 Speculators 的 `mtp_layers.0.*`，再以转换目录作为训练初始化。它是一个中间格式 checkpoint，优点是方便当时的单步 GLM 适配加载，缺点是偏离官方在线 MTP 流程，并且容易把“原生 MTP 权重”和“转换后训练权重”混淆。

正式目标是让训练器直接读取 verifier 原生 MTP 权重。当前训练器的 GLM 适配仍需要 Speculators key layout，因此过渡实现由启动脚本从 verifier 自动生成一次临时初始化目录；它不再依赖旧的固定 DRAFT，也不需要人工预先转换：

```bash
python /workspace/code2/convert_glm_mtp.py \
  --source /data1/weight/GLM-5.3-w4a8-spec \
  --teacher /data1/weight/GLM-5.3-w4a8-spec \
  --output /data1/mtp_training/runs/<run>/native_init

--verifier-name-or-path /data1/weight/GLM-5.3-w4a8-spec
--speculator-type mtp
--from-pretrained /data1/mtp_training/runs/<run>/native_init
```

该临时目录完全由原始 verifier 生成，训练前会输出 native MTP key 数量、shape、dtype、共享 embedding/head 来源和 SHA-256 摘要。后续 native loader 完成后，再移除这一步。

## 6. 训练配置

用户建议将 `num-speculative-steps` 调整为 5。本版把它解释为每个训练样本的 MTP 预测深度，不是训练更新次数：

```bash
--speculator-type mtp \
--num-speculative-steps 5 \
--target-layer-ids 78 \
--step-weight-beta 0.6 \
--on-missing generate \
--vllm-endpoint http://127.0.0.1:8000/v1 \
--total-seq-len 8192 \
--lr 1e-5 \
--optimizer adamw \
--deterministic-algorithms
```

第一阶段建议只做 1 个 optimizer update 的 5-step smoke test，再做 5 个 update 的短跑，确认每个预测步的 loss、梯度和参数更新都有限；通过后再决定完整 epoch 的步数。不能把 `num-speculative-steps=5` 直接套到当前只实现单步的 GLM 代码上。

## 7. 必须补齐的 GLM 五步代码

当前本地 GLM 适配已补入五步递归草案，但仍需要 NPU parity 验证：

```python
forward_glm_mtp(..., num_speculative_steps=5)
```

实现仍只调用 GLM 原生的一层 `model.mtp_layers[0]`，通过 teacher-forced 递归展开五步。因此在启动 5-step 正式训练前需要完成：

1. 对当前递归展开与 vLLM 原生 MTP 推理逐位置对齐；如果 vLLM 实际采用不同的 KV/位置状态复用，必须按 vLLM 语义修正，不能直接把当前草案视为 parity 通过。
2. 为每个 speculative step 构造正确的 token embedding、position ids、hidden 输入和 label shift。
3. 对 5 个 step 分别计算 vocabulary loss 和 token accuracy，并按 `step-weight-beta` 汇总。
4. 在 step 之间复用或重建符合 GLM DSA/MLA 语义的状态；不能把单步 dense causal mask 直接重复五次视作等价实现。
5. 对跨会话边界、padding、position reset 和 loss mask 做逐步截断检查。
6. 让 checkpoint 和 `stitch-mtp` 能保存/恢复五步配置、optimizer、scheduler、RNG 与原生 key 映射。

## 8. 显存和 OOM 方案

保留之前已经验证的两项修复：

- FSDP 细粒度包装 attention、router、shared expert 和每个 routed expert，避免整个 GLM MTP 层一次 all-gather；
- vocabulary logits 使用 64-token chunk-loss，并对 loss chunk 使用 non-reentrant checkpoint。

五步展开会增加激活和状态开销。容量测试顺序：1-step/512 tokens → 5-step/512 tokens → 5-step/2K → 5-step/8K。若 8K OOM，先将 chunk-loss 从 64 调到 32，再到 16，每次只改一个变量并重新做单步容量验证。

## 9. 训练后 stitch 与推理验证

```bash
speculators stitch-mtp \
  /data1/mtp_training/runs/glm53_mtp_online_v2/checkpoints/checkpoint_best \
  /data1/weight/GLM-5.3-w4a8-spec \
  --output-path /data1/mtp_training/runs/glm53_mtp_online_v2/stitched
```

推理服务必须使用与训练深度一致的配置：

```bash
vllm serve /data1/mtp_training/runs/glm53_mtp_online_v2/stitched \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --tensor-parallel-size 8 \
  --quantization ascend \
  --trust-remote-code
```

验收至少包括：服务加载、零步 logits/top-1、5 个位置的接受率、平均接受长度、输出文本一致性、吞吐和峰值显存。先与原始 verifier 做 greedy 对照，再比较增训前后 MTP。

## 10. 阶段门禁

| 阶段 | 必须通过的条件 |
|---|---|
| 教师服务 | health 200，layer 78 hidden states 可生成、可消费，启动参数和日志完整 |
| 原生初始化 | 不使用 DRAFT；native MTP key/shape/dtype/hash 检查通过 |
| 单步 parity | 原生 MTP 与训练 forward 的 hidden/logits 误差在阈值内 |
| 五步实现 | 每步 loss/梯度有限，step 1–5 label shift 和 mask 正确 |
| 容量 | 5-step 8K 单步和连续短跑无 OOM |
| checkpoint | stitch 后可加载，权重变化范围符合预期，恢复状态可审计 |
| 服务回归 | 5-token speculative service 正常，接受率和输出正确性可复现 |

