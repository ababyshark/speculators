# GLM-5.3 原生 MTP 增训执行记录

更新：2026-11-02。服务器 `172.18.32.23`，容器 `vllm_030_mtp_train`。

## 数据路径

训练数据：`/data1/mtp_training/data/prepared_8k_dflash2_official`

这是从 `.22:/data1/spec_train_data/prepared_8k_dflash2_official` 复制的通用 token IDs / loss mask 数据。原始 JSONL 在 `.22:/data1/spec_train_data/train_10k.jsonl`。

全量复验：22,070 条，训练 19,875、验证 2,195；最大 8192 tokens；非法 token/长度/mask 样本为 0；训练验证会话交集为 0，split 连续排列。训练使用 `train_ratio=0.9005663797009515`，确保切分点仍为 19,875。

使用在线教师特征，不提前生成整个数据集的 hidden states。此阶段每个 batch 只用一条样本；适配器能裁掉右侧 padding，并明确拒绝跨会话打包。

## 已完成与尚未完成

- 从原始量化 checkpoint 提取 793 个原生浮点 MTP 张量，逐张量检查有限性和保存回读 SHA-256。
- 转换为 Speculators checkpoint，791 个非词表张量写入新分片，独立 embedding/head 从原生提取目录读取。793 个张量经过实际模型加载后逐位哈希一致，保留 BF16/F32 原精度。
- MTP embedding 与主模型相同；MTP head 不同，不能替换成主模型 head。
- CPU 和 NPU 小模型组件测试通过：分块 indexer top-k、专家前向/输入及 router 梯度、首位置 embedding 屏蔽、indexer 冻结、RoPE 维度和 norm epsilon。CPU 追加 padding loss 一致及打包边界拒绝测试通过。
- 实际完整 MTP 权重短序列 NPU 前向通过，峰值分配约 22.118 GiB。输入为合成特征，loss 仅为冒烟测试值，不能用于判断训练效果或模型接受率。
- `.23` 教师服务在端口 8000 启动，健康检查 HTTP 200。
- **已完成 512、2K、8K 的单步容量验证，并完成 512、2K、8K 的连续 20 步试跑。** 细粒度 GLM FSDP 分片后，三种长度均可反向；8K 20 步 checkpoint、恢复验证均已通过。当前以该 checkpoint 为起点运行 100 步正式增训。

## 重要路径

| 用途 | 容器路径 |
| --- | --- |
| 原始权重 | `/data1/weight/GLM-5.3-w4a8-spec` |
| 原生 MTP 提取 | `/data1/mtp_training/weights/glm53_mtp_native_step0` |
| Speculators 转换产物 | `/data1/mtp_training/weights/glm53_mtp_speculators_step0` |
| 隔离 Python 环境 | `/data1/mtp_training/venv` |
| Speculators 源码 | `/workspace/speculators` |
| GLM 适配模块 | `/workspace/speculators/src/speculators/models/mtp/glm_mtp.py` |
| 启动与验证脚本 | `/workspace/code_mtp` |
| 日志 | `/workspace/code_mtp/logs` |
| 数据校验报告 | `/data1/mtp_training/provenance/dataset_audit.json` |
| 教师在线缓存 | `/data1/mtp_training/teacher/hidden_states` |

## 脚本与使用范围

- `extract_native_mtp.py`：提取原生 MTP，仅复制，不修改原始权重。
- `install_glm_mtp_adapter.py`：将 GLM 组件接入现有 MTP 配置、模型和数据入口；要求先把 `glm_mtp.py` 放入对应源码路径。
- `convert_glm_mtp.py`：对已提取权重做名称映射、shape/哈希校验和 Speculators 配置保存；输出目录必须不存在。
- `test_glm_mtp_components.py`：组件和边界检查，不能代替原生 vLLM 对齐。
- `smoke_glm_mtp_full.py`：实际全量权重加载哈希与短序列 NPU 前向；通过 `ASCEND_RT_VISIBLE_DEVICES=8` 限定设备。
- `audit_glm_mtp_data.py`：全量数据格式、split 与会话隔离复验。
- `start_glm_mtp_teacher.sh`：TP8 教师使用 NPU 0–7，包含 `HCCL_CONNECT_TIMEOUT=3600` 和 provenance 输出目录。
- `export_glm_mtp_for_vllm.py`：把训练 checkpoint 的 `mtp_layers.0.*` 反向映射为 vLLM 原生 `model.layers.78.*` 布局；补回训练 checkpoint 省略的 MTP embedding/head，使用分开的 `mtp_shared.safetensors`，并通过 `model.safetensors.index.json` 合并加载。已用 checkpoint `0` 完成 `trained=791 + shared=2` 的导出和映射校验，产物在 `/data1/mtp_training/runs/glm53_mtp_formal100_8k/exports/pilot0_vllm`。
- `glm_mtp_teacher_extension.py`：复用 SFA 排序扩展，将 capture 78 替换为实际主模型最终 norm 后的输出，匹配原生 MTP 输入。该扩展只由 MTP 专用启动脚本启用。

本地脚本保存在 `remote_scripts/`，同名文件便于审阅。原始提取脚本及报告另行同步。

## 已完成的训练试跑

- 512 token 单步：loss 2.076，验证 loss 1.217，token accuracy 0.788，checkpoint 约 112 GB。
- 512 token 连续 20 步：loss 从 1.970 降至中段 0.158，验证 loss 0.343，token accuracy 0.917；无 OOM、无 recovery error。
- 2K 单步：loss 0.601，验证 loss 0.385，token accuracy 0.893，checkpoint 约 112 GB。
- 2K 连续 20 步：loss 从 0.601 降至中段 0.247，验证 loss 0.340，token accuracy 0.892；无 OOM、无 recovery error。
- 8K 单步：loss 0.503，验证 loss 0.495，token accuracy 0.840，单步约 174 秒，其中在线特征获取约 160 秒；checkpoint 约 112 GB。
- 8K 连续 20 步：训练最终记录到 global step 19，最后一步 loss 0.591、token accuracy 0.821；验证 loss 0.416373、token accuracy 0.854651；峰值显存约 23.4 GB，无 OOM、无错误记录。
- 8K checkpoint 恢复：从 `global_step=20` 恢复模型、optimizer、scheduler 和 8 个 rank RNG，续训到 global step 21；验证 loss 0.417，无 OOM 或 traceback。恢复时 sampler 重放 20 个 batch，属于当前 sampler 能力限制。
- 正式增训：`/data1/mtp_training/runs/glm53_mtp_formal100_8k`。日志记录了 100 次有效参数更新（最后记录 `global_step=99`，loss=0.380），随后验证 loss=0.369、token accuracy=0.868；模型 shard 已写出。分布式进程在最终收尾阶段发生 SIGBUS，导致 `training_state.json` 和部分最终元数据仍显示 step20。独立参数审计显示 785/791 个训练张量发生变化，共 9,943,472,853 个元素变化；该事实和 NPU 加载冒烟验证记录在远端 `final100_verified.json`。因此可确认模型权重完成 100 次有效更新，但不能声称 optimizer/RNG 已以 step100 元数据保存。

## 当前容量诊断与 chunk-loss

- GLM MTP loss 已改为按 64 token 分块计算词表交叉熵，并对每个 loss chunk 使用 non-reentrant checkpoint；这减少了 `[seq, vocab]` logits 的峰值，但没有解决本次主要 OOM。
- 若更长序列再次出现 OOM，优先将 loss chunk 从 64 依次降为 32、16；每次调整后先做单步容量验证，再恢复连续训练。该调整只改变显存峰值和计算耗时，不改变 loss 定义。
- 若正式长跑需要按步保留中间 checkpoint，应改用小于 1 的 `checkpoint_freq`（按 epoch 比例计算）并关闭 `save_best`，否则当前训练器只在 epoch/验证边界保存。
- 2K 试跑首次失败是截断后 `loss_mask[:, 2:]` 为空；过滤窗口内无监督目标的样本后，前向和目标构造可继续。
- 初始 512/2K 试跑的 OOM 来自整层 FSDP all-gather；已改为先分片 attention、router、shared expert 和每个 routed expert，再包装外层 MTP layer。细粒度 FSDP 后 512/2K/8K 均可反向。
- GLM MTP 层已加非重入 activation checkpoint；词表 loss 使用 64 token 分块和 non-reentrant checkpoint。

## 实现约束与修复记录

1. 只支持单预测步。多步的 index/KV 复用需另行做原生对齐，不允许直接调大 speculative steps。
2. GLM 使用完整 indexer 和稀疏 MoE；RoPE 显式使用 `qk_rope_head_dim=64`，MLA norm 使用 checkpoint 的 `rms_norm_eps=1e-5`。
3. 专家保持独立模块，并按固定顺序调用；GLM 已使用 attention/router/shared expert/routed expert 的细粒度 FSDP 分片，并完成 8K 单步反向验证。
4. 现有 Speculators MTP `shift_batch_mtp` 读取 `lengths`，但 collator 已移除该字段；这是继承源码中的接口不匹配。现改为可选传递，并保留 document_ids 做边界检查。
5. 新适配在 meta 权重加载后发生 indexer 的 `requires_grad` 恢复为 True；实际权重加载测试发现后，已在加载结束显式重设冻结。它属于新适配的加载流程问题。
6. 环境使用 NumPy 2.2.6 与 PyArrow 26.0.0，符合当前仓库 NumPy>=2 要求；只安装在 `/data1/mtp_training/venv`，全局环境未改。

## 正式 100 步训练结果（2026-11-03）

- 正式命令已完成 100 个训练步，训练进程正常退出；最后阶段无 OOM、Traceback、教师请求错误或 `error_records`。
- 训练日志：`/data1/mtp_training/runs/glm53_mtp_formal100_8k/launcher.log`。
- 最终训练日志最后一步：`loss=0.380`、`token_accuracy=0.887`；固定验证 batch：`val_loss=0.369`、`val_token_accuracy=0.868`、`supervised_tokens=108.5`。
- 长序列峰值显存保持约 23.2–23.4 GB。当前 64-token chunk-loss 足够，不需要缩小；如果后续更长序列 OOM，按 `64 → 32 → 16` 逐级降低 chunk 大小，每次先做单步容量验证。
- 训练器的 `save-best` 将训练后的模型和优化器覆盖写入 `checkpoints/0`，但保留了该目录早期的 `training_state.json`/`val_metrics.json` 元数据。因此以模型文件修改时间、正式日志和独立权重哈希为准；最终模型张量已与 step0 权重抽样哈希不一致，确认确实发生了更新。
- 这次收尾失败的直接证据是 `launcher.log` 中 2026-11-02 20:00:20 UTC 的 rank0 `SIGBUS`，其余 rank 随后被 launcher 终止；内核日志显示当时 XFS 写回线程和训练进程都在等待磁盘写回。当前 `/data1` 可用空间约 152 GB，因此后续续训或导出前应先清理可恢复的临时产物，并把最终 checkpoint 元数据单独写入新文件，避免覆盖已有可用权重。
- 最终 vLLM 导出：`/data1/mtp_training/runs/glm53_mtp_formal100_8k/exports/final100_vllm`。导出报告显示 791 个训练张量 + 2 个共享张量，共 793 个键，与原生 GLM MTP 索引一致。
- 最终 checkpoint 已在 NPU 上独立加载并完成短序列前向：793 个张量加载成功，`synthetic_short_loss=11.7751`，峰值分配约 22.12 GiB；结果记录于 `/workspace/code_mtp/logs/final100_checkpoint_smoke.json`。

后续需要：启动独立的最终 MTP vLLM 服务，完成 checkpoint 加载、固定 prompts 的接受率/平均接受长度/正确性/吞吐对比，并补做最终 checkpoint 的恢复续训验证。
