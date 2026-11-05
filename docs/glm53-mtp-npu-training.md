# GLM-5.3 (GlmMoeDsa) MTP=5 增训：NPU 适配与性能修复记录

> 目标：对 GLM-5.3-W4A8（`GlmMoeDsaForCausalLM`，78 层 MoE + DSA indexer + 自带 MTP 层 `model.layers.78`）的**原生 MTP 层做 MTP=5 继续训练**（增训），训练产物可写回/挂载到 W4A8 serving 上跑投机解码。
>
> 环境：Atlas A3 16 卡（torch 2.10 + torch_npu 2.10.post4、CANN 9.1.0、 vllm-ascend v0.28.0.rc1），speculators MTP 增训框架。 speculators 的 MTP 路线此前只支持 Qwen 系（全可导 decoder）；GLM 的 DSA+MoE 组合是首次接入。

## 一、问题 → 思路 → 方法 → 效果（12 项修复）

### 1. 架构注册缺失

- **问题**：`mtp_model_classes` 只注册了 qwen3/qwen3_next/qwen3_5 系；transformers 的 `modeling_glm_moe_dsa` 没有 MTP 模块类。
- **方法**：新增 `GlmMoeDsaMTPLayer(MTPLayerMixin, GlmMoeDsaDecoderLayer)`。要点：
  - MTP 层（layer 78）是 **sparse MoE + full DSA indexer**，而 `mlp_layer_types`/ `indexer_types` 只覆盖主干 78 层 → 构造时对 `copy.copy(config)` 打补丁 （`mlp_types[0]="sparse"`、`indexer_types[0]="full"`）；
  - mixin 提供 `hidden_layernorm/token_layernorm/input_proj/final_layernorm` （对应 checkpoint 的 hnorm/enorm/eh_proj/shared_head.norm）；
  - `base_components.model_classes` 同步注册（first_layer/decoder/norm/rotary）。

### 2. warm-start 权重链路（W4A8 → BF16）

- **问题**：layer 78 权重是 ModelSlim W4A8/W8A8 量化格式（weight + weight_scale + weight_offset + scale_bias），且布局是 DeepSeek 风格（`model.layers.78.*`）而非 speculators 期望的 `mtp.*`。
- **方法**（`examples/glm53_mtp/extract_mtp_layer.py`）：
  - 反量化实测约定：无 scale → FLOAT 直拷；weight.shape[0]==scale.shape[0] → int8 逐通道 `W=q×scale`；weight.shape[0]×2==scale.shape[0] → int4 双打包 （dim0、低半字节在前）`W=unpack(q)×scale`；**offset 实测全零**； **scale_bias 不进权重**（±20 量级的逐通道加性偏置在残差流上函数不成立， 判定为融合 kernel 侧补偿项）；
  - 键映射：eh_proj→input_proj、enorm→token_layernorm、hnorm→hidden_layernorm、 shared_head.norm→final_layernorm、self_attn.*/mlp.* 直映；layer78 自己的 embed_tokens 与 shared_head.head 丢弃（drafter 用 verifier 主 embed/lm_head）；
  - 后处理：256 专家的 per-expert Linear 融合成 HF 分组参数 `experts.gate_up_proj [E,2I,H]` / `experts.down_proj [E,H,I]`；
  - vendor bias-fix 的 norm `.bias` 全零 → 丢弃（HF RMSNorm 无 bias）。
- **效果**：24GB BF16 drafter warm-start；CPU 冒烟 forward+loss 通过。

### 3. 分片索引兜底

- **问题**：verifier 索引名是非标准的 `quant_model_weights.safetensors.index.json`， `load_model_layers` 只认 `model.safetensors(.index.json)`。
- **方法**：`utils/loading.py` 回退到目录内任意 `*.safetensors.index.json`。

### 4. NPU 遥测崩溃

- **问题**：`torch.accelerator.memory_allocated()` 在 torch_npu 下触发 allocator INTERNAL ASSERT（Timer 每步调用）。
- **方法**：`trainer.py` Timer/\_mem 全部 try-except，探针改 `torch_npu.npu.*` 直连。

### 5. FSDP2 分片粒度（通信瓶颈）

- **问题**：整层 9.7B 参数是一个 all-gather 单元 → 单笔 18.5GiB 通信缓冲。
- **方法**：细粒度包裹——experts 子模块独立成 FSDP 单元，且 `reshard_after_forward=False`（分块 MoE 每层调用 16 次，若每次 unshard/reshard 19.3GB，NPU 全程等 HCCL）。
- **效果**：all-gather 每层调用 1 次常驻复用。

### 6. HCCL 通信器数量上限

- **问题**：过多 FSDP 子包裹 → 每单元新建通信器 → `hcclCommInitRootInfoConfig` error 7。
- **方法**：子包裹收敛到 experts 一个单元（layer + root + experts 共 3 个）。

### 7. MoE 计算路径（最大性能项，~300×）

- **问题演进**（AICore 全程 0% 的调查结论）：
  1. HF 朴素实现：256 次 Python 循环 + `one_hot` + `nonzero()`（每层一次 D2H 同步
     - 布尔索引）→ 步级分钟数；
  2. 稠密 einsum 版：`einsum('th,eih->tei')` 在 NPU 下沉到低效 kernel （微基准 20 TFLOPS，约 7% MFU；非连续 bmm backward 更差）→ 118s/batch （fwd 7.5s + **bwd 128s**，13.7× 病态）；
  3. `npu_grouped_matmul`：**无 autograd kernel**（实测 x.grad/w.grad=None）， 训练不可直接用（serving 专用）。
- **最终方法**：按专家排序的稀疏 dispatch—— `argsort(flat_expert)` + `bincount/cumsum`（**单次 D2H 同步**取 offsets）+ 每专家切片 GEMM（`xe @ w1[e].t()` 等，计算量为稠密的 K/E=1/32）+ `index_add_` 散射 + shared_experts 手工展开（`to_local()` 取 FSDP unshard 全量参数）。
- **效果**：fwd 7.5s→**312ms**，bwd 128s→**856ms**，batch 级 ~300×；AICore 从 全程 0% 转为正常满载。

### 8. 大词表 CE 病态 kernel（py-spy 定位 82.5% 热点）

- **问题**：`F.cross_entropy` 在 154880 词表上占步时 82.5%（py-spy record 实测）， 且中间物化导致 OOM（~1.9GiB/步）。
- **方法**：自定义 `_VocabCE`（autograd Function）：
  - forward：`logsumexp(logits.float()) - logits[target]`（原生 fast reduction）；
  - backward：`(softmax - onehot) × mask`（纯 elementwise kernel）；
  - 外层套 activation checkpointing：logits/CE 中间量（~10GB）前向即释放、 backward 重算（lm_head matmul 代价低），梯度语义不变。

### 9. 零梯度参数的 resume

- **问题**：DSA indexer 的 top-k 选择不可导 → `wq_b/wk/weights_proj` 恒零梯度 → AdamW 惰性建态跳过它们 → checkpoint 缺其优化器状态；保存宽容、resume 严格 → 续训硬崩。这是"保存宽容/加载严格"的不对称，Qwen 系全可导所以从未暴露。
- **方法**：`checkpointer.py` 零动量填充（DCP FQN 路径 + 单卡路径各一）： 按 param 顺序补 `{step, exp_avg=0, exp_avg_sq=0}`——与 epoch 1 中它们从未被 更新的实际行为完全一致。

### 10. 优化器状态/主权重精度

- **问题**：checkpoint 保存时优化器动量随模型一起转 bf16（动量精度损失跨 resume 累积）；bf16 主权重会吞 1e-5 量级的 AdamW 更新。
- **方法**：优化器状态保持 fp32 保存/加载；模型以 `dtype=torch.float32` 加载 （fp32 主权重），FSDP2 MixedPrecisionPolicy 在 unshard 时转 bf16 计算、 reduce 用 fp32。
- **代价**：epoch checkpoint 从 56GB → ~100GB；主机内存峰值 16×48GB（A3 节点 3TB RAM 实测可容纳）。

### 11. 显存 OOM 链

- `--total-seq-len` 对齐真实样本长度分布（p99=1810 → 2048），避免 padding 浪费；
- CE 中间量 checkpointing（见 #8）；
- experts 常驻 all-gather（见 #5）；
- 状态字典广播后 `empty_cache()` 清碎片。
- **效果**：fp32 下稳定运行（贴边 64-65GB/65.5GB）。

### 12. 运维杂项

- `pkill -f` 自匹配：exec 命令行含同样字串会杀死自己 → pattern 拆串； vllm worker 用 setproctitle 改名，清理需按 `VLLM::`/`EngineCore` 前缀。
- 容器 overlay 99% 满 → tar 解压静默残缺，解压后必须 grep 验证关键文件。
- 镜像 CANN 与宿主驱动兼容性：2 卡 HCCL allreduce 5 分钟判定（见 README）。

## 二、方法论

1. **最小复现探针**：每个怀疑点配一个独立小脚本，5 分钟内判定—— 2 卡 HCCL allreduce（镜像/驱动兼容性）、单算子 autograd 探针 （grouped_matmul 不可反传）、MoE 微基准（`bench_moe*.py`：einsum 1.484s vs matmul 变体 vs 稀疏 dispatch）。
2. **py-spy 采样**（record 40s + dump --native）：把"慢"归因到具体 Python 帧 （82.5% cross_entropy）与 native 等待（HCCL/idle 区分）。
3. **框架自带指标**：trainer 的 `profile/fwd_ms|bwd_ms|step_ms|fetch_ms` 直接 给出前后对比，避免凭感觉。
4. **先正确后快速**：每个优化以"loss 轨迹与基准一致"为门槛（warm-start 起点 ~13、逐步下降），微基准只决定"做不做"，训练日志决定"对不对"。

## 三、当前效果（截至本文档）

| 指标                              | 修复前         | 修复后            |
| --------------------------------- | -------------- | ----------------- |
| batch 耗时（2048 seq × 16 ranks） | ~128s+         | **~0.4-1.2s**     |
| fwd / bwd                         | 7.5s / 128s    | **312ms / 856ms** |
| AICore                            | 全程 ~0%       | 正常满载          |
| 4 epochs 总时长                   | ~15h（不可行） | **\<1h**          |

- 数据：2052 条 on-policy（UltraChat prompt + GLM-5.3 自生成回答，assistant 段 loss mask），最后一层 hidden states 离线预抽取（`extract_hidden_states` 模式）。
- loss（fp32 run）：13.66 → 11.46（epoch 1 内持续下降），warm-start 生效 （随机初始化应为 ~60+：5 步 × ln(154880)≈11.95）。
- epoch 1 checkpoint 已通过健康检查（finite、权重 diff 确认更新、 speculative_tokens=5）。
- 评测工具就绪：200 条留出 prompt、`/metrics` 每位置接受率测量、 baseline（自带 MTP）与 trained（`--spec-model` 直挂 BF16 drafter，免写回） 双 serve 脚本、layer-78 BF16 写回备用脚本。

## 四、已知限制

1. DSA indexer 与 MoE router 在 HF 参考实现中不可导，这些参数在增训中不更新 （零动量填充只保证 resume 能跑，不改变"不学习"的事实）。
2. fp32 主权重使单个 epoch checkpoint ~100GB（模型 bf16 19.9GB + 动量 fp32 79.6GB）。
3. 训练数据为快照（2052 条）；扩数据需重跑数据生成 + `generate-offline-data` 抽取阶段（~15 分钟/千条）。
4. 稀疏 dispatch 每层调用有一次 D2H 同步（offsets），层数深时流水线有停顿窗口； 如需极致吞吐可换 npu_grouped_matmul + 自定义 backward（wgrad 分组归约待实现）。

## 五、复现

见 `examples/glm53_mtp/`：数据生成（`gen_onpolicy.py`）、分词 （`tokenize_onpolicy.py`）、warm-start 抽取（`extract_mtp_layer.py`）、 训练（`train_t5.sh`，16 卡 FSDP）、评测（`prep_eval_prompts.py`/ `measure_acceptance.py`/`serve_eval_*.sh`）、备用写回（`writeback_layer78.py`）。

## 六、已知性能问题：NPU 占用率低（待改进，2026-11-04 记录）

### 现象（v5/v6/v7 三次运行一致，非回归）

| 指标                             | 实测                                     | 期望     |
| -------------------------------- | ---------------------------------------- | -------- |
| step 耗时（2048 seq × 16 ranks） | ~29-32s（fwd 12.4-14.4s + bwd 15.8-18s） | 0.4-1.2s |
| 吞吐                             | 35.5-38 tok/s                            | —        |
| AICore                           | ~35%（间歇空转）                         | 满载     |

### 根因：每专家 Python 循环 launch-bound

- 稀疏 dispatch 对 256 个专家做 Python 循环，每次迭代 ~6 个 kernel（index_select/2×GEMM/chunk/mul/index_add）→ 每次 MoE 调用 ~1500 次 launch；
- 每步 ~15 次 MoE 调用（5 预测步 × (fwd + checkpoint 重算) + backward）→ ~2 万次 launch/步；
- NPU kernel 本身极快（微基准 fwd 312ms），CPU 发射开销主导 → AICore 在 launch 间隙空转。
- 注：文档前文的 312ms/856ms 为 dispatch kernel 微基准数字；端到端步时包含上述 launch 开销。

### 已否决的替代路径

1. 稠密 einsum：NPU 沉到低效 kernel（20 TFLOPS），118s/batch，bwd 病态 128s——否决；
2. `npu_grouped_matmul`：无 autograd（x.grad/w.grad=None），训练不可直接用；
3. padded-bmm（按专家分桶 + 单 bmm）：微基准 ~900×，但两次实机运行均触发 device fault 507035（疑似路由偏斜时 max_c 失控的越界/巨块分配）——否决，根因未查清前不碰。

### 改进候选（按性价比排序，独立工作项）

1. **npu_grouped_matmul + 自定义 autograd Function**（wgrad 分组归约）：kernel 已存在，只缺反传；fwd/bwd 各一次 grouped 调用，launch 数 ~O(1)；
2. **分块 padded-bmm + 安全护栏**：专家分组（如 64/组）、组内 max_c 上限（超限回退逐专家路径）、stable argsort、空组跳过；先在单卡微基准复现并定位 507035（ASCEND_LAUNCH_BLOCKING=1）后再上训练；
3. **torch.compile（torch_npu 后端）包住 dispatch 循环**：若 dispatch 图可 capture，launch 开销可大幅压缩（需验证 torch_npu 编译支持度）；
4. **多流并行**：256 个切片 GEMM 按专家无依赖，可多流发射掩盖 launch 延迟。

### 决策记录

v5/v6/v7 的目标均为"功能正确 + 验收信号"（30s/步可接受，4 epochs ~3.5h）；性能专项在拿到验收基线后单独立项，避免再次陷入"边训边改"循环。
