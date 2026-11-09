# RFC: GLM-5.3 MTP5 投机解码 Drafter 训练与部署（实现级规格）

> **文档用途**：本文档是 GLM MTP5 工作的**规范性规格**与 agent 实现的唯一事实源。 预期读者：实现 agent / 评审者。实现 agent 在**不参考任何现有实现代码**的前提下， 仅凭本文档应能完整复现该系统。本文档不含具体代码，至多伪代码。
>
> **状态**：v3 草案（正向设计视角：动机 → 原理 → 方案对比 → 端到端 → 规格 → 验收； 由实现反推的遗留缺陷以 §12"目标语义修正"与 §13"工程经验规则"沉淀） **需求语义**：MUST / SHOULD / MAY 按 RFC 2119 解释。需求编号（`REQ-*`）、 决策（`ADR-*`）、验收 Gate（`G-A*`）、测试（`T-*`）、缺陷修正（`DFX-*`）、 工程经验（`E-*`）、开放问题（`OPEN-*`）均可被逐条引用。 **代码基线**：`glm-mtp5-v2` 分支（含 16K 现场修复 R1–R8）。日期：2026-11-09。
>
> **附录导航**：附录 A（§19）= 知识依赖地图与阅读路径（人读）；附录 B（§20）= 特性分解与 agent 开发指引（agent 按 B.1 特性树拆解、按 B.2 依赖图排序、按 B.3/B.4 验收）。

______________________________________________________________________

## 0. 术语与约定

| 术语                  | 定义                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| **verifier / target** | GLM-5.3 主模型（`GlmMoeDsaForCausalLM`），W4A8 量化部署，训练全程冻结                           |
| **drafter / MTP 层**  | 第 78 层原生 MTP 头（enorm/hnorm/eh_proj/DSA DecoderLayer/final norm + 独立词表），唯一训练对象 |
| **MAL**               | Mean Accepted Length，平均接受长度；本工作的主指标                                              |
| **HS**                | hidden states，verifier 第 78 层输出隐状态                                                      |
| **K**                 | 投机步数（`num_speculative_steps` / `speculative_tokens`），目标 K=5                            |
| **V**                 | valid_len，各步共用的监督窗口宽度，V = L − K − 1（L 为序列长）                                  |
| **post_norm**         | 捕获语义：HS 取第 78 层 final-norm **之后**的输出（GLM 正确语义）                               |
| **requant**           | 训练浮点权重按 W4A8 **存储布局**回写量化包的过程                                                |
| **边训边推**          | 训练组与 teacher 服务组共存同一节点集，训练按需向在线 teacher 请求缺失 HS（§4.3）               |

符号约定：`H ∈ R^{1×L×d}` verifier HS；`x ∈ N^{1×L}` token 序列；`m ∈ {0,1}^{1×L}` loss_mask（1=监督）； `E(·)` drafter 词表 embedding；`lm_head_v(·)` verifier 冻结共享 lm_head；步索引 `t ∈ [0,K)`；位置索引 `j ∈ [0,V)`。

______________________________________________________________________

## 1. 动机

1. GLM-5.3 = 78 层主模型（MoE + DSA 稀疏注意力 + MLA + lightning indexer）+ 第 78 层原生 MTP 头（793 张量，其中 2 个为 MTP 独立词表张量）。
2. 生产部署为 vllm-ascend + W4A8（Ascend ModelSlim）+ TP8。量化后 MTP 头接受率有损：native float MAL **2.763** → requant 路径 step0 基线 **2.551**（32 样本贪心 MTP×5 实测）。
3. 训练出的浮点 MTP 权重**不能**直接写回量化包（float 直写 acceptance ≈ 0.000，服务端反量化与存储布局强耦合）。
4. 官方 `speculators stitch-mtp` 与 GLM 路径不兼容。
5. 因此需要：**冻结主模型、仅微调 MTP 层、按存储布局 requant 回写**的完整训练-导出-服务链路。

______________________________________________________________________

## 2. 原理：MTP 微调在做什么

> 本节回答"为什么这样做有效"。实现 agent 必须理解本节，因为 §9 的每条规格 都能追溯到这里的原理；偏离原理的"优化"一律先改 RFC 再改实现。

### 2.1 原生 MTP 机制与递归语义

MTP（Multi-Token Prediction）在主模型最后挂一个小型预测头：输入 = 主模型末层输出 的上下文表征 `H` + 下一 token 的 embedding，输出 = 再下一个 token 的分布。 GLM 的 MTP 头与主 decoder 层**同构**（同一种 DSA 注意力 + MoE），但参数独立， 且规模上等价于一个 decoder 层——这就是"drafter 便宜、verifier 昂贵"的投机结构基础。

递归语义（K 步投机）：同一层被复用 K 次——

```
step t 的输入:  hidden_t (上下文表征) + token_{j+t+1} 的 embedding
step t 的输出:  hidden_{t+1}，用于预测 token_{j+t+2}，并作为 step t+1 的 hidden
```

三条关键性质（本文档所有设计的根源）：

- **性质 ①（上下文已压缩）**：`H` 已包含主模型对前缀的全部语义编码。drafter 不需要 重学语言，只需学"在这个压缩态上，下一 token 是什么"的修正函数。这就是为什么 冻结主模型、只训一层、小 LR、短周期（1 epoch）就够——与预训练的本质区别（§3.1）。
- **性质 ②（token 链决定信息边界）**：step t 的预测条件 = verifier 上下文（≤j）+ token 链 (j+1 .. j+t+1)。训练时 token 链来自**真值**（teacher forcing）；推理时来自**drafter 自己的生成**。两者的分布差异（exposure bias）是 K>1 训练的核心张力（§2.4）。
- **性质 ③（词表独立但语义必须对齐）**：MTP 头的 embed_tokens/lm_head 与主模型**不同** （793 张量中的 2 张量，sha256 可证）。训练中冻结它们 = 冻结"drafter↔verifier 的语义 对接面"；训练破坏对接面（如让 drafter 词表漂移）会直接摧毁接受率。

### 2.2 微调信号链：数据如何"蒸馏"为训练信号

一条训练样本的信号生成链（两次转化）：

```
语料 →[转化① 数据级]→ input_ids + loss_mask（哪些位置值得监督）+ 预 tokenized Arrow
     →[verifier 前向]→ H（post_norm 上下文压缩态）
     →[转化② 信号级, 训练时在线]→ 两路监督:
         (a) 直接监督:  drafter(H[j], E[x[j+1]]) 拟合条件分布, CE on x[j+2]
         (b) 蒸馏监督:  lm_head_v(H[j+t+1]) = verifier 的下一 token 分布, TV 拟合
```

**为什么必须要有 (b)**：数据 token 每个 j 只有一份（one-hot），而 drafter 的任务不是 "复述语料"，是"在任何上下文上猜中 verifier 的 argmax 并保序"。v13 实验的原理性教训： verifier argmax 与数据 token 的重合率只有 **86.7%**——纯 CE-on-data 在 13.3% 的分歧位 把 drafter 推离 verifier，出现"loss 降、接受率同降"的目标错位。(b) 把监督源从"语料" 换成"verifier"，这才是与部署指标同向的目标。

### 2.3 蒸馏目标与部署指标的对齐

- 投机解码的收益分解：`MAL = 1 + Σ_{t=1..K} P(前 t 步 argmax 全对)`。pos1 接受率 主导 MAL（基线实测 0.756 → 0.106 的深度衰减），深度项依赖递归分布的正确性（§2.4）。
- drafter 接受一步 ⇔ 其 argmax 命中 verifier argmax。TV 距离是分布一致性的总度量 （对任意事件 |p−q| ≤ TV）：压 TV ⇒ 逐 token 概率贴近 verifier ⇒ argmax 分歧概率下降。 DSpark 历史配方给出经验同源关系 `acceptance ≈ 1 − TV`，故本工作直接以 `α·TV + (1−α)·CE`（α=0.9）为主目标——训练目标与部署指标同向（ADR-5）。
- 健康度证据链：训练中 `align_step_t`（drafter↔verifier argmax 一致率）应单调上升 （v14 实测 0.737→0.758）；MAL 应跟随上升。**loss 下降而 align 不升 = 目标错位信号**， MUST 触发排查（G-A2 的异常判据之一）。

### 2.4 训练动力学要点

- **teacher forcing vs 自回归 rollout**：全真值输入训练稳定，但推理时输入是自己生成的 token 链——K 越大偏移越大。两个对策（均为 env 开关，§9.8）：
  - `agree_filter`：仅在"数据 token == verifier argmax"的位置监督。原理：这些位置上 真值链与 verifier 链重合，训练覆盖的上下文类型 ⊆ 推理会遇到的类型（v17 教训 A_float=2.908：teacher forcing 数据 token 优化了服务永不访问的上下文）。
  - `teacher_force_verifier`：输入 token 链也换成 verifier argmax 链，彻底模拟推理分布； 代价是脱离数据分布，故默认关闭（ADR-10），由 A/B 裁决（OPEN-3）。
- **固定 V**：所有步共用 V = L−K−1 的窗口，保证各步张量形状一致（编译稳定 + native 对齐）；代价是每步放弃尾部 K+1 个位置的监督（数据级可接受）。
- **步权 β^t**：`w_t ∝ β^t`（β=0.6，FastMTP Eq.2）。原理：第 t 步的 MAL 贡献被前 t−1 步 的联合命中概率衰减（0.756^t 量级），深层步天然低价值，权重应同步衰减。
- **冻结面动力学**：只训一层 + 冻结词表/indexer，LR 1e-5 + 1 epoch——防小数据上的 蒸馏崩塌（drafter 学会"模仿 verifier 的 logit 温度"而非 argmax 结构）。

______________________________________________________________________

## 3. 与预训练、DSpark 训练的区别

> 正向设计中必须明确"这不是什么"，防止实现 agent 把预训练/其他 drafter 路线的 习惯带进来（packing、全参数训练、大 LR、语料 CE 等）。

### 3.1 与预训练的区别

| 维度         | 预训练                      | MTP 微调（本方案）                                                           |
| ------------ | --------------------------- | ---------------------------------------------------------------------------- |
| 输入表征     | 原始 token embedding        | verifier 上下文 HS（已编码语义，性质①）                                      |
| 监督目标     | 语料 next-token CE          | 蒸馏 TV（主）+ 受限 CE（辅）                                                 |
| 数据形态     | 原始语料                    | 预 tokenized + loss_mask 标注的 Arrow（蒸馏管线产物）                        |
| 可训练参数   | 全模型                      | 单层 MTP（785/791 张量）；词表/indexer/verifier_norm 冻结                    |
| 训练规模     | 万亿 token / 大 LR / 长周期 | 1 epoch / 2310 步 / LR 1e-5                                                  |
| 序列组织     | packing 拼接                | **禁止 packing**（单样本，REQ-RT-2）                                         |
| 评测         | benchmark loss              | MAL / 在线 A/B 接受率                                                        |
| 典型失败模式 | 不收敛/过拟合               | **目标错位**（loss 降接受率降，v13）、exposure bias（K>1）、requant 路径损失 |

实现含义：数据加载器必须原样透传 loss_mask；训练循环不得引入语料式 CE 兜底； 任何"提升 loss"的改动必须同时看 align/MAL 是否同向。

### 3.2 与 DSpark 训练的区别

DSpark 是本库的另一条 drafter 路线（外置 block-drafter，Eagle3 生态）。两者**数据契约 同源**（`input_ids/loss_mask/seq_len` Arrow，历史配比 0.1CE+0.9TV 同源），但架构与 服务面不同：

| 维度         | DSpark                                                                                    | GLM MTP5（本方案）                                      |
| ------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| drafter 架构 | 外置 block-drafter：sequential Markov head + confidence head（低秩 logit-bias），多块堆叠 | **原生 MTP 层复用**（与主层同构的单层）                 |
| 推理接口     | Eagle3 式（SupportsEagle3），vLLM `method=dspark`                                         | `method=mtp`（AscendDeepSeekMTP，GLM 与 DeepSeek 同构） |
| 初始化       | rot-space warm start / z-lab b16 warm starts                                              | native MTP 权重直提（step0 即有 MAL 2.551 的底子）      |
| 损失族       | anchor / consistency / Markov-bias 项 + TV                                                | α·TV + (1−α)·CE + agree/teacher-force 开关              |
| 量化衔接     | 独立适配                                                                                  | 原生 MTP 布局 + ModelSlim remap 成熟（§10）             |
| 适用边界     | 无原生 MTP 头的模型                                                                       | 有原生 MTP 头的模型：服务对齐成本最低、step0 即基线     |

**为什么选原生 MTP 路线**：GLM-5.3 自带与 DeepSeek 同构的 MTP 头，服务端有现成的 量化 remap 与 proposer 实现；DSpark 路线的 warm-start 优势（从零训练 drafter）在本 场景不存在。保留 DSpark 为备选（ADR-3）。

**工程坑（已沉淀为 E-9）**：vLLM 按路径子串自动检测 draft 方法——评测/导出目录名 MUST 避开 `dspark`/`dflash` 等子串，否则 `method=mtp` 被静默覆盖。

______________________________________________________________________

## 4. 端到端方案

### 4.1 总体架构

```
┌─────────────────────────── 数据面 ────────────────────────────────────┐
│ 语料 → 预tokenized Arrow (input_ids/loss_mask/seq_len)                 │
│ GLM-5.3-W4A8 verifier (vLLM teacher, 冻结, TP8/TP16)                   │
│   └─ capture hook: 第 78 层 final-norm 后 HS (post_norm, manifest)     │
│        → hs_{i}.safetensors (bf16) 断点续填                            │
└──────────────────────────────┬─────────────────────────────────────────┘
                               ▼  (离线为主; 在线按需补缺, §4.3)
┌─────────────────────────── 训练面（speculators 库）────────────────────┐
│ MTPSpeculatorConfig(glm_moe_dsa) → MTPDraftModel                       │
│   ├─ mtp_layers.0 (唯一可训练层, §8)                                   │
│   ├─ 冻结: embed_tokens / lm_head / indexer / verifier_norm            │
│   └─ forward: teacher-forced K 步递推 (§9.2)                           │
│ loss = α·TV + (1−α)·CE, 步权 β^t (§9.4)                                │
│ 运行时: FSDP2 细粒度分片 + NO_PACK 单样本 + AC(约束 §9.9)              │
└──────────────────────────────┬─────────────────────────────────────────┘
                               ▼
┌─────────────────────────── 导出面 ─────────────────────────────────────┐
│ export: mtp_layers.0.* → model.layers.78.* 逆映射 (§10.1)              │
│ requant-w4a8: 按 W4A8 存储布局回写 (§10.2)                             │
│ + mtp_shared.safetensors (2 个独立词表张量)                             │
└──────────────────────────────┬─────────────────────────────────────────┘
                               ▼
┌─────────────────────────── 服务面（vllm-ascend）───────────────────────┐
│ vllm serve --speculative-config {"method":"mtp","num_speculative_tokens":5} │
│ 原生 AscendDeepSeekMTP + 量化 remap 补丁 (§10.3)                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 数据蒸馏转化链（工程层，§2.2 原理的落地）

```
语料 →[① 预tokenization + loss_mask 标注]→ Arrow 数据集
     →[② teacher 捕获]→ hs_{i}.safetensors (bf16 H + token_ids)
     →[③ 质量门]→ 可训练数据
```

- 捕获语义 MUST 为 post_norm（final-norm 后），捕获进程 MUST 产出 manifest （stage 声明、weight_sha256、层索引交叉校验、文件哈希）。
- 质量门四件套（MUST 保留为工具）：数据契约校验（列/长度/词表范围/mask≥1）、 数据审计（行数、session 无重叠、split 连续）、manifest 哈希校验、 **verifier argmax 重合率诊断**（输出数据 token vs verifier argmax 的重合率—— 该数字是目标设计的直接输入，86.7% 曾直接否定纯 CE 方案）。
- HS 落盘断点续填：按记录对齐 `hs_{i}.safetensors`，缺文件可重生成，MUST 幂等。
- 数据规模演进：1x（2052 行，v13/v14）→ 10x（20520 行，v15+）；split 比例 0.9006 连续切分（train/val 互补）。

### 4.3 边训边推（在线 HS 模式）

**拓扑**：同节点集分两组资源——teacher 组起 vLLM endpoint（W4A8 TP8/TP16 eager）， 训练组跑 FSDP 训练（如 16 卡 = 8 teacher 卡 + 8 训练卡）。

**按需生成（--on-missing generate）**：训练 dataloader 发现某记录无 HS 文件 → 向 endpoint 发起生成请求 → HS 实时流式回传（不落盘）→ 直接进 batch。

**降级链（MUST 实现，对应 REQ-RT-4/REQ-FWD-4）**：

```
请求超时(600s) → 重试(×2) → 仍失败 → 连续失败计数(≥20 触发熔断)
     → 空 batch 占位（document_ids 全 −1）→ 零损失步 → 后续恢复
```

**并发约束**：workers × prefetch × ranks 的乘积即瞬时并发请求数（4×4×16=256 曾 引发冷启动风暴）→ 锁超时 1800s（ADR-11）+ 事件循环 future 桥接（E-5）+ 锁等待线程池隔离。

**模式取舍**（选定 = 混合，ADR-2）：

| 模式             | 优势                             | 代价                                              |
| ---------------- | -------------------------------- | ------------------------------------------------- |
| 离线落盘（主）   | 可复现、可校验、可审计、成本可控 | 前置产能需求（teacher 机时）                      |
| 在线生成（兜底） | 无前置产能、缺啥补啥             | 把锁竞争/空 batch/网络失败耦合进训练；HS 不可审计 |
| 混合（选定）     | 离线覆盖大头 + on-missing 兜底   | 需要完整的降级链与观测（fetch_frac 指标）         |

### 4.4 导出-量化-服务闭环

训练产物（浮点 mtp_layers.0）→ 逆映射导出 → **requant 按 W4A8 存储布局回写**（唯一 服务出口）→ vllm-ascend `method=mtp` 服务 → A/B 验收（§11）。float 导出仅用于评测 探针。闭环的每一环 MUST 留 provenance 工件（REQ-RT-7）。

______________________________________________________________________

## 5. 目标与非目标

**目标**

- G1：K=5 teacher-forced 递归训练，主模型权重零参与。
- G2：训练目标与部署指标（接受率）对齐（TV 蒸馏）。
- G3：产物可直接进 vllm-ascend `method=mtp` 服务并过 A/B 验收。
- G4：全链路可复现（provenance 工件）。
- G5：GLM 路线收敛进 speculators 库（通用框架 + GLM 胶水），不留私有 fork 长期维护面。

**非目标**

- 不训练/修改主模型任何权重；不追求通用 packing 支持；不覆盖非 NPU 硬件验证；
- 不实现 DSpark block-drafter 路线（保留为备选，ADR-3）。

______________________________________________________________________

## 6. 指标与验收 Gate（规范性）

| ID   | Gate                          | 规范要求                                                                                                                                                                               | 测量方式                          |
| ---- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| G-A1 | **主验收（MAL）**             | trained 模型 MAL **MUST** > A 基线（重实现 MUST 在新环境用同协议重测 A，历史参考值 2.551）；目标逼近 native 上限（参考 2.763）；A/B 结论 MUST 附 export sha256（防"训练前快照"假验收） | §11.2 在线 A/B，32 样本贪心 MTP×5 |
| G-A2 | **训练 loss**                 | train loss 起点 ~0.50 → 终点 ≤0.40；val ≥0.45 异常；**loss 降而 align 不升 = 目标错位信号，MUST 排查**（§2.3）                                                                         | 训练日志                          |
| G-A3 | **CPU 层级 parity**           | 新实现 MTP 层 vs 基准层实现逐输出 `assert_close(rtol=1e-5)`；权重管线 793→791+2 往返无损                                                                                               | T-PAR-1                           |
| G-A4 | **native parity（go/no-go）** | 训练前向（K=1）vs vLLM native MTP 4096-token 逐张量探针：hidden cos ≥0.9997、logits cos ≥0.99997（以已测值为下限基线）                                                                 | T-PAR-2；不过则 MUST 停止         |
| G-A5 | **容量**                      | 8K 序列 ×20 步无 OOM、loss 下降、恢复后 step 号正确                                                                                                                                    | T-SMOKE-4                         |
| G-A6 | **服务冒烟**                  | 32 prompt HTTP 200；accept rate∈[0,1]；MAL∈[1,6]                                                                                                                                       | T-EVAL-3                          |
| G-A7 | **requant 健康检查**          | trained-requant 与 native-W4A8 的 MAL 差值 MUST 被报告并解释（隔离 requant 路径损失与训练损失）                                                                                        | §11.2 基线表                      |

______________________________________________________________________

## 7. 设计决策记录（ADR）

> 实现方 MUST 按以下决策执行，MUST NOT 重新开挖；变更需走 RFC 修订。

| ID     | 决策                                                                                                        | 备选与权衡                                                                                               | 状态                                  |
| ------ | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| ADR-1  | HS 语义 = **post_norm**（第 78 层 final-norm 后输出）                                                       | pre_norm 二次归一/语义错位——v12 已证伪（2.581→2.573 无效）                                               | 已定，manifest 强制校验               |
| ADR-2  | **离线 HS 为主 + 在线 on-missing 兜底**（混合模式，§4.3）                                                   | 纯在线把锁竞争/空 batch/OOM 耦合进训练；纯离线有前置产能门槛                                             | 已定                                  |
| ADR-3  | **复用 speculators 通用训练框架 + GLM 胶水**（~150 行层组合 + 注册/加载分支 + 权重三脚本）                  | .23 私有框架（813 行 diff）与库演进脱节；DSpark 路线不匹配原生 MTP 结构（§3.2）；transformers 原生栈缺失 | 已定                                  |
| ADR-4  | **自定义 export + requant**，不用 stitch-mtp                                                                | stitch 与 GLM 不兼容；requant 布局控制必须自持                                                           | 已定                                  |
| ADR-5  | **训练目标 = α·TV + (1−α)·CE，α=0.9**                                                                       | CE-on-data 目标错位（§2.2/2.3）；TV 与 acceptance=1−TV 同源；v13/v14 数据支撑                            | 已定；α 随 DFX-2 修正后复核（OPEN-2） |
| ADR-6  | **K=5 递归训练**（v14 单步 pos1-4 恶化指向递归缺失）                                                        | K=1 是唯一 parity 过的配置；K=5 需补 G-A4 级 parity 验证（OPEN-1）                                       | 已定方向，带验证条件                  |
| ADR-7  | **NO_PACK 单样本 batch**                                                                                    | GLM 前向拒绝 packing/位置重置；采样器对称丢弃防 DDP hang                                                 | 已定                                  |
| ADR-8  | **服务面复用 AscendDeepSeekMTP**（GLM MTP 与 DeepSeek 同构）                                                | 自写 proposer 成本高且偏离 vllm-ascend 主线                                                              | 已定                                  |
| ADR-9  | **长上下文 AC 策略**：≤2304 用 `ACTIVATION_CKPT=0` 绕行；16K MUST 保 AC + backport pytorch#171779           | 16K 下关 AC 必死（55.8/61.3 GiB 实测）                                                                   | 已定，REQ-RT-5                        |
| ADR-10 | **agree_filter 默认开启、teacher_force 默认关闭**（v18 组合）                                               | agree_filter 缩小监督面需与 10x 数据交互验证                                                             | 试定，A/B 后复核（OPEN-3）            |
| ADR-11 | **锁超时统一 1800s**（注册默认=全部 fallback）                                                              | 10s 默认在 256 并发冷启动下 771 次超时；fallback 因参数注册机制永不生效                                  | 已定（E-1）                           |
| ADR-12 | transformers `>=5.14,<5.19`                                                                                 | 5.14–5.18 生产验证（7 个导入符号齐全）；5.19+ 未验证                                                     | 已定                                  |
| ADR-13 | **步数解析：显式 CLI 覆盖 draft config 并 MUST 打日志**；from-pretrained 场景 SHOULD 以 checkpoint 值为默认 | 静默覆盖曾使 5 步配置实际跑 1 步（align_step_1..4 全零）                                                 | 已定（E-7）                           |

______________________________________________________________________

## 8. 模型结构规格

### 8.1 REQ-ARCH-1：MTP 层组成（唯一可训练单元）

```
MTPStep(hidden, token_emb, mask, pos):
    token_emb'  = zero_if(pos == 0, token_emb)                  # 位置 0 的 token embedding 置零
    h           = Linear_eh_proj( cat( token_layernorm(token_emb'),   # = native enorm
                                       hidden_layernorm(hidden) ))    # = native hnorm（每步都施加，native 设计）
    h           = self_attn( input_layernorm(h), pos, mask )      # DSA 注意力 + lightning indexer
    h           = post_attention_layernorm(h)
    h           = mlp(h)                                          # MoE：gate + experts + shared experts
    return final_layernorm(h)
```

- `self_attn` 与 `mlp` MUST 复用上游 `GlmMoeDsa` 实现（attention/MLP/RMSNorm/RotaryEmbedding/TopkRouter + `apply_rotary_pos_emb_interleave`），仅做子类封装；MUST NOT 重写算子。
- RMSNorm 计算 MUST 内部 fp32（`float() → rsqrt → weight.float() → cast back`）。
- indexer MUST 冻结且前向 no_grad；topk 索引 MUST 排序后使用（物理页序噪声会破坏 parity）。
- **位置 0 token embedding 置零** MUST 保留（parity 行为，OPEN-4 复核）。

### 8.2 REQ-ARCH-2：参数冻结面（穷举）

| 集合   | 张量                                                                                                                                                                | 处置                                                                                  |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 可训练 | `mtp_layers.0.{token_layernorm, hidden_layernorm, input_proj, input_layernorm, post_attention_layernorm, self_attn.*, mlp.*(gate/experts/shared), final_layernorm}` | 常规训练（formal100 实测 785/791 张量更新）                                           |
| 冻结   | `embed_tokens`、`lm_head`（来自 native `shared_head.head`）、MTP 层内 `indexer.*`、`verifier_norm`                                                                  | `requires_grad=False`；indexer 前向 no_grad                                           |
| 精度   | 全部 RMSNorm weight                                                                                                                                                 | 保持 fp32（keep-in-fp32），不参与低精度 cast                                          |
| 保存   | `embed_tokens`/`lm_head`                                                                                                                                            | MUST NOT 入 checkpoint（`_keys_to_ignore_on_save`），加载时从 native shared head 重建 |

### 8.3 REQ-ARCH-3：权重键映射（native ↔ training ↔ export，双射）

| native（GLM checkpoint）                                                     | training（drafter）               | 说明     |
| ---------------------------------------------------------------------------- | --------------------------------- | -------- |
| `model.layers.78.enorm.*`                                                    | `mtp_layers.0.token_layernorm.*`  |          |
| `model.layers.78.hnorm.*`                                                    | `mtp_layers.0.hidden_layernorm.*` |          |
| `model.layers.78.eh_proj.*`                                                  | `mtp_layers.0.input_proj.*`       |          |
| `model.layers.78.shared_head.norm.*`                                         | `mtp_layers.0.final_layernorm.*`  |          |
| `model.layers.78.{self_attn,mlp,input_layernorm,post_attention_layernorm}.*` | `mtp_layers.0.<同名>.*`           | 后缀恒等 |
| `model.layers.78.embed_tokens.weight`                                        | `embed_tokens.weight`（冻结）     | 793 之外 |
| `model.layers.78.shared_head.head.weight`                                    | `lm_head.weight`（冻结）          | 793 之外 |

- extract：逐专家 → 3D 堆叠变换；convert：恒等复制 + **双射校验**（native 键集 ↔ training 键集，缺失即报错）；export：逆映射 + 补 2 个词表张量到 `mtp_shared.safetensors`。往返 MUST 无损（G-A3）。

### 8.4 REQ-ARCH-4：config 契约

- `MTPSpeculatorConfig.transformer_layer_config.model_type == "glm_moe_dsa"` 时启用 GLM 分支；`num_nextn_predict_layers` MUST 为 1。
- `num_speculative_steps` 的唯一事实源 = `speculators_config.proposal_methods[0].speculative_tokens`。
- 捕获 stage MUST 显式声明（`post_norm`|`pre_norm`）；`post_norm` 训练路径 MUST NOT 再过 `verifier_norm`，未知 stage MUST 拒绝。

______________________________________________________________________

## 9. 训练面规格

### 9.1 REQ-DATA-1：数据契约

| 对象                  | 字段                                                                                    | 形状/类型                                                                     |
| --------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Arrow 数据集          | `input_ids`, `loss_mask`, `seq_len`                                                     | 1D token 序列；loss_mask 1=监督                                               |
| HS 文件               | `hs_{i}.safetensors`: `hidden_states`, `token_ids`                                      | `[L, d]` bf16；`[L]` long；与 Arrow 按记录对齐                                |
| 训练 batch（NO_PACK） | `input_ids`, `verifier_last_hidden_states`, `loss_mask`, `position_ids`, `document_ids` | `[1, L]`；hidden `[1, L, d]`；mask bool；document_ids −1=padding              |
| 空 batch（恢复占位）  | 同上                                                                                    | input_ids 全 0、hidden 全 0、mask 全 False、position 全 0、document_ids 全 −1 |

- `verifier_last_hidden_states` 在 GLM 分支内即 §9.2 的 `H`（post_norm 语义，无二次归一）。
- loss_mask MUST 原样透传到前向（§3.1 的实现含义）。
- split 按连续区间（train/val 互补，比例 0.9006）；val MUST 独立可开关。

### 9.2 REQ-FWD-1：多步递推索引语义（核心规格，实现 MUST 逐位对齐）

对单条 unpacked 序列（batch=1，长度 L）、步数 K：

```
V = L − K − 1                                    # 固定 valid_len，各步形状一致
若 L < K + 2:  → 零损失契约 (REQ-FWD-4)
H_step(0) = H[:, 0:V]                            # verifier HS 切片，不归一化
FOR t in 0 .. K-1:
    tok_t   = E( x[:, t+1 : t+1+V] )             # 源位置 j 的输入 token = x[j+t+1]
    out_t   = MTPStep( H_step(t)[:, 0:V], tok_t, causal_mask(V), pos=position_ids[:, 0:V] )
    label_t = x[:, t+2 : t+2+V]                  # 位置 j 的监督目标 = x[j+t+2]
    valid_t = m[:, t+2 : t+2+V]                  # mask 取"被预测 token"位置
    H_step(t+1) = out_t                          # 递推：上一步输出作为下一步 hidden
```

**工作示例（L=10, K=2, V=7）**：step0 位置 j=(H[j], token j+1) → 预测 token j+2；step1 位置 j=(out0[j], token j+2) → 预测 token j+3。验证公式：任一位置 j 的信息链 = verifier 上下文 ≤j + 真值 token j+1..j+t+1 → 预测 token j+t+2，与 native MTP teacher-forced 语义一致。

- position_ids 各步均取 `[:, 0:V]` **不随 t 推进**（fastmtp 语义）；与 vLLM proposer 的逐步推进语义差异由 `mtp_train_forward="vllm"` 变体承接（REQ-FWD-3）。
- distill 对齐：verifier 参考分布取 `lm_head_v(H[:, t+1 : t+1+V])`——H[j+t+1] 的下一 token 恰为 label j+t+2，与 CE 目标逐位对齐（实现 MUST 用此切片）。

### 9.3 REQ-FWD-2：因果 mask（内存约束版）

- 值语义：`mask[0,0,i,j] = 0 (j≤i)；= finfo(dtype).min (j>i)`，dtype 同 HS。
- 构造 MUST 单缓冲 + 行分块填充（建议 4096 行/chunk）：16K bf16 峰值 ~600MB（双全尺寸缓冲版 ~792MB）；与朴素实现逐元素等价（T-UNIT-5）。

### 9.4 REQ-LOSS-1：损失规格

```
TV_t   = mean_over_valid( 1 − Σ_v min( p_drafter(v), p_verifier(v) ) )   # 单次归一！
CE_t   = sum_over_valid( −log p_drafter(label) ) / count_valid            # 单次归一
step_t = α · TV_t + (1 − α) · CE_t                                        # α=0.9
loss   = Σ_t  w_t · step_t ,   w_t = β^t / Σ_i β^i ,  β=0.6
```

- **归一化 MUST 恰好一次**（全局 valid 计数）：分块实现中每块 MUST 累加"和"，最后除一次总数；MUST NOT 出现"块内均值再除全局"的双重归一（E-2 / DFX-2）。
- TV 的 `p_verifier` MUST 来自冻结 verifier 路径（`lm_head_v(H)` 切片，§9.2），no_grad。
- 温度 `T`（`SPECULATORS_MTP_DISTILL_TEMP`）：MUST 作用于 TV 项两侧 softmax（对称施加），默认 **1.0**；CE 恒为 T=1。`T=2.0` 为待 A/B 候选值（OPEN-5）。实现 MUST 真实接线（E-6）。
- CE 分块 chunk 默认 64，可配（REQ-CLI-3）；每块峰值 = chunk × vocab × 4B。
- 非法值：α∈[0,1]、β>0、chunk≥1，违反 MUST 报错。

### 9.5 REQ-FWD-3：目标变体（mtp_train_forward）

| 变体              | 语义                                                                                                                     | 状态                                                 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| `fastmtp`（默认） | §9.2 语义；K=1 已过 G-A4                                                                                                 | MUST 实现                                            |
| `vllm`            | head 输入 = verifier final-norm HS；draft 步 k 的注意力 = 上下文 + 仅自身草案链；位置 = anchor + k（对齐 vLLM proposer） | K>1 正确性的候选路径，SHOULD 实现（OPEN-1 的一部分） |

### 9.6 REQ-FWD-4：边界与错误语义表（规范性）

| 条件                               | 行为                                                                                                        |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `input_ids` 非 2D 或 batch≠1       | raise（"requires one unpacked sequence"）                                                                   |
| document_ids 全 −1（空恢复 batch） | **返回零损失**（非 raise）：`zero = H.sum(fp32) × 0.0`（保持 DTensor 图连通边），metrics 键齐全（REQ-RT-3） |
| document_ids 存在非右填充/多文档   | raise（"Only right padding" / "Packed documents"）                                                          |
| `L < K + 2`                        | **返回零损失**（同上契约），不 raise                                                                        |
| position 非连续/重置               | raise（"Packed/reset positions are unsupported"）                                                           |
| attention_mask 含非 1              | raise（"requires an unpadded sequence"）                                                                    |
| chunk_size < 1 / steps < 1         | raise                                                                                                       |
| valid_len ≤ 0 或 effective_steps=0 | **返回零损失**                                                                                              |

零损失返回契约：`(logits_list=[], zero, metrics)`；metrics MUST 含 `loss_sum / loss_total(=1) / loss / supervised_tokens` 四键（DFX-7 修正）；trainer 对 `requires_grad=False` 的 loss MUST 跳过 backward。

### 9.7 REQ-OBJ-1：目标开关（环境变量家族）

| 变量                                     | 默认    | 语义                                                                                                                                 |
| ---------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `SPECULATORS_MTP_NO_PACK`                | 0       | =1 时单样本 batch（GLM MUST=1）                                                                                                      |
| `SPECULATORS_MTP_DISTILL_ALPHA`          | 0       | >0 启用 TV 蒸馏分支，值=α                                                                                                            |
| `SPECULATORS_MTP_AGREE_FILTER`           | 0       | α>0 时仅在 `label == verifier_argmax` 位置计算 CE/TV；MUST 输出 `agree_rate_step_{t}` 指标                                           |
| `SPECULATORS_MTP_TEACHER_FORCE_VERIFIER` | 0       | α>0 时递推输入 token 改用 verifier argmax（`argmax(lm_head_v(H[:, t : t+V+1]))`，no_grad 分块）；slot0 喂层、slot1.. 兼作 agree 目标 |
| `SPECULATORS_MTP_CHUNK_SIZE`             | 64      | 分块宽度；显式参数 > env > 默认                                                                                                      |
| `SPECULATORS_MTP_ACTIVATION_CKPT`        | 1       | =0 跳过层 AC（≤2304 绕行用；16K 禁用，ADR-9）                                                                                        |
| `SPECULATORS_MTP_DISTILL_TEMP`           | 1.0     | TV 两侧对称温度（REQ-LOSS-1 接线）                                                                                                   |
| `SPECULATORS_MTP_FREEZE_EXPERTS`         | 0       | 冻结 MoE experts（FSDP 路径调试用）                                                                                                  |
| `SPECULATORS_FSDP_REDUCE_DTYPE`          | float32 | 归约精度；16K 内存杠杆可 bf16                                                                                                        |

### 9.8 REQ-RT-\*：训练运行时

- **REQ-RT-1 FSDP**：MoE 按 self_attn/gate/shared_experts/每 expert 细粒度 shard 再包 layer/model；GLM 属性暴露 `layers` 供 FSDP 包装。
- **REQ-RT-2 采样**：NO_PACK 采样器各 rank batch 数相等、余数对称丢弃、分区不相交完备、seed 确定性、可整 epoch 重洗（T-UNIT-6 断言清单）。
- **REQ-RT-3 空批次/恢复**：teacher 请求失败 → 空 batch → 零损失步（§9.6）；恢复流程 MUST 不产生非法 batch 崩溃；在线模式 MUST 输出 fetch 健康度指标（fetch_frac）。
- **REQ-RT-4 在线 HS 源**：锁等待 MUST 用 future 桥接异步等待（禁止事件循环内轮询，E-5）；锁超时默认 1800s（注册默认=全部 fallback 一致，ADR-11）；降级链按 §4.3。
- **REQ-RT-5 长上下文**：K 步递推 = 同模块多次 forward，嵌套 FSDP2 + 非重入 AC 触发 pytorch#153354；实现 MUST：①提供 `ACTIVATION_CKPT` 开关；②文档化 PR#171779 backport（4 行，容器易失需随 launcher 存档，E-11）；③提供内存杠杆（mask 分块、chunk_size、reduce dtype、分配器参数）。
- **REQ-RT-6 确定性**：`deterministic-algorithms` 开启；同 seed 同结果可复现。
- **REQ-RT-7 provenance**：训练 MUST 落盘 `train_command.txt`（时间戳/git SHA/world size/包版本/完整 argv）与未提交 diff；评测/发车同理（`eval_command.txt` / `vllm_command.txt` / `checkpoint_sha256.txt`），发车 MUST 支持 `--provenance-dir` 且指向关联产物目录。

______________________________________________________________________

## 10. 导出与服务规格

### 10.1 REQ-EXP-1：导出管线

```
trained checkpoint (mtp_layers.0.*)
  → reverse_key 逆映射 (§8.3 表, training→native)
  → float 路线: symlink + mtp_shared.safetensors 补 2 词表张量   # 仅评测用
  → requant-w4a8 路线: requant_writeback (唯一服务出口, ADR-4)
  → conversion_report: 张量分类计数 + 双射校验 + sha256
```

### 10.2 REQ-EXP-2：requant 存储布局（按张量类别）

| 类别                  | 处置                                    |
| --------------------- | --------------------------------------- |
| int4 权重             | 双打包写回 + per-channel scale 原样保留 |
| int8 权重             | per-channel 写回 + scale 保留           |
| offset / scale_bias   | 强制置零                                |
| float 张量（norm 等） | 原样拷贝                                |
| MTP 层外张量          | 不动                                    |

- 服务端量化配置 MUST 配合 remap：`mtp.N.*` 前缀 fallback、`glm5_next_mtp` 的 `.mtp_block.`→`.`。
- float 直写 MUST 被拒绝或显式警告（acceptance≈0 教训）。

### 10.3 REQ-SRV-1：服务契约

- vllm-ascend 原生 `AscendDeepSeekMTP`（`method=mtp`）；`num_speculative_tokens=5`。
- 发车参数基线：TP8、eager、`--quantization ascend`、`--enable-expert-parallel`、max-model-len 4096（评测）；`VLLM_VERSION` 需显式 pin（dev checkout 门禁）。
- 评测 prompt 契约：token-id 列表直传（v0.28+ 契约），MUST 校验 `/metrics` 暴露 per-position spec_decode 计数器。
- **目录命名约束（E-9）**：评测/导出目录名 MUST 避开 `dspark`/`dflash` 等方法名子串（vLLM 按路径子串自动检测 draft 方法，会静默覆盖 `method=mtp`）。

______________________________________________________________________

## 11. 评测与验证方案

### 11.1 离线验证（免服务）

- **drafter 贪心滚 K 步** vs oracle 下一 token：per-position 命中率（teacher-forced 真值）；用于训练中快速迭代。
- **parity 探针**：4096-token 逐张量对比训练前向 vs vLLM native MTP（G-A4）。

### 11.2 在线 A/B（主验收）

```
serve(evalA) → /metrics 快照 → 32 prompts → /metrics 差分 → MAL_A
serve(evalB) → 同上 → MAL_B
判据: MAL_B > MAL_A(2.551)  且  附 export sha256（E-12）
健康检查: MAL_B 与 native(2.763) 的差值 → 归因 requant vs 训练
```

per-position 接受率 = Δacceptance[pos] / Δdraft_steps；MAL = 1 + accepted/drafts。

### 11.3 基线数据（验收对照，HANDOFF 实测）

| 卷                            | MAL   | pos1-5 接受率                 |
| ----------------------------- | ----- | ----------------------------- |
| native（无 requant 上限参考） | 2.763 | 0.756/0.469/0.268/0.163/0.106 |
| A（step0 经导出路径）         | 2.551 | 0.734/0.419/0.226/0.117/0.055 |
| B_v13（CE-on-data，证伪）     | 2.426 | —                             |
| B_v14（TV 单步）              | 2.480 | pos1-4 恶化 → 递归缺失证据    |

______________________________________________________________________

## 12. 遗留缺陷的目标语义修正（DFX）

> 本 RFC 由既有实现反推整理；以下历史缺陷**已按目标语义修正**写入 §9 各规格。 重实现 MUST 采用修正后行为，MUST NOT 复刻历史行为。

| ID     | 遗留缺陷                                 | 修正后目标语义（出处）         |
| ------ | ---------------------------------------- | ------------------------------ |
| DFX-1  | 空 batch 崩溃于误导性 "Packed documents" | §9.6 空批次零损失（E-4）       |
| DFX-2  | TV 双重归一（实际权重 ≈ 名义 1/chunk）   | §9.4 单次归一（E-2）           |
| DFX-3  | `DISTILL_TEMP` 死配置                    | §9.4 对称接线，默认 1.0（E-6） |
| DFX-4  | CLI 步数静默覆盖                         | ADR-13（E-7）                  |
| DFX-5  | 锁超时 10s + fallback 失效               | ADR-11（E-1）                  |
| DFX-6  | 1ms 自旋死锁                             | §9.8 REQ-RT-4（E-5）           |
| DFX-7  | 零损失 metrics 键不全                    | §9.6 四键契约                  |
| DFX-8  | chunk_size 死参数（distill/argmax 未接） | §9.4 全通路                    |
| DFX-9  | mask 双全尺寸缓冲                        | §9.3（E-3）                    |
| DFX-10 | transformers 硬守卫拒 5.14               | ADR-12（E-8）                  |

______________________________________________________________________

## 13. 工程经验规则（E-\*：从现场缺陷提炼的共性规范）

> 由 BUG 修复（结果）提炼的共性规则（过程规范）。实现 MUST 遵守；新增代码评审 按此清单检查。每条给出"教训 → 规则"，并标注来源缺陷。

| ID       | 教训                                                                              | 规范规则                                                                                                                                       |
| -------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **E-1**  | 修改 fallback 默认值无效：参数注册机制使属性恒存在，注册默认（10s）才是唯一生效值 | **默认值单一事实源**：一个参数的默认值只允许定义一处；fallback 与注册默认 MUST 一致；测试断言两者相等                                          |
| **E-2**  | 分块 loss"块内均值再除全局"造成实际权重 ≈ 名义 1/chunk                            | **归一化恰好一次**：分块计算先累加"和"，最后除一次总数；测试必须断言分块实现与朴素实现等值                                                     |
| **E-3**  | mask 构造同时持有 bool[L,L]+zeros[L,L]，16K 峰值 792MB                            | **大张量构造禁止双全尺寸缓冲**：单缓冲 + 分块填充；内存敏感路径 MUST 写明峰值预算                                                              |
| **E-4**  | 空恢复 batch 崩溃于误导性 "Packed documents" 错误                                 | **降级输入是一等公民**：占位/恢复 batch 必须有显式契约（零损失）+ 契约测试；错误信息必须区分"拒绝的输入"（非法）与"降级的输入"（合法但无监督） |
| **E-5**  | 事件循环内 1ms 自旋等待线程池 → 16 rank eventfd 死锁                              | **禁止事件循环内轮询**：跨 future/线程等待一律用桥接原语（wrap_future）让出循环                                                                |
| **E-6**  | `DISTILL_TEMP=2.0` 写在 env.sh 但源码从未读取                                     | **配置必须接线**：任何配置项要么有"证明其生效"的测试，要么删除；dead config = 缺陷                                                             |
| **E-7**  | CLI 显式传参被 draft config 静默覆盖（5 步跑成 1 步）                             | **解析优先级显式化**：覆盖发生时 MUST 打日志（含双方值）；from-pretrained 场景默认值来源 MUST 文档化                                           |
| **E-8**  | 版本硬守卫拒掉了已验证可用的 5.14.1                                               | **特征探测优于版本硬编码**：守卫按"实际用到的 API 是否存在"设计；版本范围仅作提示                                                              |
| **E-9**  | 评测目录名含 `dspark` 子串 → vLLM 静默切换 draft 方法 → 连环崩溃                  | **路径字符串不做语义判断**：自己的代码禁止按路径/名称子串推断行为；对上游此类行为要写命名规避规范（§10.3）                                     |
| **E-10** | 9 个环境变量开关分散各处，曾出现死配置与文档漂移                                  | **开关注册表**：所有 `SPECULATORS_*` 开关在 RFC §9.7 集中登记（名称/默认/语义），实现与文档 MUST 同步，CI 可校验                               |
| **E-11** | torch backport 补丁打在容器内，容器重建即丢                                       | **容器内补丁必须随 launcher 存档**：补丁文件 + 应用脚本 + 备份说明放版本库或 launcher 目录                                                     |
| **E-12** | 曾把训练前快照当 B 卷，A/B 实为 step0-vs-step0，验收无效                          | **验收防同卷对照**：A/B 结论 MUST 附被测权重 sha256；评测流程 MUST 校验加载权重与声明一致                                                      |
| **E-13** | 一次跑挂后无法还原环境与命令                                                      | **provenance 一票否决**：无 provenance 工件的 run 不得作为验收依据（REQ-RT-7）                                                                 |

______________________________________________________________________

## 14. 开放问题（带决策标准）

| ID     | 问题                                                                                               | 决策标准 / 计划                                                                    |
| ------ | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| OPEN-1 | **K>1 递推与 vLLM proposer 的 parity**（position 不推进、递归 hnorm 是否与 native 一致待探针证实） | 补 K=2/K=5 parity 探针（G-A4 同标准）；不过 → 实现 `mtp_train_forward="vllm"` 变体 |
| OPEN-2 | TV 单次归一修正后，α=0.9 的最优值需重标                                                            | 用 §11.1 离线接受率快速扫描 α∈{0.7,0.8,0.9}，A/B 终验                              |
| OPEN-3 | agree_filter（监督面缩小）与 10x 数据的交互                                                        | A/B 对比 agree_filter on/off；看 agree_rate 曲线与 MAL                             |
| OPEN-4 | 位置 0 embedding 置零、递归 hnorm 双归一是否精确复刻 native                                        | 并入 OPEN-1 探针                                                                   |
| OPEN-5 | 蒸馏温度 T=2.0 是否有收益                                                                          | 固定其余超参 A/B                                                                   |
| OPEN-6 | 16K OOM 收尾（十一发待验证；缺口 \<400MiB 时 rollout 级 checkpoint ~750MiB / forward 2x）          | 首步出数 → step_ms×总时长估算 → 决定是否上 R8                                      |
| OPEN-7 | 上游 #1310 AsyncDataLoader（架构级解决死锁域）                                                     | 16K 跑通后二阶段评估 rebase；#1294 metrics off host 可选提速                       |
| OPEN-8 | 多机 loss 聚合为 mean-of-means（非 token 加权）                                                    | 确认偏差可接受或改 sum 聚合                                                        |
| OPEN-9 | adapter 快照与库内 glm_mtp.py 漂移                                                                 | 声明库内为单一事实源，快照归档                                                     |

______________________________________________________________________

## 15. 测试规格（实现 MUST 交付的测试矩阵）

> 每个 T-\* 为验收项；无对应测试的需求视为未实现。

**T-UNIT（CPU 单测，MUST 全绿）**

- T-UNIT-1 capability：glm_moe_dsa → post_norm；pre_norm 恰归一化一次；post_norm 路径不触碰 verifier_norm；未知 stage 拒绝。
- T-UNIT-2 dispatch：K=5 输出 loss_step_0..4；K=1 契约保留；batch 级步数覆盖被忽略；K=1 与单步包装 bit 级一致；K=2..5 各步加权 loss 分解 = 总 loss。
- T-UNIT-3 边界：短序列（\<K+2）零损失不崩；全零 mask 有限零损失；空 batch（document_ids 全 −1）零损失 + 四键 metrics + 图连通（backward 安全）；真 packing 仍 raise；右填充单文档正常。
- T-UNIT-4 loss：chunk_size 解析（显式 > env > 64）且**全通路**（CE/TV/argmax）传参；非法 chunk 报错；trainer kwargs 透传；step_weights 长度校验；**温度接线**（T≠1 改变 TV 输入分布）。
- T-UNIT-5 mask：与朴素实现逐元素相等（L∈{1,8,33,64,130} × fp32/bf16）+ 跨 chunk 等价 + 值语义。
- T-UNIT-6 采样器：等 batch、对称丢余数、不相交完备、确定性、重洗、整除时全覆盖。
- T-UNIT-7 连接器：锁超时默认 1800（注册默认= Transfer/Backend 全部 fallback 一致，E-1）；显式传参仍生效。
- T-UNIT-8 config：序列化/roundtrip；步数解析（显式覆盖 + 日志，E-7）。
- T-UNIT-9 导出：requant 四类布局逐类断言（int4 双包/int8/float 直拷/层外不动）+ report 计数 + 缺源张量拒绝。
- T-UNIT-10 开关接线（E-6/E-10）：每个 `SPECULATORS_MTP_*` 开关至少一条"证明其生效"的断言（如 agree_filter 开启后 valid_eff 缩小且输出 agree_rate）。

**T-PAR（parity，MUST 过 Gate）**

- T-PAR-1 组件 CPU parity：indexer bit 级、RMSNorm fp32 bit-exact、rotary/eps、专家 fwd/bwd、padding loss 精确。
- T-PAR-2 native parity 探针（4096-token）：G-A4 阈值；K=2/K=5 递归探针（OPEN-1）。
- T-PAR-3 权重管线：793 → 791+2 → 793 往返无损（SHA256）。

**T-SMOKE（NPU/容器）**

- T-SMOKE-1 全权重加载 + 短序列 loss 有限；T-SMOKE-2 checkpoint 无 NaN/Inf；T-SMOKE-3 训练增量有界 + 冻结面验证；T-SMOKE-4 容量冒烟（G-A5）。

**T-EVAL（验收）**

- T-EVAL-1 离线接受率流程可跑；T-EVAL-2 A/B 差分正确性（已知基线复现）+ 权重 sha256 校验（E-12）；T-EVAL-3 serving contract + sanity gate（G-A6）。

______________________________________________________________________

## 16. 里程碑（实现顺序与 go/no-go）

| 里程碑 | 内容                            | 出口条件                               |
| ------ | ------------------------------- | -------------------------------------- |
| M0     | 环境/fixture/tiny config        | T-UNIT 全绿                            |
| M1     | 模型结构 + 权重管线             | T-PAR-1 / T-PAR-3 过（G-A3）           |
| M2     | HS 捕获 + manifest + 数据质量门 | manifest 校验全绿；捕获语义 bit-exact  |
| M3     | K=1 前向 + native parity        | **G-A4 过（go/no-go）**                |
| M4     | 训练栈（FSDP/NO_PACK/AC）+ 冒烟 | T-SMOKE 过（G-A5）                     |
| M5     | TV 蒸馏 + K=5 递归 + 全量训练   | G-A2 过；OPEN-1/2 推进                 |
| M6     | 导出 + requant + A/B 验收       | **G-A1 过** + requant 健康检查（G-A7） |

______________________________________________________________________

## 17. 关键约束汇总（横切）

- 代码质量：ruff/mypy/mdformat/provenance-sync pre-commit 全套；新代码零新增错误类别；动态属性访问带显式 ignore 注释；评审按 §13 E-\* 清单检查。
- 版本：transformers `>=5.14,<5.19`；torch 2.10 容器（PR#171779 backport 易失，随 launcher 存档，E-11）。
- 数据合规：训练/评测/发车 provenance 工件齐全（REQ-RT-7 / E-13）；A/B 附 export sha256（E-12）。
- 文档同步：环境变量表（§9.7）与 README/HANDOFF MUST 一致（E-10）；本文档为 GLM MTP5 唯一事实源，实现偏离 MUST 回写本 RFC。

______________________________________________________________________

## 18. 相关工作与参考

- FastMTP: arXiv:2509.18362（步权 Eq.2）
- DeepSeek-V3 Technical Report（原生 MTP 设计；AscendDeepSeekMTP 同构依据）
- DeepSeek-V3.2 DSA（稀疏注意力/indexer 行为基线；topk 排序 parity 教训）
- GLM 技术报告（GlmMoeDsa 架构）
- Leviathan et al. 2023 / Chen et al. 2023（投机解码与 MAL 定义）
- EAGLE-1/2/3（特征层 drafter 谱系）；Medusa（多头并行对照）
- Hinton et al. 2015（蒸馏温度）
- PyTorch #153354 / #171779（FSDP2 DTensor bug 与修复）
- Ascend ModelSlim W4A8 PTQ 文档
- 内部文档：DEVELOPMENT_REQUIREMENTS / GLM53_MTP补丁实现详解 / mtp_dspark_path_analysis / direct_train PLAN+HANDOFF / pytorch_171779 backport 指引 / 16K 现场报告（十连修、R1–R8）

______________________________________________________________________

## 19. 附录 A：知识依赖地图与阅读路径（人读）

> 目的：评审者/新成员在读懂正文前需要哪些背景、读到什么深度、按什么顺序读。 深度分级：★★★ 必须精通（会阻塞实现与评审）／★★ 工作级理解（知道机制与坑）／★ 了解即可。

### A.1 知识依赖清单

| #   | 知识点                        | 一句话解释                                                                                    | 本项目用在哪                            | 深度                        | 材料                                                                   |
| --- | ----------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------- | --------------------------- | ---------------------------------------------------------------------- |
| A1  | 投机解码基础                  | draft-verify-accept 循环；MAL、per-position 接受率定义                                        | §2.3 指标对齐、§6 G-A1、§11 评测        | ★★★                         | Leviathan et al. 2023；Chen et al. 2023                                |
| A2  | MTP 与 FastMTP                | 原生 MTP 头的输入/输出与 K 步递归；步权 β^t（Eq.2）                                           | §2.1 递归语义、§9.2 索引规格、§9.4 步权 | ★★★                         | DeepSeek-V3 技术报告（MTP 一节）；FastMTP arXiv:2509.18362             |
| A3  | 知识蒸馏与 TV 距离            | KD 温度；TV = 逐 token 分布一致性总度量；acceptance ≈ 1−TV                                    | §2.2/2.3 目标设计、§9.4 损失            | ★★★                         | Hinton et al. 2015；TV 定义（任一测度论教材或 DSpark 消融报告）        |
| A4  | GLM-5.3 / GlmMoeDsa 架构      | MoE + DSA 稀疏注意力 + MLA + lightning indexer；第 78 层 MTP 头与 793 张量构成                | §1、§8 模型规格、§4.2 捕获              | ★★★（实现者）／★★（评审者） | GLM 技术报告；transformers `models/glm_moe_dsa/` 源码（作 API 文档读） |
| A5  | W4A8 量化（Ascend ModelSlim） | int4 双包/int8 per-channel/E8M0 scale；量化描述文件；requant 回写语义                         | §1 动机 3、§10.2 布局、§10.1 管线       | ★★（导出实现者 ★★★）        | ModelSlim 官方文档；仓库内 requant 工具说明                            |
| A6  | FSDP2 训练栈                  | DTensor、细粒度分片、非重入 AC、#153354 DTensor bug 与 PR#171779                              | §9.8 运行时、§10 DFX-6、16K 现场        | ★★（训练栈实现者 ★★★）      | PyTorch FSDP2 文档；pytorch#153354/#171779；RFC 引用文档 backport 指引 |
| A7  | vLLM 投机服务（vllm-ascend）  | speculative-config、method 自动检测（子串坑）、AscendDeepSeekMTP、/metrics spec_decode 计数器 | §10.3 服务契约、§11.2 A/B、E-9          | ★★★（服务面）／★★（其余）   | vLLM speculative serving 文档；vllm-ascend 源码 `speculative.py`       |
| A8  | transformers 版本面           | glm_moe_dsa 模块 7 个导入 API 的版本区间                                                      | §8.1 复用约束、ADR-12                   | ★★                          | transformers CHANGELOG + 模块源码                                      |
| A9  | Ascend/NPU 训练环境           | HCCL 超时、torch_npu 分配器、容器拓扑                                                         | §9.8、16K 现场报告                      | ★★                          | CANN/HCCL 文档；现场运维文档                                           |
| A10 | 投机解码谱系（选读）          | EAGLE 特征层 drafter / Medusa 多头——理解本方案为何走原生 MTP                                  | §3.2 方案对比                           | ★                           | EAGLE 1/2/3、Medusa 论文                                               |

### A.2 阅读路径（按角色）

| 角色                    | 路径                                                                        | 时长估计 |
| ----------------------- | --------------------------------------------------------------------------- | -------- |
| **评审者（快速）**      | A1/A2 摘要级 → §1 动机 → §2 原理 → §3 对比 → §4 端到端 → §6 Gate → §11 评测 | ~1h      |
| **训练栈实现者**        | A2/A3/A6 精读 → §2 原理 → §9 全部 → §13 E-\* → §15 测试矩阵                 | ~4h      |
| **量化/导出实现者**     | A4/A5 精读 → §8.3 键映射 → §10 全部 → T-UNIT-9/T-PAR-3                      | ~3h      |
| **服务/评测实现者**     | A1/A7 精读 → §10.3 → §11 → E-9/E-12                                         | ~2h      |
| **数据/捕获实现者**     | A4 精读 → §2.2 信号链 → §4.2 → §9.1                                         | ~2h      |
| **agent（冷启动实现）** | 附录 B 工作流（B.0）为主线，按增量查对应 REQ 与 A 条目                      | 随增量   |

### A.3 内部文档伴侣表（证据与细节，非规范性）

| 文档                         | 回答什么问题                           | 何时读                      |
| ---------------------------- | -------------------------------------- | --------------------------- |
| DEVELOPMENT_REQUIREMENTS.md  | Gate 的原始出处、.23 方案 B 的完整需求 | 评审 §6 时对照              |
| GLM53_MTP补丁实现详解.md     | 793 张量管线的实现史与 parity 教训     | 实现 F2 时                  |
| direct_train PLAN/HANDOFF    | v12→v15 实验链、全部基线数字与证伪证据 | 评审 §2.3/§11.3、复现实验时 |
| mtp_dspark_path_analysis.md  | DSpark 路线与方法误判坑的细节          | 实现 F6、规避 E-9 时        |
| pytorch_171779 backport 指引 | 16K AC 修复的完整操作                  | 16K 训练前                  |
| 16K 现场报告（十连修/R1–R8） | 全部现场缺陷的一手证据                 | 评审 §12/§13 时             |

______________________________________________________________________

## 20. 附录 B：特性分解与 agent 开发指引

> 目的：agent 依据本附录把 RFC 分解为可独立开发、可独立验收的特性增量。

### B.0 agent 工作流（每个增量的固定循环）

```
1. 读 §16 里程碑 → 确定当前增量 (M0..M6)
2. 查 B.1 特性树 → 得到本增量的 Feature 集与对应 REQ-* 清单
3. 仅按 REQ 规格 + 附录 A 对应知识实现（MUST/SHOULD 逐条满足）
4. 交付该 Feature 绑定的 T-* 测试并全绿
5. 过增量出口 Gate → 进入下一增量
6. 任何偏离/新决策/已解决 OPEN → 同一提交内回写 RFC（AGENTS.md 约定）
```

### B.1 特性树（Feature → REQ → 测试 → 出口）

| Feature                | 范围（实现什么）                                                 | 覆盖需求                | 绑定测试             | 出口              |
| ---------------------- | ---------------------------------------------------------------- | ----------------------- | -------------------- | ----------------- |
| **F7.2 测试脚手架**    | tiny config fixture、测试目录骨架                                | §15 全部 T-UNIT 前提    | —                    | M0                |
| **F7.1 CLI/config**    | schema 分组、步数解析规则、--mtp-chunk-size                      | ADR-13、§9.4 chunk 解析 | T-UNIT-8             | M0                |
| **F2.2 config/注册**   | glm_moe_dsa 分支、capability（post_norm 契约）、config roundtrip | REQ-ARCH-4              | T-UNIT-1、T-UNIT-8   | M1                |
| **F2.1 MTP 层组合**    | GlmMTP 子类封装（norm/attn/MoE/indexer 冻结/fp32）               | REQ-ARCH-1、REQ-ARCH-2  | T-PAR-1              | M1                |
| **F2.3 权重三脚本**    | extract/convert/export + 冻结词表加载 + 双射校验                 | REQ-ARCH-3              | T-PAR-3              | M1                |
| **F1.3 数据管线**      | Arrow 数据集 + HS 落盘/断点续填 + batch 组装                     | REQ-DATA-1              | T-UNIT-3（数据部分） | M2                |
| **F1.1 HS 捕获**       | post_norm hook、确定性、SFA 排序扩展                             | REQ-ARCH-4 stage 契约   | T-UNIT-1             | M2                |
| **F1.2 manifest**      | stage/sha256/交叉校验/落盘                                       | §4.2 质量门             | T-UNIT-1             | M2                |
| **F1.4 质量门工具**    | validate/audit/argmax 重合率诊断                                 | §4.2                    | SHOULD（脚本级）     | M2                |
| **F3.1 递推前向**      | K 步索引语义（K=1 先行，K=5 随 M5）                              | REQ-FWD-1               | T-UNIT-2、T-UNIT-3   | M3/M5             |
| **F3.2 因果 mask**     | 分块构造                                                         | REQ-FWD-2               | T-UNIT-5             | M3                |
| **F3.4 边界契约**      | 错误语义表 + 零损失四键                                          | REQ-FWD-4               | T-UNIT-3             | M3                |
| **F3.3 chunked 损失**  | CE/TV/argmax + 温度接线 + 归一化一次                             | REQ-LOSS-1              | T-UNIT-4、T-UNIT-10  | M3/M5             |
| **F6.3 parity 探针**   | K=1 native parity（go/no-go）；K>1 探针                          | G-A4                    | T-PAR-2              | M3/M5             |
| **F4.1 FSDP 封装**     | 细粒度分片、冻结/精度策略                                        | REQ-RT-1                | T-SMOKE-3            | M4                |
| **F4.2 NO_PACK 采样**  | 单样本采样器                                                     | REQ-RT-2                | T-UNIT-6             | M4                |
| **F4.3 在线 HS 源**    | 锁 1800s、wrap_future、降级链、fetch_frac                        | REQ-RT-3、REQ-RT-4      | T-UNIT-7             | M4                |
| **F4.4 AC/内存杠杆**   | ACTIVATION_CKPT、mask/chunk/reduce 杠杆、backport 文档           | REQ-RT-5                | T-SMOKE-4            | M4                |
| **F3.5 目标开关**      | agree_filter/teacher_force + 指标输出                            | REQ-OBJ-1               | T-UNIT-10            | M5                |
| **F4.5 provenance**    | train/eval/vllm 工件                                             | REQ-RT-7                | 流程检查（E-13）     | 全程              |
| **F3.6 vllm 前向变体** | mtp_train_forward=vllm                                           | REQ-FWD-3               | T-PAR-2（K>1）       | OPEN-1 后，SHOULD |
| **F5.1 逆映射导出**    | reverse_key + report                                             | REQ-EXP-1               | T-UNIT-9             | M6                |
| **F5.2 requant 回写**  | 四类布局 + float 拒绝                                            | REQ-EXP-2               | T-UNIT-9             | M6                |
| **F6.1 离线接受率**    | 贪心 rollout 评测                                                | §11.1                   | T-EVAL-1             | M5/M6             |
| **F6.2 A/B 验收**      | serve + metrics 差分 + sanity gate + sha256 校验                 | §11.2、G-A1/A6/A7       | T-EVAL-2/3           | M6                |

### B.2 依赖图（关键路径与可并行面）

```
F7.1/F7.2 (M0 底座) ──┬─→ F2 (模型+权重) ──→ F3.1/F3.2/F3.4 (K=1 前向) ──→ F6.3 [M3 G-A4 gate]
                      │         │
F1 (数据面) ──────────┼─────────┴──→ F3.3 (蒸馏损失) ──→ F3.5 (目标开关) ──┐
                      │                                                    ├─→ M5 全量训练 [G-A2]
F4 (运行时) ──────────┴──→ M4 冒烟 [G-A5] ─────────────────────────────────┘
F2.3 ─→ F5 (导出/量化) ─→ F6.1/F6.2 (评测) ─→ M6 [G-A1/G-A7]
```

- **关键路径**：F2 → F3.1 → F6.3（M3 parity gate）→ F3.3/F3.5 → M5 → F5 → F6.2（M6）。
- **可并行起步**：F1（数据面）∥ F2（模型）∥ F7（底座）三者无相互依赖。
- **可并行跟进**：F4 运行时仅依赖 F2 的模块结构，可与 F3.3 并行开发。

### B.3 增量的完成定义（DoD）

每个增量 MUST 同时满足：

1. B.1 表中该 Feature 的全部 MUST 需求有实现与对应 T-\* 测试全绿；
2. 无新增 ruff/mypy 错误类别（§17）；
3. §13 E-1..E-13 清单逐条自查通过；
4. §9.7 开关注册表已更新（若引入/修改开关，E-10）；
5. 运行类增量附 provenance 工件（E-13）；
6. 与 RFC 的任何偏离已回写（§17 约定）。

### B.4 实现前置条件（agent MUST 向人确认的事项）

| 事项                    | 原因                                         | 默认策略                                         |
| ----------------------- | -------------------------------------------- | ------------------------------------------------ |
| NPU/GPU 可用性与规格    | T-SMOKE、16K 训练不可纯 CPU                  | 无硬件时 M4+ 暂停，CPU 面先行                    |
| teacher 机时（数据面）  | HS 捕获与 10x 数据依赖 vLLM teacher          | 先用 fixture/tiny config 走通管线                |
| 基线重测                | G-A1 的 A/native 数值是环境相关的            | 新环境 MUST 用同协议重测 A 基线后再训（§6 注记） |
| transformers/torch 版本 | ADR-12 与 backport 约束                      | 锁定 §17 版本面，容器补丁存档（E-11）            |
| 上游同步点              | speculators 库为 fork，需定期 merge upstream | 每增量开始时检查上游（OPEN-7 相关）              |

### B.5 风险与回退

| 风险             | 触发信号                         | 回退动作                                                    |
| ---------------- | -------------------------------- | ----------------------------------------------------------- |
| M3 parity 不过   | T-PAR-2 低于 G-A4 阈值           | 停止训练线，回退修层实现（对齐 A4 知识），禁止带伤进入 M4+  |
| 目标错位复发     | loss 降而 align/MAL 不升（G-A2） | 复查 §2.2/2.3：蒸馏权重（DFX-2）、agree 配置（OPEN-3）      |
| 训练 OOM         | M4/M5 峰值超预算                 | 按 §9.8 REQ-RT-5 杠杆逐级降级；16K → 回 2304 验证逻辑正确性 |
| requant 损失过大 | MAL_B 与 native 差值异常（G-A7） | 隔离导出链逐张量对拍（T-UNIT-9 扩展为全量 SHA256）          |
