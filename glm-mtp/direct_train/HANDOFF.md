# GLM-5.3 W4A8 MTP 蒸馏增训 — 交接手册（HANDOFF）

> **固定路径（所有机器同路径可读）**：`/mnt/share_space/l00840580/speculators-work/`
> - 本手册：`speculators-work/HANDOFF.md`
> - 全部脚本+测试+工具：`speculators-work/glm-mtp-direct/`（含 env.sh、40_train.sh、
>   export_glm_mtp_for_vllm.py、eval_ab.py、tests/ 等，无需 GitHub 也可运行）
> - GitHub 权威副本：`ababyshark/speculators` 分支 `glm-mtp5-v2` @ `efc028a`

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
| **B_v15：TV 蒸馏 + steps=5 + 10x** | **2.219（失败）** | 0.713/0.318/0.132/0.037/0.018 |
| **B_v16 requant：TV 5步递归 + 1x** | 2.549（≈A，无增益） | 0.735/0.418/0.222/0.115/0.060 |
| **B_v16 float：同权重 float 写回** | **2.716（超 requant +0.167）** | 0.750/0.461/**0.278**/0.148/0.080 |
| **A_float：init drafter float serving** | **2.908（全场最高）** | 0.756/0.486/0.321/0.201/0.143 |
| verifier-argmax == 数据 token（诊断） | 监督位 86.7% | — |

### 1.6 v15/v16 复盘与三层根因（2026-11-08）

**根因 1（工程，已修）**：5 步递归从未生效。上游 vllm-project/speculators PR #1024
（commit 74fa8a1）的 cli.py 无条件 `args.num_speculative_steps = draft_model.config.num_speculative_steps`，
把 CLI 传的 5 覆盖成 v9 config 的 `speculative_tokens=1`（MTP 步数的唯一事实源是模型 config，
CLI 参数是死的）。v15 实际是"单步 + 10x 数据 + 10 倍步数"。日志 0 条 `align_step_1..4` 为铁证。
已修：显式 CLI 值写回 speculators_config 并打日志；monitor_train.py F1 检测兜底。

**根因 2（数据，已证）**：v15 的 10x 数据回答来自 Qwen3-8B regen 管线，非 GLM-5.3 分布。
训练初始 align_0 0.536（1x 数据参考值 0.737），2310 步把 drafter 拉向 Qwen 分布
（token_accuracy 0.50→0.88）。数据需换 GLM-5.3 on-policy 回答重新生成。

**根因 3（方案层，最根本，已破）**：**requant 抹平训练增量**。W4A8 int4 量化步长
~0.003-0.01，而训练后 float 权重变化 median max|Δw|/|w|=1.35%、绝对值 ~0.0007-0.004
（`tmp_v15_eval/delta_analysis.py` 可复现）——比格点小 10-100 倍，requant 后 int4 码
几乎全部回到 init 原码。这解释了 v13/v14/v16 全部训练在 eval 上不超过 A 的历史。

**解法（已在 v0.30.0 栈验证）**：MTP 层 float 写回 + quant_model_description.json 逐张量
FLOAT 标记（复用包内 eh_proj/norms 已有的 FLOAT 模式：无 scale 兄弟条目）。构建脚本
`tmp_v15_eval/build_float_hybrid.py`。**v0.28 时代"float 写回 acceptance=0.000"是误判**
——实为 v0.28 加载失败；v0.30.0 上 W4A8 主干 + FLOAT MTP 层混合加载/图捕获/serving 全部正常。

**v16 训练（修复后首个完整递归实验）**：1x 数据、5 步递归、231 步、~55 分钟（12.1s/step）。
loss 0.421→0.142，align_0 0.735→0.795，align_1..4 0.30/0.19/0.13/0.09→0.66/0.56/0.51/0.47。
注意：训练中 FSDP2+多层递归的层级 activation checkpoint 会触发 "mixed Tensor and DTensor"
（`_glm_layer_step` recompute），用 `SPECULATORS_MTP_ACTIVATION_CKPT=0` 关闭（已写入 40_train.sh）。

### 1.7 A_float 对照实验（2026-11-08 晚，结论反转）

init drafter 走同一 float 路径 serving：**A_float = 2.908，全场最高**（比 native +0.145，
比旧基线 2.551 +0.357）。三个新结论：

1. **float serving 收益 +0.145**：init 的 float 权重本身就比 W4A8 量化版强——native 2.763
   的"天花板"一直是量化损伤假象；旧基线 A=2.551 是 requant 双重削弱产物
2. **v16 训练在 float 空间为负增益（-0.192）**：B_v16_float 2.716 < A_float 2.908，
   pos1-4 恶化最重。训练时 align（teacher-forced 数据上下文上的 argmax 重合率）上升
   但 serving rollout 接受率下降——**teacher-forcing 目标与 serving 条件系统性错位**：
   86.7% 分歧位上，训练在优化 serving 永不访问的上下文（serving rollout 只在全
   verifier-correct 前缀上评估 drafter）
3. **立即可交付最优解 = A_float 路径（零训练）**：`tmp_v15_eval/build_float_hybrid_init.py`

**目标函数修复（v17，进行中）**：`SPECULATORS_MTP_AGREE_FILTER=1`（仅在 data token ==
verifier argmax 的监督位施加 loss，消除 13.3% OOD 梯度）+ `SPECULATORS_MTP_TEACHER_FORCE_VERIFIER=1`
（递归展开 teacher-force verifier argmax token，精确模拟 serving 全对上下文；每步一次
lm_head no_grad pass 同时产出 feeding tokens 与 target argmax）。已知局限：TV 目标
v_logits 仍来自数据前缀上的 verifier HS，首个分歧点之后的目标有污染——彻底修复需在
verifier 纠正序列上重新生成 HS。新基线判定标准：**对照 A_float=2.908**，非 2.551。

### 1.8 目标函数修复的隔离实验序列（2026-11-09）

v17 捆绑了两个修且结果更差（2.606），无法归因 → 拆开变量逐个验证。
基线统一对照 **A_float=2.908**（零训练最优），训练对照组是 v16=2.716。

| 实验 | 目标函数 | 结果 | 结论 |
|---|---|---|---|
| v16 | 旧目标：数据 token teacher-forcing + 全部监督位 | float 2.716 | 负增益 -0.192 |
| v17 | v16 + agree 过滤 + teacher-force | float 2.606 | 输入（verifier 链）与目标（数据前缀 HS）错配，倒退 |
| **v18（"A 实验"）** | **v16 + 仅 agree 过滤**（TEACHER_FORCE=0） | **float 2.765** | **过滤有效**：>v16（+0.049），pos0=0.757 追平 A_float；剩余差距集中在 pos2-4（深步上下文/数据错配）→ 转 onset-policy |
| B 实验（on-policy） | verifier 贪心回答重建数据（token ≡ verifier argmax）+ 自洽 HS | 数据生成中 | teacher-force 的正确打开方式 |

判读：
- v18 > 2.716 → agree 过滤有效，v16 退化部分来自 OOD 梯度，方向正确
- v18 ≤ 2.716 → 过滤无贡献，问题在数据非 verifier 分布本身 → 全力转 B
- on-policy 生成已并行启动：2052 条 1x prompt → verifier 贪心回答
  （`tmp_v15_eval/gen_onpolicy_data.py`，断点续跑 jsonl）→ `fill_hidden_states.py`
  捕获自洽 HS → v19 训练（agree 应 ≈1.0，目标函数完全自洽）

> **v15 复盘（2026-11-08）**：2.219 是最差结果，两个根因——(1) **5 步递归从未生效**：
> cli.py 无条件用 draft config 的 `speculative_tokens=1` 覆盖 `--num-speculative-steps 5`
> （日志 0 条 align_step_1..4 为铁证），实际是单步训练；(2) **10x 数据分布错位**：
> 回答来自 Qwen3-8B regen 管线而非 GLM-5.3，训练初始 align 0.536（1x 是 0.737），
> 2310 步把 drafter 拉向 Qwen 分布（token_accuracy 0.50→0.88）。已修 cli.py；
> 下一步 = 修好递归后先在 1x 数据隔离验证，10x 数据需换 GLM-5.3 on-policy 回答重生成。

---

## 2. 40 机现状与资源地图

### 1.9 B 实验终局（2026-11-09）：三对比结果与最终结论

管线全修复后（on-policy 300 条数据 + agree 过滤 + 5 epochs 自洽训练 + float 交付），
在 **on-policy 32 条评测集**（从 300 条训练 prompt 抽样，seed 7）上三对比：

| 位置 | 1. A 原始（官方 W4A8） | 2. A float（init float） | 3. B float（v19 训练） | 量化差 (2)-(1) | 训练差 (3)-(2) |
|---|---|---|---|---|---|
| pos0 | 0.823 | 0.823 | 0.800 | +0.000 | -0.023 |
| pos1 | 0.628 | 0.631 | 0.590 | +0.003 | -0.041 |
| pos2 | 0.430 | 0.433 | 0.406 | +0.003 | -0.027 |
| pos3 | 0.293 | 0.296 | 0.298 | +0.003 | +0.002 |
| pos4 | 0.214 | 0.213 | 0.210 | -0.001 | -0.003 |
| **mean** | **3.390** | **3.396** | **3.304** | **+0.006** | **-0.092** |

**结论：**
1. **训练增益 -0.092，仍为负**：全部已知缺陷修复后，通用文本 + 300 条规模微调依然劣于 init。
   劣化集中在 pos0-2（权重最大的近端），pos3 微正、pos4 噪声内
2. **MTP 层量化在此分布上零损失**（±0.003）——旧评测集的 0.145 量化差是分布相关的
3. **评测集影响 >> 一切模型改动**：同批模型在旧 32 条（2.76）vs on-policy 32 条（3.39）差 0.63——
   生产上流量分布/前缀缓存策略比训 drafter 更有杠杆
4. **最终交付建议：直接用官方 W4A8 包**（native = A_float = 3.39，无需任何改造）
5. 增训的正向空间 = 真实领域分布（verifier 行为与 init 训练分布不同）+ 10k+ on-policy 样本；
   本管线的生成/训练/评测资产可直接复用

**运维要点（本轮新增）**：
- 生成数据用 v18 float 包 + MTP×5 投机解码：输出与纯 greedy 逐 token 等价（贪心验证保证），
  300 条从 ~8h 压到 **16.5 分钟**（24x）；端点脚本 `tmp_v15_eval/serve_gen.sh`（PORT/DEVICES/TAG 环境变量）
- 此构建 completions 响应 `token_ids` 恒为 null：生成存 **text**，组装时全序列重编码
  （`assemble_onpolicy_v2.py`，roundtrip 校验 0 失配）
- `--save-best` 语义：最优 checkpoint 存到数字目录（v19 是 `checkpoints_v19/3`），旧的被自动清理；
  导出用 `checkpoint_best` 符号链接的指向，勿假定 `0`
- 引擎在持续 decode 负载下会崩（ERR99999，3 次）：投机型生成 + 12 并发实测稳定 16.5 分钟无崩溃

### 1.10 领域适配实验（v20，2026-11-10）：最终判决

数据源：/mnt/share_space/l00840580/xql-cases（内部昇腾部署案例，4588 md / 9GB，
本体为内部硬件型号/工具名/报错模式 = 真正 OOD）。管线：RAG-QA/摘要/抽取/CSV-QA
四种 prompt 风格构造 464 条（400 train / 32 in_eval / 32 held_eval，seed 42）→
双端点投机解码 on-policy 生成（400 条 8.7 分钟）→ 重编码组装（0 失配）→
v20 训练（agree 0.954、val/align 0.823 历代最高、3 epochs）→ float 导出 → 同集三对比。

| 配置 | xql in_eval 32 | xql held_eval 32 |
|---|---|---|
| A 原始（官方 W4A8） | 3.224 | 3.640 |
| A float（init float） | 3.324 | 3.729 |
| B float（v20 训练） | 3.058 | 3.380 |
| 训练差 (3)-(2) | **-0.27** | **-0.35** |

**总判决（v13-v20 八连负）**：跨 8 个变体、两种数据域（通用+内部）、全部目标/基础设施修复、
多组规模与 epoch——可及规模（≤2052 条）下任何微调都使工厂 drafter 变差，训练越重劣化越大
（领域适配 -0.27/-0.35 > 通用 -0.19）。机制：工厂 drafter 在 serving 分布处于精细最优点，
小数据梯度 = 净扰动（灾难性遗忘式劣化）。

**最终交付建议**：
1. MTP drafter 直接用官方 W4A8 包（xql 集 native ≈ A_float，float 改造收益仅 ~0.1，可选）
2. 通用与内部域文本的 MTP 微调无正向空间（除非工业级数据规模 + 官方级配方）
3. 真正的杠杆在服务侧与流量侧（评测集差异 ±0.6 实测）

**本轮新增运维要点**：
- 双端点 16 卡高并发生成：400 条 8.7 分钟（35 条/分钟），单端点 8 workers 会反复崩
- `--save-best` 导出源 = `checkpoint_best` 符号链接指向的数字目录（会轮换，勿假定 0/固定值）
- 40 机存在外部清理/资源回收机制：空闲引擎静默死亡（无日志）、大 checkpoint 目录消失——
  关键产物训练后立即转移或校验
- 两机 NFS 视图差异：/mnt/share_space/l00840580/ 下目录（speculators-work、xql-cases）
  仅部分视图可见；跨机传文件用 tar+scp（sshpass 别忘了 export SSHPASS）

### 1.11 冻结/共适应分析与 32K 长上下文方向（2026-11-10）

**共适应问题（用户提出）**：MTP 头在预训练阶段与主模型联合训练——主模型 hidden state
几何是"下游单层头可预测"目标共同塑造的，主模型权重与 MTP 头配套。冻结主模型只训头
是否"肯定出问题"？

**结论：冻结不破坏配套——只要主模型本身不变。** h 的生产者不变 → 头的任务定义不变，
方法学成立。证据：(a) lf-glm53 直连训练同样冻结主模型，d2-d5 命中率 +6.5~+17.5pt；
(b) 八连负全部是"无增益"而非"机制崩坏"。**配套真正被破坏的场景 = 主模型被 RL/SFT
更新而头不跟动**——此时必须联合训练（案例 84 verl-SpeCo 在线协同训练模式）。
主模型量化（W4A8）的偏移实测轻微（agree 0.94-0.96）。
**场景判定：新数据（主模型不变）下，冻结主模型 + 只训 MTP 头理论上可用**；
八连负的否定结论来自饱和+小数据漂移，不是冻结本身。
不建议解冻主模型：任何主模型变更 → W4A8 重校准 + verifier 全量重验 + 漂移放大。

**32K 长上下文方向（用户实测洞察）**：生产中低接受率主要出现在长上下文场景
（内部案例亦证实 64K 场景普遍）。假设：MTP 头主要在短序列上训练，32K 下
hidden state 几何与注意力模式变化导致接受率下降 → **32K 评测+训练是更合适的靶点**。

**32K 实验设计（评测先行）**：
1. 构造 32K 上下文评测集：内部案例 md 拼接成 ~30-32K token 多文档上下文 + 跨文档问题
   （OOD 内容 + 长上下文双条件）；同内容短上下文版本做对照（分离"长度效应"与"内容效应"）
2. serving 需 max-model-len 32768（当前管线 4096）——DSA 稀疏注意力 KV 压缩下预期可行
3. 测 init 接受率 @32K vs 短文本基线 3.39 → headroom 判定
4. 若 headroom 确认 → 32K on-policy 训练（显存风险：训练侧激活 ×14，层级 checkpoint
   有 DTensor bug（ACTIVATION_CKPT=0 绕过）、需 chunk-loss/重计算策略——lf-glm53
   管线已验证三级显存优化可解 64K）

### 1.12 接受率地图与项目终局（2026-11-10）

**novel-32K 评测（swezero 同仓轨迹拼接 ~30K + 续写任务，不可抄写）**：
分块 3.653/3.446/3.419/3.851，合并 **3.585**（16 条 / 287 轮）——依然高接受率。

**接受率完整地图（A 原始 / init，全部引擎实测）**：

| 任务类型 | 分布 | 上下文 | mean |
|---|---|---|---|
| 旧训练集 QA | 通用（可能已见） | ~1-2K | 2.763 |
| 案例问答 RAG-QA | 内部 OOD | ~1-2K | 3.224-3.640 |
| 指令 QA（cmb） | 通用英文 | ~0.2-1K | 3.271-3.327 |
| 32K 可抄写（拼接+统计） | 内部 OOD | 30.5K | 3.797 |
| 32K 不可抄写续写（agent 轨迹） | 内部 OOD | ~30K | 3.585 |

**最终结论（九连负 + 接受率地图）**：
1. init drafter 在所有可构造分布上（短/长、通用/内部、可抄写/不可抄写）接受率
   2.76-3.80，**低接受区（<40% 平均）无法用现有数据构造方式复现**
2. 32K 长度本身不是低接受率的原因（32K 反而最高 3.797）——决定因素是
   **任务可预测性**（格式锁定/可抄写任务 > 开放生成），且 32K 拼接文档的大量
   token 重复给了 drafter 抄写便利
3. 生产中 <40% 样本的成因在我们的数据构造空间之外——**需要真实生产样本
   （prompt 原文）才能定位与训练**；拿到样本后走 per-sample 测量 → margin 诊断
   （可学习/不可学习分类）→ 定向 on-policy 训练 → per-sample 复测（工具链全部就绪）
4. **最终交付 = 官方 W4A8 包原样**；八连负 + 直连复现负 + 接受率地图构成本项目的
   完整证据链

**40 机已清空**（全部服务停止，16 卡回到基线）。跨机大文件传输统一走 tar+scp
（/mnt/share_space 双后端脑裂：同一写入路径仅写入方视图可见，严禁跨机共享假设）。

### 1.13 服务非确定性与精度矩阵（2026-11-10/11，重大发现）

**自洽性测试（35 机新容器 vllm 0.30.0 + native W4A8 + MTP×5，temp=0 同 prompt 跑两遍）**：
- **0/24 自洽**（含短上下文对照），关 prefix caching 后**仍 0/24** → 非确定性在 kernel 层，排除 prefix cache
- 逐请求接受率两遍差 ±0.2-0.3 → **此前所有单遍评测的幅度不可信**（九连负方向仍显著：9/9 同向）

**精度×一致性矩阵（4 场景 × 6 长度 × 3 遍，可验证植入答案）**：
| 任务 | 精度（3 遍） | 一致性 |
|---|---|---|
| 英语/中文事实检索（含 32K） | **100%** | 稳定 |
| 数学大数运算 | **0-67%**，且**三遍给出三个不同的错误答案** | 全部不稳定 |
| 代码模运算 | 级联污染（乘法错→模错） | 全部不稳定 |

→ kernel 非确定性对**逐位自回归的数字生成**是毁灭性的（logit 微扰翻转数字 token）；
检索类不受影响。生产"某些场景接受率低"与此同源：目标模型自身输出不稳定。

**W8A8 对照评测进行中**：712GB 包在 35 上 4 次冷读 NFS 加载失败（30-70 分钟静默死亡），
已改走 rsync 到 /data1 本地 NVMe（5.2TB 空闲）再 serving。工具链：selfcons_test.py、
gen/run_accuracy_matrix.py（已提交 8e5b875）；分析文档 glm-mtp/docs/nondeterminism_analysis.md。

**评测口径新规（后续所有对比必须遵守）**：N=3 遍均值 ±std；差值 > 噪声带才认定方向；
新环境先过自洽性 gate。

### 1.14 最终章：N=3 降噪重评 + W4A8/W8A8 全维度对比（2026-11-11，测试闭环）

**③ N=3 修正口径重评（35 机新容器，同协议，每配置 3 遍）**：

| 评测集 | A_float（不训练） | B_v20（xql on-policy 训练） | 差值 | 显著性 |
|---|---|---|---|---|
| xql in_eval | 3.307±0.048 | 3.050±0.020 | −0.257 | 显著 |
| xql held_eval | 3.606±0.039 | 3.479±0.060 | −0.127 | 显著 |
| 通用 on-policy 32 | 3.350±0.081 | 2.681±0.006 | **−0.668** | 显著（巨大） |

- **九连负定量坐实**（降噪后三集全显著为负）
- **新模式：灾难性遗忘**——域外损失（−0.668）≫ 域内（−0.257），微调打垮通用能力
- 域内也为负 → 排除“数据不够”解释：方法上无硬伤的配方在该规模就是不行

**② W8A8 全维度（vs W4A8）**：

| 维度 | W4A8 | W8A8 | 结论 |
|---|---|---|---|
| 自洽性（32K 集两遍） | 0/24 | 0/24 | 非确定性与量化格式无关（kernel 层） |
| 检索精度（矩阵） | 100% | 100%（一致性更好） | 抄写类任务两者都好 |
| 算术精度（矩阵） | 0-67% 不稳定 | 0-33% 不稳定（含“一致地错”） | 算术崩坏两版同病 |
| 接受率（32K in/held） | 2.10-2.35 | 2.21-2.27 | 长上下文接受率两版腰斩（vs 短集 3.3+） |

**最终交付结论（全证据链）**：
1. **官方 W4A8 包原样交付**，不训练（增训=负增益+遗忘）
2. **生产低接受率场景定位 = 长上下文（32K 接受率 2.1-2.7 vs 短 3.3+）+ 数字/代码类任务（算术非确定崩坏）**，
   根因在 vllm-ascend kernel 非确定性 → 建议按此向 vllm-ascend 侧提报/排查（batch_invariant 路线）
3. 评测规范固化：新环境先过自洽性 gate；对比必须 N≥3 均值±std，差值 > 噪声带才认定方向

**产物索引**：n3 结果（35:/home/d00952671/n3_results.tgz → 本目录 n3/）、
w8a8 矩阵结果（35:/home/d00952671/w8a8_results.json）、分析文档
glm-mtp/docs/nondeterminism_analysis.md（仓库）、全部脚本在 glm-mtp/direct_train/。

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

### 2.4 训推一体成熟方案（formal100 布局，勿重头试）

> **原则：设计任何新训练前，先查本地成功记录**：`glm-mtp-direct/PLAN.md`（formal100 配方）、
> `40_train.sh`/`30_generate_hs.sh`（验证过的脚本）、`checkpoints_v*/train_command.txt`
> （实际跑过的完整命令）、`logs/`（成功证据）。v13/v14/v15/v16/v17 全部复用此布局，
> 从未失败过的是**布局本身**（失败都在代码 bug / 数据 / 目标函数层）。

16 卡分工（训推同时在线，互不抢占）：
- **卡 0-7**：FSDP 训练（`40_train.sh`，torchrun --nproc-per-node 8，参数唯一入口 `env.sh`）
- **卡 8-15**：HS 提取端点（`scripts/launch_vllm.py train`，TP8，端口 8000，
  ExampleHiddenStatesConnector 写 `--hidden-states-path`；`--on-missing generate` 时
  训练中实时补生成缺失 HS）
- 启动顺序：**先端点后训练**。训练进程启动即调 `/v1/models` 校验端点（v17 教训：
  端点未就绪时启动训练 → `openai.APIConnectionError` 直接崩）。就绪判据：
  `curl -s http://127.0.0.1:8000/v1/models` 返回 `"data"`；权重加载约 8-10 分钟
  （npu-smi 上卡 8-15 出现 ~56GB/卡的 VLLMWorker_TP 才算真就绪）。
- 杀进程规则：`pkill -f "vllm.entrypoints"` 会**连 HS 端点一起杀**（v17 教训）——
  若只需释放卡 0-7，按 npu-smi 进程表宿主 PID 定点 kill；杀后 sleep 120。
- 1x 数据（2052 条）的 HS 已全部在盘（`$RUN_ROOT/hidden_states`，~84GB），
  训练走 file backend 不依赖端点生成；但端点必须在线（trainer 启动校验 + 兜底生成）。
- 完整流水线编排见 `run_all.sh`（stage 0-4：preflight → HS → train → export+eval）。

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
# 宿主机直接跑（NFS 可读日志）；exit!=0 即有 FAIL 项；--watch 300 可持续监控。
# 日志按 global_step 回退自动分段，只分析最后一段 run。
python3 /mnt/share_space/l00840580/speculators-work/glm-mtp-direct/monitor_train.py \
  --log /mnt/weight2/l00840580-direct-v13/logs/train_v13.log \
  --expected-steps 5 --max-steps 2310 --align-ref 0.737 \
  --checkpoints /mnt/weight2/l00840580-direct-v13/checkpoints_v15
# 判读：align_step_0 应当随训练上升（TV 蒸馏生效的直接证据）；v13(CE) 是下降的
```
检测项（每项都来自实战教训，详见脚本头注释）：
- **F1 步数未生效**：请求 K 步但日志缺 `align_step_1..K-1` → 实际单步在跑
  （v15 教训：cli.py 曾无条件用 draft config 覆盖 CLI，5 步递归从未发生，白跑 4 小时）
- F2 数据分布错位：align_0 起点比参考低 0.10+（v15 教训：0.536 vs 0.737，Qwen regen 10x 数据）
- F3 CE 拟合数据 token 漂移：token_accuracy_0 超过 align_0 0.08+
- F4 吞吐劣化（step_ms 增长 1.5x+，查 HS 端点 on_missing=generate）｜F5 Traceback
- F6 checkpoint 迟迟不落盘｜F7 HCCL(EI0015)+OOM 签名｜F8 HS 端点存活｜F9 diag 数据重合率

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
| 训练日志只有 `align_step_0`，无 step_1..K-1 指标 | **v15 教训**：cli.py 无条件 `args.num_speculative_steps = draft_model.config.num_speculative_steps`，CLI 传的 5 被覆盖为 v9 config 的 1 | 已修（显式 CLI 值写回 speculators_config 并打日志）；重训前用 monitor_train.py --expected-steps 5 验证 |
| 训练 align_0 起点 0.536（1x 参考是 0.737） | 10x 数据回答来自 Qwen3-8B regen，与 verifier 分布错位 | 数据换 GLM-5.3 on-policy 回答；训练前先跑 diag_verifier_vs_data.py |
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

---

## 8. W8A8 交叉验证结论（2026-11-10，新增）

GLM-5.3-W8A8（712GB 官方包）已按 §6 同协议完成 native MTP 接受率评测，并以 W4A8@TP16
做严格配对（TP8 历史基线跨 TP 有 ±0.03-0.05 数值差）。详见
`glm53-w8a8-eval/GLM5.3-W8A8-HANDOFF.md` §11。

**结论**：W8A8 vs W4A8@TP16 六集 5 集统计等价、平均 mal Δ = +0.030（W8A8 方向）、
无任何一集 W4A8 显著更优；唯一显著差异 cmb3k_held32 = +0.147（3.6σ，W8A8 优）。
→ MTP 接受率对 drafter 量化位宽（4bit vs 8bit）不敏感，±0.03 量级即噪声；
  §7 中 "A=2.551 / native=2.763" 的差距主要来自 requant 双重量化路径而非位宽本身。
