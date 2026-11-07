# GLM-5.3 MTP 直连训练方案（v13）

> 结论先行：v12（.23 框架 + 离线 HS）已证伪——隔离实验 2.581→2.573，训练无效。
> 本方案回到 **2026-11-02 已验证成功的 formal100 直连配方**（speculators 仓库原生 trainer），
> 全流程在 `/mnt/share_space/l00840580/my-contributions/speculators` 代码路径内完成。

## 1. 为什么直连是对的

| | v12（废弃） | 直连 v13（本方案） |
|---|---|---|
| trainer | .23 框架 | `src/speculators/train/__main__.py --speculator-type mtp` |
| HS 来源 | .23 离线捕获管线（自定义 pre_norm/layer78 语义） | **vLLM serving 引擎直出**（同权重同前后处理，`kv_transfer_params` 经 hs_connectors 落盘） |
| verifier | 不在训练图内 | 训练进程内 FSDP 加载，参与前向 |
| 语义 | 与 serving 错位（已实锤） | 构造上零漂移 |

## 2. 资产重建表（旧资产已被管理员清理）

| 资产 | formal100 当时 | 现在 |
|---|---|---|
| verifier | /data1/weight/GLM-5.3-w4a8-spec ❌ | `/mnt/weight2/GLM-5.3-W4A8` ✅ |
| drafter init | /data1/mtp_training/weights/glm53_mtp_speculators_step0 ❌ | `dspark-glm53/drafter_init/glm53-mtp-speculators-v9` ✅（同源，speculators mtp_layers.0.* 格式） |
| Arrow 数据 | /data1/mtp_training/data/prepared_8k_dflash2_official ❌ | `dspark-glm53/data_prep_rot` ✅（2052 行，input_ids/loss_mask/seq_len，契约完全一致，**无需重新准备**） |
| HS 文件 | ❌ | 待生成：3 层 × 2052 条 ≈ **84GB** → `/mnt/weight2/l00840580-direct-hs-v13/`（100T 大盘 37T 空闲） |
| 训练环境 | glm-mtp-910c 容器（exec 卡死） | 同镜像（nightly-releases-v0.30.0-a3）重建干净容器 + `pip install -e` + adapter 安装 |

## 3. 数据规模与切分

- 2052 行全量使用（覆盖全），seq_len 214–2281（p50=1108），总 2.3M tokens
- `--train-data-ratio 0.9006` → **1848 train / 204 val**（ArrowDataset 内部按行切，无需拆文件）
- `--total-seq-len 2304`（≥ max 2281；formal100 用 8192 是因为其数据是 8k 长序列；此处调小仅减少 padding 算力，语义中性 → 单步耗时降 ~3.5x）
- HS 存储 = 2.3M tokens × 3 层 × 6144 × 2B ≈ 84GB

## 4. 执行流程（5 阶段，每阶段可独立重跑/断点续跑）

| 阶段 | 脚本 | 依赖 NPU | 内容 |
|---|---|---|---|
| 0 preflight | `10_preflight.sh` | 否 | 资产/仓库状态/容器环境/数据契约校验 |
| 1 data | `20_prepare_data.sh` | 否 | 数据契约校验 + HS 目录初始化 + 记录清单 |
| 2 HS | `30_generate_hs.sh` | **是**（8 卡 vLLM 端点） | `launch_vllm.py train` 起 HS 端点 + `fill_hidden_states.py` 驱动补齐 2052×3 层 |
| 3 train | `40_train.sh` | **是**（8 卡 FSDP） | formal100 配方 100 步（可调 EPOCHS/MAX_STEPS），`--on-missing raise`（HS 已齐） |
| 4 eval | `50_export_requant_eval.sh` | **是**（8 卡 serving） | convert 导出 → requant_layer78 → preflight → serve → eval_ab → sanity |

阶段 2/3/4 串行复用同一 8 卡窗口（预计合计 ~3-4h：HS 1-2h + 训练 1h + 导出评测 40min）。

## 5. 关键决策点

1. `num_speculative_steps=1`：formal100 已验证配方。serving 是 5-token 递归，但训练按验证过的单步走；
   若直连训练有效，后续可试 steps=5 + `step_weight_beta`（MTPArgs 支持）作为增益实验。
2. drafter init 用 v9（= formal100 的 step0 同源，native 提取）。**不**用 v12 的 epoch3（错误语义下训过的）。
3. 数据**不**混入 v12 的任何产物；data_prep_rot 是纯 token 数据，与 .23 训练语义无关。
4. 导出→requant 的双重量化损失（~7%）仍在；若直连训练有效且要上线，后续需优化 requant（单独议题）。

## 6. 验收标准

- 训练：loss 收敛 + checkpoint smoke 通过（复刻 formal100 验证）
- **接受率 A/B（同一套工具链）**：B(direct-trained) vs A(epoch0 init) 必须显著为正，
  逐位 pos1-5 记录；同时与 native 2.763 对比给出 requant 路径损失的健康检查
- 全部产物入库 + 测试通过（test_eval_sanity 等）

## 7. 机器窗口

- 35 前 8 卡（今天不可用 → 明天窗口）
- 或本机 8 卡（qwen38 benchmark 结束后）
- 脚本以 `VISIBLE_DEVICES` + `HOST` 参数化，两端皆可跑
