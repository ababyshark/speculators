# GLM-5.3 W4A8 MTP 增训 — 测试设计文档

> 目标：把 GLM-5.3-W4A8（`GlmMoeDsaForCausalLM`，78 层 MoE + DSA indexer + 原生 MTP 层 `model.layers.78`）MTP=5 增训全链路的**每一个已踩过的坑固化为测试**。 原则：每个单元测试对应一个真实发生过的事故；每个功能测试对应一条端到端契约； NPU 集成测试只在有卡环境手动/门控执行。

## 一、流水线分解 → 测试映射

```
[W4A8 checkpoint]                                    [speculators 增训]
 ① 抽取 warm-start ──────────────► BF16 drafter ──► ② 数据生成/rot 变换
    extract_mtp_layer.py              │               gen_onpolicy / transform_hs_rot
    (反量化约定/键映射/专家融合)        │                       │
                                      ▼                       ▼
 ⑧ requant 写回 ◄──────────────── 训练 ③ FSDP fp32 增训 (④ sparse dispatch
    requant_layer78.py               speculators.train         ⑤ _VocabCE ⑥ 冻结
    (int4 打包/scale/描述一致性)       (checkpoint/resume)       ⑦ zero-momentum fill)
                                      │
                                      ▼
 ⑨ vllm-ascend serving (rot/MTP 挂载) ──► ⑩ 验收测量 (measure_acceptance)
```

## 二、单元测试矩阵（tests/unit/，CPU，全绿 27 项）

### 已固化 ✅

| #   | 测试文件                                | 锁定的不变量                                                                                                                                                                                                                                                                                                                                                                                                                      | 对应事故                                                                                                                                                     |
| --- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| U1  | `test_glm_moe_dsa_mtp_layer.py` (8)     | glm_moe_dsa 注册；MTP 层配置补丁（sparse MoE + full indexer）；**state-dict 键不得改名**（wrapper 会把 mlp.*→mlp.moe.*，warm-start 专家静默归零）；dispatch 以实例级 MethodType 绑定（FSDP2 unshard 钩子必须触发）；**dispatch 自包含契约**（路由由父级算好传入、shared_experts 由父级加——违反即 AttributeError/重复叠加）；mixin 组合契约（enorm→embed、hnorm→HS、拼接顺序、final_layernorm 位置 = serving 的 DeepSeekMTP 语义） | 事故:键重命名致 warm-start 失效；AttributeError: no attribute 'gate'；shared_experts 双重叠加                                                                |
| U2  | `test_glm_sparse_dispatch.py` (7)       | dispatch ≡ 稠密参考实现（随机/偏斜/空专家路由）；梯度只流向输入（冻结权重无梯度）；**路由权重必须按排序后的 job 顺序对齐**；topk 形状无关（[N,K] 与 [B,S,K]）；单次 D2H 同步契约                                                                                                                                                                                                                                                  | **事故×2（测试当场抓出）**：flat_w 未按 argsort 重排（每 token 的专家输出被乘上别的 token 的路由权重，4 个 epoch 训练目标被污染）；K=shape[1] 对 3D 输入取错 |
| U3  | `test_glm_vocab_ce.py` (3)              | \_VocabCE 前向/反向 ≡ F.cross_entropy；ignore_index 语义；全 mask 步不 NaN                                                                                                                                                                                                                                                                                                                                                        | 事故:大词表 CE 占步时 82.5%                                                                                                                                  |
| U4  | `test_glm_checkpointer_recovery.py` (4) | 零梯度参数的零动量填充（单卡 + DCP FQN 两路）；幂等；冻结参数不发明状态；**convert_float_dtype(None) 保持 fp32 动量**                                                                                                                                                                                                                                                                                                             | 事故:resume 严格加载崩溃；bf16 动量跨 resume 精度损失                                                                                                        |
| U5  | `test_glm_loading_fallback.py` (2)      | 非标准索引名（quant_model_weights.safetensors.index.json）回退；后缀匹配                                                                                                                                                                                                                                                                                                                                                          | 事故:vendor checkpoint 加载 FileNotFoundError                                                                                                                |
| U6  | `test_glm_expert_freeze.py` (3)         | 冻结命中所有含分组权重的模块且仅这些；FSDP 候选选择（experts 独立单元、已分片跳过）                                                                                                                                                                                                                                                                                                                                               | 事故:61GB 卡梯度 OOM；HCCL 通信器上限                                                                                                                        |

### 缺口（按优先级，待补）

| #   | 测试                                                                                 | 锁定的不变量                                                                                                                                                                                                                                                                                                                                                                          | 优先级                                      |
| --- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| U7  | 抽取反量化约定（`extract_mtp_layer` 的核心函数移入 `src/speculators/convert/` 后测） | int8 逐通道 `W=q×s`；int4 双打包（dim0、低半字节在前）`W=unpack(q)×s`；offset=0 语义；scale_bias 不进权重（±20 量级 vs 权重 ±0.076，任何简单组合都毁掉权重——已有张量级实验）；键映射表（eh_proj→input_proj、enorm→token_layernorm、hnorm→hidden_layernorm、shared_head.norm→final_layernorm、self_attn/mlp 直映、layer78 embed/head 丢弃）；专家融合 gate/up 顺序；norm bias 全零丢弃 | **P0**（P1 事故的源头：抽取有损 0.79→0.26） |
| U8  | requant 往返                                                                         | `requant(dequant(x)) ≈ x`（int8/int4）；scale=amax/127、amax/8；打包顺序一致；**quant_model_description 保持（is_rot_used 等标志不得丢失）**；symlink shard 一致性                                                                                                                                                                                                                    | **P0**（写回协议闭环的前置）                |
| U9  | rot 变换                                                                             | `H @ R^T` ≡ `nn.Linear(H)`（serving 语义）；bf16 舍入界                                                                                                                                                                                                                                                                                                                               | P1                                          |
| U10 | 训练目标对齐契约                                                                     | step k：`embed(input_ids[:, k+1]) + HS → target input_ids[:, k+2]`；loss_mask 正确置 -100；data.py 的 HS[:-1]/input_ids[2:] 切片契约                                                                                                                                                                                                                                                  | **P0**（serving pos0 语义对齐的前提）       |
| U11 | trainer 遥测守卫                                                                     | torch.accelerator.memory_allocated 等探针在 torch_npu 下不崩                                                                                                                                                                                                                                                                                                                          | P2                                          |

## 三、功能测试（CPU 集成，可进 CI）

| #   | 测试                            | 内容                                                                                                                | 门槛                                 |
| --- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| F1  | tiny 端到端训练冒烟             | tiny GlmMoeDsa 配置 → 合成 warm-start → MTPDraftModel 2 步训练 → loss 有限下降 → checkpoint 保存 → 严格 resume 成功 | loss 有限；resume 无缺失键           |
| F2  | 抽取→requant 全往返             | 合成 W4A8 层 → 抽取 BF16 → requant 写回 → dequant 对照原始                                                          | 逐张量 cos ≥ 0.999；描述文件逐键一致 |
| F3  | 保存/续训等价                   | 2 步 + save + resume 的 loss 轨迹 ≡ 连续 4 步                                                                       | 相对差 < 1e-6                        |
| F4  | warm-start 对照协议（离线半边） | requant(warm-start) 张量往返 ≈ warm-start                                                                           | 0.99998（现有实测值）                |

## 四、NPU 集成测试（门控/手动，preflight_check.sh 前置）

| #   | 测试                     | 内容                                                                                      | 通过判据                                                                                                     |
| --- | ------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| N1  | 预检                     | `preflight_check.sh`（15 项：孤儿进程/HBM/磁盘/数据/代码标记/模型完整性）                 | ALL PASS                                                                                                     |
| N2  | 单步训练体检             | 16 卡 1 epoch 前 3 步                                                                     | `[sparse] w1 (256,4096,6144)` 全量；loss_step_0 < 8（warm-start 生效）；无 NaN；fwd/bwd 无 launch-bound 回归 |
| N3  | **写回协议（强制顺序）** | ① requant(**warm-start 对照**) → serve → 验收 ② 通过后才 requant(训练权重) → serve → 验收 | ① ≈ A 卷基线（0.79 级）② 对比 ① 给出训练增益                                                                 |
| N4  | 直挂 BF16 drafter        | `--spec-model` 绕开量化域（需 MTPDraftModel 的 vllm adapter，待开发）                     | 区分 P2（量化域 gap）与训练语义问题                                                                          |
| N5  | 验收测量                 | `measure_acceptance.py` 200 条留出集，显式 `--model`                                      | 每位置接受率 + mean，落 eval_results/                                                                        |

## 五、事故 → 测试映射总表（回归防线）

| 事故                                | 根因                     | 防线                                        |
| ----------------------------------- | ------------------------ | ------------------------------------------- |
| warm-start 专家静默归零             | wrapper 改名键           | U1 键不变式                                 |
| AttributeError: no attribute 'gate' | dispatch 自算路由        | U1 自包含契约                               |
| shared_experts 双重叠加             | dispatch 内加 shared     | U1 自包含契约                               |
| 训练目标被污染（新发现）            | flat_w 未按 argsort 对齐 | U2 等价测试                                 |
| 3D 输入崩溃（潜伏）                 | K=shape[1]               | U2 形状无关测试                             |
| resume 硬崩                         | 零梯度参数缺动量         | U4 填充测试                                 |
| 动量精度损失                        | fp32→bf16 转换           | U4 dtype 测试                               |
| vendor checkpoint 加载失败          | 非标准索引名             | U5 回退测试                                 |
| 梯度 OOM / HCCL error 7             | 冻结/FSDP 粒度           | U6                                          |
| 抽取有损（0.79→0.26）               | scale_bias/约定          | U7（待补）+ N3 强制对照                     |
| serving 0 接受误诊 3 天             | 无 ctrl 基线 + 无效测量  | **N3 强制顺序 + N5 显式 --model**           |
| epoch 保存 OOM（差 5.9MB）          | gather 前未清缓存        | checkpointer empty_cache（N2 覆盖首保存点） |

## 六、运行方式

```bash
# 单元 + 功能（CPU，CI 可跑）
python -m pytest tests/unit/models/test_glm_*.py tests/unit/models/test_glm_vocab_ce.py \
                  tests/unit/train/test_glm_*.py tests/unit/utils/test_glm_loading_fallback.py -q

# NPU 集成（35 机器容器内）
bash /mnt/share_space/l00840580/dspark-glm53/preflight_check.sh   # N1
# N2/N3/N4/N5 按 run_pipeline_v2.sh 各阶段手动触发
```
