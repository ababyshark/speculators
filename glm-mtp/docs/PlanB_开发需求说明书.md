# GLM-5.3 原生 MTP 增训 —— 方案 B 开发需求说明书

> 版本：v1.0（2026-11-03）
> 适用场景：在一台**全新的开发/训练机**上，从零实现"薄适配层 + 上游组件 + 框架共享前向"的 GLM-5.3 MTP 训练路径。
> 本文档自包含：不依赖任何对话上下文。所有"已核实"标注的事实均已对照源码确认，开发者直接引用，不要重新发现。

---

## 0. 一页速览

**做什么**：让 vllm-project/speculators 的 MTP 训练框架支持 GLM-5.3（`glm_moe_dsa`）。不 fork 框架算法、不复制模型组件，只写一个**薄适配层**（~300 行），使 GLM 走框架的 generic 训练前向。

**为什么不是直接照抄 .23 旧实现**：.23（POD1-A3-23 机器）有一套已跑通 100 步训练的实现，但它把 GLM 层整体私有复制（237 行私有组件 + 813 行框架 diff），脱离上游演进路线，每个上游改进都要手工嫁接。方案 B 用上游 transformers 现成组件 + 框架 generic 前向，代码量更小、后续免维护税。**.23 的语义结论（词表、norm、冻结等）全部继承，其实现仅作 parity 参照物。**

**交付物**：speculators 的一个开发分支（可提上游 PR 的形态）+ GLM-5.3 MTP 权重管线三脚本 + 一份在 Ascend NPU 上的 formal100 等价训练 + serving 验收 A/B 报告。

**周期**：Gate 1（CPU 纯开发+测试）1.5-2 天；Gate 2（NPU 机器）3-5 天 + ~2 天机器时间。

---

## 1. 背景

### 1.1 模型与训练目标

GLM-5.3 = `GlmMoeDsaForCausalLM`（transformers `model_type="glm_moe_dsa"`）：78 层主模型（MoE + DSA 稀疏注意力 + MLA + lightning indexer）+ **第 78 层原生 MTP 头**（`model.layers.78.*`，793 个张量）。MTP 头结构 = enorm/hnorm 双 RMSNorm + eh_proj 拼接下投影 + 一个 DSA DecoderLayer + shared_head（norm + head）+ **独立的 embed_tokens**。

训练流程（speculators 框架的 MTP 算法）：teacher vLLM 服务导出 verifier 第 78 层（final-norm 后）隐状态 → MTP 头做 teacher-forced 微调（step k 的输入 = 隐状态 + 真实 token t+k+1 的 embedding，监督目标是 t+k+2）→ 导出缝回 vLLM 布局 → serving 投机解码验收（Mean acceptance length）。

当前**只支持单预测步**（`num_speculative_steps=1`）。多步是 Phase 3 可选项，见 §6.3。

### 1.2 基线与参考实现（.23）

.23 机器（Ascend 16×910B）在 speculators@`7ed657f`（"Async dataloader (#1310)"，0.10.0a28）上打过一套本地补丁并跑通 100 步：loss 0.503→0.380，val 0.369，token acc 0.868，峰值显存 23.4GB/卡。**它的权重语义全部正确、已经 native parity 验证**，但代码组织是"整体私有复制"，方案 B 要以它的语义为规格、以它的代码为 parity 参照，重写组织形式。

.23 全部材料的字节级归档：`glm_patch_bundle.tar.gz`（239KB，sha256 `1c5fd40906ed39bcf117c6c4cb9736f88cce73e6d295f80cd8e6916652ff594a`）。解压后目录见 §2.4。

---

## 2. 环境与材料清单

### 2.1 软件基线（钉死，不许浮动）

| 组件 | 版本 | 说明 |
|---|---|---|
| 容器镜像 | `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` | 含 vllm 0.30.0+empty / vllm-ascend 0.30.0rc2 / torch 2.10.0+cpu / python 3.12.13 |
| transformers | **5.14.1** | 容器现役版本。✅ 已核实该版本 `GlmMoeDsaIndexer`/`GlmMoeDsaAttention` 支持 `past_key_values=None` 的无 cache 训练前向（attention_mask=None 时用 position_ids 自建因果掩码），**无需升级** |
| speculators | checkout `7ed657f`（main） | 基线 commit |
| torch_npu | 2.10.0.post7 | ⚠️ .23 用 post7；.26 容器自带 post4，**开工前核对并统一到 post7**（post7 wheel 曾��于 /tmp/torch_npu/，已不在，需另寻） |
| venv 内追加 | datasets==5.1.0, numpy==2.2.6, pyarrow==26.0.0, setuptools-git-versioning | 完整清单见 bundle 内 `venv_packages.txt` |

环境搭建命令（NPU 训练机上）：

```bash
python -m venv --system-site-packages /data1/mtp_training/venv
source /data1/mtp_training/venv/bin/activate
pip install -e speculators -e speculators/hs_connectors --no-deps --no-build-isolation
pip install datasets==5.1.0 numpy==2.2.6 pyarrow==26.0.0 setuptools-git-versioning
```

### 2.2 硬件与数据

- NPU 训练：16 × Ascend 910B（每 die 64GB HBM）。teacher 用卡 0-7（TP8+EP），训练用卡 8-15（FSDP 8 进程）。
- Gate 1 **不需要 NPU**：任何有 python 3.12 + pip 的机器即可（CPU 组件测试）。
- verifier 权重：`/data1/weight/GLM-5.3-w4a8-spec`（Ascend w4a8 量化包，106 分片；MTP 层张量本身是浮点）。.26 机器上可经 `/mnt/data1/weight/GLM-5.3-w4a8-spec`（.27 的 sshfs 挂载）访问——config.json sha256 前缀 `22189a89ed44f45a`，已核对与 .23 同一份。sshfs 有 D-state 卡死前科，长任务前先 `sha256sum` 完整校验。
- 训练数据：`prepared_8k_dflash2_official`（22,070 条 8K：train 19,875 / val 2,195，Arrow 格式，含 token ids + loss mask；`train_data_ratio=0.9005663797009516`）。在 .23 的 `/data1/mtp_training/data/` 下，需 scp 过来（~416MB）。
- 磁盘：extract+convert 产物 ~45GB + checkpoint ~112GB/run，预留 ≥300GB。

### 2.3 上游 vendor 依赖：PR #1226

speculators 的 [PR #1226](https://github.com/vllm-project/speculators/pull/1226)（"Add opt-in vLLM-aligned training forward"，open 未合并，538 行 / 4 文件）**必须 vendor**（作为本地 patch 携带；它合并后改为 rebase）。它提供：

1. **监督位置 CE**：LM head + CE 只在有监督位置算，2048-token 分块 + per-chunk checkpoint（`_head_ce_chunked`）——比 .23 的全窗口 64-chunk 更省更快，GLM 免费获得
2. **冻结 `verifier_norm`**：trainer 侧补 final-norm（`resolve_verifier_norm_class`）——⚠️ GLM 有双重 norm 规避，见 §7 坑 1
3. **`_forward_ttt`**：vLLM proposer 对齐的多步前向语义规格（step-k 只看 step-0 KV + 本链 KV；flex-attention 实现，NPU 不可用，Phase 3 需替换实现但**语义照抄**）
4. **训练期 accept 指标**：`accept_pos_{k}_sum/total`（累积接受率，与 vLLM `/metrics` 的 per-position acceptance 对齐）
5. 返回契约 logits→draft_tokens、`--mtp-train-forward {fastmtp,vllm}` CLI

获取：`curl -L https://github.com/vllm-project/speculators/pull/1226.diff -o pr1226.diff`，`git apply pr1226.diff`（基于 7ed657f 应能干净套上；若 context 漂移，手工对齐 4 个文件：`src/speculators/models/mtp/core.py`、`src/speculators/train/config/schema.py`、`tests/unit/models/test_mtp_model.py`、`tests/unit/train/test_cli_args.py`）。

### 2.4 .23 归档内容（bundle/）

| 路径 | 内容 | 用途 |
|---|---|---|
| `speculators_git.diff` | .23 对 7ed657f 的完整 diff（813 行） | **语义参照**：训练稳健性改动（FSDP/checkpointer/dataloader/trainer）部分**原样沿用**，见 §5 W4 |
| `glm_mtp.py` | .23 的私有 GLM 层（237 行） | **parity 参照物**：Gate 1 逐位比对的基准实现 |
| `code_mtp/` | extract/convert/export 三脚本、install 接线、teacher 启动、parity compare 工具链、`test_glm_mtp_components.py`、`README_glm53_mtp_23.md`、formal100 脚本 | W3/W5/W7 的底稿；**test_glm_mtp_components.py 的堆叠变换代码直接复用** |
| `code2/` | teacher 扩展链（sorted SFA 确定性、final-norm hook、native capture）、v2 交付文档 | W5/W6 底稿 |
| `formal100_train_command.txt` / `run.yaml` / `final100_verified.json` | formal100 完整 provenance | W7 复刻参数来源 |
| `base_commit.txt` / `venv_packages.txt` | 基线与依赖记录 | 环境核对 |

---

## 3. 目标架构（方案 B）

```
┌─ transformers 5.14.1（模型知识，零改动）────────────────────┐
│  GlmMoeDsaDecoderLayer / GlmMoeDsaAttention / GlmMoeDsaIndexer │
│  GlmMoeDsaMoE(Experts=3D堆叠) / GlmMoeDsaRMSNorm / Rotary      │
└───────────────────────────────────────────────────────────┘
            ▲ 组合（不是复制）
┌─ speculators 开发分支 ─────────────────────────────────────┐
│  models/mtp/glm_glue.py        ← 新文件 ~150 行（本需求核心）  │
│    GlmMTPLayer = MTPLayerMixin + GlmMoeDsaDecoderLayer        │
│  models/mtp/model_definitions.py  + glm_moe_dsa 注册（~15 行）│
│  models/mtp/core.py               + GLM 分支（~40 行）          │
│  train/* 的 .23 稳健性 diff      ← 原样沿用（W4）              │
│  + vendored PR #1226                                          │
└───────────────────────────────────────────────────────────┘
            ▲ 调用
┌─ 权重管线（外围脚本，改造自 code_mtp/）──────────────────────┐
│  extract_native_mtp.py（+堆叠变换）→ convert → export          │
└───────────────────────────────────────────────────────────┘
```

**原则**：GLM 专属知识只出现在两处——`glm_glue.py`（层组合）和权重管线（key 映射 + 堆叠变换）。算法（loss/step weights/优化器/调度）全部走框架 generic 路径，不写一行 GLM loss。

---

## 4. 已核实事实（开发依据，直接引用）

以下每一条都已对照源码逐行确认过（transformers==5.14.1 / speculators@7ed657f），**不要重新验证、不要凭记忆推翻**：

**F1 接口契约**。`MTPLayerMixin.forward`（speculators `models/mtp/model_definitions.py:95-116`）签名 `forward(hidden_states, token_embeddings, attention_mask=None, position_ids=None, position_embeddings=None, **kwargs)`，内部：`input_proj(cat([token_layernorm(token_embeddings), hidden_layernorm(hidden_states)]))` → `super().forward(proj, attention_mask=..., position_ids=..., position_embeddings=...)` → `final_layernorm(output)`。`.23` 私有层（bundle `glm_mtp.py:73-78`）与此**逐行语义相同**。

**F2 上游层可无 cache 前向**。5.14.1 `GlmMoeDsaDecoderLayer.forward` 的 `past_key_values=None` 路径：indexer 在 `attention_mask=None` 时用 position_ids 自建因果掩码（与 .23 手写逻辑同构）；attention 同样支持。MTP 层构造时需覆写 `config.indexer_types=["full"]`、`config.mlp_layer_types=["sparse"]`（.23 做法，保留）。

**F3 注册契约**。`ModelComponents(first_layer_class, norm_class, rotary_emb_class)`（`models/base_components.py:17` NamedTuple）。`MTPDraftModel.__init__` 用 `first_layer_class(tc, layer_idx=0)` 构造层、`rotary_emb_class(flatten_rope_parameters(tc))` 构造 rotary（`core.py:128-131`）。工厂函数 `_create_mtp_layer_class(name, decoder_class, norm_class, hybrid=...)` 现成（`model_definitions.py:119-150`），GLM 复用它。

**F4 generic 前向可吃 GLM 层**。`MTPDraftModel.forward`（`core.py:158-279`）对层只传 `create_causal_mask(...)` 的 4D mask、`position_ids`、`position_embeddings`、`hidden_states`/`token_embeddings`——GlmMoeDsaDecoderLayer 签名全部接受（`prev_topk_indices` 走默认 None，单层自含 indexer）。**两个待验证点**（Gate 1 任务）：`create_causal_mask` 对 `GlmMoeDsaConfig` 的输出形状/dtype；`flatten_rope_parameters` 是否保留 `rope_parameters` dict（rotary 需要 `rope_type`/`rope_theta`）。若后者丢失：注册的 rotary 类做成"单参构造 + 从 config 直接读 dict"。

**F5 专家布局差异（本需求最大的坑）**。native checkpoint 是**逐专家**布局：`model.layers.78.mlp.experts.{i}.gate_proj/up_proj/down_proj.weight`（768 张量）。5.14.1 上游 `GlmMoeDsaExperts` 是 **3D 堆叠**：`experts.gate_up_proj [E, 2I, H]` + `experts.down_proj [E, H, I]`（2 个大参数）。**堆叠变换代码已存在**：bundle `code_mtp/test_glm_mtp_components.py:39-43`——`gate_up_proj[i] = cat([gate_proj.weight, up_proj.weight])`（gate 在前），`down_proj[i] = down_proj.weight`。变换放在 extract 脚本（一次性 + 哈希校验）。

**F6 key 映射（方案 B 版）**。native → 训练态（speculators 布局）：

| native（`model.layers.78.` 前缀下） | 训练态 | 备注 |
|---|---|---|
| `self_attn.*`、`mlp.gate.*`、`mlp.shared_experts.*` | `mtp_layers.0.` + 同名 | **恒等**（子模块名与上游一致） |
| `mlp.experts.{i}.gate_proj/up_proj/down_proj.weight` | `mtp_layers.0.mlp.experts.gate_up_proj` / `.down_proj` 堆叠 | §F5 |
| `enorm.*` / `hnorm.*` / `eh_proj.weight` / `shared_head.norm.*` | `token_layernorm.*` / `hidden_layernorm.*` / `input_proj.weight` / `final_layernorm.*` | mixin 模块名 |
| `shared_head.head.weight` → `lm_head.weight`；`embed_tokens.weight` → `embed_tokens.weight` | 顶层 2 张量 | **独立词表**（F7） |

**F7 独立词表（.23 最重要的语义发现）**。GLM MTP 的 `embed_tokens` 和 `shared_head.head` 与主模型**不同**（extract 时 sha256 比对不一致，已证实）。因此：不能用框架默认的"从 verifier 拷词表"；必须保留 .23 的 `mtp_shared_weights_path` 机制——config 加该字段（str），`load_verifier_shared_weights` 的 GLM 分支从该路径（native 提取目录）读这两个张量、shape 校验、`nn.Parameter(..., requires_grad=False)` 冻结加载。这 2 张量不进训练 checkpoint，export 时从 native 目录补回。**此部分 .23 代码原样移植**（bundle `glm_mtp.py:90-101`）。

**F8 三处加载/构建期修复必须保留**（.23 踩坑记录）：(a) 加载完成后**显式重冻结 indexer**（`requires_grad_(False)`，meta 加载流程会意外解冻）；(b) 量化包会给 RMSNorm 全局挂可选全零 bias，加载时摘除（native capture 扩展的 `load_weights_with_optional_norm_bias`）；(c) position-0 的 token embedding 置零（MTP 序列起点无前 token）。

**F9 teacher 隐状态语义**。vLLM `--target-layer-ids 78` 原生 capture 抓的是第 78 层 pre-final-norm 输出，而 GLM MTP 吃 `model.norm` **之后**的 hidden。.23 用 vLLM worker 扩展 hook 把连接器输出替换为 final-norm 后张量（已 parity 验证）。**方案 B 保留 teacher hook**（见坑 1 的对比说明）。

**F10 #1226 已合并的通用改动 GLM 免费获得**：监督位置 CE、TTT 语义、accept 指标、draft_tokens 契约。GLM 需要的增量只有一个 flag（坑 1）。

---

## 5. 工作包分解

依赖关系：W1→W2→W3（CPU 可完成）→ W4 → W5 → W6 → W7 → W8（NPU）。

### W1 核心：`glm_glue.py` 层组合（~150 行，CPU 可开发）

**位置**：`src/speculators/models/mtp/glm_glue.py`

**内容**：
1. `GlmMTPRMSNorm(GlmMoeDsaRMSNorm)`：fp32 前向（`.float()` 计算、weight `.float()`、回 cast）——.23 现成（bundle `glm_mtp.py:11-15`），原样搬。理由：NPU bf16 RMSNorm 与 vLLM `AscendRMSNorm` 舍入不一致会毁 parity。
2. `GlmMTPRotaryEmbedding(GlmMoeDsaRotaryEmbedding)`：`copy.deepcopy(config)` 后 `config.head_dim = config.qk_rope_head_dim`（rope 只作用于 64 维 rope 分量）。注意签名 `(config, device=None)` 保持 `device=None` 默认以兼容 `core.py` 单参构造。若 F4 的 `flatten_rope_parameters` 验证失败，在此类里从 config 直接读 `rope_parameters` dict。
3. `GlmMTPLayer = _create_mtp_layer_class("GlmMTPLayer", GlmMoeDsaDecoderLayer, GlmMTPRMSNorm)` 之后的子类覆写，共 3 点：
   - `__init__`：调 super 后设置 `config.indexer_types=["full"]`、`config.mlp_layer_types=["sparse"]`（注意在 super().__init__ **之前**改 copy 的 config，参照 .23 `glm_mtp.py:67`），并对 `self.self_attn.indexer.requires_grad_(False)`
   - `forward`：调 mixin 前加一行 position-0 置零：`token_embeddings = torch.where(position_ids[..., None] == 0, 0, token_embeddings)`，其余全部走 mixin
   - 加载后重冻结钩子（配合 W2 的 core 分支）
4. 组件级单元测试（CPU）：`tests/unit/models/test_glm_mtp_glue.py`——
   - 构造 fake `GlmMoeDsaConfig`（小尺寸：hidden 64、8 experts、index_topk 4、q_lora_rank 16、kv_lora_rank 8、`rms_norm_eps=1e-5`），实例化 `GlmMTPLayer`
   - **parity 主测试**：同一 state_dict（含逐专家→堆叠变换）分别喂 `.23` 的 `GlmMTPLayer`（从 bundle 拷入测试目录）和新层，比较输出 `torch.testing.assert_close(rtol=1e-5, atol=1e-5)`——.23 的 `test_glm_mtp_components.py` 提供变换代码和测试骨架，反向复用
   - indexer 冻结断言：构造后和 `load_state_dict` 后 `requires_grad` 均为 False
   - position-0 置零断言：`position_ids` 首位 0 时输出不受首位 embedding 影响
   - 无 cache 前向：`past_key_values=None` + `attention_mask=None` 路径能跑通且与显式 4D 因果 mask 路径输出一致

**验收**：`pytest tests/unit/models/test_glm_mtp_glue.py -q` 全绿；与 .23 层 parity 逐位通过。

### W2 注册与 core 分支（~55 行）

**内容**：
1. `model_definitions.py`：`mtp_model_classes["glm_moe_dsa"] = base_components.override_components("glm_moe_dsa", first_layer_class=GlmMTPLayer, norm_class=GlmMTPRMSNorm, rotary_emb_class=GlmMTPRotaryEmbedding)`。注意 `base_components.model_classes` 需先有 `glm_moe_dsa` 条目（若上游 base_components 无此 model_type，用 `ModelComponents(...)` 直接构造）。
2. `config.py`（mtp 的）：`+ mtp_shared_weights_path: str | None = None`。
3. `core.py`：
   - `load_verifier_shared_weights` 开头 GLM 分支：无 `mtp_shared_weights_path` 即 raise；有则调 `load_native_shared_weights(self)`（.23 `glm_mtp.py:90-101` 原样搬，挪进 glm_glue.py）后 return，**跳过**框架默认的 verifier 词表拷贝
   - 加载完成后 `self.mtp_layers[0].self_attn.indexer.requires_grad_(False)`
   - （配合 #1226）GLM 的 `hidden_states` 已是 post-norm，`_forward_ttt` 入口加 per-model-type 判断跳过 `self.verifier_norm(...)`——实现为一个注册表 flag（如 `ModelComponents` 加字段 `verifier_hidden_is_post_norm: bool = False`，GLM 传 True），**不要写 if model_type == "glm_moe_dsa" 散弹**

**验收**：`MTPDraftModel(MTPSpeculatorConfig(transformer_layer_config=GlmMoeDsaConfig(...)))` 能实例化；`resolve_model_type("glm_moe_dsa")` 通过；fake 权重 `save_pretrained`→重新加载，`embed_tokens`/`lm_head` 不在 checkpoint、`mtp_shared_weights_path` 生效。

### W3 权重管线三脚本（改造 `code_mtp/` 底稿，~200 行改动）

**extract_native_mtp.py**（底稿：bundle `code_mtp/extract_native_mtp.py`）：
- 保留：只读源、793 张量计数断言、浮点断言、finite 检查、分片写 + 读回 sha256 逐张量校验、独立词表对比报告（embed/lm_head vs 主模型，预期不一致）
- **新增**：逐专家→3D 堆叠变换（F5 公式），变换后再做一次 sha256 校验；index 里写堆叠 key（训练态命名，F6 表）
- 产物：`glm53_mtp_native/`（含 `model.safetensors.index.json` 用**训练态 key**——这样 convert 就是恒等复制）

**convert_glm_mtp.py**：大幅简化——meta 实例化新 MTPDraftModel → expected keys；从提取目录拷张量（key 已是训练态，理论恒等，仍做逐张量 shape+sha256 校验）；config 写入 `mtp_shared_weights_path`；排除 2 个词表张量。

**export_glm_mtp_for_vllm.py**（底稿保留）：逆映射回 `model.layers.78.*`（堆叠→逐专家逆变换）+ `mtp_shared.safetensors` 补 2 个共享张量 + symlink 不复制大文件 + export_report.json 逐张量 sha256。**vLLM 服务读的是逐专家 native 布局，堆叠逆变换必须有。**

**验收**：extract→convert→(空训练)→export 往返：export 目录与 native 提取物逐张量 sha256 一致（除 2 个共享张量来自原始提取）；793 == 791 训练 + 2 共享。

### W4 训练稳健性 diff 原样应用（0 新代码）

从 bundle `speculators_git.diff` 里挑以下文件改动**原样套用**（它们与 GLM 专属代码正交，已实测）：`train/distributed.py`（细粒度 FSDP：GLM 分支改为 `children = [layer.self_attn, layer.mlp.gate, layer.mlp.shared_experts, layer.mlp.experts]`——注意方案 B 下 `mlp.experts` 是单模块不是 ModuleList，这是唯一需要改的一行）、`train/checkpointer.py`（fp32 落盘、分片加载、CPU 加载不换 dtype）、`train/dataloader.py`（`SPECULATORS_MTP_NO_PACK=1` 单样本采样器——**GLM 训练必须开**）、`train/trainer.py`（fp32 master、RNG 断点、validation_rng）、`train/rng_state.py`（新文件）、`train/cli.py` + `config/schema.py`（deterministic_algorithms / eval-only / max-val-batches）、`hs_connectors/transfer.py`（lock-timeout）。

**验收**：`pytest tests/unit/train -q` 无回归；`git diff` 里 GLM 专属内容只出现在 W1/W2 文件。

### W5 Teacher 服务 + parity（NPU，工具链全现成）

`code2/` 的 teacher 扩展链**原样使用**：(a) `GLMTeacherDeterminism`（worker 内 `torch.use_deterministic_algorithms(True)`）；(b) `sorted_sfa`（`AscendSFAImpl` 打补丁，稀疏注意力 topk 索引按逻辑位置排序——**Ascend SFA 结果依赖物理页序，不排序 parity 必失败**）；(c) `final_hidden_hook`（capture 换 final-norm 后输出，F9）。

parity 验收（不可省）：`start_glm_mtp_native_parity.sh`（起 native deepseek_mtp 投机服务）+ `glm_mtp_native_capture_extension`（抓 native MTP 前向输入/输出）+ `request_glm_mtp_native_4096.py` + `compare_glm_mtp_native.py`——**新实现的 GLM 层 forward vs vLLM native MTP forward 逐张量比对**。这是方案 B 的 go/no-go：parity 不过，后面的训练全无意义。

teacher 启动命令以 bundle `code_mtp/start_glm_mtp_teacher.sh` 为准（卡 0-7，TP8+EP，`--enforce-eager`，`--max-model-len 8193`，`--quantization ascend`，`--attention_config.indexer_kv_dtype int8`，seed 1024，LCCL_DETERMINISTIC=1 等；`launch_vllm.py` **必须传 `--provenance-dir`**）。

**验收**：compare_glm 在 4096-token 探针上全绿（tolerance 以 .23 当时通过的门限为准，见 code_mtp README）。

### W6 容量阶梯（NPU）

512 → 2K → 8K，各：单步反向 + 20 步连续 + checkpoint 恢复。OOM 降级顺序：`mtp_index_query_chunk`（若 W1 决定保留分块 indexer 子类则用之，否则先降 CE chunk 2048→1024→512）。同时核对：堆叠 experts 的 FSDP all-gather 峰值（风险 §7 R3）。

**验收**：8K×20 步无 OOM、loss 下降、恢复后 step 号正确（.23 基线：8K 单步 174s、峰值 23.4GB/卡——方案 B 因监督位置 CE 应**快于或等于**此值，显存**低于或等于**此值）。

### W7 formal100 等价训练（NPU）

参数以 bundle `formal100_train_command.txt` 为准（lr 1e-5、adamw eager、cosine 100 步、`--num-speculative-steps 1`、`--max-val-batches 1`、`--fsdp-shard --deterministic-algorithms`、reduce dtype bf16、NO_PACK=1、`--on-missing generate` + file backend）。改动点：`--from-pretrained` 指向 W3 新 convert 产物；建议同时开 async dataloader 预取（`--num-workers >0`，基线 #1310 本身支持；.23 当时用 0 没吃红利，8K 每步 160s 在等 teacher）。

**验收**：100 次有效更新；loss 轨迹与 .23 相似量级（起点 ~0.50、终点 ≤0.40；不要求逐位复现，但 val loss ≥ 0.45 视为异常，需排查 parity）；provenance 齐全（train_command.txt、speculators.patch 自动生成）。

### W8 export + serving 验收 A/B（NPU）

export（W3 脚本）→ `vllm serve <exports_dir> --speculative_config '{"method":"mtp","num_speculative_tokens":1}'`（TP8）。**A = step0 基线，B = trained**，两服务同 prompt 集，从 `/metrics` 读 `SpecDecoding` Mean acceptance length / Per-position acceptance rate。

⚠️ **.23 重大教训（必读）**：.23 的最终验收服务实际加载了**训练前**的快照（stitched 目录 mtime 早于训练开始），"final vs baseline"探针数字几乎相同是 step0 vs step0，**验收无效**。本工作包的验收服务**必须指向 W3 export 的产物目录**，并在报告中附 export_report.json 的 sha256 证明加载的是训练权重。

**验收**：trained 的 MAL **显著高于** step0 基线（.23 的 step0 基线 MAL ~1.9-2.0，训练后期望 ≥2.3；具体门限以 20 步 pilot 的趋势外推）。发布时附全套 provenance（train_command.txt / vllm_command.txt / checkpoint_sha256.txt / patches）。

---

## 6. 阶段划分与 Go/No-Go

### 6.1 Gate 1（CPU，1.5-2 天）——先做这个
W1 + W2 + W3 的 extract/convert 部分 + 单元测试。**不需要 NPU、不需要真实权重**（用 fake config + 随机权重；真实权重校验可用 sshfs 只读跑 extract 的校验逻辑）。
- **Go 条件**：新层 vs .23 层 CPU parity 逐位通过；793→791+2 权重管线往返一致；`create_causal_mask`/`flatten_rope_parameters` 两个契约点验证通过
- **No-Go 止损**：若 `create_causal_mask` 对 GlmMoeDsaConfig 不可用且无法在注册层修（比如需要改 masking_utils），保留 .23 的 `_glm_causal_mask`（10 行）作为 GLM 分支的 mask 来源——这是允许的兜底，不构成 No-Go。真正的 No-Go 只有：新层与 .23 层语义无法对齐（此时退回方案 A 照抄 .23，损失锁定在 Gate 1 的 2 天）

### 6.2 Gate 2（NPU 训练机，3-5 天 + 2 天机器时间）
W4 → W5 parity → W6 阶梯 → W7 formal100 → W8 验收。
- **Go/No-Go 门槛在 W5**：native parity 不过则停止，回到 Gate 1 排查

### 6.3 Phase 3（可选，1-2 周）：多步（num_speculative_steps 3-5）
- 语义规格**照抄** #1226 `_ttt_mask_mod`/`_forward_ttt`（step-0 KV 只读 + 本链 KV；位置每步 +step）
- 实现**必须替换** flex-attention（NPU 无）：物化 bool mask（8K×5 步 ~335MB 可承受）+ SDPA/eager
- **indexer 多步语义是最大未知数**：先用 bundle `code2/glm_mtp_native_capture_extension.py` 抓 vLLM native MTP 在 step≥1 的实际输入（input_ids/positions/topk_indices/hidden），compare 对齐后再写训练路径
- MLA 的 `_TTTCache` 变体：GlmMoeDsaAttention 在 `past_key_values` 非 None 时 update 的是**展开后** KV（[B,H,S,qk_head_dim]/[B,H,S,v_head]）——cache 存展开 KV，5 步 ~1GB/层可承受
- vLLM 端多步 KV 成本参考：社区报告 native MTP 5 steps KV 0.0922 MiB/token（speculators issue #1175）

---

## 7. 坑清单（每一条都是 .23 真金白银踩出来的）

1. **双重 norm**：teacher hook 已输出 final-norm 后张量（F9）；#1226 的 vllm 前向又会过 `verifier_norm`。GLM **必须跳过**（W2 的 flag）。备选：撤 hook 改用 trainer 侧 norm——不推荐（要额外提取 `model.norm.weight`、量化包 norm 可能带全零 bias、与 serving 时 bf16 norm 有数值差）。
2. **indexer 加载后意外解冻**：meta 加载流程会恢复 requires_grad——加载完显式重冻结，且在测试里断言（W1）。
3. **打包**：GLM 前向只接受未打包单序列。训练必须 `SPECULATORS_MTP_NO_PACK=1`、batch=1。数据 `document_ids` 断言只允许右 padding 单文档。
4. **`loss_mask[:, 2:]` 可能为空**：2K 截断样本若无监督目标会 NaN——保留 .23 的过滤守卫（forward 里 valid_count clamp + 空 mask 提前返回零 loss）。
5. **Ascend SFA 物理页序噪声**：teacher 端不排序就开训，parity 和训练本身都会被随机噪声污染——sorted_sfa 扩展链必须挂。
6. **w4a8 量化包的 RMSNorm 可选 bias**：加载时摘除（native capture 扩展里的 `load_weights_with_optional_norm_bias` 已有实现）。
7. **磁盘写回竞争**：.23 formal100 收尾 SIGBUS（XFS + 磁盘只剩 152G），模型 shard 写出但 training_state.json 停在 step20。开跑前清盘 ≥300G；checkpoint 频率 20 步起步，磁盘吃紧就拉大。
8. **验收目录指向**（重复强调）：serving A/B 的 B 侧必须指向 export 产物并留 sha256 证据。
9. **`--include-last-layer` off-by-one**（speculators issue #705）：`launch_vllm.py` 对 78 层模型会把 78 当最后层索引（应为 77）。.23 用 `--target-layer-ids 78` + hook 恰好绕开；**不要"修复"成 include-last-layer**，保持现状。
10. **`rms_norm_eps` 必须读 config**（=1e-5），不许硬编码 1e-6。
11. **kv dtype**：teacher 端 `--attention_config.indexer_kv_dtype int8` 是 parity 门限的一部分，不要动。

---

## 8. 上游 PR 规约（如果决定把 W1/W2 提给 speculators）

遵循仓库 `AGENTS.md`/`CONTRIBUTING.md`：
- 读 `CONTRIBUTING.md`；装 dev 依赖后 `pre-commit install`
- PR 模板：**Purpose** 1-3 句讲为什么（"GLM-5.3 ships a native MTP layer at model.layers.78; register it so MTP training works for glm_moe_dsa"），不重述 diff
- **Tests**：给可运行命令（`pytest tests/unit/models/test_glm_mtp_glue.py -q`）+ 是否真跑过。**没跑过的数字不许写**
- 不写 emoji 标题、不用 "comprehensive/robust"；描述长度与 diff 成比例；不留硬换行
- "I (a human) have written or reviewed the code" 复选框**不勾**
- 任何 `launch_vllm.py` 使用都带 `--provenance-dir`；发布结果附 train_command.txt / vllm_command.txt / checkpoint_sha256.txt / patches 全套
- 建议拆分：PR-1 = 注册 + glue 层 + CPU 测试（无 NPU 依赖，reviewer 可复现）；PR-2 = 权重管线脚本（或放 examples/）。#1226 若已合并则 rebase 到 main。

---

## 9. 交付物清单

| # | 交付物 | 形式 |
|---|---|---|
| D1 | speculators 开发分支（W1/W2/W4 + vendored #1226） | git 分支 + 可提 PR 的 commit 序列 |
| D2 | 权重管线三脚本（W3） | 独立脚本 + 各自 `_report.json` |
| D3 | 组件测试 + parity 测试 | pytest，全绿 |
| D4 | NPU parity 记录（W5） | compare_glm 输出 + provenance |
| D5 | formal100 run（W7） | checkpoint + train_command.txt + speculators.patch + run.yaml |
| D6 | 验收 A/B 报告（W8） | MAL/acceptance 对比 + export_report sha256 证明 |
| D7 | 本文档的执行记录 | 各 Gate 的 go/no-go 结论 |

---

## 10. 快速参考：关键数字

- 793 = 791 可训练 + 2 独立词表张量
- 词表 154,880；hidden/qk_rope_head_dim=64（rotary 只用 64 维分量）；78 层主模型 + layer 78 MTP
- formal100 基线（.23）：loss 0.503→0.380 / val 0.369 / token acc 0.868 / 峰值 23.4GB/卡 / 8K 单步 174s（160s 等 teacher）/ checkpoint ~112GB
- step0 基线 MAL ~1.9-2.0（.23 探针，无效验收的产物，仅作下限参考）
- 数据 22,070 条（19,875 train / 2,195 val），`train_data_ratio=0.9005663797009516`
- bundle sha256 `1c5fd40906ed39bcf117c6c4cb9736f88cce73e6d295f80cd8e6916652ff594a`
