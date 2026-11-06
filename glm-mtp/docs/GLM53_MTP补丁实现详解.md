# GLM-5.3 原生 MTP 增训补丁 —— 实现详解与重现指南

> 来源：`.23`（POD1-A3-23，172.18.32.23）容器 `vllm_030_mtp_train`（镜像 `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`）内 `/workspace/speculators` 工作树 + `/workspace/code_mtp`、`/workspace/code2` 脚本目录，2026-11-02/03 已端到端跑通 100 步正式训练。 本地归档：`glm53_mtp_patch_study/bundle/`（字节级 tar 包 `glm_patch_bundle.tar.gz`，sha256 `1c5fd40906ed39bcf117c6c4cb9736f88cce73e6d295f80cd8e6916652ff594a`）。

______________________________________________________________________

## 0. 一页总览

### 0.1 训练拓扑（.23 实测布局）

```
172.18.32.23 单机 16×Ascend910（每 die 64GB HBM）
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│ NPU 0–7：Teacher vLLM（TP8+EP）    │  │ NPU 8–15：torchrun 8 进程 FSDP     │
│  端口 8000，--enforce-eager        │  │  speculators.train (MTP, 单步)    │
│  extract_hidden_states 连接器      │  │  --on-missing generate            │
│  输出: /data1/mtp_training/        │  │  每 batch 向 8000 请求隐状态       │
│    teacher/hidden_states/*.safetensors ──file backend──▶ 训练端读取      │
└──────────────────────────────────┘  └──────────────────────────────────┘
```

- 在线隐状态：**file 后端**（不用 mooncake），`ExampleHiddenStatesConnector`（kv_producer）写 safetensors + `.lock` 文件，训练端轮询读取。
- 训练端 `--on-missing generate`：缓存未命中时实时向 teacher 请求。8K 单步 ≈174s，其中 ≈160s 是等 teacher 出特征；峰值显存 ≈23.4GB/卡。

### 0.2 版本基线（重现时必须对齐）

| 组件                       | 版本                                                                | 说明                                                               |
| -------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 容器镜像                   | `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`                          | Python 3.12.13                                                     |
| speculators 基线           | **`7ed657f`**（main，"Async dataloader (#1310)"，版本号 0.10.0a28） | 所有改动叠加在此 commit 之上                                       |
| vllm / vllm-ascend         | 0.30.0+empty / 0.30.0rc2                                            | vLLM 侧含 `extract_hidden_states` + `ExampleHiddenStatesConnector` |
| torch / torch_npu          | 2.10.0+cpu / **2.10.0.post7**（venv 内）                            | ⚠️ .26 容器是 post4，需核对                                        |
| transformers               | 5.14.1                                                              | 必须含 `transformers.models.glm_moe_dsa.modeling_glm_moe_dsa`      |
| datasets / numpy / pyarrow | 5.1.0 / 2.2.6 / 26.0.0（venv 内）                                   | 只装 venv，全局不动                                                |
| verifier 权重              | `GLM-5.3-w4a8-spec`（106 分片，含原生 MTP 层）                      | Ascend w4a8 量化包，MTP 张量本身是浮点                             |

### 0.3 补丁全景（12 个已跟踪文件改动 + 5 个新文件）

| 类别           | 文件                                              | 作用                                                                            |
| -------------- | ------------------------------------------------- | ------------------------------------------------------------------------------- |
| **核心新文件** | `src/speculators/models/mtp/glm_mtp.py`（237 行） | GLM MTP 全部组件 + 训练 forward + key 映射                                      |
| 新文件         | `src/speculators/train/rng_state.py`（44 行）     | per-rank RNG 保存/恢复 + 固定 seed 验证上下文                                   |
| 框架接线       | `models/mtp/config.py`                            | + `mtp_shared_weights_path`                                                     |
| 框架接线       | `models/mtp/model_definitions.py`                 | 注册 `glm_moe_dsa`                                                              |
| 框架接线       | `models/mtp/core.py`                              | fp32 严格模块集合；GLM 加载分支；GLM forward 分发                               |
| 框架接线       | `models/mtp/data.py`                              | `shift_batch_mtp` 透传 `document_ids`、`lengths` 变可选                         |
| 训练稳健性     | `train/checkpointer.py`                           | fp32 默认落盘；分片加载；不隐式换 dtype                                         |
| 训练稳健性     | `train/cli.py` + `train/config/schema.py`         | `deterministic_algorithms`、`eval_only`、`eval_before_train`、`max_val_batches` |
| 训练稳健性     | `train/dataloader.py`                             | `SPECULATORS_MTP_NO_PACK=1` 单样本采样器                                        |
| 训���稳健性    | `train/distributed.py`                            | GLM 细粒度 FSDP；reduce dtype 可配                                              |
| 训练稳健性     | `train/trainer.py`                                | fp32 master 参数；RNG 断点续训；mid-epoch 语义修复                              |
| 传输           | `hs_connectors/transfer.py`                       | `--hidden-states-lock-timeout`                                                  |

完整 diff 见 `bundle/speculators_git.diff`（813 行，267 insertions / 61 deletions）。

______________________________________________________________________

## 1. 为什么官方 speculators MTP 流程跑不了 GLM-5.3

官方 MTP 路径（`docs/user_guide/algorithms/mtp.md`）只支持 `qwen3` / `qwen3_next` / `qwen3_5_text` / `qwen3_5_moe_text`：一个标准 decoder layer + `input_proj`（2·hidden→hidden），**复用主模型的 embed_tokens 和 lm_head**。

GLM-5.3（`GlmMoeDsaForCausalLM`，78 层主模型 + 第 78 层原生 MTP）有四个硬差异：

1. **注意力是 DSA 稀疏注意力**：每层带 Indexer（lightning indexer top-k 选 token），MTP 层用 `indexer_types=["full"]` + MLA 注意力（`q_lora_rank`/`kv_lora_rank` 压缩 + `qk_rope_head_dim=64` rope 分量），不是标准 MHA/GQA。
2. **FFN 是稀疏 MoE**：`GlmMoeDsaTopkRouter` + 256 个 routed experts + shared experts，不是密集 MLP。
3. **词表权重独立**：原生 MTP 层自带 `embed_tokens` 和 `shared_head.head`，**与主模型的 embedding/head 不同**（extract 脚本里 `shared_weight_comparison` 的 `exact=False` 证实）。官方路径假设共享 verifier 词表，直接不成立。
4. **vLLM 服务端语义**：vLLM-Ascend 用 `AscendDeepSeekMTP`（deepseek_mtp 方法）实现 GLM-5.3 的 MTP 推理（见 `tools/glm_mtp_native_capture_extension.py` 导入 `vllm_ascend.models.deepseek_mtp.AscendDeepSeekMTP`）——GLM MTP 与 DeepSeek V3.2 MTP 同构，训练侧语义必须对齐它。

所以补丁的思路：**不动官方 Qwen 路径，新开 `glm_moe_dsa` 分支**——用 transformers 的 `glm_moe_dsa` 原生组件搭训练层，权重从 verifier 量化包里原样提取，forward 语义用 vLLM native 前向做 parity 校验。

## 2. GLM 原生 MTP 层的语义（从权重与 capture 反推）

原生 key 布局（verifier 量化包内，共 **793** 个张量，全部 `model.layers.78.*` 前缀）：

```
model.layers.78.enorm.*              # token embedding 的 RMSNorm
model.layers.78.hnorm.*              # verifier hidden 的 RMSNorm
model.layers.78.eh_proj.weight       # [2*hidden, hidden] 下投影
model.layers.78.self_attn.*          # MLA 注意力（含 indexer）
model.layers.78.mlp.{gate,experts,shared_experts}.*   # 稀疏 MoE
model.layers.78.shared_head.norm.*   # final RMSNorm
model.layers.78.shared_head.head.weight  # MTP 专属 lm_head（独立词表权重）
model.layers.78.embed_tokens.weight     # MTP 专属 embedding（独立词表权重）
```

单步 forward 数学（teacher-forced 训练视角，与 native 一致）：

```
h_t   = verifier 最后一个 norm 之后的输出（teacher 抓取点！）
e_{t+1} = MTP_embed_tokens(input_ids[t+1])          # position 0 处强制置 0
x     = eh_proj( concat(enorm(e_{t+1}), hnorm(h_t)) )
x     = x + MLA_attn(x)                             # indexer 冻结，top-k 稀疏上下文
x     = x + MoE(x)                                  # router + 256 experts + shared
out_t = shared_head.norm(x)
logits_{t+2} = shared_head.head(out_t)               # 监督 target = input_ids[t+2]
```

两个容易踩的语义点：

- **teacher hidden 必须是 final norm 之后的**：`launch_vllm.py --target-layer-ids 78` 默认抓的是第 78 层原始输出（pre-final-norm）。补丁用 vLLM worker 扩展把它替换成 final norm 后的输出（§6）。
- **position 0 的 token embedding 置零**：`torch.where(position_ids[...,None]==0, 0, token_embeddings)`——MTP 在序列起点没有"前一个 token"，原生实现同样如此（`test_glm_mtp_components.py` 有专门测试）。

## 3. 核心适配器 `glm_mtp.py` 逐组件精读

> 原文见 `bundle/glm_mtp.py`（= 容器 `/workspace/speculators/src/speculators/models/mtp/glm_mtp.py`）。依赖全部来自 `transformers.models.glm_moe_dsa.modeling_glm_moe_dsa`，没有任何手写算子——这是补丁能保持数值 parity 的关键。

### 3.1 `GlmMTPRMSNorm` —— fp32 化的 RMSNorm

```python
class GlmMTPRMSNorm(GlmMoeDsaRMSNorm):
    def forward(self, hidden_states):
        dtype = hidden_states.dtype; x = hidden_states.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (x * self.weight.float()).to(dtype)
```

继承 `GlmMoeDsaRMSNorm`（拿到 `variance_epsilon` 等），但 forward 强制 fp32 计算。原因：(a) NPU bf16 RMSNorm 与 vLLM `AscendRMSNorm` 数值不一致会导致 parity 永远对不上；(b) `rms_norm_eps` 用 checkpoint 里的 **1e-5**（README 明确记录）。同时 `core.py` 里把 `{input_layernorm, post_attention_layernorm, q_a_layernorm, kv_a_layernorm, k_norm, e_score_correction_bias}` 列为 `_keep_in_fp32_modules_strict`——FSDP 混合精度下这些模块禁止被降精度。

### 3.2 `GlmMTPRotaryEmbedding` —— head_dim 覆写

```python
class GlmMTPRotaryEmbedding(GlmMoeDsaRotaryEmbedding):
    def __init__(self, config, device=None):
        config = copy.deepcopy(config); config.head_dim = config.qk_rope_head_dim
        super().__init__(config, device=device)
```

MTP 层的 rope 只作用于每个 head 的 rope 分量（`qk_rope_head_dim=64`），而基类默认用完整 `head_dim` 预计算 cos/sin。覆写后 RoPE 维度正确（组件测试里有专门校验）。

### 3.3 `GlmMTPIndexer` —— 训练态 indexer（冻结）

原生 `GlmMoeDsaIndexer.forward` 是为推理写的（依赖 KV cache 分页）。训练态重写：

```python
@torch.no_grad()
def forward(self, hidden_states, q_resid, position_embeddings, attention_mask, position_ids, past_key_values=None):
    if past_key_values is not None: raise NotImplementedError(...)
    b, s, _ = hidden_states.shape; cos, sin = position_embeddings
    q = self.wq_b(q_resid).view(b, s, self.n_heads, self.head_dim)
    k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
    dims = [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim]
    qr, qp = q.split(dims, dim=-1); kr, kp = k.split(dims, dim=-1)
    qr, kr = apply_rotary_pos_emb_interleave(qr, kr, cos, sin, unsqueeze_dim=2)  # 只给 rope 分量上 RoPE
    q = torch.cat([qr, qp], dim=-1); k = torch.cat([kr, kp], dim=-1).squeeze(2)
    weights = self.weights_proj(...).float() * self.n_heads ** -0.5
    # 128 token 分块算 top-k（config.mtp_index_query_chunk），控 [chunk, seq] 分数矩阵峰值
    for start in range(0, s, chunk):
        scores = relu(q @ k^T * softmax_scale); scores = weights @ scores
        # attention_mask 缺失时用 position_ids 构造因果掩码（支持右 padding 截断后的序列）
        result.append(scores.topk(min(self.index_topk, s), dim=-1).indices.to(torch.int32))
    return torch.cat(result, dim=1)
```

要点：`@torch.no_grad()` + 构造时 `requires_grad_(False)` —— **indexer 永久冻结**（README：除非单独定义可微目标，否则不许训 indexer；实测发现加载后 `requires_grad` 会意外恢复 True，所以 `load_native_shared_weights` 末尾**再冻一次**）。`weights_proj` 输入显式 `.to(self.weights_proj.weight.dtype)` 防 FSDP 下 dtype 漂移。

### 3.4 `GlmMTPMoE` —— 显式专家循环

```python
class GlmMTPMoE(nn.Module):
    def __init__(self, config):
        self.gate = GlmMoeDsaTopkRouter(config)
        self.experts = nn.ModuleList([GlmMoeDsaMLP(config, config.moe_intermediate_size) for _ in range(config.num_local_experts)])
        self.shared_experts = GlmMoeDsaMLP(config, config.moe_intermediate_size * config.n_shared_experts)
        self.checkpoint_experts = False
    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, shape[-1]); _, weights, indices = self.gate(flat)
        result = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token, slot = torch.where(indices == expert_id); current = flat[token]
            if self.checkpoint_experts and self.training:
                output = checkpoint(expert, current, use_reentrant=False)   # 可选 per-expert 重算
            else:
                output = expert(current)
            output = output * weights[token, slot, None].to(output.dtype)
            result = result.index_add(0, token, output.to(result.dtype))
        return result.view(shape) + self.shared_experts(hidden_states)
```

为什么不用 fused MoE：Ascend 上没有与 vLLM 路由语义逐位一致的 fused 实现，显式循环 + `index_add` 保 parity；专家保持独立模块、固定顺序调用（README 实现约束 3）。`checkpoint_experts` 默认关（8K 20 步实测峰值 23.4GB 不需要），是留给更长序列的备用开关。

### 3.5 `GlmMTPLayer` —— 层组装

```python
class GlmMTPLayer(nn.Module):
    def __init__(self, config, layer_idx=0):
        config = copy.deepcopy(config); config.indexer_types = ["full"]; config.mlp_layer_types = ["sparse"]
        self.self_attn = GlmMoeDsaAttention(config, 0)
        self.self_attn.indexer = GlmMTPIndexer(config, 0)          # 替换原生 indexer
        self.self_attn.indexer.requires_grad_(False)
        self.self_attn.q_a_layernorm = GlmMTPRMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.self_attn.kv_a_layernorm = GlmMTPRMSNorm(config.kv_lora_rank, config.rms_norm_eps)
        self.mlp = GlmMTPMoE(config)
        # 5 个 norm：input/post_attention/hidden/token/final
        self.input_proj = nn.Linear(2*config.hidden_size, config.hidden_size, bias=False)
    def forward(self, hidden_states, token_embeddings, attention_mask=None, position_ids=None, position_embeddings=None, **kwargs):
        if position_ids is None: raise ValueError(...)
        token_embeddings = torch.where(position_ids[..., None] == 0, 0, token_embeddings)  # 起点 embedding 置零
        hidden = self.input_proj(torch.cat([self.token_layernorm(token_embeddings), self.hidden_layernorm(hidden_states)], dim=-1))
        attention, _, _ = self.self_attn(self.input_layernorm(hidden), position_embeddings=position_embeddings, ...)
        hidden = hidden + attention
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return self.final_layernorm(hidden)
```

注意 norm 结构与 Qwen MTP 的差异：Qwen 版是 `MTPLayerMixin`（hidden_normed 与 embed_normed 各自 norm 后 concat）；GLM 版位置等价但 norm 命名对应原生（enorm→token_layernorm、hnorm→hidden_layernorm、eh_proj→input_proj、shared_head.norm→final_layernorm），保证 key 映射后权重原样加载。

### 3.6 `native_key_to_training` —— key 映射（convert/export 的共用语义）

```python
def native_key_to_training(key, layer_index=78):
    prefix = f"model.layers.{layer_index}."
    key = key[len(prefix):]
    if key == "embed_tokens.weight":        return "embed_tokens.weight"
    if key == "shared_head.head.weight":     return "lm_head.weight"
    for source, target in {"enorm.": "token_layernorm.", "hnorm.": "hidden_layernorm.",
                           "eh_proj.": "input_proj.", "shared_head.norm.": "final_layernorm."}.items():
        if key.startswith(source): key = target + key[len(source):]; break
    return "mtp_layers.0." + key
```

793 native key → 793 training key 的双射（`shared_head.head.weight`→`lm_head.weight` 例外，2 个词表张量映射到顶层）。export 脚本里 `reverse_key` 做完全相同的逆映射。

### 3.7 `load_native_shared_weights` —— 独立词表权重加载

```python
def load_native_shared_weights(model):
    path = Path(model.config.mtp_shared_weights_path)
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    layer_index = model.config.transformer_layer_config.num_hidden_layers   # =78
    for source, target in ((f"model.layers.{layer_index}.embed_tokens.weight", model.embed_tokens),
                           (f"model.layers.{layer_index}.shared_head.head.weight", model.lm_head)):
        weight = safe_open(...).get_tensor(source)
        if tuple(weight.shape) != tuple(target.weight.shape): raise ValueError(...)
        target.weight = nn.Parameter(weight.to(device=...), requires_grad=False)   # 冻结！
    if hasattr(model, "verifier_lm_head"): del model.verifier_lm_head
    model.mtp_layers[0].self_attn.indexer.requires_grad_(False)   # 防加载后解冻
```

在 `MTPDraftModel.load_verifier_shared_weights`（core.py）的 GLM 分支调用：**跳过官方的 "NaN sentinel + 从 verifier 拷词表" 逻辑**，改从 `mtp_shared_weights_path`（= native 提取目录）读 GLM 自己的 embed/head 并冻结。这两个张量也因此**不进训练 checkpoint**（793 - 2 = 791 个可训练/加载张量），export 时再补回来。

### 3.8 `forward_glm_mtp` —— teacher-forced 递归 forward（训练核心）

```
输入约束（全部显式断言，违反即 raise）:
  - batch 必须是 1 条未打包序列（input_ids.shape[0]==1）
  - document_ids: 只允许右 padding、且 padding 前必须单文档（打包直接报错）
  - position_ids 必须连续 +1（有 reset 即报错 → 必须关 packing）
  - attention_mask 若给必须全 1（不接受 padding mask）
  - seq_len >= steps + 2
有效长度: valid_len = seq_len - effective_steps - 1   # 全步固定，图形态稳定
递归:
  current_hidden = hidden_states                      # step0 用 verifier 特征
  for step in range(effective_steps):                 # 单步训练时 effective_steps=1
      token_embeddings = embed(input_ids[:, step+1 : step+1+valid_len])
      output = checkpoint(mtp_layers[0], current_hidden[:, :valid_len], token_embeddings,
                          causal_mask, pos, use_reentrant=False)     # 非重入激活重算
      labels = input_ids[:, step+2 : step+2+valid_len]
      valid  = loss_mask[:, step+2 : ...]  (无 loss_mask 则全 1)
      step_loss, step_acc, count = _glm_chunked_ce(output, labels, valid, chunk=64)
      loss += step_loss * step_weights[step]          # beta^i 归一化（FastMTP）
      current_hidden = output                        # 下一步吃 draft 输出（递归）
```

**64-token chunk-loss**（`_glm_chunked_ce`）：词表 154,880，整条 `[seq, vocab]` logits 是 OOM 主凶。做法：把输出按 64 token 切块，每块在 non-reentrant checkpoint 内做 `lm_head → float CE(sum, ignore_index=-100) → argmax acc`，累加后除以 `valid_count`。只算 loss 数值不存 logits，峰值从"整层 logits"降到"64×154,880"。README 的降级顺序：再 OOM 就 64→32→16，只影响峰值不影响 loss 定义。

**step weights**：`raw = [beta**i for i in range(steps)]`，归一化后按步加权（formal100 单步时只有一个 step0）。5 步递归版本（`forward_glm_mtp(num_speculative_steps=5)`）代码在但**没做 vLLM parity，不许直接用**（README 实现约束 1 + CODE_DELIVERY v2 "未启动五步训练"）。

`core.py` 的分发（注意现状代码里有一段**不可达的重复 if**——先 dispatch 到 `forward_glm_single_step` 并 return，后面的 `forward_glm_mtp` 分支是历史残留，重现时可以只保留第一个）。

## 4. speculators 框架侧改动逐文件说明

### 4.1 `models/mtp/config.py`（+4 行）

新增 `mtp_shared_weights_path: str | None`——指向 native 提取目录（convert 时写入 config，训练加载时读）。

### 4.2 `models/mtp/model_definitions.py`（+7 行）

```python
from speculators.models.mtp.glm_mtp import (GlmMTPLayer, GlmMTPRMSNorm, GlmMTPRotaryEmbedding)
mtp_model_classes["glm_moe_dsa"] = base_components.ModelComponents(
    GlmMTPLayer, GlmMTPLayer, GlmMTPRMSNorm, GlmMTPRotaryEmbedding)
```

`model_type` 从 verifier config 自动解析（`AutoConfig` → `model_type: "glm_moe_dsa"`），注册后 `MTPDraftModel` 直接实例化 GLM 组件。

### 4.3 `models/mtp/core.py`（+25 行）

1. `glm_moe_dsa` 的 `_keep_in_fp32_modules_strict` 集合（§3.1 列的 6 个模块名）。
2. `load_verifier_shared_weights` 开头加 GLM 分支：无 `mtp_shared_weights_path` 即 raise；否则 `load_native_shared_weights(self)` 后直接 return。
3. `forward` 开头加 GLM 分支：dispatch 到 `forward_glm_single_step`。

### 4.4 `models/mtp/data.py`（+3/-1 行）

`shift_batch_mtp` 原来硬读 `batch["lengths"]`，但新 collator 已不再产出该字段（继承源码的接口不匹配，README 修复记录 4）。改为：`lengths` 存在才透传；`document_ids` 存在才透传（GLM forward 用它做打包边界断言）。

### 4.5 `train/dataloader.py`（+31 行）—— 单样本采样器

```python
if os.environ.get("SPECULATORS_MTP_NO_PACK") == "1":
    class _SingleSampleSampler:
        def __iter__(self):
            indices = list(range(get_dp_rank(), len(dataset), get_dp_size()))   # rank 轮转
            if max_batches is not None: indices = indices[:max_batches]
            return iter(([i] for i in indices))                                # batch=1
```

GLM forward 只接受未打包单序列 → 训练必须 `SPECULATORS_MTP_NO_PACK=1`。另外 `DataLoader(..., generator=torch.Generator().manual_seed(seed))` 固定加载器 RNG；`max_val_batches` 传给 val loader（formal100 用 `--max-val-batches 1`，验证只跑 1 个 batch，指标是 smoke 级估计）。

### 4.6 `train/distributed.py`（+17 行）—— 细粒度 FSDP（本补丁最关键的显存改动）

```python
reduce_name = os.environ.get("SPECULATORS_FSDP_REDUCE_DTYPE", "float32")  # formal100 用 bf16
mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
model_type = ...transformer_layer_config.model_type
for layer in model.layers:
    if model_type == "glm_moe_dsa":
        children = [layer.self_attn, layer.mlp.gate, layer.mlp.shared_experts, *layer.mlp.experts]
        for child in children:
            fully_shard(child, mp_policy=mp_policy)     # 先分片子模块
    fully_shard(layer, mp_policy=mp_policy)             # 再包整层
fully_shard(model, mp_policy=mp_policy)
```

为什么：GLM MTP 是"一个巨大的稀疏 MoE 层"，整层 all-gather 会在反向时把 256 个专家参数整体拉到每张卡——512/2K 试跑因此连续 OOM。先分片 attention/router/shared expert/每个 routed expert，再包外层 MTP layer，三种长度（512/2K/8K）都能反向（README 修复记录）。

### 4.7 `train/trainer.py`（+90 行）—— 精度与断点语义

- **参数 dtype 策略**（构造函数里）：`requires_grad` 的参数 `.to(torch.float32)`（fp32 master，配 adamw eager），冻结参数 `.to(hidden_states_dtype)`（bf16 压缩）。冻结大头是 indexer 和词表。
- **RNG 断点续训**：`_save_rng_state` 每个 rank 写 `rng_state_rank{rank}.pt`（python/numpy/torch/NPU 设备 RNG 全量）；resume 时恢复，警告"无 RNG 状态则续训不精确"。配套新文件 `rng_state.py`。
- **验证确定性**：`val_epoch` 包在 `validation_rng(seed=0)` 里（验证前固定 seed、验证后还原），`batch_sampler.set_epoch(0)`——两次验证可比。
- `train_epoch` 返回 `local_step`；`clip_grad_norm_(..., error_if_nonfinite=True)`；mid-epoch checkpoint 的 `maybe_update_best` 修复（save-best 覆盖权重时同步刷新 metrics/训练状态；无 save-best 时从全部 `val_metrics.json` 重算 best 链接）；`run_training` 支持 `eval_before_train`（step0 基线指标）和 `max_steps` 提前退出。
- ⚠️ 已知残留：formal100 收尾时 rank0 SIGBUS（XFS 写回竞争，当时 /data1 只剩 152G），模型 shard 已写出但 `training_state.json` 停在 step20——**复现长跑前先清盘 + 落盘节奏放慢**。

### 4.8 `train/checkpointer.py`（47 行改动）

- 默认保存 dtype **bf16→fp32**（配 fp32 master）。
- `load_safetensors_state_dict` 支持 `.index.json` 分片 checkpoint（native 提取目录就是分片的）。
- 加载一律走 CPU、不做隐式 dtype 转换（`float_dtype is not None` 才转）——防止优化器状态被 FSDP 参数 dtype 污染。

### 4.9 `train/cli.py` + `train/config/schema.py`（50 行）

- `set_seed(..., deterministic_algorithms)` → `torch.use_deterministic_algorithms(True)`（NPU 算子不支持会直接报错，比 silently 错强）。
- `build_draft_model` 的 `from_pretrained(..., dtype=torch.float32)`——draft 全 fp32 初始化（后续 Trainer 再压冻结部分）。
- 新 CLI 字段：`--max-val-batches`、`--eval-only`（只评估写 `save_path/evaluation/metrics.json`）、`--eval-before-train`、`--deterministic-algorithms`。
- provenance：`cfg.save()` 在 eval_only 时写到 `save_path/evaluation`。

### 4.10 `hs_connectors/transfer.py`（47 行）

`--hidden-states-lock-timeout`（默认 10s，formal100 传 180）：teacher 返回 HTTP 响应后 safetensors 是**异步落盘**的，训练端要按 `.lock` 文件等；没有上限的 `wait_for_lock` 会在 teacher 卡死时把训练一起挂死。FP8 backend 复用 File backend 的参数。

## 5. 权重三段流水线（convert → train → export）

### 5.1 `extract_native_mtp.py` —— 从量化包提取原生 MTP（只读不写源）

```
source: /data1/weight/GLM-5.3-w4a8-spec   (quant_model_weights.safetensors.index.json)
keys  : model.layers.78.*  → 必须恰好 793 个
校验  : assert is_floating_point（量化包里 MTP 张量本身就是浮点）
        分块 4M 元素 finite 检查
        每 2GB 分片写 mtp-NNN.safetensors + 读回 sha256 逐张量比对
对比  : layers.78.embed_tokens vs model.embed_tokens → sha256 不一致！
        layers.78.shared_head.head vs lm_head → 不一致！
        （证实 GLM MTP 有独立词表权重——官方 MTP 假设直接失效）
产物  : glm53_mtp_native_step0/{mtp-*.safetensors, index, config.json, extraction_report.json}
```

### 5.2 `convert_glm_mtp.py` —— 提取物 → Speculators checkpoint

```
1. MTPSpeculatorConfig(
     transformer_layer_config = AutoConfig.from_pretrained(提取目录),   # glm_moe_dsa
     mtp_shared_weights_path  = 提取目录,                              # 共享词表权重回读地址
     speculators_config       = SpeculatorsConfig(algorithm='mtp',
         proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
         verifier=VerifierConfig(name_or_path=<teacher路径>, architectures=['GlmMoeDsaForCausalLM'])))
2. torch.device('meta') 下实例化 MTPDraftModel → 拿 expected state_dict keys/shapes
3. mapping = native_key_to_training(每个 key)；断言双射 + 集合与 expected 完全一致
4. 逐张量: shape 校验 → sha256 记录 → 写分片（排除 embed_tokens/lm_head 两个词表张量）
5. 读回逐张量 hash 复验；写 index + conversion_report.json（semantics_parity: NOT_YET_VALIDATED）
产物: glm53_mtp_speculators_step0/（训练的 --from-pretrained 起点，19GB）
```

### 5.3 `export_glm_mtp_for_vllm.py` —— 训练 checkpoint → vLLM 可服务目录

```
1. reverse_key: mtp_layers.0.* → model.layers.78.*（enorm/hnorm/eh_proj/shared_head.norm 逆映射）
2. 断言: 训练 791 张量 + 2 共享张量 == native index 793（缺的必须恰好是 embed_tokens 和 shared_head.head）
3. shape 与 native 逐张量比对
4. model.safetensors = symlink → 训练 checkpoint（不复制大文件！）
   mtp_shared.safetensors = 从 native 提取目录补 2 个共享张量
   model.safetensors.index.json: 791→model.safetensors + 2→mtp_shared.safetensors
5. 复制 target(config.json/tokenizer/chat_template/generation_config) + export_report.json（逐张量 sha256）
产物: exports/<run>_vllm/  →  vllm serve <dir> --speculative_config '{"method":"mtp","num_speculative_tokens":1}'
```

注意：**官方 `speculators stitch-mtp` 不兼容 GLM 路径**（它假设共享 verifier 词表、bf16 落盘、无 mtp_shared 语义），CODE_DELIVERY v2 把"接官方 stitch"列为待办，当前正式出口就是 export 脚本。

## 6. Teacher 隐状态服务（vLLM 侧）

### 6.1 启动命令（`tools/start_glm_mtp_teacher.sh`）

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="/workspace/code_mtp:/workspace/code2"          # teacher 扩展链在这两个目录
export PATH=/data1/mtp_training/venv/bin:$PATH                     # venv 里有 editable speculators
export HCCL_BUFFSIZE=200 HCCL_CONNECT_TIMEOUT=3600
export LCCL_DETERMINISTIC=1 HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0 ATB_LLM_LCOC_ENABLE=0 HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python scripts/launch_vllm.py train /data1/weight/GLM-5.3-w4a8-spec \
  --target-layer-ids 78 \
  --hidden-states-path $RUN_DIR/hidden_states \
  --provenance-dir $RUN_DIR/provenance_$(date +%Y%m%d_%H%M%S) \
  --no-hash-checkpoints -- \
  --host 0.0.0.0 --port 8000 --api-server-count 1 --renderer-num-workers 1 \
  --tensor-parallel-size 8 --data-parallel-size 1 \
  --worker-extension-cls glm_mtp_teacher_extension.GLMMTPTeacher \
  --enable-expert-parallel --seed 1024 \
  --served-model-name "$MODEL" \
  --max-num-seqs 2 --max-model-len 8193 --max-num-batched-tokens 4096 \
  --trust-remote-code --gpu-memory-utilization 0.90 \
  --quantization ascend --kv-cache-dtype auto \
  --attention_config.indexer_kv_dtype int8 \
  --no-enable-prefix-caching --enforce-eager
```

`launch_vllm.py train`（provenance 按容器内 AGENTS.md 约定带 `--provenance-dir`）自动注入：

```json
--speculative_config {"method":"extract_hidden_states","num_speculative_tokens":1,
                      "draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[78]}}}
--kv_transfer_config {"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer",
                      "kv_connector_extra_config":{"shared_storage_path":"<hidden-states-path>"}}
```

### 6.2 `GLMMTPTeacher` 扩展链（三层，全部 import 副作用生效）

1. `glm_teacher_worker_extension.GLMMTPTeacher` 基类 `GLMTeacherDeterminism`：import 即 `torch.use_deterministic_algorithms(True)`——每个 spawn 出来的 worker 都要执行一次。
2. `glm_teacher_sorted_sfa_extension`：import 时给 `AscendSFAImpl._execute_sparse_flash_attention_process` 打补丁，进 sparse flash attention 前用 `sort_sparse_indices`（`glm_sfa_determinism.py`：把 topk 索引按逻辑位置升序、-1 padding 沉底）排序。原因：**Ascend 稀疏注意力的舍入结果依赖物理 KV 页顺序**，同样一组逻辑 token 不同页排布会出不同浮点值，parity 会被随机噪声毁掉。
3. `glm_mtp_teacher_extension.GLMMTPTeacher`：包装 `NPUWorker.load_model`，加载后断言 `uses_extract_hidden_states()` 且 PP==1，给模型挂 forward hook：
   ```python
   def final_hidden_hook(module, args, output):
       final_hidden, auxiliary = output
       return final_hidden, [final_hidden]      # 把 aux capture 替换成 final-norm 后输出
   ```
   **这是 teacher 语义的核心修正**：`--target-layer-ids 78` 的原生 capture 语义抓的是第 78 层（最后主层）的原始输出；而 GLM 原生 MTP 吃的是 `model.norm` **之后**的 hidden。hook 把连接器送出去的张量换成 final-norm 输出，与 `AscendDeepSeekMTP` 的输入语义对齐（probe + compare_teacher_inputs 验证）。

### 6.3 Parity 验证（重现时不可省）

- `start_glm_mtp_native_parity.sh` + `request_glm_mtp_native_4096.py`：起一个**带原生 deepseek_mtp 投机解码**的服务（`--speculative_config {"method":"deepseek_mtp",...}`），用 `glm_mtp_native_capture_extension.GLMMTPTeacher`（在 native MTP forward 前后抓 input_ids/positions/teacher_hidden/pre/post 输出/topk_indices）。
- `compare_glm_mtp_native.py` / `compare_glm_mtp_chunk4096.py`：同一批请求下，训练适配器前向 vs native vLLM MTP 前向逐张量比对（4096 chunk）。
- 附加发现（native capture 扩展里的 `load_weights_with_optional_norm_bias`）：主模型量化会给 RMSNorm 全局分配可选 bias，MTP 的 norm 没有对应权重 → 加载后把全零 bias 显式摘除，否则 forward 数值不纯。

## 7. 训练启动（实测脚本要点）

`tools/train_glm_mtp_formal100_8k.sh`（与 pilot20 的差异：reduce dtype、max-steps、checkpoint 节奏）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export HCCL_CONNECT_TIMEOUT=3600 HCCL_BUFFSIZE=200
export SPECULATORS_FSDP_REDUCE_DTYPE=bfloat16        # formal 用 bf16 梯度归约（pilot 默认 fp32）
export LCCL_DETERMINISTIC=1 HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0 ATB_LLM_LCOC_ENABLE=0 HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SPECULATORS_MTP_NO_PACK=1                     # 必须：关打包
export PYTHONPATH=/workspace/speculators/src:/workspace/speculators/hs_connectors/src
export GLOO_SOCKET_IFNAME=lo MASTER_ADDR=127.0.0.1 MASTER_PORT=29501
python -m torch.distributed.run --nproc_per_node=8 -m speculators.train \
  --speculator-type mtp \
  --verifier-name-or-path /data1/weight/GLM-5.3-w4a8-spec --trust-remote-code \
  --from-pretrained $ROOT/weights/glm53_mtp_speculators_step0 \
  --data-path $ROOT/data/prepared_8k_dflash2_official \
  --hidden-states-backend file --hidden-states-path $ROOT/online_hidden_states_formal100_8k \
  --vllm-endpoint http://127.0.0.1:8000/v1 --on-missing generate \
  --request-timeout 600 --max-retries 2 --generation-validation-retries 2 \
  --max-consecutive-generation-failures 20 --hidden-states-lock-timeout 180 \
  --total-seq-len 8192 --train-data-ratio 0.9005663797009516 --max-val-batches 1 \
  --hidden-states-dtype bfloat16 --num-workers 0 --prefetch-factor 1 --max-transfer-workers 1 \
  --num-speculative-steps 1 \
  --optimizer adamw --optimizer-implementation eager --lr 1e-5 --weight-decay 0.1 \
  --scheduler-type cosine --scheduler-warmup-steps 1 --scheduler-total-steps 100 \
  --epochs 1 --checkpoint-freq 20 --save-best \
  --fsdp-shard --deterministic-algorithms \
  --max-steps 100 --log-freq 1 --logger "" --log-dir $RUN/logs
```

实测数据（8-epoch 之外的 MTP formal100，2026-11-02/03）：

- 数据：`prepared_8k_dflash2_official`（22,070 条 8K，train 19,875 / val 2,195；`train_data_ratio=0.9005663797009516` 保证切分点重合；来源 DFlash2 官方 8k 管线）
- 100 步：loss 0.503→0.380，token accuracy 0.887；验证 loss 0.369 / acc 0.868（1 个 val batch）
- 峰值显存 23.2–23.4 GB/卡；checkpoint ~112GB（fp32 全量 + optimizer）
- 785/791 训练张量发生变化，9.94e9 元素，max|Δ|=3.07e-4（`parameter_update_audit_step20.json`）

## 8. 精度与确定性策略（一条线贯穿全补丁）

| 层           | 手段                                                                                                           |
| ------------ | -------------------------------------------------------------------------------------------------------------- |
| Teacher 前向 | `--enforce-eager`、SFA 索引排序、`torch.use_deterministic_algorithms(True)`、`LCCL_DETERMINISTIC=1`、seed 1024 |
| 数据         | 单样本采样器 + 固定 DataLoader generator + `validation_rng(seed=0)`                                            |
| 训练前向     | fp32 RMSNorm、indexer 冻结、chunk-loss fp32 CE                                                                 |
| 优化         | fp32 master 参数 + adamw eager + fp32/bf16 可选梯度归约                                                        |
| 落盘/恢复    | fp32 checkpoint、per-rank RNG 状态、CPU 加载不换 dtype                                                         |
| 断点续训     | mid-epoch local_step + training_state.json + scheduler 重建（DSpark 归档验证过同套机制）                       |

## 9. 已验证结论 vs 未完成项（重现者必读）

**已验证（.23 实测）**：

- 提取/转换 793 张量逐位哈希一致；CPU/NPU 组件测试全过（分块 indexer top-k、专家前向/梯度、首位置 embedding 屏蔽、indexer 冻结、RoPE 维度、norm eps）。
- 完整 MTP 权重 NPU 前向冒烟（峰值 22.1GiB）；512/2K/8K 单步反向 + 各 20 步连续 + 8K checkpoint 恢复读进 step21。
- formal100：100 次有效更新、模型 shard 写出、独立加载冒烟通过。

**未完成/已知问题**：

1. **单步限制**：`--num-speculative-steps 1` 是唯一 parity 过的配置；5 步递归 forward 是草案（CODE_DELIVERY v2 §3.3），vLLM 多步 KV/位置语义未对齐，**不许直接调大**。
2. **正式 stitch 未接**：出口是自定义 export，官方 `stitch-mtp` 不兼容。
3. **验收 A/B 实际无效（本次复盘新发现）**：`stitched_final100/` 目录全部文件时间戳是 11-02 14:10 CST——**早于 formal100 训练开始（23:57 CST）**。`start_final_mtp_service.sh` 加载的是这个**训练前**的快照，所以 final vs baseline 探针数字几乎相同（MAL 1.90–2.00 vs 1.95–2.00）是"step0 vs step0"。真正的训练产物在 `exports/final100_vllm/`（11-03 04:02，791+2 张量），**从未被服务加载过**。重跑验收时必须把服务指向 exports 目录（或重新做 stitch），A/B 结论才有意义。
4. SIGBUS 收尾（XFS 写回竞争 + 磁盘余量 152G），`training_state.json` 停在 step20；长跑前清盘、关 `--save-best` 或降低 checkpoint 频率、最终元数据写独立文件。
5. 2K 试跑首次失败：截断后 `loss_mask[:, 2:]` 全空 → 过滤窗口内无监督目标的样本（forward 里 `valid_len<=0` 返回零 loss 的守卫已加）。
6. torch_npu post7（.23/.27）vs .26 容器 post4——未验证差异，重现前建议对齐。

## 10. 重现清单（step by step）

```text
[ ] 1. 容器：vllm-ascend v0.30.0rc2-a3（或同 torch 2.10/transformers 5.14.1 栈），16 卡可见
[ ] 2. speculators checkout 到 7ed657f（vllm-project/speculators main）
[ ] 3. 放入 src/speculators/models/mtp/glm_mtp.py、src/speculators/train/rng_state.py
[ ] 4. 应用 4 处接线 edit（等价于 install_glm_mtp_adapter.py，幂等）:
       models/mtp/config.py        + mtp_shared_weights_path
       models/mtp/model_definitions.py + glm_moe_dsa 注册
       models/mtp/core.py          + GLM 加载分支、GLM forward 分发
       models/mtp/data.py          + document_ids/lengths 可选化
[ ] 5. 应用训练稳健性 diff（对照 speculators_git.diff）:
       train/{checkpointer,cli,config/schema,dataloader,distributed,trainer}.py
       hs_connectors/transfer.py
[ ] 6. venv：python -m venv --system-site-packages（继承容器 torch/vllm），
       pip install -e speculators -e hs_connectors --no-deps --no-build-isolation
       + datasets==5.1.0 + numpy==2.2.6 + pyarrow==26.0.0 + setuptools-git-versioning
[ ] 7. extract_native_mtp.py → weights/glm53_mtp_native_step0（核对 793、finite、hash、独立词表）
[ ] 8. convert_glm_mtp.py --source <提取目录> --teacher <verifier> --output <speculators_step0>
[ ] 9. 组件测试 test_glm_mtp_components.py + 全权重冒烟 smoke_glm_mtp_full.py（单卡）
[ ] 10. teacher: start_glm_mtp_teacher.sh（卡 0-7）→ probe_glm_mtp_teacher.py 校验
[ ] 11. parity: start_glm_mtp_native_parity.sh + request_glm_mtp_native_4096.py
        + compare_glm_mtp_native.py（适配器 vs native deepseek_mtp 逐张量）
[ ] 12. 容量阶梯: 512 单步 → 512×20 → 2K×20 → 8K 单步 → 8K×20（OOM 时先降 chunk 64→32→16）
[ ] 13. 正式训练（卡 8-15, NO_PACK=1, fsdp-shard, max-val-batches 1）→ checkpoints/<N>
[ ] 14. export_glm_mtp_for_vllm.py --checkpoint checkpoints/0 --output exports/<run>_vllm
[ ] 15. 验收 A/B：baseline(step0) 与 trained 各起一个 vllm serve（ speculative_config
        mtp / num_speculative_tokens=1, TP8），同一 prompt 集对比
        /metrics 的 SpecDecoding Mean acceptance length / Per-position acceptance rate
        ⚠️ 训练侧服务必须指向 exports/<run>_vllm（.23 的教训，见 §9.3）
```

## 11. 本地归档索引（`glm53_mtp_patch_study/bundle/`）

| 文件                                                                            | 内容                                                                                                                                   |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `speculators_git.diff`                                                          | 12 文件完整 diff（813 行）——框架侧改动的唯一权威来源                                                                                   |
| `glm_mtp.py` / `rng_state.py`                                                   | 两个新文件原样拷贝                                                                                                                     |
| `base_commit.txt` / `git_status.txt`                                            | 基线 commit 7ed657f 与工作树状态                                                                                                       |
| `tools/`                                                                        | 运行现场：extract/convert/export、install 接线、teacher/parity/验收脚本、README_glm53_mtp_23.md、formal100_train_command.txt、run.yaml |
| `adapter/`                                                                      | 交付与计划：CODE_DELIVERY/PLAN v2（5 步草案）、teacher 扩展链、SFA 确定性、v2 启动脚本                                                 |
| `transfers/`                                                                    | 中间补丁工具（fine-FSDP、single-sample loader、pilot 数据构造）                                                                        |
| `formal100_run.yaml` / `formal100_train_command.txt` / `final100_verified.json` | formal100 run 的完整 provenance                                                                                                        |
| `venv_packages.txt`                                                             | .23 venv 完整包清单（版本对齐用）                                                                                                      |

原 tar 包：`glm53_mtp_patch_study/glm_patch_bundle.tar.gz`。
