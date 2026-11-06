# vLLM MTP/DSPark 方法自动检测问题分析

日期：2026-11-06。环境：vllm-ascend v0.28.0.rc1-a3（vllm v0.28.0）。

## 现象

同一台机器、同一脚本、同一显式参数 `--speculative-config '{"method":"mtp",...}'`：

|  serving 模型路径 | SpeculativeConfig 解析结果 | 结果 |
|---|---|---|
| `/mnt/weight2/l00840580-glm53-w4a8-ws-signed` | `method='mtp'` | 正常启动，原生 MTP 递归 |
| `/mnt/share_space/l00840580/dspark-glm53/hybrids/evalA_init-v12` | `method='dspark'` | 加载 DSpark draft，DS-v4 专有字段缺失，连环崩溃 |

config.json、quant_model_description.json、safetensors index **三者逐键完全一致**，唯一差异是**目录名**。

## 根因

`vllm/config/speculative.py`（约 940-965 行）的 draft 方法自动推断：

```python
elif (
    "dflash" in self.draft_model_config.model.lower()
    or ...
):
    self.method = "dflash"
elif (
    "dspark" in self.draft_model_config.model.lower()
    or "Qwen3DSparkModel" in self.draft_model_config.architectures
    or "Gemma4DSparkModel" in self.draft_model_config.architectures
):
    self.method = "dspark"
```

**`model` 字段是模型路径字符串**，用 `in` 做**子串匹配**。我们的工作目录叫
`dspark-glm53`，路径里恰好包含 `"dspark"` 子串，方法被强制覆盖为 dspark。
`/mnt/weight2/` 下的 checkpoint 不含该子串，推断为 mtp。

## 它有多死板

1. **路径字符串当语义用**。`"dspark" in path` 是最脆弱的检测方式：
   目录名、挂载点、甚至 checkout 位置都可能改变推理行为。
2. **用户显式参数会被覆盖**。CLI 已显式给出 `method="mtp"`，
   推断链仍然执行并把解析结果改写为 dspark（观察行为），显式声明失去意义。
3. **错误被放大成无关崩溃**。方法一旦判成 dspark，draft 模型按 DS-v4
   DSparkDeepseekV4ForCausalLM 构建，需要 `hc_mult / o_groups / o_lora_rank /
   rope_parameters[factor|beta_fast|beta_slow]` 等一系列 DS-v4 专有 config 字段；
   GLM-5.3（MLA + DSA indexer + 单层 MTP）一个都没有，于是产生
   AttributeError / KeyError / 空张量 reshape 等五层连环错误，
   每一层都指向不同模块（config、heterogeneity wrapper、quant description、
   权重名映射、o-group 投影），与真正的根因（方法判错）完全无关。
4. **报错不回指根因**。没有任何日志提示 "method 被推断为 dspark"；
   `llm_base_proposer.py` 的 `Loading draft model: method=dspark` INFO 行
   是唯一的线索，淹没在 20 万字符日志里。

## 修复

零代码修复：把 serving 路径挪到不含 "dspark" 的目录。

```bash
mkdir -p /mnt/share_space/l00840580/glm53-eval
ln -s .../dspark-glm53/hybrids/evalA_init-v12 .../glm53-eval/evalA
# serve MODEL=.../glm53-eval/evalA  →  method='mtp'  ✓
```

## 建议（给上游）

1. 方法推断应基于 `architectures`/`model_type` 等结构化字段，不应匹配路径子串。
2. 用户显式传入 `method` 时应完全跳过推断链（fail-fast 而不是静默覆盖）。
3. 推断发生时打 INFO 日志说明 "auto-detected method=X from Y"，
   并在 override 用户显式值时升级为 WARNING。

## 附：调试过程中沉淀的其他修复（glm-mtp/ 下存档）

- `dspark_glm_patch.py`：GLM config 缺失字段的运行时注入 + `n_mtp_layers=1`
- `modelslim_mtp_remap_patch.py`：draft 权重名 `mtp.N.*` → `model.layers.78.*`
  的量化描述回退映射
- 二者在 mtp 正确路径下不会触发，保留作为 DSpark 路径的兼容垫片。
