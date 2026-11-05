# GLM-5.3 MTP 在线增训代码交付审视稿 v2

状态：代码审视稿；已加入五步 teacher-forced 递归草案，未启动五步训练。

## 1. 交付内容

本次交付分为三部分：

1. 教师服务启动脚本：改用 `python scripts/launch_vllm.py MODEL --target-layer-ids 78 -- ...`，在线返回 target layer hidden states。
2. 在线训练脚本：以 verifier 原始权重为唯一来源，启动时自动生成本次运行的 native_init；设置 `num-speculative-steps=5`。
3. GLM MTP 适配补丁说明：将现有单步 forward 扩展为五步递归/状态传递，并接入官方的多步 loss、stitch 和推理配置。

## 2. 当前代码基线

当前本地代码文件：

- `remote_scripts/glm_mtp.py`
- `remote_scripts/install_glm_mtp_adapter.py`
- `remote_scripts/convert_glm_mtp.py`
- `remote_scripts/export_glm_mtp_for_vllm.py`
- `remote_scripts/start_glm_mtp_teacher.sh`
- `remote_scripts/train_glm_mtp_pilot20.sh`
- `remote_scripts/apply_glm_mtp_v2_remote.sh`

当前代码已经具备：

- GLM `glm_moe_dsa` MTP layer 注册；
- native `model.layers.78.*` 到 Speculators key 的转换；
- GLM DSA indexer、MLA attention、稀疏 MoE 和 norm 适配；
- 细粒度 FSDP 和 64-token chunk-loss；
- 在线 file backend hidden-state 传输；
- 单步 zero-step/短序列/8K 容量验证记录。

当前代码仍有以下限制：

- 五步实现目前是 `forward_glm_mtp` 的 teacher-forced 递归草案，只执行 `mtp_layers[0]`，尚未完成 vLLM 端到端 parity；
- 当前初始化仍需要本次运行自动生成的 `native_init`，但不再依赖旧的固定 DRAFT 目录；
- 训练脚本暂时保留 `--from-pretrained`，直到 native loader 直接接入 verifier；
- 训练后使用的是自定义 export，而不是官方 `stitch-mtp` 流程；
- v2 明确使用官方在线训练脚本的 `launch_vllm.py MODEL -- VLLM_ARGS` 形式。
- `train_glm_mtp_online_v2.sh` 中的 `--target-layer-ids` 和 `--step-weight-beta` 是待实现的配置字段；在 CLI/schema/模型 forward 落地前，脚本仅作为接口草案，不能直接启动。

## 3. 目标代码接口

### 3.1 教师服务

`--provenance-dir` 是审计目录参数，不是模型权重目录，也不是 hidden-state 缓存目录。官方参考脚本不依赖它；当前服务脚本默认不传，避免把旧 wrapper 参数混入普通 `launch_vllm.py`。

目标入口：

```bash
python scripts/launch_vllm.py \
  /data1/weight/GLM-5.3-w4a8-spec \
  --target-layer-ids 78 \
  -- \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 8
```

实现要求：

- hidden state 由 live vLLM 按请求提取并返回 endpoint；
- 训练端使用 `--on-missing generate` 按 batch 获取特征；
- 不生成全量预计算 hidden-state 数据集；
- 服务日志单独保存，便于重放和审计；如需要额外审计，可由外层启动器保存 provenance。

### 3.2 训练初始化

删除对旧固定 DRAFT 目录的依赖：

```bash
DRAFT=/data1/mtp_training/weights/glm53_mtp_speculators_step0
--from-pretrained "$DRAFT"
```

当前可启动的过渡实现会从原始 verifier 自动生成本次运行专属的 `native_init`：

```bash
python /workspace/adapter/convert_glm_mtp.py \
  --source /data1/weight/GLM-5.3-w4a8-spec \
  --teacher /data1/weight/GLM-5.3-w4a8-spec \
  --output /data1/mtp_training/runs/<run>/native_init
```

训练保留：

```bash
--verifier-name-or-path /data1/weight/GLM-5.3-w4a8-spec
--speculator-type mtp
--from-pretrained /data1/mtp_training/runs/<run>/native_init
```

建议在 `MTPDraftModel` 初始化处新增 `load_native_mtp_from_verifier()`：

1. 读取 verifier 的量化索引文件；
2. 只定位 `model.layers.78.*` 的 native MTP keys；
3. 将 key 映射到训练模块，或直接在 native key layout 下加载；
4. 共享 embedding/head 从同一 verifier 读取并冻结；
5. 输出 key、shape、dtype、有限性和 SHA-256 报告；
6. 当前过渡实现允许落地本次运行专属的 native_init；最终 native loader 完成后再移除该中间目录。

### 3.3 五步 forward

建议新增接口：

```python
def forward_glm_mtp(
    model,
    input_ids,
    hidden_states,
    position_ids,
    loss_mask,
    num_speculative_steps=5,
    step_weight_beta=0.6,
    document_ids=None,
):
    ...
```

已落地的本地草案位置：`remote_scripts/glm_mtp.py:forward_glm_mtp`。实现要点：

- 逐步生成 step 0 到 step 4 的 draft hidden/output；
- 每一步使用正确的 token embedding 和 position；
- 每一步只监督对应的 shifted target；
- 使用 `w_s = beta ** s` 汇总五步 loss；
- 返回 `loss_step_0` 到 `loss_step_4`、每步 token accuracy 和 supervised token 数；
- 保留 indexer 冻结策略，除非单独定义可微训练目标；
- 继续使用细粒度 FSDP 和 chunk-loss；
- 对 packed document、padding 和 position reset 明确报错或正确截断。

远程容器内应用代码（不需要 VS Code）：

```bash
bash /workspace/adapter/remote_scripts/apply_glm_mtp_v2_remote.sh
```

该脚本把 GLM MTP 模块安装到 `/workspace/speculators`，执行适配器修改，并运行 Python compile 检查。完成后才使用训练入口。

### 3.4 checkpoint 和 stitch

训练 checkpoint 需要保存：

- native MTP 参数；
- 五步配置和 `step_weight_beta`；
- optimizer/scheduler/RNG/训练位置；
- verifier 路径和 native key manifest。

优先实现与官方接口一致的：

```bash
speculators stitch-mtp CHECKPOINT VERIFIER --output-path STITCHED
```

旧的 `export_glm_mtp_for_vllm.py` 可保留为诊断和兼容工具，但不再作为正式训练流程唯一出口。

## 4. 建议的新版训练命令

```bash
cd /workspace/speculators
python -m speculators.train \
  --verifier-name-or-path /data1/weight/GLM-5.3-w4a8-spec \
  --speculator-type mtp \
  --num-speculative-steps 5 \
  --step-weight-beta 0.6 \
  --data-path /data1/spec_train_data/prepared_8k_dflash2_official \
  --vllm-endpoint http://127.0.0.1:8000/v1 \
  --on-missing generate \
  --total-seq-len 8192 \
  --hidden-states-lock-timeout 180 \
  --fsdp-shard \
  --gradient-checkpointing \
  --deterministic-algorithms \
  --lr 1e-5 \
  --optimizer adamw \
  --max-steps 1 \
  --save-path /data1/mtp_training/runs/glm53_mtp_v2_smoke/checkpoints
```

这里的 `--max-steps 1` 只是第一道 smoke test；它与 `--num-speculative-steps 5` 是两个不同概念。当前脚本仍是审视草案，必须先完成 CLI/schema 接入、编译检查和 512-token 五步 parity，再将 max-steps 增加到 5 或更高。

## 5. 审视重点

请重点确认：

1. GLM 原生 MTP 是否允许通过同一层递归展开到 5 个 speculative steps，还是必须构造 5 个独立 MTP layers；
2. `target-layer-ids=78` 返回的 hidden state 是否正好对应 GLM MTP 的输入语义；
3. 官方 `launch_vllm.py` 当前版本是否接受 hidden-state/provenance 参数，是否仍需要 `train` 子命令和 `GLMMTPTeacher` extension；
4. 是否保留 8K prepared 数据，还是按官方 `prepare-data` 从 `train_10k.jsonl` 重建；
5. 五步训练的显存预算和 chunk-loss 大小；
6. 是否同意先采用“原始 verifier 自动生成 native_init”的过渡路径，待 native loader 完成后再切换为完全直读。

## 6. 未执行事项

本交付稿没有启动教师服务或训练，也没有宣称五步 GLM MTP 已通过 parity。当前本地已补入递归 forward 草案；远程源码同步由 `apply_glm_mtp_v2_remote.sh` 完成，随后仍需做 vLLM 位置/KV 语义对齐和 NPU 验证。

