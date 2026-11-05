# GLM-5.3 MTP 增训管线 — 依赖与部署清单（deps）

> 目标：在任何一台兼容的 Atlas A3 服务器上快速重建训练-评测管线。
> 本目录 = 管线全部运维脚本；speculators 主体修改随本仓库（glm-mtp 分支）分发。

## 1. 硬件与宿主机要求

| 项 | 要求 |
|---|---|
| NPU | Atlas A3，16× Ascend 910B4（8 板 × 2 chip），davinci0-15 |
| 宿主驱动 | npu-smi **26.1.1**（硬约束：CANN 9.1.x 必配；9.2 会 HCCL error 19） |
| 内存 | ≥ 2TB（16 卡 DCP 保存时峰值 ~750GB） |
| 共享存储 | NFS：权重卷（biasfix checkpoint 378G）+ 工作区卷（≥ 500G：checkpoints 4×21G + 数据） |
| 网络 | 容器 host 网络；github 走华为代理（git config http.proxy） |

## 2. 容器镜像（serving + 训练执行环境）

```
quay.io/ascend/vllm-ascend:v0.28.0.rc1-a3     # CANN 9.1.0, vllm 0.28.0, vllm-ascend v0.28.0.rc1
```

- **上游 tag 已被 quay 清理**（"revive via time machine"）→ 从已有主机导出：
  `docker save quay.io/ascend/vllm-ascend:v0.28.0.rc1-a3 | gzip > v028.tar.gz`
  注意 /var/lib/docker 需 ≥ 25G 临时空间（先清 `docker/tmp/*`）；目标机 `docker load`。
- **已否决镜像**（勿用）：
  - `v0.30.0rc1-a3` / CANN 9.2 → 宿主驱动 26.1.1 不兼容，2 卡 allreduce 即 HCCL error 19
  - `v0.31.0-a3-v1027` → 内部 vllm/vllm-ascend 快照不同步，首次 forward 必挂
  - `nightly-main-openeuler-1010` → 在 A3 上 CANN 算子 JSON 缺失（aclnnInplaceOne 561103）

## 3. 训练 venv（Python 3.12.13）

```text
torch==2.10.0            # +cpu 轮子 + torch_npu 2.10.0.post4（Ascend 后端）
torch_npu==2.10.0.post4
transformers==5.14.1     # glm_moe_dsa 建模来自这里
numpy==2.0.2             # 不要 2.5+（scipy 错位告警；2.0.2 已验证）
datasets==4.4.0
pyarrow==21.0.0
safetensors==0.8.0
accelerate               # transformers 依赖
speculators==0.9.0a0     # 本仓库 editable：pip install -e <本仓库根>
vllm==0.28.0（容器内 /vllm-workspace/vllm，仅供评测脚本 import）
```

venv 创建（在容器内）：
```bash
python3 -m venv /mnt/<workspace>/dspark-glm53/venv
venv/bin/pip install torch==2.10.0 numpy==2.0.2
venv/bin/pip install torch_npu==2.10.0.post4   # Ascend 源或本地轮子
venv/bin/pip install transformers==5.14.1 datasets pyarrow safetensors accelerate
venv/bin/pip install -e <本仓库根>              # speculators + GLM 适配
```

## 4. speculators fork

- 分支 `glm-mtp` @ cb37927（github.com/ababyshark/speculators）
- `pip install -e .` 后训练入口：`python -m speculators.train ...`
- 关键适配清单见 `docs/glm53-mtp-npu-training.md`（12 项修复 + 已知限制）

## 5. 模型与数据产物（NFS 路径，按环境替换）

| 产物 | 路径 | 说明 |
|---|---|---|
| verifier W4A8 checkpoint | `/mnt/weight2/l00840580-glm53-w4a8-biasfix` | 378G ModelSlim W4A8，含 rot.safetensors + 原生 MTP 层 |
| signed warm-start | `<workspace>/drafter_init/glm53-mtp-warmstart-signed` | 由 `extract_mtp_layer.py` 抽取（**必须含 int16 符号修复**） |
| 训练数据 | `<workspace>/data_prep_rot` | 2052 条 rot 空间 HS（extract_hidden_states 模式抽取 + transform_hs_rot） |
| 留出评测集 | `<workspace>/data/eval_prompts.jsonl` | 200 条 UltraChat（与训练集零重叠） |
| 训练 checkpoint | `<workspace>/checkpoints/glm53_mtp_v2` | 4 epochs，~21G/epoch |
| requant 输出 | `<workspace>/hybrids/v8-w4a8` | 训练后写回的 W4A8（weight2 可能只读，输出到工作区卷） |

## 6. 环境变量（所有训练/评测脚本统一前缀）

```bash
export OMP_NUM_THREADS=4                     # 训练（serving 用 1）
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_BUFFSIZE=256                     # serving 用
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ENGINE_STARTUP_TIMEOUT=1800      # serving：慢 NFS 加载防超时
```

## 7. 一键部署顺序（新服务器）

1. 挂载 NFS（weight 卷 + 工作区卷），确认驱动 26.1.1：`npu-smi info`
2. 导入镜像：`docker load < v028.tar.gz`
3. 起容器：`bash run_v028.sh`（按宿主路径微调 -v 挂载）
4. 容器内：HCCL 冒烟 `python3 -m torch.distributed.run --standalone --nproc-per-node 2 hccl_test.py`
   （预期 `allreduce OK`；失败 = 镜像/驱动不兼容，5 分钟排除法）
5. 建 venv（§3）+ `pip install -e <本仓库根>`
6. 抽取 signed warm-start：`python examples/glm53_mtp/extract_mtp_layer.py --model <biasfix> --output <workspace>/drafter_init/glm53-mtp-warmstart-signed`
   （验证：专家权重负值占比 ~0.43；0% = 踩了无符号 bug）
7. 数据准备：`gen_onpolicy.py` → `tokenize_onpolicy.py` → serving 抽 HS（`serve_extract.sh`）→ `transform_hs_rot.py`
8. 预检：`bash preflight_check.sh`（15 项，--fix 清孤儿）
9. 训练+评测全链：`bash run_pipeline_v2.sh`（训练 4 epochs → requant → serve B → 200 条验收）

## 8. 已知坑速查（详见 docs/glm53-mtp-npu-training.md）

- pkill 自匹配：pattern 拆串（`P="speculators.trai"; pkill -f "${P}n"`）
- 杀 serve 后 worker 变僵尸占 HBM：按 `VLLM::`/`EngineCore` 前缀清理，npu-smi 确认归零再起
- PYTHONPATH 覆盖 acl：必须 `${PYTHONPATH:-}` 追加
- NFS 只读切换/外部清理：requant 输出与 checkpoint 一律写工作区卷
- 容器 overlay 满：docker save 前清 /var/lib/docker/tmp
- e_score_correction_bias 必须 fp32 全链路（bf16 舍入 = 75% top8 选择翻转）
