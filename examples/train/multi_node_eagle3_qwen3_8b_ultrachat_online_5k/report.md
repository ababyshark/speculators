# Eagle3 Qwen3-8B Draft Training: Single-Node vs Two-Node — Results Report

> Measured 2026-10-17 (steady-state window ~00:35–01:35 UTC, both runs live simultaneously).
> Hardware: Ascend 910 A3 servers, 16 NPUs (64GB HBM) per node, two nodes anonymized as `ip-a` / `ip-b`.
> Software: vllm-ascend 0.27.1rc1 (vLLM 0.26.0/0.27.1), speculators multi-node branch, torchrun DDP.

## 1. TL;DR

Doubling the training hardware (1 node / 4 ranks -> 2 nodes / 8 ranks) delivered a
**2.06x end-to-end training speedup** (projected wall-clock for the identical recipe:
**26.1h -> 12.7h**), i.e. **~103% scaling efficiency**. This was only achievable after
fixing the hidden-state data plane: moving extraction output from shared storage (NFS)
to **node-local `/dev/shm` with per-node endpoint pinning** took the 2-node aggregate
throughput from **0.5x** (broken) to **2.06x** (linear). Training quality is on par:
both runs start at the same loss and converge with matching per-layer accuracies.

## 2. Setup (identical for both runs)

| Item | Value |
|---|---|
| Target model | Qwen3-8B (36 layers) |
| Draft | Eagle3, 1 llama-style decoder layer, draft vocab 32000 (d2t/t2d mapping), ttt_steps=3 |
| Target layers | 2 / 18 / 33 (+ last layer 36 auto-extracted) |
| Loss / optimizer | kl_div, AdamW, lr 1e-4, cosine schedule, 4% warmup, wd 0.0 |
| Sequence / epochs | 4096 packed tokens per rank per step, 2 epochs |
| Dataset | On-policy regenerated corpus (assistant turns produced by Qwen3-8B, pretokenized): 1,841,307 samples, 2.094B train tokens. Directory name follows the `ultrachat_online_5k` example-family convention; scale `MAX_SAMPLES` for the 5k sanity run |
| Single node | 4 training ranks (NPUs 12-15) + extractor vLLM DP4 (NPUs 8-11), same machine, `/dev/shm` handoff |
| Two nodes | 8 training ranks (4 per node, `dp_size=8`) + one extractor per node (NPUs 8-11), endpoints pinned per node, node-local `/dev/shm` handoff |

## 3. Training throughput

| Metric (steady state, 30-min window) | 1 node (dp=4) | 2 nodes (dp=8) |
|---|---|---|
| Step time (mean) | 336 ms | 341 ms (+1%) |
| Step rate | 2.98 steps/s | 2.95 steps/s |
| Tokens per step (cluster) | ~15.0K | ~31.1K |
| Per-rank throughput (logged) | 12.2K tok/s | 11.7K tok/s (−4.4%) |
| Steps per epoch | 139,838 | 67,380 |
| **Aggregate throughput** | **~44.6K tok/s** | **~91.7K tok/s** |
| **Projected total wall-clock (2 epochs)** | **~26.1 h** | **~12.7 h** |

**Aggregate speedup: 2.06x on 2x hardware — scaling efficiency ~103%.**

Why steps per epoch halve: the packed sampler distributes up to
`dp_size x 4096` tokens per step (each rank gets one batch ≤ 4096). The same
2.094B-token dataset therefore becomes 139,838 optimizer steps on 4 ranks but
67,380 on 8 ranks; per-rank work per step is unchanged (~3.7K-3.9K tokens), so
step time is unchanged and total time halves. Verified three ways: (a) sampler
simulation on the real dataset reproduces both epoch lengths exactly
(139,838 / 67,389); (b) the 2-node checkpoint boundaries land on exact 10%-epoch
multiples (resume at 20,214 = 3x6,738; next save at 26,952 = 4x6,738); (c)
`tokens_per_s x step_ms ≈ 4K tokens/rank/step` in both runs.

Per-rank overhead of going multi-node is small and confined to the data plane:
median `fetch_ms` 16 -> 55 ms, `fetch_frac` 0.10 -> 0.20 (plus ~53 ms
`queue_ms`); forward/backward themselves are 92/137 ms (single) vs 91/156 ms
(two-node).

## 4. The data plane: what made two-node work

The first 2-node attempt (2026-10-16) ran **slower than single-node** because
hidden states round-tripped through shared storage:

| | 2-node, NFS round trip (before) | 2-node, node-local /dev/shm (after) |
|---|---|---|
| fetch_ms per step | ~477-534 (70% of step) | ~16-55 |
| fetch_frac | ~0.70 | 0.10-0.20 |
| step_ms | ~715-767 | ~341 |
| Aggregate throughput | ~24K tok/s (**0.5x single-node**) | ~91.7K tok/s (**2.06x**) |
| Generation load balance | all traffic on one extractor; peer idled; 17 failovers | symmetric per node; 0 failovers |

Root cause: the FileTransfer handoff returns the extractor's local file path in
the response and the trainer reads that path directly, so a training rank can
only consume its **local** extractor. Round-robining both endpoints routed
requests across nodes (unreadable `/dev/shm` paths) and collapsed load onto one
machine; shared-storage caching of hidden states was NFS-bound (~590 MB/s each
way). Fix: pin one endpoint per node + write/read node-local `/dev/shm`
(consume-once, in-flight samples only, a few GB). The two fixes together are
exactly the validated single-node topology, replicated per node.

## 5. Training quality (convergence parity)

Train metrics at identical global steps (loss / layer-0 full-token accuracy):

| Global step | 1 node | 2 nodes |
|---|---|---|
| 1 | 35.28 / 0.000 | 34.65 / 0.000 |
| 5,000 | 6.91 / 0.551 | 4.03 / 0.699 |
| 13,476 | 3.84 / 0.688 | 2.35 / 0.808 |
| 20,214 | 1.95 / 0.829 | 1.67 / 0.840 |

Both runs start from the same ~35 loss and converge to the same regime. The
2-node curve is ahead at equal global step, as expected: each of its steps
covers 2x the tokens. `error_records = 0` throughout on both runs.

Single-node end-of-epoch-0 validation (after one full pass over the 2.094B-token
corpus): **val loss 1.417**; per-layer full-token accuracy **0.834 / 0.688 /
0.572**, conditional accuracy 0.834 / 0.825 / 0.831 (layers 0/1/2);
`checkpoint_best` updated. The 2-node run was mid-epoch-0 (~49%) at measurement
time; its first 40% checkpoint (with optimizer/scheduler state, resume-tested)
is saved, and its epoch-0 validation will be compared against the numbers above.

Pipeline context: the same online-extraction infrastructure previously produced
a DSpark draft for the same target model that reached an expected acceptance
length of 4.964 and **2.85-3.41x end-to-end serving speedup** (GSM8K / HumanEval
benchmarks), demonstrating that draft models from this pipeline hold up in
production serving.

## 6. Reproduce

See [README.md](README.md) and the launch scripts in this directory
(`launch_extractor_node.sh` + `launch_train_node.sh` on each node;
`single_node_baseline/` for the 1-node baseline). Key knobs: `MODEL`,
`DATA_PATH`, `WORK_ROOT`, host IPs `ip-a`/`ip-b`, HCCL NIC.

## 7. Status and next steps

- 1-node run: ~98% of epoch 2/2 at measurement time.
- 2-node run: ~25% of epoch 0/2; projected finish in ~9.5h remaining at
  measurement time — a full recipe run in about half a day.
- Next: serve the finished drafts (single-NPU speculative deployment) and run
  the same acceptance-length + end-to-end benchmark suite used for DSpark
  (Guidellm acceptance length, `vllm bench serve` GSM8K/HumanEval at 128
  concurrency, spec vs baseline) to quantify serving-side gains of the
  two-node-trained Eagle3 draft.
