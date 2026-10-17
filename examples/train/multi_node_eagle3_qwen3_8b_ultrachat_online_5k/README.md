# Eagle3 Qwen3-8B online training, one node vs two nodes (online extraction)

Eagle3 draft training for Qwen3-8B against a live hidden-state extractor, in two
topologies that share model, data, and hyperparameters:

```text
single node (baseline)                    two nodes (this example, single_node_baseline/ is the baseline)

NPUs  8-11  extractor vLLM DP4            node ip-a                    node ip-b
NPUs 12-15  draft training, 4 ranks      NPUs 8-11  extractor DP4     NPUs 8-11  extractor DP4
             |                            NPUs 12-15 trainer rank 0-3  NPUs 12-15 trainer rank 4-7
             |                                \ dp_size=8 DDP, torchrun nnodes=2 /
             v                                            |
      /dev/shm (same container)                       node-local /dev/shm
                                                      (endpoint pinned: rank set of a node
                                                       only talks to that node's extractor)
```

This is a worked example from a validated 2-node Ascend 910 A3 deployment
(16 NPUs per node, vllm-ascend 0.27.1rc1), not a portable harness. Hosts are
anonymized as `ip-a` (node rank 0) and `ip-b` (node rank 1); replace them and
`<mgmt-nic>` (the HCCL/Gloo NIC) in `env.sh` / `launch_train_node.sh`. Results
and the single-vs-two-node comparison are in [report.md](report.md).

## Requirements

- One environment per node with vLLM (vllm-ascend), `speculators` (multi-node
  branch) plus `hs_connectors`, and `torchrun`.
- Verifier weights at `/models/Qwen3-8B`, or set `MODEL`.
- Shared storage (NFS or equivalent) visible on both nodes for checkpoints and
  logs only — hidden states never touch it.

## The components

### 0. Prepare data and token frequencies

The corpus must be on-policy regenerated (assistant turns produced by the
target model, pretokenized `input_ids` + `loss_mask`) so training and extraction
stay aligned:

```bash
speculators prepare-data --model Qwen/Qwen3-8B \
    --data <regenerated.jsonl> --output $DATA_PATH --seq-length 4096
bash build_token_freq.sh   # writes $DATA_PATH/token_freq.pt
```

### 1. Extractor — one instance per node

```bash
bash launch_extractor_node.sh
```

Hidden states are written to node-local `/dev/shm` (`HS_SHM_DIR`) and consumed
once by the local training ranks. Wait for `/health` on port 9019 on both nodes
before starting training.

### 2. Trainer — one launch per node

```bash
NODE_RANK=0 bash launch_train_node.sh   # on ip-a (head)
NODE_RANK=1 bash launch_train_node.sh   # on ip-b
```

The script waits for its local extractor, infers `NODE_RANK` from the host IP
when unset, and resumes mid-epoch from `$EAGLE3_MULTI_DIR/checkpoints/0` if
present. Stop order is the reverse of start order (trainers first, then
extractors).

### Single-node baseline

`single_node_baseline/` runs the identical recipe on one machine
(`launch_extractor.sh` on NPUs 8-11, `launch_train.sh` on NPUs 12-15, both
hitting `localhost:9019`). Use it for A/B comparisons; see report.md for the
measured difference.

## Operational notes (learned the hard way)

- **Endpoint pinning is mandatory.** Round-robining both extractors sends
  requests to the peer node; the response carries a `/dev/shm` path unreadable
  locally, and load collapses onto one machine (measured: 0.5x aggregate
  throughput, 17 failovers). One endpoint per node = pass-through behavior.
- **Hidden states live on node-local `/dev/shm`, never on shared storage.** A
  shared-storage round trip cost ~500ms per step (70% of step time). Local RAM
  reads: ~15-55ms.
- **Pin the HCCL/Gloo NIC** (`HCCL_SOCKET_IFNAME`, `HCCL_IF_IP`): docker bridges
  that exist on only one node make HCCL pick an unreachable IP and cross-node
  `connect()` hangs forever.
- `/dev/shm` does not survive restarts; that is fine (in-flight samples only,
  a few GB) — the extractor recreates it on start.
