# Multi-Node Training + Multi-Node Data Routing — Design Document

> Status: Draft | Branch: `multi-node` | Base commit: `55ebbd5`
>
> Corresponding roadmap item: *"Multi-node support (ToDo) — Improved routing from multiple nodes —
> train a smaller model using a multi-node set-up to see how quickly we can speed-up training"*

## 1. Background and Goals

The current speculators training topology is "single-node extraction + single-node
training". When the draft model's training corpus is large, single-node DP becomes the
training-throughput bottleneck. This design covers two things:

1. **Multi-node training side**: `speculators train` supports 2 nodes × 8 NPUs (16 DP
   ranks), statically wired with torchrun.
2. **Multi-node data routing**: in online generate mode, the trainer's hidden-state
   generation requests are client-side load-balanced across multiple vLLM extractor
   nodes (round-robin + failure eviction).

**Non-goals (out of scope for this iteration)**: elastic/fault tolerance, Slurm
orchestration, DCP sharded checkpoints, queue-depth-aware LB (interface reserved),
multi-endpoint support for offline batch generation (`generate_offline_data.py`).

## 2. Requirements Decisions (output of TODO-1)

| Decision | Outcome |
|---|---|
| Launch method | Static torchrun `--nnodes/--node-rank/--master-addr/--master-port` |
| Target scale | 2 nodes × 8 NPUs, pure DP scaling |
| Checkpoint | Keep rank0 full-state disk writes, assuming shared FS (documented + startup probe) |
| Routing | `--vllm-endpoints` multi-endpoint, client-side RR + failure eviction / automatic recovery |
| Validation environment | Ascend NPU (HCCL); code stays `torch.accelerator` device-agnostic |
| Acceptance | 2-node vs 1-node training throughput ≥ 1.7x; 2-extractor vs 1-extractor generation throughput ≥ 1.6x; multi-node loss curve aligned with single-node |

## 3. Current State and Gaps

Already supported: multi-node vLLM extraction (`--nnodes` pass-through), cross-node
hidden-state transfer (`hs_connectors`: file / mooncake), single-node multi-device
training (`torchrun --standalone` + FSDP2 + SP/DP groups).

| # | Gap | Location |
|---|---|---|
| G1 | `init_process_group` has no timeout; slow multi-node startup (weight loading / NCCL link setup) times out | `src/speculators/train/distributed.py:156` |
| G2 | All docs/examples use single-node `torchrun --standalone`; no multi-node launch docs or reference scripts | `docs/user_guide/tutorials/multi_node_training.md` |
| G3 | No shared-FS validation on multi-node; after rank0 writes, resume on other nodes fails outright with costly debugging | `src/speculators/train/checkpointer.py` |
| G4 | `--vllm-endpoint` is single-endpoint; every DP rank's generate requests hit the same extractor — egress bandwidth caps throughput | `src/speculators/train/data.py:153`, `config/schema.py:258` |
| G5 | Multi-node never validated | — |

## 4. Overall Architecture

```text
  extractor node group (scales independently)      trainer node group (this iteration: 2 x 8)
  ┌─────────────────────────┐            ┌──────────────────────────────┐
  │ node E0: launch_vllm.py │            │ node T0: torchrun rank 0-7   │
  │   vLLM TP8, port 8000   │            │   FSDP2, DP16 (cross-node)   │
  │ node E1: launch_vllm.py │            │ node T1: torchrun rank 8-15  │
  │   vLLM TP8, port 8000   │            │                              │
  └───────────┬─────────────┘            └──────────────┬───────────────┘
              │  HTTP generate (EndpointRouter RR+evict)│
              │◀────────────────────────────────────────┤
              │                                         │
              └──────────▶ mooncake master ◀────────────┘
                  hidden states streamed back via KV store P2P
                  (keys globally visible, independent of which
                   extractor produced them)
              Shared FS: data-path / save-path (NFS/Lustre)
```

Layered responsibilities (aligned with the existing code):

- **HTTP layer (routing target)**: the `--vllm-endpoints` list → `EndpointRouter`
  selects an endpoint per request.
- **Transfer layer (no routing)**: hidden-state retrieval goes through
  `HiddenStatesTransfer` (mooncake P2P or shared-FS file backend); handles just need
  to be globally visible, regardless of extractor.
  **Hard constraint: all extractors must point to the same mooncake master / the same
  shared-FS path.**
- **Training layer**: `distributed.py` topology unchanged; DP groups naturally span
  nodes (strided global ranks).

## 5. Detailed Design

### 5.1 Multi-Node Launch and Initialization (G1/G2/G5)

#### 5.1.1 Timeout (G1)

`maybe_setup_distributed()` gains a timeout parameter, passed through to
`init_process_group` (same idea as Megatron, see
`megatron/training/initialize.py:341-355`):

```python
# distributed.py
def maybe_setup_distributed(sp_size: int = 1, timeout_minutes: float = 30.0) -> None:
    ...
    dist.init_process_group(
        backend, device_id=local_rank,
        timeout=timedelta(minutes=timeout_minutes),
    )
```

- `TrainConfig` gains `distributed_timeout_minutes: float = 30` (schema.py);
  `cli.py main()` passes
  `maybe_setup_distributed(timeout_minutes=args.distributed_timeout_minutes)`.
- With 30s-interval multi-node rendezvous plus large-model loading, 30 minutes is a
  safe default; increase it when HCCL link setup is slow on NPU.

#### 5.1.2 Reference Launch Scripts (G2)

Add `examples/train/multi_node/` (env-var driven style, mirroring Megatron
`examples/llama/train_llama3_8b_h100_fp8.sh:21-53`):

- `launch_train_node.sh`: driven by
  `GPUS_PER_NODE/NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT`

  ```bash
  torchrun \
    --nnodes "$NNODES" --node-rank "$NODE_RANK" \
    --nproc-per-node "$GPUS_PER_NODE" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    -m speculators.train ...
  ```

- `launch_extractor_node.sh`: one `launch_vllm.py` instance per node, `NODE_RANK=0`
  as head, the rest `--headless` (same pattern as kimi_k3_dspark).
- `run_multi_node.md`: topology notes plus Mooncake multi-NIC gotchas
  (`MOONCAKE_LOCAL_HOSTNAME` / `FABRIC_SUBNET` — already bitten by these in the kimi
  example).

Rank semantics unchanged: `RANK/WORLD_SIZE` come from torchrun's global assignment;
the SP/DP subgroups in `distributed.py` are partitioned by global rank, so at 16 ranks
the DP group spans nodes automatically — no topology code changes needed.

#### 5.1.3 Shared-FS Startup Probe (G3)

When multi-node (`is_distributed() and get_world_size() > 1`), run right after
initialization in `cli.py main()`:

1. rank0 writes a probe file `.fs_probe_<uuid>` under `save_path`, then
   `dist.barrier()`;
2. each node's local rank0 (`LOCAL_RANK == 0`) checks the file's visibility;
3. `all_reduce(SUM)` aggregates results; any node that cannot see the file → raise,
   with a message that "save-path/data-path must be a shared filesystem visible to
   all nodes";
4. clean up the probe. Provide an `--skip-fs-probe` escape hatch.

### 5.2 Client-Side Multi-Endpoint Routing (G4, core addition)

#### 5.2.1 Configuration and Interfaces

| Layer | Change |
|---|---|
| `config/schema.py` `GenerationArgs` | Add `vllm_endpoints: list[str] = []`; keep `vllm_endpoint` (backward compatible, single value) |
| `config/resolution.py` | Merge rules, see below |
| `cli.py` | Pass the merged `list[str]` down to `create_train_val_loaders` |
| `train/dataloader.py` | Rename parameter `vllm_endpoint: str` → `vllm_endpoints: list[str]` |
| `train/data.py` `ArrowDataset` | Same; `_setup_client` becomes lazy per-endpoint client creation |

Merge rules (centralized in the resolution layer):

- Both empty → error (generate mode requires at least one endpoint);
- Only `vllm_endpoint` → `[vllm_endpoint]` (fully backward compatible);
- Only `vllm_endpoints` → used as-is;
- Both set → error (ambiguity), prompting to pass only one;
- Deduplicate, preserving order.

#### 5.2.2 `EndpointRouter` (new file `src/speculators/data_generation/endpoint_router.py`)

State (one per process; DataLoader workers are spawn processes, each holding its own
router — no cross-process/cross-node synchronization needed):

```python
class EndpointRouter:
    def __init__(
        self,
        endpoints: list[str],
        model: str | None = None,
        failure_threshold: int = 2,      # evict after N consecutive failures
        cooldown_seconds: float = 60.0,  # eviction duration, exponential (x2, cap 600s)
        client_factory: Callable[[str], Client] = ...,  # openai.Client(base_url=ep, max_retries=0)
    ): ...

    def pick(self) -> tuple[Client, str]:        # RR over healthy endpoints, returns (client, endpoint)
    def report_success(self, endpoint: str) -> None   # reset failure count, half-open → healthy
    def report_failure(self, endpoint: str) -> None   # count += 1, enter cooldown at threshold
    def all_down(self) -> bool
```

Per-endpoint state machine:

```text
HEALTHY ──consecutive failures ≥ threshold──▶ COOLDOWN(t) ──t expires──▶ HALF-OPEN ──success──▶ HEALTHY
   ▲                                             │                          │
   └────────────── success ◀─────────────────────┴────── failure ──▶ COOLDOWN(2t, cap 600s)
```

Scheduling semantics:

- `pick()`: round-robin over HEALTHY endpoints only (cursor advances past COOLDOWN);
  when everything is in COOLDOWN, **no full-eviction fallback** — return the endpoint
  with the earliest cooldown expiry (dead-end fallback) so training never deadlocks on
  misconfiguration or transient outages;
- Lazy initialization: create an `openai.Client` per endpoint on first `pick()`
  (`max_retries=0` — retry semantics move up to the router layer); model discovery
  runs in `_setup_endpoints()` (before the first sample generation) against every
  endpoint: endpoints whose model id does not match are **evicted permanently**
  (`router.evict` — misconfiguration is not a transient fault); if all fail, raise and
  let the outer round-trip retry handle it;
- Threading: single thread (sync path) or single loop (async path) inside a DataLoader
  worker — no locks; if `generate_offline_data.py` reuses this with async later,
  re-evaluate then.

#### 5.2.3 Integration with Existing Retry/Recovery Mechanisms (key semantics)

Current call chain: `ArrowDataset._generate_hidden_states_once` (data.py:227)
→ `generate_hidden_states()` (vllm_client.py:243, `@with_retries` same-endpoint
backoff retry)
→ `transfer.get_generated()` → `check_hidden_states()` → `GenerationRecoveryGuard`.

Refactor into **router-aware outer retries** (`vllm_client.generate_hidden_states`
keeps its name; it now receives a pre-selected client):

```python
# data.py: _generate_hidden_states_once (pseudocode)
for attempt in range(self.max_router_attempts):        # = max_retries + 1, semantics moved up
    client, endpoint = self.router.pick()
    try:
        handle, params = generate_hidden_states(client, model, item,
                                                max_retries=0,   # retries move up
                                                timeout=self.request_timeout)
    except InvalidResponseError:
        self.router.report_success(endpoint)   # content validation failure ≠ endpoint fault
        raise                                   # left to GenerationRecoveryGuard counting
    except openai.APIStatusError as e:
        if e.status_code in {408, 429, 502, 503, 504}:
            self.router.report_failure(endpoint); continue  # endpoint-level fault
        self.router.report_success(endpoint); raise      # 4xx semantic errors fail fast
    except openai.APIConnectionError as e:      # includes APITimeoutError
        self.router.report_failure(endpoint); continue
    self.router.report_success(endpoint)
    ...  # transfer.get_generated / check_hidden_states; failures follow the InvalidResponseError path
```

Boundary with recovery (aligned with `recovery.py` semantics):

- **The router only tracks endpoint health** and does not count toward
  `GenerationRecoveryGuard`'s per-sample failure counter; a sample is exhausted only
  after all router attempts are used up;
- All endpoints evicted and the dead-end fallback also fails → the guard's
  `max_consecutive_generation_failures=20` circuit breaker + `BatchRecoveryCoordinator`
  cross-rank sync (recovery.py:200 `all_reduce`) kick in — same behavior as a
  single-endpoint outage;
- In-flight requests failing at eviction time are absorbed by the retry semantics
  above and do not trigger batch recovery.

Logging: maintain `{requests, failures}` counters per endpoint and emit a `rank`-prefixed
log line every 100 requests (or on evict/recover events) to ease multi-node debugging.

#### 5.2.4 Preconditions for LB Effectiveness

RR is approximately uniform under "independent cursor per rank": 16 ranks × 12 workers
interleave requests enough to spread load. vLLM on the extractor side queues requests
internally and absorbs short-term imbalance — this is why RR was chosen over
queue-depth-aware LB. A subclassing point is reserved on `EndpointRouter`: swap
`pick()` for a queue-depth-aware implementation polling `/metrics` (interface only in
this iteration).

### 5.3 Checkpoint (unchanged + explicit constraints)

- `DistributedCheckpointer` (checkpointer.py:363-442) logic unchanged:
  `full_state_dict=True, cpu_offload=True`, gather to rank0 and write. Draft models are
  small (≤1B params) so the full gather fits in memory; no new risk on 2 nodes with
  shared FS.
- The probe in 5.1.3 moves "shared FS not visible" from a mid-training error to a
  startup error.
- Resume: `_get_previous_epoch` reads the shared path, consistent across nodes.
- Future option (recorded, not implemented): `--sharded-checkpoint` via
  `torch.distributed.checkpoint` with per-rank shards (same as Megatron
  `fully_parallel_save`, see `megatron/training/checkpointing.py:471-792`).

### 5.4 Networking and Mooncake Notes

- `MOONCAKE_LOCAL_HOSTNAME`: must be set explicitly on multi-NIC clusters (already
  documented in multi_node_training.md:135);
- With 16 ranks fetching concurrently, consider raising `--mooncake-global-segment-gib`
  from 4.0 (value to be set by benchmarking);
- Single-master bandwidth is a candidate bottleneck: benchmark item (see §8); evaluate
  multi-master sharding only if it falls short (non-goal this iteration);
- fcntl file locks (vllm_client.py:154 `wait_for_lock`) are used by the file backend
  only; semantics on NFS are best-effort — multi-node risk recorded (the mooncake
  backend does not use file locks).

## 6. Code Change List

| File | Change | Size |
|---|---|---|
| `src/speculators/data_generation/endpoint_router.py` | **New** EndpointRouter state machine | ~150 lines |
| `src/speculators/train/data.py` | ArrowDataset multi-endpoint + router integration + retry semantics moved up | ~80 lines |
| `src/speculators/train/config/schema.py` | `vllm_endpoints` + `distributed_timeout_minutes` fields | ~15 lines |
| `src/speculators/train/config/resolution.py` | Endpoint merge validation | ~20 lines |
| `src/speculators/train/cli.py` | Pass-through + FS probe call | ~30 lines |
| `src/speculators/train/dataloader.py` | Parameter rename | ~10 lines |
| `src/speculators/train/distributed.py` | Timeout parameter | ~5 lines |
| `src/speculators/train/utils.py` | FS probe function | ~40 lines |
| `docs/user_guide/tutorials/multi_node_training.md` | Multi-node training section update | docs |
| `examples/train/multi_node/*` | **New** launch scripts + README | scripts |
| `tests/` | Router unit tests + integration tests | ~200 lines |

## 7. Compatibility and Rollback

- Single-endpoint users see zero change: `--vllm-endpoint` behaves as before, the
  router degenerates to pass-through;
- Single-node multi-device unaffected: the probe only triggers when multi-node
  (world_size > 1);
- Rollback switches: omit `--vllm-endpoints` to return to the status quo;
  `--skip-fs-probe` skips the probe.

## 8. Test Plan (TODO-4)

| Level | Case | Pass criteria |
|---|---|---|
| Unit | EndpointRouter: RR order / evict at threshold / cooldown recovery / half-open failure doubles / all-down fallback / model-id mismatch eviction | Full state-machine coverage |
| Unit | resolution merge rules (compat / conflict / dedup) | All 4 rules |
| Integration (single-node NPU) | Mock two endpoints: 1 healthy + 1 always-503 → training completes normally | Loss curve matches single-endpoint baseline; logs show eviction |
| Integration (single-node NPU) | Kill one of two endpoints → training uninterrupted, recovery after restart | No fatal BatchRecovery errors |
| Multi-node smoke (2×8 NPU) | Small dataset, torchrun multi-node | Startup succeeds; loss aligned with single-node at same seed; checkpoint resumes across nodes |
| Performance | Fixed data+model, sweep nnodes ∈ {1,2} | Training throughput speedup ≥ 1.7x |
| Performance | Fixed training scale, sweep extractors ∈ {1,2} | Online generation throughput ≥ 1.6x; record mooncake master bandwidth headroom |
| Correctness | Multi-node 5 epochs vs single-node 5 epochs | Loss curves align step-by-step (same seed) |

## 9. Milestones

1. **M1 (TODO-3a)**: EndpointRouter + multi-endpoint config chain + unit tests —
   routing capability first, developable and verifiable on a single node;
2. **M2 (TODO-3b)**: timeout + FS probe + launch scripts + docs — multi-node launch
   capability;
3. **M3 (TODO-4)**: 2×8 NPU smoke → correctness alignment → two-dimensional
   performance benchmarking; feed the numbers into the roadmap initiative
   ("how quickly we can speed-up training").

## 10. Megatron-LM Cross-Reference (design rationale)

| Design decision here | Megatron-LM reference |
|---|---|
| Static torchrun wiring params | `examples/llama/train_llama3_8b_h100_fp8.sh:47-53` |
| Configurable init timeout | `megatron/training/initialize.py:341-355` (`--distributed-timeout-minutes`) |
| DP group spanning nodes (strided) | `megatron/core/parallel_state.py` (our distributed.py is already isomorphic) |
| Sharded checkpoint (future) | `megatron/training/checkpointing.py:471-792` |
| FT / queue-depth LB (non-goal, reserved) | `megatron/training/ft_integration.py` |
