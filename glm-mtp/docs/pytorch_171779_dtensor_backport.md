# Long-context (16K) FSDP2 + activation checkpoint DTensor failure: root cause and backport guide

## Symptom

With `SPECULATORS_MTP_ACTIVATION_CKPT=1` (default), GLM MTP training with K>1 recursive rollout inside nested FSDP2 crashes in backward recompute:

```
RuntimeError: aten.matmul.default: got mixed torch.Tensor and DTensor
```

## Root cause

Upstream PyTorch bug [pytorch#153354](https://github.com/pytorch/pytorch/issues/153354): nested FSDP2 + non-reentrant activation checkpointing + the same module forwarded multiple times (the MTP K-step recursion is exactly this pattern; the community reproduces it as DPO double-forward). When recompute enters, the parameters have already been resharded by a previous post-backward stage, so saved (plain tensor) inputs meet DTensor weights.

Official fix: [pytorch#171779](https://github.com/pytorch/pytorch/pull/171779) — 4 lines in `torch/distributed/fsdp/_fully_shard/_fsdp_state.py`: in `_pre_forward`'s PRE_BACKWARD branch, check `is_unsharded` and re-unshard before the early return.

## Workarounds and when they apply

| Approach                            | Mechanism                                                                                                 | Applies                                                                                                                        |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `SPECULATORS_MTP_ACTIVATION_CKPT=0` | skips the per-step layer checkpoint so backward never recomputes it; per-step layer activations stay live | works at ~2304 context (validated: formal100 / v15 runs); **OOMs at 16K** (measured 55.8/61.3 GiB forward on a 61.27 GiB card) |
| torch backport PR#171779 + AC on    | recompute unshards parameters correctly                                                                   | required for 16K x K=5; backward recompute peaks ~59.2 GiB, needs the memory levers below                                      |

## Backport procedure (torch 2.10.0 container)

1. Back up `_fsdp_state.py`.
2. In `_pre_forward`, locate the PRE_BACKWARD early-return branch and add the `is_unsharded` check + re-unshard (see upstream PR#171779 diff).
3. Restart training with AC **enabled** (remove `SPECULATORS_MTP_ACTIVATION_CKPT=0`).

The patch lives inside the container image — **it is lost on container rebuild**; keep a copy of the patched file next to the launcher script.

## Memory levers for 16K x K=5 (61.27 GiB card)

- Chunked causal-mask construction (`_glm_causal_mask`, row chunk 4096): ~792MB -> ~600MB construction peak.
- `--mtp-chunk-size 32` (or `SPECULATORS_MTP_CHUNK_SIZE=32`): per-chunk lm_head logits peak halved (chunk x vocab x 4B).
- `SPECULATORS_FSDP_REDUCE_DTYPE=bfloat16`, allocator `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256`.
- Reserve lever if still short (\<400MiB gap): checkpoint the whole rollout and recompute all K steps in backward (~750MB saved, forward x2 cost).

## Related env vars

- `SPECULATORS_MTP_ACTIVATION_CKPT` (default `1`): set `0` to skip the layer checkpoint (short-context workaround above).
- `SPECULATORS_MTP_CHUNK_SIZE` / `--mtp-chunk-size`: chunk token count for `_glm_chunked_ce` / `_glm_chunked_distill` / `_glm_chunked_argmax`.
