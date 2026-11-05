from pathlib import Path

p=Path("/workspace/speculators/src/speculators/train/dataloader.py")
s=p.read_text()
old="""    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=get_dp_size(),
        rank=get_dp_rank(),
        seed=seed,
        max_batches=max_batches,
    )
"""
new="""    if os.environ.get("SPECULATORS_MTP_NO_PACK") == "1":
        class _SingleSampleSampler:
            def __iter__(self):
                indices = list(range(get_dp_rank(), len(dataset), get_dp_size()))
                if max_batches is not None:
                    indices = indices[:max_batches]
                return iter(([i] for i in indices))
            def __len__(self):
                n = max(0, (len(dataset) - get_dp_rank() + get_dp_size() - 1) // get_dp_size())
                return min(n, max_batches) if max_batches is not None else n
        batch_sampler = _SingleSampleSampler()
    else:
        batch_sampler = MultipackDistributedBatchSamplerV2(
            batch_max_length=total_seq_len,
            lengths=dataset.approx_lengths,
            num_replicas=get_dp_size(),
            rank=get_dp_rank(),
            seed=seed,
            max_batches=max_batches,
        )
"""
if new not in s:
    assert s.count(old)==1
    p.write_text(s.replace(old,new))
print("single-sample loader patch applied")
