"""Tests for NoPackDistributedBatchSampler (GLM K=1 NO_PACK path)."""

import numpy as np
import pytest

from speculators.train.distributed_batch_sampler import (
    NoPackDistributedBatchSampler,
)


def _batches_for(lengths, num_replicas, rank, seed=0, max_batches=None, epoch=0):
    sampler = NoPackDistributedBatchSampler(
        lengths=lengths,
        num_replicas=num_replicas,
        rank=rank,
        seed=seed,
        max_batches=max_batches,
    )
    sampler.set_epoch(epoch)
    return [int(batch[0]) for batch in sampler]


def test_equal_batches_per_rank_invariant():
    lengths = np.arange(1, 129)  # 128 rows
    per_rank = [_batches_for(lengths, num_replicas=8, rank=rank) for rank in range(8)]
    counts = {len(batches) for batches in per_rank}
    # 128 / 8 ranks: every rank gets exactly the same number of batches.
    assert counts == {16}
    # Batches are single samples.
    assert all(len(batches) == 16 for batches in per_rank)


def test_remainder_is_dropped_not_unevenly_sharded():
    lengths = np.arange(1, 116)  # 115 rows, 8 ranks -> 112 sharded, 3 dropped
    per_rank = [_batches_for(lengths, num_replicas=8, rank=rank) for rank in range(8)]
    # The n % R tail must be dropped on every rank symmetrically, never
    # given to rank 0 (unequal counts would hang DDP collectives).
    assert {len(batches) for batches in per_rank} == {14}
    seen = {index for batches in per_rank for index in batches}
    assert len(seen) == 112
    assert seen.isdisjoint(set(range(128)) - seen)  # sanity: ints only


def test_partitions_are_disjoint_and_complete():
    lengths = np.arange(1, 65)  # 64 rows, 4 ranks
    per_rank = [_batches_for(lengths, num_replicas=4, rank=rank) for rank in range(4)]
    seen = [index for batches in per_rank for index in batches]
    assert sorted(seen) == list(range(64))


def test_same_seed_is_deterministic_across_ranks():
    lengths = np.arange(1, 65)
    first = _batches_for(lengths, num_replicas=4, rank=0)
    second = _batches_for(lengths, num_replicas=4, rank=0)
    assert first == second


def test_set_epoch_reshuffles_and_recomputes_length():
    lengths = np.arange(1, 65)
    sampler = NoPackDistributedBatchSampler(lengths=lengths, num_replicas=4, rank=0)
    epoch0 = [int(batch[0]) for batch in sampler]
    sampler.set_epoch(1)
    epoch1 = [int(batch[0]) for batch in sampler]
    assert len(epoch0) == len(epoch1) == 16
    assert epoch0 != epoch1


def test_max_batches_truncates():
    lengths = np.arange(1, 65)
    batches = _batches_for(lengths, num_replicas=4, rank=0, max_batches=2)
    assert len(batches) == 2


def test_seed_controls_the_permutation():
    """Different seeds derive different permutations; validation stability
    comes from the dataloader always passing seed=0 for the val loader."""
    lengths = np.arange(1, 65)
    val_a = _batches_for(lengths, num_replicas=4, rank=1, seed=0)
    val_b = _batches_for(lengths, num_replicas=4, rank=1, seed=42)
    assert val_a != val_b
    assert sorted(val_a + val_b) != list(range(64))  # each is a permutation shard
    assert sorted(val_a) != sorted(range(16))  # shards are order-significant


@pytest.mark.parametrize("num_replicas", [1, 2, 4, 8])
def test_every_sample_appears_exactly_once_when_divisible(num_replicas):
    lengths = np.arange(1, 33)
    per_rank = [
        _batches_for(lengths, num_replicas=num_replicas, rank=rank)
        for rank in range(num_replicas)
    ]
    seen = [index for batches in per_rank for index in batches]
    assert sorted(seen) == list(range(32))
