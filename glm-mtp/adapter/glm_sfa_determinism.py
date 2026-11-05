"""Canonical ordering for logically equivalent sparse attention selections."""
import torch


def sort_sparse_indices(indices: torch.Tensor) -> torch.Tensor:
    """Sort selected logical token IDs; keep invalid padding at the end.

    The valid selected multiset is unchanged. A fixed logical order avoids the
    observed Ascend sparse-attention rounding dependency on physical KV pages.
    """
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("Sparse indices must use int32 or int64")
    sentinel = torch.iinfo(indices.dtype).max
    ordered = torch.where(indices >= 0, indices, sentinel).sort(dim=-1).values
    return torch.where(ordered == sentinel, -1, ordered)
