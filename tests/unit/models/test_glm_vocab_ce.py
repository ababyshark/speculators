"""_VocabCE: the large-vocab cross-entropy used inside the MTP draft training.

functional.cross_entropy hits pathologically slow kernel paths on NPU for very large
vocabs (154880) and materializes ~10GB of intermediates. _VocabCE decomposes
CE into logsumexp - target_logit (forward) and (softmax - onehot) * mask
(backward) — mathematically identical, NPU-fast. These tests lock that
equivalence in both directions, including the ignore_index semantics.
"""

import torch
from torch.nn import functional

from speculators.models.mtp.core import _IGNORE_INDEX, _VocabCE


def test_forward_matches_cross_entropy() -> None:
    torch.manual_seed(0)
    vocab = 97
    logits = torch.randn(32, vocab)
    targets = torch.randint(0, vocab, (32,))
    targets[::5] = _IGNORE_INDEX
    nll = _VocabCE.apply(logits, targets, _IGNORE_INDEX)
    ref = functional.cross_entropy(
        logits, targets, ignore_index=_IGNORE_INDEX, reduction="none"
    )
    assert torch.allclose(nll, ref, rtol=1e-4, atol=1e-5)
    # ignored positions are exactly zero
    assert (nll[targets == _IGNORE_INDEX] == 0).all()


def test_backward_matches_cross_entropy() -> None:
    torch.manual_seed(1)
    vocab = 97
    logits = torch.randn(24, vocab, requires_grad=True)
    targets = torch.randint(0, vocab, (24,))
    targets[::4] = _IGNORE_INDEX
    grad_out = torch.randn(24)

    nll = _VocabCE.apply(logits, targets, _IGNORE_INDEX)
    (nll * grad_out).sum().backward()

    ref_logits = logits.detach().clone().requires_grad_(True)
    ref = functional.cross_entropy(
        ref_logits, targets, ignore_index=_IGNORE_INDEX, reduction="none"
    )
    (ref * grad_out).sum().backward()

    assert torch.allclose(logits.grad, ref_logits.grad, rtol=1e-3, atol=1e-4)
    # no gradient flows to logits at ignored positions
    assert (logits.grad[targets == _IGNORE_INDEX] == 0).all()


def test_all_ignored_is_safe() -> None:
    """A fully-masked step (all targets ignored) must not produce NaN."""
    logits = torch.randn(4, 11, requires_grad=True)
    targets = torch.full((4,), _IGNORE_INDEX)
    nll = _VocabCE.apply(logits, targets, _IGNORE_INDEX)
    assert torch.isfinite(nll).all()
    (nll.sum()).backward()  # grad exists and is finite
    assert torch.isfinite(logits.grad).all()
