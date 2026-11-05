"""Static shape/logic test for the GLM five-step teacher-forced path.

This does not load GLM weights. It validates the index arithmetic used by the
NPU implementation before uploading the patch to the training container.
"""
from __future__ import annotations


def ranges(seq_len: int, steps: int):
    if seq_len < steps + 2:
        raise ValueError
    valid_len = seq_len - steps - 1
    out = []
    for step in range(steps):
        start = step + 1
        out.append({
            "hidden": (0, valid_len),
            "tokens": (start, start + valid_len),
            "labels": (start + 1, start + 1 + valid_len),
        })
    return out


def main() -> None:
    got = ranges(8192, 5)
    assert len(got) == 5
    assert all(item["hidden"] == (0, 8186) for item in got)
    assert got[0]["tokens"] == (1, 8187)
    assert got[4]["tokens"] == (5, 8191)
    assert got[0]["labels"] == (2, 8188)
    assert got[4]["labels"] == (6, 8192)
    print("GLM_MTP_FIVE_STEP_INDEX_TEST_PASSED")


if __name__ == "__main__":
    main()
