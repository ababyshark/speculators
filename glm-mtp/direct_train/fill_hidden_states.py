#!/usr/bin/env python3
"""Pre-generate and PERSIST all hidden states for direct MTP training.

Why: the trainer's --on-missing generate consumes HS in memory only (no
persistence path exists upstream), and formal100 relied on a 16-card layout
(endpoint on 8 + FSDP on 8). With a single 8-card window we must fill
hs_{idx}.safetensors first, then train fully offline with --on-missing raise.

Reuses the trainer's own machinery for generation/validation:
  VllmHiddenStatesSource (vllm round trip + check_hidden_states + recovery)
  FileTransfer.get_cached (resume-safe existence check)

Writes hs_{storage_index}.safetensors = {"hidden_states": [seq, L, h] bf16,
"token_ids": [seq] int64} — the exact payload _build_sample consumes.

Usage (inside the training container, endpoint already up):
  python3 fill_hidden_states.py <data_path> <hs_path> <served_model_name> <endpoint> [chunk]
"""
# ruff: noqa
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/speculators/src")

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from hs_connectors import FileTransfer  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from speculators.train.data_types import DatasetRecord  # noqa: E402
from speculators.train.hidden_states_source import VllmHiddenStatesSource  # noqa: E402

DATA, HS_PATH, MODEL, ENDPOINT = sys.argv[1:5]
CHUNK = int(sys.argv[5]) if len(sys.argv) > 5 else 16


def main():
    d = load_from_disk(DATA)
    n = len(d)
    transfer = FileTransfer(hidden_states_path=Path(HS_PATH))
    source = VllmHiddenStatesSource(
        transfer=transfer,
        model=MODEL,
        vllm_endpoint=ENDPOINT,
        request_timeout=600.0,
        max_retries=2,
        max_transfer_workers=8,
    )
    missing = [i for i in range(n) if transfer.get_cached(i) is None]
    print(f"{n} records, {n - len(missing)} cached, {len(missing)} to generate", flush=True)

    t0 = time.time()
    done = 0
    for start in range(0, len(missing), CHUNK):
        batch_idx = missing[start:start + CHUNK]
        records = [DatasetRecord.from_arrow_row(d[i], storage_index=i) for i in batch_idx]
        results = source.get_batch(records)
        for i, rec, hs in zip(batch_idx, records, results):
            if hasattr(hs, "reason") or hs is None:
                raise RuntimeError(f"storage index {i} unavailable: {getattr(hs, 'reason', '?')}")
            payload = {
                "hidden_states": hs["hidden_states"].to(torch.bfloat16).contiguous(),
                "token_ids": rec.input_ids.to(torch.int64),
            }
            tmp = Path(HS_PATH) / f"hs_{i}.safetensors.tmp"
            save_file(payload, str(tmp))
            tmp.rename(Path(HS_PATH) / f"hs_{i}.safetensors")
            done += 1
        rate = done / max(time.time() - t0, 1) * 60
        eta = (len(missing) - done) / max(rate, 1e-9)
        print(f"[{done}/{len(missing)}] {rate:.1f}/min, ETA {eta:.0f} min", flush=True)

    source.close()
    left = [i for i in range(n) if transfer.get_cached(i) is None]
    if left:
        print(f"INCOMPLETE: {len(left)} missing -> {left[:10]}", flush=True)
        sys.exit(1)
    print(f"ALL {n} HS PRESENT in {HS_PATH}", flush=True)


if __name__ == "__main__":
    main()
