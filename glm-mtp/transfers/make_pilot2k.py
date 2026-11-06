import shutil
from pathlib import Path

from datasets import Dataset, load_from_disk

src = Path("/data1/mtp_training/data/prepared_8k_dflash2_official")
out = Path("/data1/mtp_training/data/prepared_2k_dflash2_pilot")
shutil.rmtree(out, ignore_errors=True)
d = load_from_disk(src)
kept = []
dropped = 0
for row in d:
    ids = row["input_ids"][:2048]
    mask = row["loss_mask"][:2048]
    if len(ids) < 3 or int(mask[2:].sum()) == 0:
        dropped += 1
        continue
    kept.append({**row, "input_ids": ids, "loss_mask": mask, "seq_len": len(ids)})
out_d = Dataset.from_list(kept)
out_d.set_format(type="torch", columns=["input_ids", "loss_mask", "seq_len"])
print("source", len(d), "kept", len(kept), "dropped", dropped, "format", out_d.format)
out_d.save_to_disk(out)
print("saved", out)
