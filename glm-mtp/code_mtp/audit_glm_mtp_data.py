import json
from pathlib import Path
from datasets import load_from_disk

root = Path('/data1/mtp_training')
data = load_from_disk(root / 'data/prepared_8k_dflash2_official').with_format(None)
checks = dict(count=len(data), max_length=0, invalid=0, train_count=0, validation_count=0)
sessions = {'train': set(), 'validation': set()}
for row in data:
    ids, mask = row['input_ids'], row['loss_mask']
    valid = (3 <= len(ids) <= 8192 and len(ids) == len(mask)
             and sum(mask) >= 16 and min(ids) >= 0 and max(ids) < 154880)
    checks['invalid'] += int(not valid)
    checks['max_length'] = max(checks['max_length'], len(ids))
    split = row['split']
    checks[split + '_count'] += 1
    sessions[split].add(row['session_hash'])
checks['overlapping_sessions'] = len(sessions['train'] & sessions['validation'])
splits = list(data['split'])
checks['split_contiguous'] = (all(s == 'train' for s in splits[:19875])
                              and all(s == 'validation' for s in splits[19875:]))
checks['status'] = ('passed' if checks['invalid'] == 0 and len(data) == 22070
                    and checks['overlapping_sessions'] == 0 and checks['split_contiguous'] else 'failed')
(root / 'provenance/dataset_audit.json').write_text(json.dumps(checks, indent=2))
print(json.dumps(checks), flush=True)
assert checks['status'] == 'passed'
