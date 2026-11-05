"""Probe online teacher on a few inputs; keep only parity fixtures, not a dataset cache."""
import json
import time
from pathlib import Path

import openai
import torch
from datasets import load_from_disk
from safetensors.torch import save_file
from hs_connectors import FileTransfer
from speculators.data_generation.vllm_client import generate_hidden_states

root = Path('/data1/mtp_training/parity')
root.mkdir(parents=True, exist_ok=True)
data = load_from_disk('/data1/mtp_training/data/prepared_8k_dflash2_official').with_format(None)
long_index = next(i for i, value in enumerate(data['seq_len']) if value == 8192)
cases = [('short_a', data[0]['input_ids'][:64]), ('interleaved', data[1]['input_ids'][:96]),
         ('short_b', data[0]['input_ids'][:64]), ('long8k', data[long_index]['input_ids'])]
client = openai.OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='EMPTY', max_retries=0)
transfer = FileTransfer(Path('/data1/mtp_training/teacher/hidden_states'), lock_timeout=180)
results = {}
for name, ids in cases:
    started = time.monotonic()
    handle, params = generate_hidden_states(client, '/data1/weight/GLM-5.3-w4a8-spec',
                                            {'input_ids': ids}, timeout=600, max_retries=0)
    payload = transfer.get_generated(params)
    assert torch.equal(payload['token_ids'].long(), torch.tensor(ids))
    h = payload['hidden_states']
    assert h.shape == (len(ids), 1, 6144), h.shape
    assert torch.isfinite(h).all()
    save_file({k: v.contiguous() for k, v in payload.items()}, str(root / f'teacher_{name}.safetensors'))
    transfer.delete(handle)
    results[name] = dict(tokens=len(ids), seconds=time.monotonic()-started,
                         dtype=str(h.dtype), rms=float(h.float().square().mean().sqrt()))
    if name == 'short_a':
        first = h.clone()
    if name == 'short_b':
        results['repeat'] = dict(bitwise_equal=torch.equal(first, h),
                                 max_abs_diff=float((first.float()-h.float()).abs().max()))
    print(name, results[name], flush=True)
results['long_dataset_index'] = long_index
(root / 'teacher_probe.json').write_text(json.dumps(results, indent=2))
assert results['repeat']['bitwise_equal'], results['repeat']
print('TEACHER_PROBE_PASSED', flush=True)
