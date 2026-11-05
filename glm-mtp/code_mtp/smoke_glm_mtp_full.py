"""Check actual converted checkpoint loading and short NPU inference (not native parity)."""
import hashlib
import json
import time
from pathlib import Path
import torch
import torch_npu
from speculators.models.mtp import MTPDraftModel

torch.set_num_threads(8)
torch.manual_seed(13)
torch.npu.set_device(0)
torch.use_deterministic_algorithms(True)
torch.npu.matmul.allow_hf32 = False
root = Path('/data1/mtp_training/weights/glm53_mtp_speculators_step0')
start = time.monotonic()
model = MTPDraftModel.from_pretrained(root, dtype=torch.bfloat16, local_files_only=True)
mapping = json.loads((root / 'conversion_report.json').read_text())['mapping']
state = model.state_dict()
assert set(state) == set(mapping), (set(state) - set(mapping), set(mapping) - set(state))
for key, tensor in state.items():
    actual = hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert str(tensor.dtype) == mapping[key]['dtype'], (key, tensor.dtype, mapping[key]['dtype'])
    assert actual == mapping[key]['sha256'], key
print('All 793 loaded tensors match converted checkpoint bitwise', flush=True)
assert not model.embed_tokens.weight.requires_grad
assert not model.lm_head.weight.requires_grad
assert all(not p.requires_grad for p in model.mtp_layers[0].self_attn.indexer.parameters())
model = model.to('npu:0').eval()
ids = torch.tensor([[100, 101, 102, 103, 104, 105, 106, 107]], device='npu:0')
hidden = torch.randn(1, 8, 6144, device='npu:0', dtype=torch.bfloat16)
with torch.no_grad():
    _, loss, metrics = model(ids, hidden, loss_mask=torch.ones_like(ids))
torch.npu.synchronize()
assert torch.isfinite(loss)
report = dict(status='passed', loaded_tensors_bitwise_exact=793,
              synthetic_short_loss=float(loss), seconds=time.monotonic()-start,
              peak_allocated_gib=torch.npu.max_memory_allocated()/2**30,
              native_parity='NOT_YET_VALIDATED')
Path('/workspace/code_mtp/logs/full_weight_smoke.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
