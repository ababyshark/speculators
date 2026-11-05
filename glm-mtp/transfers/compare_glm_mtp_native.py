"""Measure trained-code step-zero against captures from the native Ascend MTP path."""
import json
import time
from pathlib import Path
import torch
import torch_npu
from safetensors.torch import load_file
from speculators.models.mtp import MTPDraftModel

torch.set_num_threads(8)
torch.manual_seed(13)
torch.npu.set_device(0)
torch.npu.matmul.allow_hf32 = False
torch.use_deterministic_algorithms(True)
root = Path('/data1/mtp_training/parity')
model = MTPDraftModel.from_pretrained('/data1/mtp_training/weights/glm53_mtp_speculators_step0',
                                    dtype=torch.bfloat16, local_files_only=True).to('npu:0').eval()

def metrics(actual, expected):
    a, b = actual.double().flatten(), expected.double().flatten()
    d = a - b
    return dict(max_abs=float(d.abs().max()), mean_abs=float(d.abs().mean()),
                relative_l2=float(d.norm()/b.norm().clamp_min(1e-12)),
                cosine=float(torch.nn.functional.cosine_similarity(a[None], b[None])[0]),
                exact_fraction=float((a==b).double().mean()), finite=bool(torch.isfinite(a).all()))

results = {}
with torch.no_grad():
    for name, teacher_name in [('short64', 'short_a'), ('long8190', 'long8k')]:
        fixture = load_file(str(root / f'native_{name}.safetensors'))
        length = fixture['native_post'].shape[0]
        ids = fixture['input_ids'][:length].long()[None].to('npu:0')
        pos = fixture['positions'][:length].long()[None].to('npu:0')
        hidden = fixture['teacher_hidden'][:length][None].to('npu:0')
        mask = torch.zeros(1, 1, length, length, dtype=torch.bfloat16, device='npu:0')
        causal = torch.ones(length, length, dtype=torch.bool, device='npu:0').triu(1)
        mask.masked_fill_(causal, torch.finfo(torch.bfloat16).min)
        start = time.monotonic()
        output = model.mtp_layers[0](
            hidden, model.embed_tokens(ids), attention_mask=mask, position_ids=pos,
            position_embeddings=model.rotary_emb(hidden, pos),
        )
        logits = model.lm_head(output[:, -1]).float()
        torch.npu.synchronize()
        out = output[0].cpu()
        expected_logits = fixture['native_last_logits'].float()
        row = dict(hidden=metrics(out, fixture['native_post']),
                   logits=metrics(logits.cpu(), expected_logits),
                   top1_actual=int(logits.argmax(-1)), top1_native=int(expected_logits.argmax(-1)),
                   seconds=time.monotonic()-start,
                   peak_allocated_gib=torch.npu.max_memory_allocated()/2**30)
        teacher = load_file(str(root / f'teacher_{teacher_name}.safetensors'))['hidden_states'][:length, 0]
        row['teacher_capture_vs_native_input'] = metrics(teacher, fixture['teacher_hidden'][:length])
        if 'native_topk' in fixture:
            topk = fixture['native_topk']
            row['native_topk_range'] = [int(topk.min()), int(topk.max())]
        row['status'] = 'measured_pending_review'
        results[name] = row
        (root / 'native_parity_comparison.json').write_text(json.dumps(results, indent=2))
        print(name, json.dumps(row), flush=True)
