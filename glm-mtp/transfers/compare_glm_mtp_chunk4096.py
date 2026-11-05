import torch, json
from pathlib import Path
from safetensors.torch import load_file
from speculators.models.mtp import MTPDraftModel
torch.npu.set_device(0); torch.npu.matmul.allow_hf32=False; torch.use_deterministic_algorithms(True)
root=Path('/data1/mtp_training/parity'); f=load_file(str(root/'native_chunk4096.safetensors'))
m=MTPDraftModel.from_pretrained('/data1/mtp_training/weights/glm53_mtp_speculators_step0',dtype=torch.bfloat16,local_files_only=True).to('npu:0').eval()
n=f['native_post'].shape[0]; ids=f['input_ids'][:n].long()[None].to('npu:0'); pos=f['positions'][:n].long()[None].to('npu:0'); h=f['teacher_hidden'][:n][None].to('npu:0')
mask=torch.zeros(1,1,n,n,dtype=torch.bfloat16,device='npu:0'); mask.masked_fill_(torch.ones(n,n,dtype=torch.bool,device='npu:0').triu(1),torch.finfo(torch.bfloat16).min)
with torch.no_grad(): out=m.mtp_layers[0](h,m.embed_tokens(ids),attention_mask=mask,position_ids=pos,position_embeddings=m.rotary_emb(h,pos)); logits=m.lm_head(out[:,-1]).float()
def met(a,b):
    a,b=a.double().flatten(),b.double().flatten(); d=a-b
    return dict(max_abs=float(d.abs().max()),mean_abs=float(d.abs().mean()),rel_l2=float(d.norm()/b.norm()),cos=float(torch.nn.functional.cosine_similarity(a[None],b[None])[0]))
r=dict(hidden=met(out.cpu(),f['native_post']),logits=met(logits.cpu(),f['native_last_logits']),top1=[int(logits.argmax()),int(f['native_last_logits'].argmax())])
print(json.dumps(r,indent=2)); (root/'native_chunk4096_comparison.json').write_text(json.dumps(r,indent=2))
