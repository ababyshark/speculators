"""Extract native GLM MTP tensors without changing names/dtypes; verify readback."""
import collections,hashlib,json,struct,time,shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
source=Path('/data1/weight/GLM-5.3-w4a8-spec')
out=Path('/data1/mtp_training/weights/glm53_mtp_native_step0')
out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(8)
index=json.loads((source/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
keys=sorted(k for k in index if k.startswith('model.layers.78.'))
assert len(keys)==793
report={'source':str(source),'output':str(out),'started':time.time(),'tensors':{},'passed':False}
newindex={};buffer={};size=0;shard=0

def digest(t):
 return hashlib.sha256(t.contiguous().view(torch.uint8).numpy()).hexdigest()
def flush():
 global buffer,size,shard
 if not buffer:return
 shard+=1;name=f'mtp-{shard:03d}.safetensors'
 save_file(buffer,str(out/name))
 with safe_open(out/name,framework='pt') as f:
  for key in buffer:
   assert digest(f.get_tensor(key))==report['tensors'][key]['sha256'],key
   newindex[key]=name
 buffer={};size=0
 print('SHARD_VERIFIED',name,flush=True)
for file in sorted(set(index[k] for k in keys)):
 with safe_open(source/file,framework='pt') as f:
  for key in keys:
   if index[key]!=file:continue
   t=f.get_tensor(key)
   assert t.is_floating_point(),key
   assert all(torch.isfinite(c).all().item() for c in t.reshape(-1).split(4_000_000)),key
   n=t.numel()*t.element_size()
   if size+n>2*2**30:flush()
   report['tensors'][key]={'shape':list(t.shape),'dtype':str(t.dtype),'numel':t.numel(),'sha256':digest(t),'source_shard':file}
   buffer[key]=t.clone();size+=n
flush()
report['shared_weight_comparison']={}
for own,main in [('model.layers.78.embed_tokens.weight','model.embed_tokens.weight'),('model.layers.78.shared_head.head.weight','lm_head.weight')]:
 with safe_open(source/index[main],framework='pt') as f:
  h=digest(f.get_tensor(main))
 report['shared_weight_comparison'][own]={'main_key':main,'main_sha256':h,'exact':h==report['tensors'][own]['sha256']}
shutil.copy2(source/'config.json',out/'config.json')
(out/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':sum(v['numel']*(4 if v['dtype']=='torch.float32' else 2) for v in report['tensors'].values())},'weight_map':newindex},indent=2))
report.update(passed=True,ended=time.time(),tensor_count=len(keys),numel=sum(v['numel'] for v in report['tensors'].values()),format='Native names, not yet a Speculators model checkpoint')
(out/'extraction_report.json').write_text(json.dumps(report,indent=2))
print('EXTRACTION_PASSED',len(keys),report['shared_weight_comparison'],flush=True)
