import torch
from safetensors.torch import load_file
p='/data1/mtp_training/parity'
a=load_file(p+'/native_long8190.safetensors')['teacher_hidden']
b=load_file(p+'/teacher_long8k.safetensors')['hidden_states'][:8190,0]
print('shapes',a.shape,b.shape,'rms',float(a.float().square().mean().sqrt()),float(b.float().square().mean().sqrt()))
d=a.float()-b.float()
print('diff',float(d.abs().max()),float(d.abs().mean()),float(d.norm()/b.float().norm()),float(torch.nn.functional.cosine_similarity(a.float().flatten()[None],b.float().flatten()[None])[0]))
row=d.abs().amax(-1); vals,idx=torch.topk(row,10); print('rows',list(zip(idx.tolist(),vals.tolist())))
for i in [0,1,64,4095,8189]: print(i,a[i,:4].tolist(),b[i,:4].tolist())
