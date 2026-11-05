"""GLM native MTP components; single-step training is the initial contract."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaAttention, GlmMoeDsaIndexer, GlmMoeDsaMLP, GlmMoeDsaRMSNorm,
    GlmMoeDsaRotaryEmbedding, GlmMoeDsaTopkRouter, apply_rotary_pos_emb_interleave,
)

class GlmMTPRMSNorm(GlmMoeDsaRMSNorm):
    def forward(self, hidden_states):
        dtype=hidden_states.dtype; x=hidden_states.float()
        x=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+self.variance_epsilon)
        return (x*self.weight.float()).to(dtype)

class GlmMTPRotaryEmbedding(GlmMoeDsaRotaryEmbedding):
    def __init__(self, config, device=None):
        config=copy.deepcopy(config); config.head_dim=config.qk_rope_head_dim
        super().__init__(config, device=device)

class GlmMTPIndexer(GlmMoeDsaIndexer):
    @torch.no_grad()
    def forward(self, hidden_states, q_resid, position_embeddings, attention_mask, position_ids, past_key_values=None):
        if past_key_values is not None: raise NotImplementedError("Training indexer does not accept KV caches")
        b,s,_=hidden_states.shape; cos,sin=position_embeddings
        q=self.wq_b(q_resid).view(b,s,self.n_heads,self.head_dim)
        k=self.k_norm(self.wk(hidden_states)).unsqueeze(2)
        dims=[self.qk_rope_head_dim,self.head_dim-self.qk_rope_head_dim]
        qr,qp=q.split(dims,dim=-1); kr,kp=k.split(dims,dim=-1)
        qr,kr=apply_rotary_pos_emb_interleave(qr,kr,cos,sin,unsqueeze_dim=2)
        q=torch.cat([qr,qp],dim=-1); k=torch.cat([kr,kp],dim=-1).squeeze(2)
        weights=self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float()*self.n_heads**-0.5
        result=[]; chunk=getattr(self.config,"mtp_index_query_chunk",128)
        for start in range(0,s,chunk):
            stop=min(start+chunk,s)
            scores=(q[:,start:stop].float()@k.float().transpose(-1,-2).unsqueeze(1))
            scores=F.relu(scores*self.softmax_scale)
            scores=(weights[:,start:stop].unsqueeze(-2)@scores).squeeze(-2)
            if attention_mask is not None: scores=scores+attention_mask[:,start:stop]
            else:
                causal=torch.arange(s,device=q.device)[None,None,:]>position_ids[:,start:stop,None]
                scores=scores.masked_fill(causal,float("-inf"))
            result.append(scores.topk(min(self.index_topk,s),dim=-1).indices.to(torch.int32))
        return torch.cat(result,dim=1)

class GlmMTPMoE(nn.Module):
    def __init__(self, config):
        super().__init__(); self.gate=GlmMoeDsaTopkRouter(config)
        self.experts=nn.ModuleList([GlmMoeDsaMLP(config,config.moe_intermediate_size) for _ in range(config.num_local_experts)])
        self.shared_experts=GlmMoeDsaMLP(config,config.moe_intermediate_size*config.n_shared_experts)
        self.checkpoint_experts=False
    def forward(self, hidden_states):
        shape=hidden_states.shape; flat=hidden_states.reshape(-1,shape[-1]); _,weights,indices=self.gate(flat)
        result=torch.zeros_like(flat)
        for expert_id,expert in enumerate(self.experts):
            token,slot=torch.where(indices==expert_id); current=flat[token]
            if self.checkpoint_experts and self.training:
                from torch.utils.checkpoint import checkpoint
                output=checkpoint(expert,current,use_reentrant=False)
            else: output=expert(current)
            output=output*weights[token,slot,None].to(output.dtype); result=result.index_add(0,token,output.to(result.dtype))
        return result.view(shape)+self.shared_experts(hidden_states)

class GlmMTPLayer(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__(); config=copy.deepcopy(config); config.indexer_types=["full"]; config.mlp_layer_types=["sparse"]
        self.self_attn=GlmMoeDsaAttention(config,0); self.self_attn.indexer=GlmMTPIndexer(config,0); self.self_attn.indexer.requires_grad_(False)
        self.self_attn.q_a_layernorm=GlmMTPRMSNorm(config.q_lora_rank,config.rms_norm_eps); self.self_attn.kv_a_layernorm=GlmMTPRMSNorm(config.kv_lora_rank,config.rms_norm_eps)
        self.mlp=GlmMTPMoE(config)
        for name in ("input_layernorm","post_attention_layernorm","hidden_layernorm","token_layernorm","final_layernorm"): setattr(self,name,GlmMTPRMSNorm(config.hidden_size,config.rms_norm_eps))
        self.input_proj=nn.Linear(2*config.hidden_size,config.hidden_size,bias=False)
    def forward(self, hidden_states, token_embeddings, attention_mask=None, position_ids=None, position_embeddings=None, **kwargs):
        if position_ids is None: raise ValueError("Native GLM MTP requires explicit position_ids")
        token_embeddings=torch.where(position_ids[...,None]==0,0,token_embeddings)
        hidden=self.input_proj(torch.cat([self.token_layernorm(token_embeddings),self.hidden_layernorm(hidden_states)],dim=-1))
        attention,_,_=self.self_attn(self.input_layernorm(hidden),position_embeddings=position_embeddings,attention_mask=attention_mask,position_ids=position_ids)
        hidden=hidden+attention; hidden=hidden+self.mlp(self.post_attention_layernorm(hidden)); return self.final_layernorm(hidden)

def native_key_to_training(key, layer_index=78):
    prefix=f"model.layers.{layer_index}."; 
    if not key.startswith(prefix): raise ValueError(f"Not a native MTP tensor: {key}")
    key=key[len(prefix):]
    if key=="embed_tokens.weight": return "embed_tokens.weight"
    if key=="shared_head.head.weight": return "lm_head.weight"
    for source,target in {"enorm.":"token_layernorm.","hnorm.":"hidden_layernorm.","eh_proj.":"input_proj.","shared_head.norm.":"final_layernorm."}.items():
        if key.startswith(source): key=target+key[len(source):]; break
    return "mtp_layers.0."+key

def load_native_shared_weights(model):
    import json
    from pathlib import Path
    from safetensors import safe_open
    path=Path(model.config.mtp_shared_weights_path); index=json.loads((path/"model.safetensors.index.json").read_text())["weight_map"]
    layer_index=model.config.transformer_layer_config.num_hidden_layers
    for source,target in ((f"model.layers.{layer_index}.embed_tokens.weight",model.embed_tokens),(f"model.layers.{layer_index}.shared_head.head.weight",model.lm_head)):
        with safe_open(str(path/index[source]),framework="pt") as f: weight=f.get_tensor(source)
        if tuple(weight.shape)!=tuple(target.weight.shape): raise ValueError(f"Shared weight shape mismatch: {source}")
        target.weight=nn.Parameter(weight.to(device=target.weight.device if not target.weight.is_meta else "cpu"),requires_grad=False)
    if hasattr(model,"verifier_lm_head"): del model.verifier_lm_head
    model.mtp_layers[0].self_attn.indexer.requires_grad_(False)

def forward_glm_single_step(model,input_ids,hidden_states,attention_mask=None,position_ids=None,loss_mask=None,step_weights=None,document_ids=None,**kwargs):
    if model.config.num_speculative_steps!=1: raise NotImplementedError("GLM multi-step index/KV reuse needs separate parity validation")
    if input_ids.ndim!=2 or input_ids.shape[0]!=1: raise ValueError("Initial GLM MTP implementation requires one unpacked sequence")
    if document_ids is not None:
        real=document_ids>=0; length=int(real.sum())
        if not bool(real[:,:length].all()) or bool(real[:,length:].any()): raise ValueError("Only right padding is supported")
        docs=document_ids[:,:length]
        if length==0 or not bool((docs==docs[...,:1]).all()): raise ValueError("Packed documents are unsupported; use one sample per batch")
        input_ids=input_ids[:,:length]; hidden_states=hidden_states[:,:length]; document_ids=docs
        if position_ids is not None: position_ids=position_ids[:,:length]
        if loss_mask is not None: loss_mask=loss_mask[:,:length]
        if attention_mask is not None: attention_mask=attention_mask[:,:length]
    seq_len=input_ids.shape[1]
    if seq_len<3: raise ValueError("MTP needs at least three tokens")
    if position_ids is None: position_ids=torch.arange(seq_len,device=input_ids.device)[None]
    if position_ids.shape!=input_ids.shape or not bool((position_ids[:,1:]-position_ids[:,:-1]==1).all()): raise ValueError("Packed/reset positions are unsupported; disable packing")
    if document_ids is not None and not bool((document_ids==document_ids[...,:1]).all()): raise ValueError("Packed documents are unsupported; disable packing")
    if attention_mask is not None and not bool((attention_mask==1).all()): raise ValueError("Initial GLM MTP path requires an unpadded sequence")
    length=seq_len-2; hidden=hidden_states[:,:length]; pos=position_ids[:,:length]
    causal=torch.ones((length,length),dtype=torch.bool,device=hidden.device).triu(1); mask=torch.zeros((1,1,length,length),dtype=hidden.dtype,device=hidden.device); mask.masked_fill_(causal,torch.finfo(hidden.dtype).min)
    token_embeddings=model.embed_tokens(input_ids[:,1:-1].long()); position_embeddings=model.rotary_emb(hidden,pos)
    if model.training and getattr(model.config,"mtp_activation_checkpoint",True):
        from torch.utils.checkpoint import checkpoint
        def layer_forward(h,tok,attn_mask,positions,rotary): return model.mtp_layers[0](h,tok,attention_mask=attn_mask,position_ids=positions,position_embeddings=rotary)
        output=checkpoint(layer_forward,hidden,token_embeddings,mask,pos,position_embeddings,use_reentrant=False)
    else: output=model.mtp_layers[0](hidden,token_embeddings,attention_mask=mask,position_ids=pos,position_embeddings=position_embeddings)
    targets=input_ids[:,2:].long(); valid=torch.ones_like(targets,dtype=torch.bool) if loss_mask is None else loss_mask[:,2:].bool(); count=valid.sum()
    if not bool(count): raise ValueError("MTP sample has no supervised shifted targets")
    from torch.utils.checkpoint import checkpoint
    total=output.new_zeros((),dtype=torch.float32); correct=output.new_zeros((),dtype=torch.float32)
    for start in range(0,length,64):
        end=min(start+64,length); labels=targets[:,start:end].masked_fill(~valid[:,start:end],-100)
        def chunk_loss(h,labels):
            logits=model.lm_head(h).float(); loss=F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100,reduction="sum"); acc=((logits.argmax(-1)==labels)&(labels!=-100)).sum().float(); return loss,acc
        value,acc=checkpoint(chunk_loss,output[:,start:end],labels,use_reentrant=False); total=total+value; correct=correct+acc.detach()
    weight=1.0 if step_weights is None else step_weights[0]; loss=total/count*weight
    return [],loss,{"loss_sum":loss.detach(),"loss_total":loss.new_tensor(1),"loss_step_0":loss.detach(),"token_accuracy_step_0":correct/count,"supervised_tokens":count.detach()}
