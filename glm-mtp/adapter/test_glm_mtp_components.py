"""CPU component checks; these do NOT replace native vLLM parity."""
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaIndexer,
    GlmMoeDsaMoE,
)

spec = importlib.util.spec_from_file_location("glm_adapter", "/workspace/speculators/src/speculators/models/mtp/glm_mtp.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.manual_seed(13)
torch.set_num_threads(2)
c = json.loads(Path("/data1/weight/GLM-5.3-w4a8-spec/config.json").read_text())
c.update(hidden_size=16, intermediate_size=24, moe_intermediate_size=12,
         num_attention_heads=2, num_key_value_heads=2, n_routed_experts=4,
         num_experts_per_tok=2, q_lora_rank=8, kv_lora_rank=4,
         qk_rope_head_dim=4, qk_nope_head_dim=4, qk_head_dim=8, head_dim=4,
         v_head_dim=4, index_n_heads=2, index_head_dim=8, index_topk=3,
         vocab_size=32, pad_token_id=0, mtp_index_query_chunk=2)
c.pop("model_type")
cfg = AutoConfig.for_model("glm_moe_dsa", **c)
cfg._attn_implementation = "sdpa"
layer = module.GlmMTPLayer(cfg)
rotary = module.GlmMTPRotaryEmbedding(cfg)
x = torch.randn(1, 7, 16)
pos = torch.arange(7)[None]
pe = rotary(x, pos)
mask = torch.zeros(1, 7, 7).masked_fill(torch.ones(7, 7, dtype=torch.bool).triu(1), float("-inf"))
ref_index = GlmMoeDsaIndexer(cfg, 0)
ref_index.load_state_dict(layer.self_attn.indexer.state_dict())
q = torch.randn(1, 7, 8)
expected = ref_index(x, q, pe, mask, pos)
actual = layer.self_attn.indexer(x, q, pe, mask, pos)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
reference = GlmMoeDsaMoE(cfg)
reference.gate.load_state_dict(layer.mlp.gate.state_dict())
reference.shared_experts.load_state_dict(layer.mlp.shared_experts.state_dict())
with torch.no_grad():
    for i, expert in enumerate(layer.mlp.experts):
        reference.experts.gate_up_proj[i].copy_(torch.cat([expert.gate_proj.weight, expert.up_proj.weight]))
        reference.experts.down_proj[i].copy_(expert.down_proj.weight)
    layer.mlp.gate.weight.normal_(0, .1)
    reference.gate.weight.copy_(layer.mlp.gate.weight)
x1 = x.clone().requires_grad_()
x2 = x.clone().requires_grad_()
y1, y2 = layer.mlp(x1), reference(x2)
torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-6)
y1.square().sum().backward()
y2.square().sum().backward()
torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-5, atol=1e-6)
torch.testing.assert_close(layer.mlp.gate.weight.grad, reference.gate.weight.grad, rtol=1e-5, atol=1e-6)
embeds = torch.randn_like(x)
output = layer(x, embeds, mask[:, None], pos, pe)
changed = embeds.clone()
changed[:, 0] = 1000
torch.testing.assert_close(output, layer(x, changed, mask[:, None], pos, pe), atol=0, rtol=0)
assert torch.isfinite(output).all()
output.square().mean().backward()
assert all(p.grad is None for p in layer.self_attn.indexer.parameters())
assert layer.input_proj.weight.grad is not None
assert layer.self_attn.q_a_layernorm.variance_epsilon == 1e-5
assert rotary.inv_freq.numel() == cfg.qk_rope_head_dim // 2
from types import SimpleNamespace

dummy = SimpleNamespace(
    config=SimpleNamespace(num_speculative_steps=1),
    mtp_layers=[layer], rotary_emb=rotary,
    embed_tokens=torch.nn.Embedding(32, 16), lm_head=torch.nn.Linear(16, 32, bias=False),
)
dummy.embed_tokens.requires_grad_(False)
dummy.lm_head.requires_grad_(False)
ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
mask_loss = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
_, loss, _ = module.forward_glm_single_step(dummy, ids, x, loss_mask=mask_loss)
pad_ids = torch.cat([ids, torch.zeros(1, 3, dtype=torch.long)], dim=1)
pad_hidden = torch.cat([x, torch.zeros(1, 3, 16)], dim=1)
docs = torch.tensor([[0, 0, 0, 0, 0, 0, 0, -1, -1, -1]])
pad_mask = torch.cat([mask_loss, torch.zeros(1, 3, dtype=torch.long)], dim=1)
_, pad_loss, _ = module.forward_glm_single_step(dummy, pad_ids, pad_hidden,
                                               loss_mask=pad_mask, document_ids=docs)
torch.testing.assert_close(loss, pad_loss, atol=0, rtol=0)
try:
    module.forward_glm_single_step(dummy, ids, x, document_ids=torch.tensor([[0, 0, 0, 1, 1, 1, 1]]))
except ValueError as error:
    assert "Packed" in str(error)
else:
    raise AssertionError("Packed sequence was not rejected")
loss.backward()
assert dummy.embed_tokens.weight.grad is None and dummy.lm_head.weight.grad is None
report = dict(status="passed", indexer_chunk_exact=True, expert_forward_backward=True,
              first_position_mask=True, indexer_frozen=True, eps_and_rope=True,
              padding_loss_exact=True, packed_boundary_rejected=True,
              native_vllm_parity="NOT_YET_VALIDATED")
Path("/workspace/tools/logs/component_tests.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
