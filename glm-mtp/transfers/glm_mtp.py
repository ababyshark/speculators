from pathlib import Path

p = Path("remote_scripts/glm_mtp.py")
s = p.read_text()
old = """    output = model.mtp_layers[0](
        hidden, model.embed_tokens(input_ids[:, 1:-1].long()), attention_mask=mask,
        position_ids=pos, position_embeddings=model.rotary_emb(hidden, pos),
    )
"""
new = """    token_embeddings = model.embed_tokens(input_ids[:, 1:-1].long())
    position_embeddings = model.rotary_emb(hidden, pos)
    # Attention and sparse-MoE activations dominate the NPU peak. Recompute the
    # complete MTP layer during backward when training, while retaining the
    # forward output needed by the chunked vocabulary loss.
    if model.training and getattr(model.config, "mtp_activation_checkpoint", True):
        from torch.utils.checkpoint import checkpoint
        def layer_forward(h, tok, attn_mask, positions, rotary):
            return model.mtp_layers[0](
                h, tok, attention_mask=attn_mask,
                position_ids=positions, position_embeddings=rotary,
            )
        output = checkpoint(
            layer_forward, hidden, token_embeddings, mask, pos, position_embeddings,
            use_reentrant=False,
        )
    else:
        output = model.mtp_layers[0](
            hidden, token_embeddings, attention_mask=mask,
            position_ids=pos, position_embeddings=position_embeddings,
        )
"""
assert old in s
p.write_text(s.replace(old, new))
