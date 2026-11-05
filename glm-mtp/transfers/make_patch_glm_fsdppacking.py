from pathlib import Path
p=Path('remote_scripts/patch_glm_fsdppacking.py')
p.write_text(r'''from pathlib import Path
p=Path('/workspace/speculators/src/speculators/train/distributed.py')
s=p.read_text()
old='''    for layer in model.layers:  # type: ignore[union-attr]
        fully_shard(layer, mp_policy=mp_policy)

    fully_shard(model, mp_policy=mp_policy)
'''
new='''    # GLM native MTP has one very large sparse-MoE layer. Sharding only the
    # whole layer makes every rank all-gather the full ~10B-parameter block
    # before the forward pass. Shard its attention, router, shared expert and
    # each routed expert first; the enclosing layer then owns only its small
    # direct parameters while nested FSDP modules gather one component at a time.
    model_type = getattr(getattr(model.config, "transformer_layer_config", None), "model_type", None)
    for layer in model.layers:  # type: ignore[union-attr]
        if model_type == "glm_moe_dsa":
            children = [layer.self_attn, layer.mlp.gate, layer.mlp.shared_experts, *layer.mlp.experts]
            for child in children:
                fully_shard(child, mp_policy=mp_policy)
        fully_shard(layer, mp_policy=mp_policy)

    fully_shard(model, mp_policy=mp_policy)
'''
if old not in s: raise SystemExit('target not found')
p.write_text(s.replace(old,new))
''', encoding='utf-8')
