from pathlib import Path

p = Path("remote_scripts/apply_patch_glm_fsdppacking.py")
p.write_text(
    '''from pathlib import Path
p=Path("/workspace/speculators/src/speculators/train/distributed.py")
s=p.read_text()
old="""    for layer in model.layers:  # type: ignore[union-attr]
        fully_shard(layer, mp_policy=mp_policy)

    fully_shard(model, mp_policy=mp_policy)
"""
new="""    # GLM native MTP has one very large sparse-MoE layer. Shard attention,
    # router, shared expert and each routed expert before the enclosing layer.
    model_type = getattr(getattr(model.config, \"transformer_layer_config\", None), \"model_type\", None)
    for layer in model.layers:  # type: ignore[union-attr]
        if model_type == \"glm_moe_dsa\":
            children = [layer.self_attn, layer.mlp.gate, layer.mlp.shared_experts, *layer.mlp.experts]
            for child in children:
                fully_shard(child, mp_policy=mp_policy)
        fully_shard(layer, mp_policy=mp_policy)

    fully_shard(model, mp_policy=mp_policy)
"""
if old not in s: raise SystemExit("target not found")
p.write_text(s.replace(old,new))
''',
    encoding="utf-8",
)
