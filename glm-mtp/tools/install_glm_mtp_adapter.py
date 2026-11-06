"""Apply scoped, guarded integration edits to the inherited Speculators tree."""

from pathlib import Path

root = Path("/workspace/speculators/src/speculators")


def replace(path, old, new):
    text = path.read_text()
    if new in text:
        return
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one edit location: {path}")
    path.write_text(text.replace(old, new))


replace(
    root / "models/mtp/config.py",
    "    num_nextn_predict_layers: int = Field(",
    "    mtp_shared_weights_path: str | None = Field(\n"
    '        default=None, description="Native MTP checkpoint containing independent frozen vocab weights"\n'
    "    )\n\n    num_nextn_predict_layers: int = Field(",
)
replace(
    root / "models/mtp/model_definitions.py",
    "mtp_model_classes: dict[str, base_components.ModelComponents] = {",
    "from speculators.models.mtp.glm_mtp import (\n"
    "    GlmMTPLayer, GlmMTPRMSNorm, GlmMTPRotaryEmbedding,\n)\n\n"
    "mtp_model_classes: dict[str, base_components.ModelComponents] = {\n"
    '    "glm_moe_dsa": base_components.ModelComponents(\n'
    "        GlmMTPLayer, GlmMTPLayer, GlmMTPRMSNorm, GlmMTPRotaryEmbedding\n"
    "    ),",
)
replace(
    root / "models/mtp/core.py",
    "        with torch.no_grad():\n            self.embed_tokens.weight.fill_(torch.nan)",
    '        if self.config.transformer_layer_config.model_type == "glm_moe_dsa":\n'
    "            if not self.config.mtp_shared_weights_path:\n"
    '                raise ValueError("GLM requires its independent MTP shared weights")\n'
    "            from speculators.models.mtp.glm_mtp import load_native_shared_weights\n"
    "            load_native_shared_weights(self)\n"
    "            return\n"
    "        with torch.no_grad():\n            self.embed_tokens.weight.fill_(torch.nan)",
)
replace(
    root / "models/mtp/core.py",
    "        input_ids = input_ids.long()",
    '        if self.config.transformer_layer_config.model_type == "glm_moe_dsa":\n'
    "            from speculators.models.mtp.glm_mtp import forward_glm_single_step\n"
    "            return forward_glm_single_step(\n"
    "                self, input_ids, hidden_states, attention_mask=attention_mask,\n"
    "                position_ids=position_ids, loss_mask=loss_mask,\n"
    "                step_weights=step_weights, **kwargs,\n"
    "            )\n"
    "        input_ids = input_ids.long()",
)
replace(
    root / "models/mtp/data.py",
    "    return {\n",
    '    return {\n        **({"document_ids": batch["document_ids"]} if "document_ids" in batch else {}),\n',
)
print("GLM MTP scoped integration edits applied")
