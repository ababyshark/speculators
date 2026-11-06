from pathlib import Path

p = Path("/workspace/speculators/src/speculators/models/mtp/glm_mtp.py")
s = p.read_text()
s = s.replace(
    '            print("GLM_DOC_DEBUG", tuple(document_ids.shape), int(length), torch.unique(docs).tolist(), flush=True)\n',
    "",
)
p.write_text(s)
