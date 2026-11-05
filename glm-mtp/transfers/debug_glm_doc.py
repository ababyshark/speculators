from pathlib import Path
p=Path('/workspace/speculators/src/speculators/models/mtp/glm_mtp.py')
s=p.read_text()
old='''        if length == 0 or not bool((docs == docs[..., :1]).all()):
            raise ValueError("Packed documents are unsupported; use one sample per batch")
'''
new='''        if length == 0 or not bool((docs == docs[..., :1]).all()):
            print("GLM_DOC_DEBUG", tuple(document_ids.shape), int(length), torch.unique(docs).tolist(), flush=True)
            raise ValueError("Packed documents are unsupported; use one sample per batch")
'''
assert old in s
p.write_text(s.replace(old,new))
