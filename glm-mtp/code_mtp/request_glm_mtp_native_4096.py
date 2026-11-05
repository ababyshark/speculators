import json
import time
from pathlib import Path

import httpx
from safetensors.torch import load_file

root=Path("/data1/mtp_training/parity")
while True:
    try:
        if httpx.get("http://127.0.0.1:8001/health",timeout=3).status_code==200: break
    except httpx.HTTPError: pass
    time.sleep(5)
ids=load_file(str(root/"teacher_long8k.safetensors"))["token_ids"].tolist()[:4096]
arm=root/"native_capture_arm.json"; tmp=arm.with_suffix(".tmp")
tmp.write_text(json.dumps({"name":"chunk4096","length":4096})); tmp.replace(arm)
res=httpx.post("http://127.0.0.1:8001/v1/completions",json={"model":"/data1/weight/GLM-5.3-w4a8-spec","prompt":ids,"temperature":0,"max_tokens":2,"ignore_eos":True,"return_token_ids":True},timeout=600)
res.raise_for_status(); arm.unlink(missing_ok=True)
assert (root/"native_chunk4096.safetensors").is_file(), res.text[:500]
print("CAPTURED_CHUNK4096",flush=True)
