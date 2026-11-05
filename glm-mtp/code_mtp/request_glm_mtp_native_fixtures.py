"""Trigger two native prefill MTP captures after the local parity service is ready."""
import json
import time
from pathlib import Path
import httpx
from safetensors.torch import load_file

root = Path('/data1/mtp_training/parity')
started = time.monotonic()
while time.monotonic() - started < 600:
    try:
        if httpx.get('http://127.0.0.1:8001/health', timeout=3).status_code == 200:
            break
    except httpx.HTTPError:
        pass
    time.sleep(5)
else:
    raise TimeoutError('Native parity service did not become healthy')

results = {}
for name, source, length in [('short64', 'short_a', 64), ('long8190', 'long8k', 8190)]:
    ids = load_file(str(root / f'teacher_{source}.safetensors'))['token_ids'].tolist()[:length]
    arm = root / 'native_capture_arm.json'
    temporary = arm.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(name=name, length=length)))
    temporary.replace(arm)
    before = time.monotonic()
    try:
        response = httpx.post('http://127.0.0.1:8001/v1/completions', json=dict(
            model='/data1/weight/GLM-5.3-w4a8-spec', prompt=ids,
            temperature=0, max_tokens=2, ignore_eos=True, return_token_ids=True,
        ), timeout=600)
        response.raise_for_status()
        assert (root / f'native_{name}.safetensors').is_file(), response.text[:1000]
        results[name] = dict(seconds=time.monotonic()-before, response=response.json())
        print(name, 'CAPTURED', results[name]['seconds'], flush=True)
    finally:
        arm.unlink(missing_ok=True)
(root / 'native_capture_requests.json').write_text(json.dumps(results, indent=2))
