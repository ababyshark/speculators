import json
import time
import requests

base = 'http://127.0.0.1:8010'
prompts = [
    '请解释二分查找算法，并给出 Python 实现。',
    'Write a Python function that checks whether a string is a palindrome.',
    '计算 12345 * 678，并说明计算过程。',
]
records = []
for prompt in prompts:
    started = time.perf_counter()
    response = requests.post(
        base + '/v1/chat/completions',
        json={
            'model': 'glm53-mtp-final100',
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0,
            'top_p': 1.0,
            'max_tokens': 64,
            'stream': False,
        },
        timeout=600,
    )
    elapsed = time.perf_counter() - started
    try:
        body = response.json()
    except Exception:
        body = response.text
    records.append({'prompt': prompt, 'status': response.status_code, 'elapsed_s': elapsed, 'body': body})
print(json.dumps(records, ensure_ascii=False, indent=2))
print('---METRICS---')
metrics_text = requests.get(base + '/metrics', timeout=30).text
print(metrics_text)
Path = __import__('pathlib').Path
Path('/workspace/code_mtp/logs/final_acceptance_probe.json').write_text(
    json.dumps({'records': records, 'metrics': metrics_text}, ensure_ascii=False, indent=2)
)

