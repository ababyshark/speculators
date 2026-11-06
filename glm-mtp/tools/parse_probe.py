import json
import re
import sys

payload = json.load(open(sys.argv[1]))
metrics = payload["metrics"]
for name in [
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
]:
    match = re.search(rf"{re.escape(name)}[^\\n]*\\s([0-9.]+)", metrics)
    print(name, match.group(1) if match else "MISSING")
for record in payload["records"]:
    print("elapsed", f"{record['elapsed_s']:.3f}", "status", record["status"])
