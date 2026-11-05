"""Generate on-policy responses for UltraChat prompts via the live vLLM server.

Writes conversations jsonl in speculators prepare-data format:
{"conversations": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
"""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
import pyarrow.parquet as pq

MODEL = "/mnt/weight2/l00840580-glm53-w4a8-biasfix"


async def gen_one(session, sem, url, prompt, out_f, lock, idx, stats):
    async with sem:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.8,
        }
        for attempt in range(3):
            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=600)
                ) as r:
                    data = await r.json()
                content = data["choices"][0]["message"]["content"]
                if data["choices"][0].get("finish_reason") == "length":
                    content = content  # keep truncated
                row = {
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": content},
                    ]
                }
                async with lock:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                stats["ok"] += 1
                if stats["ok"] % 200 == 0:
                    print(f"done {stats['ok']}", flush=True)
                return
            except Exception as e:
                if attempt == 2:
                    stats["fail"] += 1
                    print(f"[{idx}] failed: {type(e).__name__} {e}", flush=True)
                else:
                    await asyncio.sleep(5)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num", type=int, default=4000)
    ap.add_argument("--url", default="http://127.0.0.1:8100/v1/chat/completions")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument(
        "--resume", action="store_true", help="skip prompts already present in --output"
    )
    args = ap.parse_args()

    seen = set()
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen.add(r["conversations"][0]["content"])
                except Exception:
                    pass
        print(f"resume: {len(seen)} prompts already done")

    table = pq.read_table(args.parquet, columns=["prompt", "messages"])
    prompts = []
    for p, msgs in zip(
        table.column("prompt").to_pylist(), table.column("messages").to_pylist()
    ):
        if p and len(p) < 4096 and p not in seen:
            seen.add(p)
            prompts.append(p)
        if len(prompts) >= args.num:
            break
    print(f"prompts to generate: {len(prompts)}")

    stats = {"ok": 0, "fail": 0}
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    out_path = Path(args.output)
    conn = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(connector=conn) as session:
        with open(out_path, "a") as out_f:
            await asyncio.gather(
                *[
                    gen_one(session, sem, args.url, p, out_f, lock, i, stats)
                    for i, p in enumerate(prompts)
                ]
            )
    print("stats:", stats)


if __name__ == "__main__":
    asyncio.run(main())
