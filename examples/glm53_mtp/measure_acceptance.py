"""Measure speculative-decoding acceptance on a served endpoint.

Runs eval prompts (temperature 0), then scrapes vLLM /metrics spec-decode
counters and reports per-position acceptance + mean accepted length.
"""

import argparse
import asyncio
import json
import re
import urllib.request

import aiohttp


async def run_prompt(session, sem, url, prompt, model, stats):
    async with sem:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0,
        }
        for _ in range(3):
            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=600)
                ) as r:
                    await r.json()
                stats["ok"] += 1
                return
            except Exception:
                stats.setdefault("err", 0)
                stats["err"] += 1
                await asyncio.sleep(3)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prompts",
        default="/mnt/share_space/l00840580/dspark-glm53/data/eval_prompts.jsonl",
    )
    ap.add_argument("--url", default="http://127.0.0.1:8100/v1/chat/completions")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8100/metrics")
    ap.add_argument("--model", default="/mnt/weight2/l00840580-glm53-w4a8-biasfix")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    prompts = [json.loads(l)["prompt"] for l in open(args.prompts)]
    print(f"[{args.label}] running {len(prompts)} prompts")
    stats = {"ok": 0}
    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession() as s:
        await asyncio.gather(
            *[run_prompt(s, sem, args.url, p, args.model, stats) for p in prompts]
        )
    print(f"[{args.label}] done: {stats}")

    text = urllib.request.urlopen(args.metrics_url).read().decode()
    per_pos = {}
    total_draft = total_accepted = 0.0
    for m in re.finditer(
        r'^vllm:spec_decode_num_accepted_tokens_per_pos\{.*?position="(\d+)".*?\} (\S+)',
        text,
        re.M,
    ):
        per_pos[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(
        r"^vllm:spec_decode_num_draft_tokens\{.*?\} (\S+)", text, re.M
    ):
        total_draft += float(m.group(1))
    for m in re.finditer(
        r"^vllm:spec_decode_num_accepted_tokens\{.*?\} (\S+)", text, re.M
    ):
        total_accepted += float(m.group(1))

    print(f"[{args.label}] per-position accepted tokens:")
    keys = sorted(per_pos)
    prev = None
    acc_rates = []
    for i, k in enumerate(keys):
        denom = per_pos[keys[i - 1]] if i > 0 else total_draft
        rate = per_pos[k] / denom if denom else 0.0
        acc_rates.append(rate)
        print(f"  pos{i}: {rate:.4f}  (accepted {per_pos[k]:.0f})")
    if acc_rates:
        mean_len = 1.0 + sum(acc_rates)
        print(f"[{args.label}] mean accepted length ~= {mean_len:.3f}")
        print(
            f"[{args.label}] overall accepted/draft = {total_accepted / total_draft if total_draft else 0:.4f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
