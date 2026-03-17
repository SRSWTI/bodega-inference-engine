"""
Direct manual test: fire C concurrent requests at raw-test-4b,
measure real tokens generated per request and system throughput.
No model reload between concurrency levels.
"""
import asyncio, time, json
import httpx

URL      = "http://localhost:44468"
MODEL_ID = "raw-test-4b"
MAX_TOK  = 128
PROMPTS  = [
    "What is the capital of France?",
    "Explain the theory of relativity briefly.",
    "Write a haiku about the ocean.",
    "What are the main causes of World War I?",
    "Describe how photosynthesis works.",
    "What is machine learning?",
    "Write a short poem about autumn.",
    "Explain Newton's laws of motion.",
    "What is the significance of the Magna Carta?",
    "How does the internet work?",
]


async def fire_one(client: httpx.AsyncClient, prompt: str, idx: int):
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOK,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with client.stream("POST", f"{URL}/v1/chat/completions",
                             json=payload, timeout=120.0) as resp:
        buf = ""
        async for chunk in resp.aiter_text():
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                s = line[6:]
                if s == "[DONE]":
                    break
                try:
                    d = json.loads(s)
                except Exception:
                    continue
                delta = d.get("choices", [{}])[0].get("delta", {})
                txt = delta.get("content") or delta.get("reasoning_content")
                if txt:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tokens += 1
    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    return idx, tokens, ttft, total


async def run_concurrency(c: int):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(c)]
    async with httpx.AsyncClient() as client:
        t_wall = time.perf_counter()
        results = await asyncio.gather(*[fire_one(client, p, i) for i, p in enumerate(prompts)])
        wall = time.perf_counter() - t_wall

    total_tokens = sum(r[1] for r in results)
    ttfts = [r[2] * 1000 for r in results]
    token_counts = [r[1] for r in results]
    sys_tps = total_tokens / wall if wall > 0 else 0

    ok = sum(1 for t in token_counts if t >= 10)
    print(f"\n  C={c:2d} │ wall={wall:.2f}s │ sys_tps={sys_tps:.0f} tok/s │ "
          f"total_tokens={total_tokens} │ ok_reqs(≥10tok)={ok}/{c}")
    print(f"         tokens/req: {token_counts}")
    ttft_sorted = sorted(ttfts)
    print(f"         TTFTs(ms):  min={ttft_sorted[0]:.0f} "
          f"p50={ttft_sorted[len(ttft_sorted)//2]:.0f} "
          f"max={ttft_sorted[-1]:.0f}")
    return ok == c  # True if all requests generated proper tokens


async def main():
    print(f"\n{'='*60}")
    print(f"  Raw test: {MODEL_ID}  max_tokens={MAX_TOK}")
    print(f"{'='*60}")

    # Warmup
    print("\n  [Warmup] 1 request...")
    async with httpx.AsyncClient() as client:
        _, tok, ttft, total = await fire_one(client, "Hello!", 0)
        print(f"  warmup: {tok} tokens, ttft={ttft*1000:.0f}ms, total={total:.2f}s")

    results = {}
    for c in [4, 8, 16, 32]:
        ok = await run_concurrency(c)
        results[c] = ok

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for c, ok in results.items():
        status = "✅ OK" if ok else "❌ BROKEN"
        print(f"  C={c:2d}: {status}")


asyncio.run(main())
