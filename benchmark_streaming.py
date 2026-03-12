#!/usr/bin/env python3
"""
Streaming Benchmark for bodega_mlx_engine (HTTP API)
====================================================

Benchmarks streaming sequence performance and calculates Inter-Token Latency (ITL).
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import List

import httpx

PROMPTS = [
    "Write a detailed 5 paragraph essay on the history of Rome.",
    "Explain the concepts of quantum entanglement.",
    "Write a short python script to manage dependencies."
]

@dataclass
class StreamingResult:
    prompt: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    itl_ms: float = 0.0
    end_to_end_s: float = 0.0
    tps: float = 0.0
    error: str = None

async def run_streaming_benchmark(client: httpx.AsyncClient, base_url: str, model_id: str, prompt: str, max_tokens: int) -> StreamingResult:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    t0 = time.perf_counter()
    ttft = None
    last_token_time = None
    inter_token_times = []
    
    prompt_tokens = 0
    completion_tokens = 0

    try:
        async with client.stream("POST", url, json=payload, timeout=120.0) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                return StreamingResult(prompt=prompt, error=f"HTTP {resp.status_code}")

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except:
                    continue

                if "usage" in data and data["usage"]:
                    if data["usage"].get("prompt_tokens"):
                        prompt_tokens = data["usage"]["prompt_tokens"]
                    if data["usage"].get("completion_tokens"):
                        completion_tokens = data["usage"]["completion_tokens"]
                    
                choices = data.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    completion_tokens += 1
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    elif last_token_time is not None:
                        inter_token_times.append(now - last_token_time)
                    last_token_time = now

        total_time = time.perf_counter() - t0
        itl_ms = (statistics.mean(inter_token_times) * 1000) if inter_token_times else 0
        tps = completion_tokens / (total_time - ttft) if ttft and float(total_time - ttft) > 0 else 0
        
        return StreamingResult(
            prompt=prompt[:30]+"...",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=(ttft or 0) * 1000,
            itl_ms=itl_ms,
            end_to_end_s=total_time,
            tps=tps
        )
    except Exception as exc:
        return StreamingResult(prompt=prompt[:30]+"...", error=str(exc))

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:44468", help="Server base URL")
    parser.add_argument("--model-path", default="srswti/bodega-raptor-90m", help="Model path")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens")
    args = parser.parse_args()

    model_id = "streaming-bench"
    print(f"Starting Streaming Benchmark on {args.base_url}")
    
    async with httpx.AsyncClient() as client:
        print("Loading model via HTTP API...", end=" ", flush=True)
        from benchmark_continuous_batching import manage_model
        success = await manage_model(client, args.base_url, "load", args.model_path, model_id, continuous_batching=True)
        if not success:
            print("FAILED")
            return
        print("done.")

        print("\nRunning prompts...")
        results = []
        for p in PROMPTS:
            print(f"  -> {p[:40]}...")
            res = await run_streaming_benchmark(client, args.base_url, model_id, p, args.max_tokens)
            results.append(res)
            
        print("\nUnloading model...", end=" ", flush=True)
        await manage_model(client, args.base_url, "unload", args.model_path, model_id)
        print("done.")
        
    print("\nRESULTS:")
    print(f"{'Prompt':<35} | {'Tokens':<8} | {'TTFT(ms)':<10} | {'ITL(ms)':<8} | {'TPS'}")
    print("-" * 75)
    for r in results:
        if r.error:
            print(f"{r.prompt:<35} | ERROR: {r.error}")
        else:
            print(f"{r.prompt:<35} | {r.completion_tokens:<8} | {r.ttft_ms:<10.0f} | {r.itl_ms:<8.1f} | {r.tps:.1f}")

if __name__ == "__main__":
    asyncio.run(main())
