#!/usr/bin/env python3
"""
HTTP Concurrency Benchmark for bodega_mlx_engine
================================================

Benchmarks the running server (localhost:44468) with different concurrency levels
to measure continuous batching benefits. Tests max_concurrency 8, 16, 32 with
the same total number of queries for fair comparison.

Usage:
    # Benchmark 90m model at concurrency 8, 16, 32 (default)
    python scripts/benchmark_http_concurrency.py --base-url http://localhost:44468

    # Custom model and query count
    python scripts/benchmark_http_concurrency.py \
        --model bodega-raptor-90m \
        --num-queries 32 \
        --max-tokens 128

    # Compare with sequential (requires reloading model without CB first)
    python scripts/benchmark_http_concurrency.py --compare-sequential

Prerequisites:
    - Server running with bodega-raptor-90m loaded (with continuous_batching: true)
    - For clean benchmark: restart with config_benchmark_90m.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# Test prompts (varied lengths for realistic load)
PROMPTS = [
    "Hello, how are you?",
    "What is 2+2?",
    "Say hello in Spanish.",
    "What is the capital of France and why is it historically significant?",
    "Write a Python function to calculate fibonacci numbers with memoization.",
    "Explain the difference between a list and a tuple in Python.",
    (
        "Explain quantum computing in detail, covering: qubits, superposition, "
        "entanglement, potential applications in cryptography and drug discovery."
    ),
    (
        "Write a comprehensive guide to building a production REST API in Python. "
        "Include: project structure, routing, authentication with JWT."
    ),
]


@dataclass
class RequestResult:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    total_time_s: float = 0.0
    tps: float = 0.0
    error: str | None = None


@dataclass
class ConcurrencyResult:
    concurrency: int
    num_requests: int
    total_wall_time_s: float
    results: list[RequestResult] = field(default_factory=list)

    @property
    def successful(self) -> list[RequestResult]:
        return [r for r in self.results if r.error is None]

    @property
    def throughput_tps(self) -> float:
        total_out = sum(r.completion_tokens for r in self.successful)
        return total_out / self.total_wall_time_s if self.total_wall_time_s > 0 else 0

    @property
    def mean_ttft_ms(self) -> float:
        ttfts = [r.ttft_ms for r in self.successful]
        return statistics.mean(ttfts) if ttfts else 0

    @property
    def p95_ttft_ms(self) -> float:
        ttfts = sorted(r.ttft_ms for r in self.successful)
        if not ttfts:
            return 0
        idx = min(int(len(ttfts) * 0.95), len(ttfts) - 1)
        return ttfts[idx]

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.successful)


async def run_one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Run a single non-streaming request and measure TTFT from response time."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }

    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=120.0)
        total_time = time.perf_counter() - t0

        if resp.status_code != 200:
            return RequestResult(
                error=f"HTTP {resp.status_code}: {resp.text[:80]}",
                total_time_s=total_time,
            )

        data = resp.json()
        choices = data.get("choices", [])
        usage = data.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # For non-streaming we don't get per-token TTFT; use total_time as proxy
        # (first token arrives with full response in non-streaming)
        ttft_ms = total_time * 1000  # Approximate for non-streaming
        gen_time = total_time
        tps = (completion_tokens - 1) / gen_time if gen_time > 0 and completion_tokens > 1 else 0

        return RequestResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            total_time_s=total_time,
            tps=tps,
        )
    except Exception as e:
        return RequestResult(
            error=str(e),
            total_time_s=time.perf_counter() - t0,
        )


async def run_streaming_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Run a streaming request to measure actual TTFT."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = 0
    completion_tokens = 0
    last_content = ""

    try:
        async with client.stream("POST", url, json=payload, timeout=120.0) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                return RequestResult(
                    error=f"HTTP {resp.status_code}: {text.decode()[:80]}",
                    total_time_s=time.perf_counter() - t0,
                )

            buffer = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        last_content += content
                        completion_tokens += 1

                    usage = data.get("usage")
                    if usage:
                        if usage.get("prompt_tokens"):
                            prompt_tokens = usage["prompt_tokens"]
                        if usage.get("completion_tokens"):
                            completion_tokens = usage["completion_tokens"]

                    if choices[0].get("finish_reason"):
                        break

        total_time = time.perf_counter() - t0
        if ttft is None:
            ttft = total_time * 1000

        # Approximate completion tokens if not in usage
        if completion_tokens == 0 and last_content:
            completion_tokens = max(1, len(last_content.split()))

        gen_time = total_time - (ttft / 1000) if total_time > ttft / 1000 else total_time
        tps = (completion_tokens - 1) / gen_time if gen_time > 0 and completion_tokens > 1 else 0

        return RequestResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft,
            total_time_s=total_time,
            tps=tps,
        )
    except Exception as e:
        return RequestResult(
            error=str(e),
            total_time_s=time.perf_counter() - t0,
        )


async def manage_model(client: httpx.AsyncClient, base_url: str, action: str, model_path: str, model_id: str) -> bool:
    """Helper to dynamically load/unload the model for benchmarks."""
    full_url = base_url.rstrip("/")
    if action == "load":
        print(f"  [+] Loading model {model_path} into {model_id}...")
        payload = {
            "model_path": model_path,
            "model_id": model_id,
            "model_type": "lm",
            "context_length": 8192,
            "continuous_batching": True,
            "cb_max_num_seqs": 128
        }
        resp = await client.post(f"{full_url}/v1/admin/load-model", json=payload, timeout=120.0)
        if resp.status_code == 409:
            print(f"      [✓] Model already loaded. Continuing.")
            return True
        elif resp.status_code not in [200, 201]:
            print(f"      [!] Load failed: {resp.status_code} {resp.text}")
        return resp.status_code in [200, 201]
    elif action == "unload":
        print(f"  [-] Unloading model {model_id}...")
        resp = await client.delete(f"{full_url}/v1/admin/unload-model/{model_id}", timeout=30.0)
        if resp.status_code not in [200, 204]:
            print(f"      [!] Unload failed: {resp.status_code} {resp.text}")
        return resp.status_code in [200, 204]


async def benchmark_concurrency(
    base_url: str,
    model: str,
    concurrency: int,
    num_queries: int,
    max_tokens: int,
    stream: bool = True,
) -> ConcurrencyResult:
    """Run num_queries with at most `concurrency` in-flight at once."""
    prompts = (PROMPTS * ((num_queries // len(PROMPTS)) + 1))[:num_queries]
    run_fn = run_streaming_request if stream else run_one_request
    sem = asyncio.Semaphore(concurrency)

    async def run_with_sem(prompt: str) -> RequestResult:
        async with sem:
            async with httpx.AsyncClient() as client:
                return await run_fn(client, base_url, model, prompt, max_tokens)

    t_start = time.perf_counter()
    tasks = [run_with_sem(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - t_start

    return ConcurrencyResult(
        concurrency=concurrency,
        num_requests=len(prompts),
        total_wall_time_s=wall_time,
        results=list(results),
    )


def print_result(r: ConcurrencyResult) -> None:
    ok = r.successful
    failed = len(r.results) - len(ok)
    print(f"\n  Concurrency {r.concurrency} ({len(ok)}/{r.num_requests} succeeded, {failed} failed)")
    print(f"    Wall time:      {r.total_wall_time_s:.2f}s")
    print(f"    Throughput:     {r.throughput_tps:.1f} tok/s (system)")
    print(f"    Mean TTFT:      {r.mean_ttft_ms:.0f}ms")
    print(f"    P95 TTFT:       {r.p95_ttft_ms:.0f}ms")
    print(f"    Total tokens:   {r.total_completion_tokens:,}")
    if failed > 0:
        first_error = next(res.error for res in r.results if res.error)
        print(f"    [red]First error: {first_error}[/red]")


def print_comparison_table(results: list[ConcurrencyResult]) -> None:
    print("\n" + "=" * 70)
    print("  CONTINUOUS BATCHING BENCHMARK — Concurrency Comparison")
    print("=" * 70)
    print(f"  {'Concurrency':<14} {'Wall Time':<12} {'Throughput':<14} {'Mean TTFT':<12} {'P95 TTFT':<10}")
    print("-" * 70)
    for r in results:
        print(f"  {r.concurrency:<14} {r.total_wall_time_s:>9.2f}s   {r.throughput_tps:>10.1f} tok/s  "
              f"{r.mean_ttft_ms:>8.0f}ms   {r.p95_ttft_ms:>6.0f}ms")
    print("=" * 70)

    # Show scaling benefit
    if len(results) >= 2:
        base = results[0]
        print("\n  Scaling vs concurrency 1 (approximate):")
        for r in results[1:]:
            speedup = base.throughput_tps / r.throughput_tps if r.throughput_tps > 0 else 0
            # Actually we want: as concurrency increases, throughput should increase with CB
            tp_gain = r.throughput_tps / base.throughput_tps if base.throughput_tps > 0 else 0
            print(f"    Concurrency {r.concurrency}: {tp_gain:.2f}x throughput vs {base.concurrency}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTTP concurrency benchmark for bodega_mlx_engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:44468",
        help="Server base URL (default: http://localhost:44468)",
    )
    parser.add_argument(
        "--model",
        default="srswti/bodega-raptor-90m",
        help="Model ID (default: srswti/bodega-raptor-90m)",
    )
    parser.add_argument(
        "--concurrencies",
        default="8,16,32",
        help="Comma-separated concurrency levels (default: 8,16,32)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=32,
        help="Total number of queries per run (default: 32)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens per request (default: 128)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Use non-streaming requests (TTFT less accurate)",
    )
    args = parser.parse_args()

    concurrencies = [int(x.strip()) for x in args.concurrencies.split(",") if x.strip()]

    print("=" * 70)
    print("  bodega_mlx_engine — HTTP Concurrency Benchmark")
    print("=" * 70)
    print(f"  Base URL:    {args.base_url}")
    print(f"  Model:      {args.model}")
    print(f"  Concurrency: {concurrencies}")
    print(f"  Num queries: {args.num_queries} (same for all runs)")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Streaming:   {not args.no_stream}")
    print()

    # Health check
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{args.base_url.rstrip('/')}/health", timeout=5.0)
            if r.status_code != 200:
                print("⚠ Health check returned non-200. Proceeding anyway.")
            else:
                print("✓ Server health OK")
    except Exception as e:
        print(f"✗ Cannot reach server at {args.base_url}: {e}")
        sys.exit(1)

    results: list[ConcurrencyResult] = []

    # Dynamically load the model via admin API
    async with httpx.AsyncClient() as client:
        success = await manage_model(client, args.base_url, "load", args.model, args.model)
        if not success:
            print("✗ Failed to load model. Exiting.")
            sys.exit(1)

    for concurrency in concurrencies:
        print(f"\n--- Running concurrency {concurrency} ---")
        res = await benchmark_concurrency(
            base_url=args.base_url,
            model=args.model,
            concurrency=concurrency,
            num_queries=args.num_queries,
            max_tokens=args.max_tokens,
            stream=not args.no_stream,
        )
        results.append(res)
        print_result(res)

    print_comparison_table(results)

    # Unload after testing
    async with httpx.AsyncClient() as client:
        await manage_model(client, args.base_url, "unload", args.model, args.model)


if __name__ == "__main__":
    asyncio.run(main())
