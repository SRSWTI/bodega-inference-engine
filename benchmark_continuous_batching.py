#!/usr/bin/env python3
"""
Continuous Batching Benchmark for bodega_mlx_engine (HTTP API)
==============================================================

Benchmarks continuous batching performance against the server (localhost:44468) by dynamically
loading the model through the admin API, running the workload, and unloading it.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import subprocess
import os
from dataclasses import dataclass, field
from typing import List
import httpx
from loguru import logger

logger.remove()

PROMPTS = [
    "Hello, how are you?",
    "What is 2+2?",
    "Say hello in Spanish.",
    "What is the capital of France and why is it historically significant?",
    "Write a Python function to calculate fibonacci numbers with memoization.",
    "Explain the difference between a list and a tuple in Python.",
    "Explain quantum computing in detail, covering: qubits, superposition, entanglement, potential applications in cryptography and drug discovery.",
    "Write a comprehensive guide to building a production REST API in Python. Include: project structure, routing, authentication with JWT, etc.",
]

@dataclass
class RequestResult:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float = 0.0
    total_time_s: float = 0.0
    tps: float = 0.0
    error: str = None

    def __post_init__(self):
        if self.completion_tokens > 1 and self.total_time_s > self.ttft_s:
            gen_time = self.total_time_s - self.ttft_s
            self.tps = (self.completion_tokens - 1) / gen_time if gen_time > 0 else 0

@dataclass
class BenchmarkSummary:
    model: str
    mode: str  # "batched" or "sequential"
    concurrency: int
    num_requests: int
    max_tokens: int
    total_wall_time_s: float
    results: List[RequestResult] = field(default_factory=list)

    @property
    def successful(self):
        return [r for r in self.results if r.error is None]

    @property
    def throughput_tps(self) -> float:
        total_out = sum(r.completion_tokens for r in self.successful)
        return total_out / self.total_wall_time_s if self.total_wall_time_s > 0 else 0

    @property
    def mean_ttft_ms(self) -> float:
        ttfts = [r.ttft_s * 1000 for r in self.successful]
        return statistics.mean(ttfts) if ttfts else 0

    @property
    def p95_ttft_ms(self) -> float:
        ttfts = sorted(r.ttft_s * 1000 for r in self.successful)
        if not ttfts:
            return 0
        idx = min(int(len(ttfts) * 0.95), len(ttfts) - 1)
        return ttfts[idx]

    @property
    def mean_tps(self) -> float:
        tpss = [r.tps for r in self.successful if r.tps > 0]
        return statistics.mean(tpss) if tpss else 0

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.successful)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.successful)

async def _run_one_http(client: httpx.AsyncClient, base_url: str, model_id: str, prompt: str, max_tokens: int) -> RequestResult:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = 0
    completion_tokens = 0

    try:
        async with client.stream("POST", url, json=payload, timeout=120.0) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                return RequestResult(error=f"HTTP {resp.status_code}: {text.decode()[:40]}", total_time_s=time.perf_counter() - t0)

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
                    if choices and choices[0].get("delta", {}).get("content"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        completion_tokens += 1
                    
                    if "usage" in data and data["usage"]:
                        if data["usage"].get("prompt_tokens"):
                            prompt_tokens = data["usage"]["prompt_tokens"]
                        if data["usage"].get("completion_tokens"):
                            completion_tokens = data["usage"]["completion_tokens"]
                        
        total_time = time.perf_counter() - t0
        return RequestResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft or total_time,
            total_time_s=total_time,
        )
    except Exception as exc:
        return RequestResult(error=str(exc), total_time_s=time.perf_counter() - t0)

class IncompatibleModelError(Exception):
    pass

def open_mactop_window():
    """Opens mactop in a new Terminal window side-by-side via osascript."""
    script = 'tell application "Terminal" to do script "mactop"'
    ret = os.system(f"osascript -e '{script}' >/dev/null 2>&1")
    if ret != 0:
        # Fallback: silent, don't block the benchmark
        pass

async def get_model_type(client: httpx.AsyncClient, model_path: str) -> str:
    """Check HuggingFace API: verify MLX tag and return 'lm' or 'multimodal'.
    Raises IncompatibleModelError if the model has no MLX tag.
    """
    if "/" not in model_path:
        return "lm"  # local model, assume OK
    try:
        resp = await client.get(f"https://huggingface.co/api/models/{model_path}", timeout=6.0)
        if resp.status_code == 200:
            data = resp.json()
            tags = [t.lower() for t in data.get("tags", [])]
            pipeline_tag = data.get("pipeline_tag", "")

            has_mlx = any("mlx" in t for t in tags)
            if not has_mlx:
                raise IncompatibleModelError(
                    f"Model '{model_path}' does not have an MLX tag on HuggingFace. "
                    f"Only MLX-format models are compatible with the Bodega Inference Engine."
                )

            if pipeline_tag == "image-text-to-text":
                return "multimodal"
    except IncompatibleModelError:
        raise
    except Exception:
        pass
    return "lm"

async def manage_model(client: httpx.AsyncClient, base_url: str, action: str, model_path: str, model_id: str, **kwargs) -> bool:
    if action == "load":
        url = f"{base_url.rstrip('/')}/v1/admin/load-model"

        try:
            primary_type = await get_model_type(client, model_path)
        except IncompatibleModelError as e:
            print(f"\n  ✗ {e}")
            return False

        # Try primary detected type, then fallback to the other
        types_to_try = [primary_type, "multimodal" if primary_type == "lm" else "lm"]
        for mtype in types_to_try:
            payload = {"model_path": model_path, "model_id": model_id, "model_type": mtype, "context_length": 8192}
            payload.update(kwargs)
            resp = await client.post(url, json=payload, timeout=120.0)
            if resp.status_code == 409:
                print(f"  (Already loaded as {mtype})", flush=True)
                if mtype == "multimodal" and kwargs.get("continuous_batching"):
                    print("  [!] Note: Continuous batching for 'multimodal' models is coming soon to Bodega.\n"
                          "      The engine currently falls back to sequential execution for vision models.", flush=True)
                return True
            if resp.status_code in [200, 201]:
                print(f"  (Loaded as {mtype})", flush=True)
                if mtype == "multimodal" and kwargs.get("continuous_batching"):
                    print("  [!] Note: Continuous batching for 'multimodal' models is coming soon to Bodega.\n"
                          "      The engine currently falls back to sequential execution for vision models.", flush=True)
                return True
            if resp.status_code == 500:
                # Try next type
                continue
            # Any other error (4xx etc.) — fatal
            try:
                msg = resp.json().get("error", {}).get("message", resp.text[:120])
            except Exception:
                msg = resp.text[:120]
            print(f"  ✗ Load failed ({resp.status_code}): {msg}")
            return False

        print(f"  ✗ Model '{model_path}' could not be loaded as 'lm' or 'multimodal'. "
              f"This model may not be compatible with the Bodega Inference Engine.")
        return False

    elif action == "unload":
        url = f"{base_url.rstrip('/')}/v1/admin/unload-model/{model_id}"
        resp = await client.delete(url, timeout=30.0)
        return resp.status_code in [200, 204]

async def benchmark_batched_http(base_url: str, model_path: str, prompts: List[str], concurrency: int, max_tokens: int, **cb_args) -> BenchmarkSummary:
    model_id = "bench-model-batched"
    print(f"\n{'─'*60}\n  [BATCHED] {model_path}\n  concurrency={concurrency}  max_tokens={max_tokens}\n{'─'*60}")
    
    async with httpx.AsyncClient() as client:
        print("  Loading model via HTTP API...", end=" ", flush=True)
        t_load = time.perf_counter()
        success = await manage_model(client, base_url, "load", model_path, model_id, continuous_batching=True, **cb_args)
        if not success:
            print("FAILED")
            return BenchmarkSummary(model_path, "batched", concurrency, len(prompts), max_tokens, 0)
        print(f"done in {time.perf_counter()-t_load:.1f}s")
        
        test_prompts = (prompts * ((concurrency // len(prompts)) + 1))[:concurrency]
        print(f"  Running {len(test_prompts)} concurrent requests...", flush=True)
        
        t_start = time.perf_counter()
        results = await asyncio.gather(*[_run_one_http(client, base_url, model_id, p, max_tokens) for p in test_prompts])
        wall_time = time.perf_counter() - t_start

        try:
            if os.system("command -v mactop >/dev/null 2>&1") == 0:
                res = subprocess.run(["mactop", "--headless", "--count", "1"], capture_output=True, text=True, timeout=2.0)
                if res.returncode == 0:
                    data = json.loads(res.stdout)
                    if isinstance(data, list) and len(data) > 0:
                        sm = data[0].get("soc_metrics", {})
                        cpu_p = sm.get("cpu_power", 0)
                        gpu_p = sm.get("gpu_power", 0)
                        sys_p = sm.get("system_power", 0)
                        gpu_freq = sm.get("gpu_freq_mhz", 0)
                        gpu_temp = sm.get("gpu_temp", 0)
                        
                        mem = data[0].get("memory", {})
                        ram_used = mem.get("used", 0) / (1024**3)
                        ram_tot = mem.get("total", 0) / (1024**3)
                        
                        print(f"  [Telemetry] RAM: {ram_used:.1f}GB/{ram_tot:.0f}GB | Pwr: {sys_p:.1f}W (CPU: {cpu_p:.1f}W GPU: {gpu_p:.1f}W {gpu_freq}MHz) | GPU Temp: {gpu_temp:.1f}°C")
        except Exception:
            pass

        print("  Unloading model...", end=" ", flush=True)
        await manage_model(client, base_url, "unload", model_path, model_id)
        print("done.")

    return BenchmarkSummary(model_path, "batched", concurrency, len(test_prompts), max_tokens, wall_time, list(results))

async def benchmark_sequential_http(base_url: str, model_path: str, prompts: List[str], concurrency: int, max_tokens: int) -> BenchmarkSummary:
    model_id = "bench-model-seq"
    print(f"\n{'─'*60}\n  [SEQUENTIAL] {model_path}\n  concurrency={concurrency}  max_tokens={max_tokens}\n{'─'*60}")
    
    async with httpx.AsyncClient() as client:
        print("  Loading model via HTTP API...", end=" ", flush=True)
        t_load = time.perf_counter()
        success = await manage_model(client, base_url, "load", model_path, model_id, continuous_batching=False)
        if not success:
            print("FAILED")
            return BenchmarkSummary(model_path, "sequential", concurrency, len(prompts), max_tokens, 0)
        print(f"done in {time.perf_counter()-t_load:.1f}s")
        
        test_prompts = (prompts * ((concurrency // len(prompts)) + 1))[:concurrency]
        print(f"  Running {len(test_prompts)} concurrent requests (server queued)...", flush=True)
        
        t_start = time.perf_counter()
        results = await asyncio.gather(*[_run_one_http(client, base_url, model_id, p, max_tokens) for p in test_prompts])
        wall_time = time.perf_counter() - t_start

        print("  Unloading model...", end=" ", flush=True)
        await manage_model(client, base_url, "unload", model_path, model_id)
        print("done.")

    return BenchmarkSummary(model_path, "sequential", concurrency, len(test_prompts), max_tokens, wall_time, list(results))

def print_summary(s: BenchmarkSummary):
    ok = s.successful
    failed = len(s.results) - len(ok)
    sep = "═" * 62
    print(f"\n{sep}\n  {'RESULTS':^58}\n  Model:        {s.model}\n  Mode:         {s.mode.upper()}  (concurrency={s.concurrency})\n{sep}")
    print(f"  Requests:     {len(ok)}/{s.num_requests} succeeded  ({failed} failed)")
    if ok:
        print(f"  Prompt tokens:     {s.total_prompt_tokens:>8,}\n  Completion tokens: {s.total_completion_tokens:>8,}\n")
        print(f"  ── Latency ──────────────────────────────────────────\n  Wall time (all reqs):   {s.total_wall_time_s:.2f}s\n  Mean TTFT:              {s.mean_ttft_ms:.0f}ms\n  P95 TTFT:               {s.p95_ttft_ms:.0f}ms\n")
        print(f"  ── Throughput ───────────────────────────────────────\n  Mean per-req TPS:       {s.mean_tps:.1f} tok/s\n  System throughput:      {s.throughput_tps:.1f} tok/s  ← total output/wall time\n")
        print(f"  ── Per-request breakdown ────────────────────────────\n  {'#':<4} {'Prompt':>6} {'Output':>6} {'TTFT':>8} {'TPS':>8}  Status")
        for i, r in enumerate(s.results, 1):
            if r.error: print(f"  {i:<4} {'—':>6} {'—':>6} {'—':>8} {'—':>8}  ❌ {r.error[:40]}")
            else: print(f"  {i:<4} {r.prompt_tokens:>6} {r.completion_tokens:>6} {r.ttft_s*1000:>7.0f}ms {r.tps:>7.1f}  ✅")
    print(sep)

def print_comparison(batched: BenchmarkSummary, sequential: BenchmarkSummary):
    ratio = sequential.total_wall_time_s / batched.total_wall_time_s if batched.total_wall_time_s > 0 else 0
    tp_ratio = batched.throughput_tps / sequential.throughput_tps if sequential.throughput_tps > 0 else 0
    print(f"\n{'═'*62}\n  BATCHED vs SEQUENTIAL COMPARISON\n{'═'*62}")
    print(f"  {'Metric':<30} {'Sequential':>12} {'Batched':>12}\n  {'─'*54}")
    print(f"  {'Wall time':.<30} {sequential.total_wall_time_s:>11.2f}s {batched.total_wall_time_s:>11.2f}s")
    print(f"  {'Mean TTFT':.<30} {sequential.mean_ttft_ms:>10.0f}ms {batched.mean_ttft_ms:>10.0f}ms")
    print(f"  {'P95 TTFT':.<30} {sequential.p95_ttft_ms:>10.0f}ms {batched.p95_ttft_ms:>10.0f}ms")
    print(f"  {'System throughput (tok/s)':.<30} {sequential.throughput_tps:>12.1f} {batched.throughput_tps:>12.1f}\n  {'─'*54}")
    print(f"  Wall time speedup:  {ratio:.2f}x\n  Throughput gain:    {tp_ratio:.2f}x\n{'═'*62}")

async def main():
    parser = argparse.ArgumentParser(description="Benchmark continuous batching via HTTP API")
    parser.add_argument("--base-url", default="http://localhost:44468", help="Server base URL")
    parser.add_argument("--model", dest="models", action="append", default=[], help="Model path")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens")
    parser.add_argument("--cb-max-num-seqs", type=int, default=64)
    parser.add_argument("--cb-prefill-batch-size", type=int, default=4)
    parser.add_argument("--cb-chunked-prefill-tokens", type=int, default=2048)
    parser.add_argument("--compare", action="store_true", help="Compare with sequential mode")
    args = parser.parse_args()

    models = args.models or ["srswti/bodega-raptor-90m"]
    print("=" * 62 + "\n  bodega_mlx_engine — Continuous Batching Benchmark\n" + "=" * 62)
    print(f"  Base URL:    {args.base_url}\n  Models:      {models}\n  Concurrency: {args.concurrency}\n  Max tokens:  {args.max_tokens}\n")

    for model_path in models:
        batched = await benchmark_batched_http(
            args.base_url, model_path, PROMPTS, args.concurrency, args.max_tokens,
            cb_max_num_seqs=args.cb_max_num_seqs, cb_prefill_batch_size=args.cb_prefill_batch_size, cb_chunked_prefill_tokens=args.cb_chunked_prefill_tokens
        )
        print_summary(batched)

        if args.compare:
            seq = await benchmark_sequential_http(args.base_url, model_path, PROMPTS, args.concurrency, args.max_tokens)
            print_summary(seq)
            print_comparison(batched, seq)

if __name__ == "__main__":
    asyncio.run(main())
