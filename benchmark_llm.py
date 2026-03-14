
"""
Bodega LLM Performance Benchmark
=================================

Comprehensive language-model benchmark for the Bodega Inference Engine.
Communicates exclusively through the HTTP API at localhost:44468 — no internal
engine imports required.

Measures 
  TTFT          Time to First Token (ms) — latency until the model starts generating
  TPOT          Time Per Output Token (ms) — time between each generated token
  Gen TPS       Output tokens/second (generation phase only, excludes first token)
  Proc TPS      Input tokens/second — how fast the prompt was ingested (prompt_tokens / TTFT)
  Latency       End-to-end wall time per request
  Throughput    (input + output) tokens / total wall time
  System TPS    Total output tokens / wall time across ALL concurrent requests

Two modes:
  batched     Load with continuous_batching=True, fire all requests concurrently.
              This is the preferred mode for throughput benchmarks.
  sequential  Load with continuous_batching=False; the server queues and serialises
              requests. Use --compare to run both and see the speedup.

Usage:
    # Quick start — loads, benchmarks, unloads
    python benchmark_llm.py --model srswti/bodega-raptor-90m

    # More prompts and tokens
    python benchmark_llm.py --model srswti/bodega-raptor-0.9b --prompts 10 --max-tokens 256

    # Concurrency sweep (mirrors the README tables)
    python benchmark_llm.py --model srswti/bodega-raptor-0.9b --concurrencies 4,8,16,32

    # CB vs sequential side-by-side comparison
    python benchmark_llm.py --model srswti/bodega-raptor-8b --compare

    # Use a model already loaded in the server (skip auto load/unload)
    python benchmark_llm.py --model-id bodega-raptor-8b --no-manage

    # Save full structured JSON results
    python benchmark_llm.py --model srswti/bodega-raptor-0.9b --output results.json

    # Test against a different host
    python benchmark_llm.py --model srswti/bodega-raptor-0.9b --base-url http://192.168.1.10:44468

Environment:
    BODEGA_SKIP_TELEMETRY=1   Skip the mactop telemetry window.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:44468"

# 10-prompt suite: 3 short · 3 medium · 4 long  )
PROMPTS: list[str] = [
    # Short (~5-15 tokens) — 3 prompts
    "Hello, how are you?",
    "What is 2+2?",
    "Say hello in Spanish.",
    # Medium (~30-60 tokens) — 3 prompts
    "What is the capital of France and why is it historically significant? Include some interesting facts about the city.",
    "Write a Python function to calculate fibonacci numbers using memoization. Explain how it works.",
    "Explain the difference between a list and a tuple in Python. When should you use each one?",
    # Long (100+ tokens) — 4 prompts
    (
        "Explain quantum computing in comprehensive detail. Cover all of the following thoroughly:\n"
        "1. What are qubits and how do they differ from classical bits?\n"
        "2. What is quantum superposition and how does it enable parallel computation?\n"
        "3. What is quantum entanglement and why is it crucial for quantum algorithms?\n"
        "4. What are the most promising applications in cryptography, drug discovery, and optimisation?\n"
        "5. What are the current hardware limitations, error correction challenges, and decoherence problems?\n"
        "6. Compare the approaches of IBM, Google, and other major players in quantum computing research."
    ),
    (
        "Write a comprehensive guide to building a production-ready REST API with Python Flask. Include:\n"
        "1. Project structure with blueprints, configuration management, and environment variables\n"
        "2. Creating routes and endpoints following RESTful conventions with proper HTTP methods\n"
        "3. Handling JSON requests and responses with validation using marshmallow or pydantic\n"
        "4. Authentication and authorisation with JWT tokens and role-based access control\n"
        "5. Error handling best practices with custom exception handlers\n"
        "6. Writing tests with pytest — unit tests and integration tests\n"
        "7. Logging, monitoring, and API documentation with Swagger/OpenAPI"
    ),
    (
        "Describe the complete process of photosynthesis with scientific detail. Cover:\n"
        "1. Light-dependent reactions in the thylakoid membrane, including photosystems I and II\n"
        "2. The electron transport chain and chemiosmosis for ATP synthesis\n"
        "3. The Calvin cycle (light-independent reactions) and carbon fixation by RuBisCO\n"
        "4. The role of chlorophyll a, chlorophyll b, and accessory pigments like carotenoids\n"
        "5. How light intensity, CO2 concentration, and temperature affect the photosynthesis rate\n"
        "6. The importance of photosynthesis for life on Earth and its role in the carbon cycle\n"
        "7. C3, C4, and CAM photosynthesis adaptations in different plant species"
    ),
    (
        "You are a senior software architect. Design a microservices architecture for a large-scale "
        "e-commerce platform handling millions of users. Include:\n"
        "1. Service breakdown: User, Product catalog, Inventory, Order, Payment, Notification\n"
        "2. Database choices per service with justification (PostgreSQL vs MongoDB vs Redis)\n"
        "3. Inter-service communication: synchronous REST/gRPC vs asynchronous message queues\n"
        "4. API gateway design with rate limiting, authentication, and request routing\n"
        "5. Caching strategy with Redis for sessions, product data, and search results\n"
        "6. Message queue architecture with RabbitMQ or Kafka for event-driven communication\n"
        "7. Kubernetes deployment with horizontal pod autoscaling and rolling updates\n"
        "8. CI/CD pipeline with GitHub Actions, testing stages, and blue-green deployments"
    ),
]


# ---------------------------------------------------------------------------
# Resource metrics  
# ---------------------------------------------------------------------------

@dataclass
class ResourceMetrics:
    """Hardware/memory snapshot during a benchmark run.

    Fields map to what the Bodega engine exposes via GET /v1/admin/loaded-models
    (metal_* fields) and mactop --headless (power/temp/system-RAM fields).
    The names are kept compatible with the ResourceMetrics dataclass
    so JSON output is directly comparable.
    """
    # From GET /v1/admin/loaded-models  (per-model Metal Unified Memory)
    process_memory_gb: float = 0.0        # model subprocess RSS
    mlx_active_memory_gb: float = 0.0    # metal_active_mb  (in-flight compute)
    mlx_cache_gb: float = 0.0            # metal_cache_mb   (retained KV cache)
    mlx_peak_memory_gb: float = 0.0      # metal_peak_mb    (high-water mark)
    # From mactop --headless (system-level)
    system_memory_used_gb: float = 0.0
    system_memory_total_gb: float = 0.0
    gpu_power_w: float = 0.0
    cpu_power_w: float = 0.0
    system_power_w: float = 0.0
    gpu_temp_c: float = 0.0


class ResourceMonitor:
    """Poll the Bodega HTTP API and mactop during benchmark runs.

    Mirrors the interface ofResourceMonitor so callers work the
    same way, but all data comes from HTTP rather than in-process mlx calls.
    """

    def __init__(self, client: httpx.AsyncClient, base_url: str, model_id: str):
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self.samples: list[ResourceMetrics] = []

    async def sample(self) -> ResourceMetrics:
        metrics = ResourceMetrics()

        # -- Metal memory from /v1/admin/loaded-models ---------------------
        try:
            resp = await self._client.get(
                f"{self._base_url}/v1/admin/loaded-models", timeout=5.0
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                for m in data:
                    if m.get("id") == self._model_id:
                        mem = m.get("memory", {})
                        metrics.process_memory_gb = mem.get("rss_mb", 0) / 1024
                        metrics.mlx_active_memory_gb = mem.get("metal_active_mb", 0) / 1024
                        metrics.mlx_cache_gb = mem.get("metal_cache_mb", 0) / 1024
                        metrics.mlx_peak_memory_gb = mem.get("metal_peak_mb", 0) / 1024
                        break
        except Exception:
            pass

        # -- System RAM + power from mactop --------------------------------
        try:
            res = subprocess.run(
                ["mactop", "--headless", "--count", "1"],
                capture_output=True, text=True, timeout=3.0,
            )
            if res.returncode == 0 and res.stdout.strip():
                mdata = json.loads(res.stdout)
                if isinstance(mdata, list) and mdata:
                    sm = mdata[0].get("soc_metrics", {})
                    mem = mdata[0].get("memory", {})
                    metrics.gpu_power_w = sm.get("gpu_power", 0.0)
                    metrics.cpu_power_w = sm.get("cpu_power", 0.0)
                    metrics.system_power_w = sm.get("system_power", 0.0)
                    metrics.gpu_temp_c = sm.get("gpu_temp", 0.0)
                    metrics.system_memory_used_gb = mem.get("used", 0) / (1024 ** 3)
                    metrics.system_memory_total_gb = mem.get("total", 0) / (1024 ** 3)
        except Exception:
            pass

        self.samples.append(metrics)
        return metrics

    def get_summary(self) -> ResourceMetrics:
        if not self.samples:
            return ResourceMetrics()
        peak = ResourceMetrics(
            process_memory_gb=max(s.process_memory_gb for s in self.samples),
            mlx_active_memory_gb=max(s.mlx_active_memory_gb for s in self.samples),
            mlx_cache_gb=max(s.mlx_cache_gb for s in self.samples),
            mlx_peak_memory_gb=max(s.mlx_peak_memory_gb for s in self.samples),
            gpu_power_w=max(s.gpu_power_w for s in self.samples),
            cpu_power_w=max(s.cpu_power_w for s in self.samples),
            system_power_w=max(s.system_power_w for s in self.samples),
            gpu_temp_c=max(s.gpu_temp_c for s in self.samples),
        )
        latest = self.samples[-1]
        peak.system_memory_used_gb = latest.system_memory_used_gb
        peak.system_memory_total_gb = latest.system_memory_total_gb
        return peak


# ---------------------------------------------------------------------------
# Per-request result 
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Metrics for a single request.

    Derived fields are computed automatically in __post_init__ using the same
    arithmetic — TPOT and generation_tps exclude the
    first token (already captured by TTFT), and processing_tps measures prompt
    ingestion speed.
    """
    prompt: str
    prompt_tokens: int
    generated_tokens: int

    # Primary timing (seconds)
    ttft: float        # Time to First Token
    total_time: float  # End-to-end latency

    # Derived (filled by __post_init__)
    tpot: float = 0.0              # Time Per Output Token (s)
    generation_tps: float = 0.0   # Output tok/s  (excludes first token)
    processing_tps: float = 0.0   # Prompt tok/s  = prompt_tokens / TTFT

    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.error:
            return
        if self.generated_tokens > 1:
            # Exclude the first token from generation speed — it is already
            # captured by TTFT.  
            generation_time = self.total_time - self.ttft
            self.tpot = (
                generation_time / (self.generated_tokens - 1)
                if self.generated_tokens > 1 else 0.0
            )
            self.generation_tps = (
                (self.generated_tokens - 1) / generation_time
                if generation_time > 0 else 0.0
            )
        if self.ttft > 0:
            self.processing_tps = self.prompt_tokens / self.ttft


# ---------------------------------------------------------------------------
# Aggregate summary 
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSummary:
    """Aggregate statistics across all requests in a benchmark run."""

    model_name: str
    mode: str          # "batched" | "sequential"
    concurrency: int
    num_runs: int
    total_prompt_tokens: int
    total_generated_tokens: int
    total_time: float  # wall time for the entire batch

    # TTFT distribution (seconds)
    ttft_mean: float
    ttft_min: float
    ttft_max: float
    ttft_p50: float
    ttft_p95: float

    # TPOT distribution (seconds)
    tpot_mean: float
    tpot_min: float
    tpot_max: float

    # Token throughput
    generation_tps_mean: float   # mean per-request generation speed
    generation_tps_max: float
    processing_tps_mean: float   # mean prompt ingestion speed

    # End-to-end latency distribution (seconds)
    latency_mean: float
    latency_min: float
    latency_max: float
    latency_p50: float
    latency_p95: float

    # System-level throughput
    total_throughput_tps: float   # (input + output) tok / wall time
    system_throughput_tps: float  # output tok / wall time  (CB key metric)
    requests_per_second: float

    # Hardware (populated from mactop / platform at run time)
    hardware_chip: str = ""
    hardware_memory_gb: float = 0.0

    # Resource snapshot
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)

    # Raw per-request results (for per-run breakdown table)
    results: list[BenchmarkResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calculate_percentile(data: list[float], percentile: float) -> float:
    """Percentile helper"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    index = min(index, len(sorted_data) - 1)
    return sorted_data[index]


def build_summary(
    model_name: str,
    mode: str,
    concurrency: int,
    results: list[BenchmarkResult],
    wall_time: float,
    resources: ResourceMetrics,
    hardware_chip: str = "",
    hardware_memory_gb: float = 0.0,
) -> BenchmarkSummary:
    """Aggregate a list of BenchmarkResult into a BenchmarkSummary."""
    ok = [r for r in results if r.error is None]
    if not ok:
        return BenchmarkSummary(
            model_name=model_name, mode=mode, concurrency=concurrency,
            num_runs=0, total_prompt_tokens=0, total_generated_tokens=0,
            total_time=wall_time, ttft_mean=0, ttft_min=0, ttft_max=0,
            ttft_p50=0, ttft_p95=0, tpot_mean=0, tpot_min=0, tpot_max=0,
            generation_tps_mean=0, generation_tps_max=0, processing_tps_mean=0,
            latency_mean=0, latency_min=0, latency_max=0, latency_p50=0,
            latency_p95=0, total_throughput_tps=0, system_throughput_tps=0,
            requests_per_second=0, hardware_chip=hardware_chip,
            hardware_memory_gb=hardware_memory_gb, resources=resources,
            results=results,
        )

    ttfts = [r.ttft for r in ok]
    tpots = [r.tpot for r in ok if r.tpot > 0]
    latencies = [r.total_time for r in ok]
    gen_tps = [r.generation_tps for r in ok if r.generation_tps > 0]
    proc_tps = [r.processing_tps for r in ok if r.processing_tps > 0]

    total_prompt = sum(r.prompt_tokens for r in ok)
    total_gen = sum(r.generated_tokens for r in ok)

    return BenchmarkSummary(
        model_name=model_name,
        mode=mode,
        concurrency=concurrency,
        num_runs=len(ok),
        total_prompt_tokens=total_prompt,
        total_generated_tokens=total_gen,
        total_time=wall_time,
        # TTFT
        ttft_mean=statistics.mean(ttfts),
        ttft_min=min(ttfts),
        ttft_max=max(ttfts),
        ttft_p50=calculate_percentile(ttfts, 50),
        ttft_p95=calculate_percentile(ttfts, 95),
        # TPOT
        tpot_mean=statistics.mean(tpots) if tpots else 0.0,
        tpot_min=min(tpots) if tpots else 0.0,
        tpot_max=max(tpots) if tpots else 0.0,
        # TPS
        generation_tps_mean=statistics.mean(gen_tps) if gen_tps else 0.0,
        generation_tps_max=max(gen_tps) if gen_tps else 0.0,
        processing_tps_mean=statistics.mean(proc_tps) if proc_tps else 0.0,
        # Latency
        latency_mean=statistics.mean(latencies),
        latency_min=min(latencies),
        latency_max=max(latencies),
        latency_p50=calculate_percentile(latencies, 50),
        latency_p95=calculate_percentile(latencies, 95),
        # Throughput
        total_throughput_tps=(total_prompt + total_gen) / wall_time if wall_time > 0 else 0,
        system_throughput_tps=total_gen / wall_time if wall_time > 0 else 0,
        requests_per_second=len(ok) / wall_time if wall_time > 0 else 0,
        hardware_chip=hardware_chip,
        hardware_memory_gb=hardware_memory_gb,
        resources=resources,
        results=results,
    )


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def _detect_hardware_from_mactop() -> tuple[str, float]:
    """Return (chip_name, total_memory_gb) from mactop --headless."""
    try:
        res = subprocess.run(
            ["mactop", "--headless", "--count", "1"],
            capture_output=True, text=True, timeout=4.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if isinstance(data, list) and data:
                sm = data[0].get("soc_metrics", {})
                mem = data[0].get("memory", {})
                chip = sm.get("chip_name", "")
                total_gb = mem.get("total", 0) / (1024 ** 3)
                return chip, total_gb
    except Exception:
        pass
    return platform.processor() or "Apple Silicon", 0.0


# ---------------------------------------------------------------------------
# Model lifecycle via admin API
# ---------------------------------------------------------------------------

class IncompatibleModelError(Exception):
    pass


async def _hf_model_type(client: httpx.AsyncClient, model_path: str) -> str:
    """Resolve HuggingFace pipeline_tag → bodega model_type.
    Raises IncompatibleModelError if the model lacks an mlx tag.
    """
    if "/" not in model_path:
        return "lm"
    try:
        resp = await client.get(
            f"https://huggingface.co/api/models/{model_path}", timeout=8.0
        )
        if resp.status_code == 200:
            info = resp.json()
            tags = [t.lower() for t in info.get("tags", [])]
            if not any("mlx" in t for t in tags):
                raise IncompatibleModelError(
                    f"'{model_path}' has no MLX tag on HuggingFace — only MLX-format "
                    "models are compatible with the Bodega Inference Engine."
                )
            if info.get("pipeline_tag") == "image-text-to-text":
                return "multimodal"
    except IncompatibleModelError:
        raise
    except Exception:
        pass
    return "lm"


async def load_model(
    client: httpx.AsyncClient,
    base_url: str,
    model_path: str,
    model_id: str,
    continuous_batching: bool,
    cb_max_num_seqs: int,
    cb_prefill_batch_size: int,
    cb_completion_batch_size: int,
    cb_chunked_prefill_tokens: int,
    context_length: int,
) -> bool:
    url = f"{base_url}/v1/admin/load-model"
    try:
        model_type = await _hf_model_type(client, model_path)
    except IncompatibleModelError as exc:
        print(f"\n  ✗ {exc}")
        return False

    payload: dict[str, Any] = {
        "model_path": model_path,
        "model_id": model_id,
        "model_type": model_type,
        "context_length": context_length,
    }
    if continuous_batching:
        payload.update(
            continuous_batching=True,
            cb_max_num_seqs=cb_max_num_seqs,
            cb_prefill_batch_size=cb_prefill_batch_size,
            cb_completion_batch_size=cb_completion_batch_size,
            cb_chunked_prefill_tokens=cb_chunked_prefill_tokens,
            cb_enable_prefix_cache=True,
        )

    resp = await client.post(url, json=payload, timeout=180.0)
    if resp.status_code == 409:
        print("  (already loaded)", flush=True)
        return True
    if resp.status_code in (200, 201):
        print(f"  (loaded as {model_type})", flush=True)
        return True
    try:
        msg = resp.json().get("error", {}).get("message", resp.text[:120])
    except Exception:
        msg = resp.text[:120]
    print(f"  ✗ load failed ({resp.status_code}): {msg}")
    return False


async def unload_model(
    client: httpx.AsyncClient, base_url: str, model_id: str
) -> None:
    try:
        await client.delete(
            f"{base_url}/v1/admin/unload-model/{model_id}", timeout=30.0
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Single streaming request
# ---------------------------------------------------------------------------

async def _stream_one(
    client: httpx.AsyncClient,
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> BenchmarkResult:
    """Stream a single chat completion and measure TTFT/total_time precisely."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # Request usage in the final chunk — honoured by Bodega and newer
        # OpenAI-compatible servers (LM Studio, etc.).  Ignored if unsupported.
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    prompt_tokens = 0
    completion_tokens = 0
    truncated_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt

    try:
        async with client.stream("POST", url, json=payload, timeout=300.0) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return BenchmarkResult(
                    prompt=truncated_prompt, prompt_tokens=0, generated_tokens=0,
                    ttft=0.0, total_time=time.perf_counter() - t0,
                    error=f"HTTP {resp.status_code}: {body.decode()[:60]}",
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
                    if choices:
                        delta = choices[0].get("delta", {})
                        # Accept both standard content and reasoning_content
                        # (reasoning models like bodega-raptor emit the latter).
                        token_text = delta.get("content") or delta.get("reasoning_content")
                        if token_text:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            completion_tokens += 1

                    # Usage chunk — present in Bodega; present in LM Studio only
                    # when stream_options.include_usage=true is sent.
                    usage = data.get("usage")
                    if usage:
                        pt = usage.get("prompt_tokens")
                        ct = usage.get("completion_tokens")
                        if pt is not None and pt > 0:
                            prompt_tokens = pt
                        # Prefer server count over our increment counter
                        if ct is not None and ct > 0:
                            completion_tokens = ct

        total_time = time.perf_counter() - t0
        if ttft is None:
            ttft = total_time

        return BenchmarkResult(
            prompt=truncated_prompt,
            prompt_tokens=prompt_tokens,
            generated_tokens=completion_tokens,
            ttft=ttft,
            total_time=total_time,
        )
    except Exception as exc:
        return BenchmarkResult(
            prompt=truncated_prompt, prompt_tokens=0, generated_tokens=0,
            ttft=0.0, total_time=time.perf_counter() - t0,
            error=str(exc)[:80],
        )


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

async def run_benchmark(
    base_url: str,
    model_path: str,
    model_id: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    temperature: float,
    warmup_runs: int,
    continuous_batching: bool,
    manage_model_lifecycle: bool,
    cb_max_num_seqs: int,
    cb_prefill_batch_size: int,
    cb_completion_batch_size: int,
    cb_chunked_prefill_tokens: int,
    context_length: int,
) -> BenchmarkSummary:
    mode = "batched" if continuous_batching else "sequential"
    chip, mem_gb = _detect_hardware_from_mactop()

    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  [{mode.upper()}]  {model_path or model_id}")
    print(f"  concurrency={concurrency}  max_tokens={max_tokens}  prompts={len(prompts)}")
    print(sep)

    async with httpx.AsyncClient() as client:
        # ── health check ───────────────────────────────────────────────────
        try:
            hc = await client.get(f"{base_url}/health", timeout=5.0)
            if hc.status_code != 200:
                print("  ⚠ Health check returned non-200; proceeding anyway.")
        except Exception as exc:
            print(f"  ✗ Cannot reach server at {base_url}: {exc}")
            sys.exit(1)

        # ── model load ─────────────────────────────────────────────────────
        if manage_model_lifecycle:
            print("  Loading model via API...", end=" ", flush=True)
            t_load = time.perf_counter()
            ok = await load_model(
                client, base_url, model_path, model_id, continuous_batching,
                cb_max_num_seqs, cb_prefill_batch_size, cb_completion_batch_size,
                cb_chunked_prefill_tokens, context_length,
            )
            if not ok:
                return BenchmarkSummary(
                    model_name=model_path, mode=mode, concurrency=concurrency,
                    num_runs=0, total_prompt_tokens=0, total_generated_tokens=0,
                    total_time=0, ttft_mean=0, ttft_min=0, ttft_max=0,
                    ttft_p50=0, ttft_p95=0, tpot_mean=0, tpot_min=0, tpot_max=0,
                    generation_tps_mean=0, generation_tps_max=0, processing_tps_mean=0,
                    latency_mean=0, latency_min=0, latency_max=0, latency_p50=0,
                    latency_p95=0, total_throughput_tps=0, system_throughput_tps=0,
                    requests_per_second=0, hardware_chip=chip,
                    hardware_memory_gb=mem_gb,
                )
            print(f"  Model load time: {time.perf_counter()-t_load:.1f}s")

        monitor = ResourceMonitor(client, base_url, model_id)

        # ── warmup ─────────────────────────────────────────────────────────
        if warmup_runs > 0:
            print(f"  Warmup ({warmup_runs} run(s))...", end=" ", flush=True)
            for _ in range(warmup_runs):
                await _stream_one(
                    client, base_url, model_id, "Hello, how are you?",
                    max_tokens=20, temperature=temperature,
                )
            await monitor.sample()
            print("done")

        # ── build prompt list ──────────────────────────────────────────────
        test_prompts = (prompts * ((concurrency // len(prompts)) + 1))[:concurrency]

        print(f"\n  Prompt distribution ({len(test_prompts)} requests):")
        short_n  = sum(1 for p in test_prompts if len(p) < 100)
        medium_n = sum(1 for p in test_prompts if 100 <= len(p) < 500)
        long_n   = sum(1 for p in test_prompts if len(p) >= 500)
        print(f"    Short  (<100 chars):  {short_n}")
        print(f"    Medium (100-500):     {medium_n}")
        print(f"    Long   (500+):        {long_n}")
        print()

        # ── concurrent requests ────────────────────────────────────────────
        print(f"  Firing {len(test_prompts)} concurrent request(s)...", flush=True)
        t_start = time.perf_counter()
        raw_results = await asyncio.gather(
            *[
                _stream_one(client, base_url, model_id, p, max_tokens, temperature)
                for p in test_prompts
            ]
        )
        wall_time = time.perf_counter() - t_start

        # ── resource snapshot ──────────────────────────────────────────────
        await monitor.sample()

        # ── mactop live telemetry line ─────────────────────────────────────
        _try_print_mactop_line()

        # ── unload model ───────────────────────────────────────────────────
        if manage_model_lifecycle:
            print("  Unloading model...", end=" ", flush=True)
            await unload_model(client, base_url, model_id)
            print("done.")

    return build_summary(
        model_name=model_path or model_id,
        mode=mode,
        concurrency=concurrency,
        results=list(raw_results),
        wall_time=wall_time,
        resources=monitor.get_summary(),
        hardware_chip=chip,
        hardware_memory_gb=mem_gb,
    )


# ---------------------------------------------------------------------------
# Mactop telemetry helper
# ---------------------------------------------------------------------------

def _try_print_mactop_line() -> None:
    if os.environ.get("BODEGA_SKIP_TELEMETRY") == "1":
        return
    if not shutil.which("mactop"):
        return
    try:
        res = subprocess.run(
            ["mactop", "--headless", "--count", "1"],
            capture_output=True, text=True, timeout=3.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if isinstance(data, list) and data:
                sm = data[0].get("soc_metrics", {})
                mem = data[0].get("memory", {})
                ram_used = mem.get("used", 0) / (1024 ** 3)
                ram_tot  = mem.get("total", 0) / (1024 ** 3)
                print(
                    f"  [Telemetry] RAM {ram_used:.1f}/{ram_tot:.0f} GB  "
                    f"| Pwr {sm.get('system_power',0):.1f}W "
                    f"(CPU {sm.get('cpu_power',0):.1f}W  "
                    f"GPU {sm.get('gpu_power',0):.1f}W)  "
                    f"| GPU {sm.get('gpu_temp',0):.0f}°C"
                )
    except Exception:
        pass


def open_mactop_window() -> None:
    if os.environ.get("BODEGA_SKIP_TELEMETRY") == "1":
        return
    if not shutil.which("mactop"):
        return
    os.system("osascript -e 'tell application \"Terminal\" to do script \"mactop\"' >/dev/null 2>&1")


# ---------------------------------------------------------------------------
# Print functions
# ---------------------------------------------------------------------------

def print_summary(s: BenchmarkSummary) -> None:
    sep = "═" * 66
    ok  = [r for r in s.results if r.error is None]
    err = [r for r in s.results if r.error is not None]

    print(f"\n{sep}")
    print("  BENCHMARK RESULTS")
    print(sep)
    # Overview
    rows = [
        ("Model",         s.model_name),
        ("Mode",          f"{s.mode.upper()}  (concurrency={s.concurrency})"),
        ("Hardware",      f"{s.hardware_chip} ({s.hardware_memory_gb:.0f} GB)" if s.hardware_chip else "—"),
        ("Requests",      f"{len(ok)}/{s.num_runs + len(err)} succeeded  ({len(err)} failed)"),
        ("Input tokens",  f"{s.total_prompt_tokens:,}"),
        ("Output tokens", f"{s.total_generated_tokens:,}"),
        ("Wall time",     f"{s.total_time:.2f}s"),
    ]
    for label, val in rows:
        print(f"  {label:<20} {val}")

    if not ok:
        print(sep)
        return

    # Performance metrics
    print(f"\n  {'─'*62}")
    print("  Performance Metrics")
    print(f"  {'─'*62}")
    perf_rows = [
        ("TTFT mean",         f"{s.ttft_mean*1000:.1f} ms",
         f"p50={s.ttft_p50*1000:.1f}ms  p95={s.ttft_p95*1000:.1f}ms"),
        ("TTFT range",        f"{s.ttft_min*1000:.1f}ms → {s.ttft_max*1000:.1f}ms", ""),
        ("TPOT mean",         f"{s.tpot_mean*1000:.2f} ms/tok",
         f"max={s.tpot_max*1000:.2f}ms"),
        ("Gen TPS mean",      f"{s.generation_tps_mean:.1f} tok/s",
         f"max={s.generation_tps_max:.1f} tok/s"),
        ("Proc TPS mean",     f"{s.processing_tps_mean:.1f} tok/s", "input ingestion"),
        ("Latency mean",      f"{s.latency_mean:.3f}s",
         f"p50={s.latency_p50:.3f}s  p95={s.latency_p95:.3f}s"),
    ]
    for label, val, note in perf_rows:
        note_str = f"  ← {note}" if note else ""
        print(f"  {label:<22} {val:<22}{note_str}")

    print(f"\n  {'─'*62}")
    print("  Throughput")
    print(f"  {'─'*62}")
    print(f"  {'System TPS':<22} {s.system_throughput_tps:.1f} tok/s   ← output tokens / wall time")
    print(f"  {'Total TPS':<22} {s.total_throughput_tps:.1f} tok/s   ← (input+output) / wall time")
    print(f"  {'Req/sec':<22} {s.requests_per_second:.2f}")

    # Resource usage
    res = s.resources
    if res.mlx_peak_memory_gb > 0 or res.system_memory_total_gb > 0:
        print(f"\n  {'─'*62}")
        print("  Resource Usage")
        print(f"  {'─'*62}")
        if res.mlx_peak_memory_gb > 0:
            print(f"  {'Metal peak':<22} {res.mlx_peak_memory_gb:.2f} GB")
        if res.mlx_active_memory_gb > 0:
            print(f"  {'Metal active':<22} {res.mlx_active_memory_gb:.2f} GB")
        if res.mlx_cache_gb > 0:
            print(f"  {'Metal cache (KV)':<22} {res.mlx_cache_gb:.2f} GB")
        if res.process_memory_gb > 0:
            print(f"  {'Process RSS':<22} {res.process_memory_gb:.2f} GB")
        if res.system_memory_total_gb > 0:
            pct = (res.system_memory_used_gb / res.system_memory_total_gb) * 100
            print(f"  {'System RAM':<22} {res.system_memory_used_gb:.1f} / {res.system_memory_total_gb:.0f} GB  ({pct:.0f}%)")
        if res.system_power_w > 0:
            print(f"  {'Power (peak)':<22} {res.system_power_w:.1f}W  (CPU {res.cpu_power_w:.1f}W  GPU {res.gpu_power_w:.1f}W  {res.gpu_temp_c:.0f}°C)")

    # Per-request breakdown
    print(f"\n  {'─'*62}")
    print("  Per-Request Breakdown")
    print(f"  {'─'*62}")
    print(f"  {'#':<4} {'In':>6} {'Out':>6} {'TTFT':>10} {'TPOT':>10} {'Gen TPS':>9}  Status")
    for i, r in enumerate(s.results, 1):
        if r.error:
            print(f"  {i:<4} {'—':>6} {'—':>6} {'—':>10} {'—':>10} {'—':>9}  ✗ {r.error[:36]}")
        else:
            print(
                f"  {i:<4} {r.prompt_tokens:>6} {r.generated_tokens:>6} "
                f"{r.ttft*1000:>9.1f}ms {r.tpot*1000:>9.2f}ms "
                f"{r.generation_tps:>8.1f}  ✓"
            )
    print(sep)


def print_comparison(batched: BenchmarkSummary, sequential: BenchmarkSummary) -> None:
    sep = "═" * 66
    bt = batched.total_time or 1
    st = sequential.total_time or 1
    wall_ratio = st / bt
    tp_ratio   = (batched.system_throughput_tps / sequential.system_throughput_tps
                  if sequential.system_throughput_tps > 0 else 0)

    print(f"\n{sep}")
    print("  BATCHED vs SEQUENTIAL  —  COMPARISON")
    print(sep)
    print(f"  {'Metric':<32} {'Sequential':>14} {'Batched (CB)':>14}")
    print(f"  {'─'*60}")

    def row(label: str, seq_val: str, bat_val: str) -> None:
        print(f"  {label:<32} {seq_val:>14} {bat_val:>14}")

    row("Wall time",
        f"{sequential.total_time:.2f}s",
        f"{batched.total_time:.2f}s")
    row("Mean TTFT",
        f"{sequential.ttft_mean*1000:.0f}ms",
        f"{batched.ttft_mean*1000:.0f}ms")
    row("P95 TTFT",
        f"{sequential.ttft_p95*1000:.0f}ms",
        f"{batched.ttft_p95*1000:.0f}ms")
    row("Mean TPOT",
        f"{sequential.tpot_mean*1000:.2f}ms",
        f"{batched.tpot_mean*1000:.2f}ms")
    row("Mean Gen TPS (per-req)",
        f"{sequential.generation_tps_mean:.1f} tok/s",
        f"{batched.generation_tps_mean:.1f} tok/s")
    row("System throughput",
        f"{sequential.system_throughput_tps:.1f} tok/s",
        f"{batched.system_throughput_tps:.1f} tok/s")

    print(f"  {'─'*60}")
    faster = "faster" if wall_ratio >= 1 else "slower"
    print(f"  Wall time speedup:   {wall_ratio:.2f}x  (CB is {faster})")
    print(f"  Throughput gain:     {tp_ratio:.2f}x")
    print(sep)


def print_concurrency_table(summaries: list[BenchmarkSummary]) -> None:
    sep = "═" * 80
    print(f"\n{sep}")
    print("  CONTINUOUS BATCHING  —  CONCURRENCY SWEEP")
    print(f"  {summaries[0].model_name}")
    print(sep)
    print(
        f"  {'Concurrency':<14} {'Wall Time':>12} {'Sys TPS':>12} "
        f"{'Mean TTFT':>12} {'P95 TTFT':>10} {'Req/s':>8}"
    )
    print(f"  {'─'*76}")
    for s in summaries:
        print(
            f"  {s.concurrency:<14} {s.total_time:>11.2f}s "
            f"{s.system_throughput_tps:>11.1f} "
            f"{s.ttft_mean*1000:>11.0f}ms "
            f"{s.ttft_p95*1000:>9.0f}ms "
            f"{s.requests_per_second:>7.2f}"
        )
    if len(summaries) >= 2:
        base = summaries[0]
        print(f"\n  Throughput scaling vs concurrency={base.concurrency}:")
        for s in summaries[1:]:
            gain = (s.system_throughput_tps / base.system_throughput_tps
                    if base.system_throughput_tps > 0 else 0)
            print(f"    concurrency {s.concurrency:>3}:  {gain:.2f}x throughput")
    print(sep)


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _metrics_to_dict(r: ResourceMetrics) -> dict[str, Any]:
    return {
        "process_memory_gb":     r.process_memory_gb,
        "mlx_active_memory_gb":  r.mlx_active_memory_gb,
        "mlx_cache_gb":          r.mlx_cache_gb,
        "mlx_peak_memory_gb":    r.mlx_peak_memory_gb,
        "system_memory_used_gb": r.system_memory_used_gb,
        "system_memory_total_gb":r.system_memory_total_gb,
        "gpu_power_w":           r.gpu_power_w,
        "cpu_power_w":           r.cpu_power_w,
        "system_power_w":        r.system_power_w,
        "gpu_temp_c":            r.gpu_temp_c,
    }


def _result_to_dict(r: BenchmarkResult) -> dict[str, Any]:
    return {
        "prompt":          r.prompt,
        "prompt_tokens":   r.prompt_tokens,
        "generated_tokens":r.generated_tokens,
        "ttft_ms":         r.ttft * 1000,
        "total_time_s":    r.total_time,
        "tpot_ms":         r.tpot * 1000,
        "generation_tps":  r.generation_tps,
        "processing_tps":  r.processing_tps,
        "error":           r.error,
    }


def _summary_to_dict(s: BenchmarkSummary) -> dict[str, Any]:
    return {
        "model_name":           s.model_name,
        "mode":                 s.mode,
        "concurrency":          s.concurrency,
        "num_runs":             s.num_runs,
        "hardware": {
            "chip":      s.hardware_chip,
            "memory_gb": s.hardware_memory_gb,
        },
        "total_prompt_tokens":      s.total_prompt_tokens,
        "total_generated_tokens":   s.total_generated_tokens,
        "total_wall_time_s":        s.total_time,
        "ttft_ms": {
            "mean": s.ttft_mean * 1000,
            "min":  s.ttft_min  * 1000,
            "max":  s.ttft_max  * 1000,
            "p50":  s.ttft_p50  * 1000,
            "p95":  s.ttft_p95  * 1000,
        },
        "tpot_ms": {
            "mean": s.tpot_mean * 1000,
            "min":  s.tpot_min  * 1000,
            "max":  s.tpot_max  * 1000,
        },
        "tokens_per_second": {
            "generation_mean":   s.generation_tps_mean,
            "generation_max":    s.generation_tps_max,
            "processing_mean":   s.processing_tps_mean,
            "system_throughput": s.system_throughput_tps,
            "total_throughput":  s.total_throughput_tps,
        },
        "latency_seconds": {
            "mean": s.latency_mean,
            "min":  s.latency_min,
            "max":  s.latency_max,
            "p50":  s.latency_p50,
            "p95":  s.latency_p95,
        },
        "requests_per_second": s.requests_per_second,
        "resources":  _metrics_to_dict(s.resources),
        "results":    [_result_to_dict(r) for r in s.results],
    }


def save_json(summaries: list[BenchmarkSummary], path: str) -> None:
    payload: dict[str, Any] = {
        "type": "llm",
        "runs": [_summary_to_dict(s) for s in summaries],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Results saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bodega LLM Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model / server
    p.add_argument("--model", default="", metavar="MODEL_PATH",
                   help="HuggingFace repo ID or local path (auto load/unload)")
    p.add_argument("--model-id", default="bench-llm",
                   help="Model ID used in API requests (default: bench-llm)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help=f"Server base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--no-manage", action="store_true",
                   help="Skip auto load/unload — use a model already running in the server")

    # Workload
    p.add_argument("--prompts", type=int, default=5,
                   help="Number of prompts to use from the built-in suite (default: 5, max 10)")
    p.add_argument("--max-tokens", type=int, default=256,
                   help="Max output tokens per request (default: 256)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (default: 0.7)")
    p.add_argument("--warmup", type=int, default=1,
                   help="Warmup requests before measuring (default: 1)")

    # Concurrency
    p.add_argument("--concurrency", type=int, default=4,
                   help="Concurrent requests for a single run (default: 4)")
    p.add_argument("--concurrencies", default="",
                   help="Comma-separated concurrency levels for a sweep, e.g. 4,8,16,32")

    # CB tuning (passed to /v1/admin/load-model)
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--cb-max-num-seqs", type=int, default=256)
    p.add_argument("--cb-prefill-batch-size", type=int, default=8)
    p.add_argument("--cb-completion-batch-size", type=int, default=32)
    p.add_argument("--cb-chunked-prefill-tokens", type=int, default=2048)

    # Modes
    p.add_argument("--compare", action="store_true",
                   help="Run batched + sequential and print a side-by-side comparison")
    p.add_argument("--sequential-only", action="store_true",
                   help="Run sequential mode only (no continuous batching)")

    # Output
    p.add_argument("--output", default="",
                   help="Save full JSON results to this file")

    return p.parse_args()


async def _main() -> None:
    args = parse_args()

    if not args.model and not args.no_manage:
        print("  ✗  Provide --model MODEL_PATH  or  --no-manage --model-id LOADED_MODEL_ID")
        sys.exit(1)

    model_path = args.model
    model_id   = args.model_id if args.model_id != "bench-llm" else (
        model_path.split("/")[-1] if model_path else "bench-llm"
    )
    manage     = not args.no_manage
    base_url   = args.base_url.rstrip("/")

    # Prompt subset
    num_prompts = max(1, min(args.prompts, len(PROMPTS)))
    prompts = PROMPTS[:num_prompts]

    # Concurrency levels
    if args.concurrencies:
        concurrencies = [int(x.strip()) for x in args.concurrencies.split(",") if x.strip()]
    else:
        concurrencies = [args.concurrency]

    # Header
    chip, mem_gb = _detect_hardware_from_mactop()
    print("=" * 66)
    print("  Bodega LLM Performance Benchmark")
    print("=" * 66)
    print(f"  Server:       {base_url}")
    print(f"  Model:        {model_path or model_id}")
    print(f"  Hardware:     {chip} ({mem_gb:.0f} GB)" if chip else "  Hardware:     —")
    print(f"  Prompts:      {num_prompts}")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Concurrency:  {concurrencies}")
    print(f"  Mode:         {'sequential only' if args.sequential_only else 'batched (CB)' + (' + sequential comparison' if args.compare else '')}")
    print()

    open_mactop_window()

    all_summaries: list[BenchmarkSummary] = []

    common = dict(
        base_url=base_url,
        model_path=model_path,
        model_id=model_id,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        warmup_runs=args.warmup,
        manage_model_lifecycle=manage,
        cb_max_num_seqs=args.cb_max_num_seqs,
        cb_prefill_batch_size=args.cb_prefill_batch_size,
        cb_completion_batch_size=args.cb_completion_batch_size,
        cb_chunked_prefill_tokens=args.cb_chunked_prefill_tokens,
        context_length=args.context_length,
    )

    for conc in concurrencies:
        if not args.sequential_only:
            # ── Batched (CB) run ───────────────────────────────────────────
            s_batched = await run_benchmark(
                concurrency=conc, continuous_batching=True, **common
            )
            print_summary(s_batched)
            all_summaries.append(s_batched)

        if args.compare or args.sequential_only:
            # ── Sequential run ─────────────────────────────────────────────
            s_seq = await run_benchmark(
                concurrency=conc, continuous_batching=False, **common
            )
            print_summary(s_seq)
            all_summaries.append(s_seq)

            if not args.sequential_only:
                print_comparison(s_batched, s_seq)

    # Concurrency sweep table (batched only, multiple levels)
    batched_sweep = [s for s in all_summaries if s.mode == "batched"]
    if len(batched_sweep) > 1:
        print_concurrency_table(batched_sweep)

    if args.output:
        save_json(all_summaries, args.output)


if __name__ == "__main__":
    asyncio.run(_main())
