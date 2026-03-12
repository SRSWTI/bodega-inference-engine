#!/usr/bin/env python3
"""
Continuous Batching Configuration Sweep (HTTP API)
==================================================
Runs a grid of CB configurations via the HTTP /v1/admin/load-model endpoint
to find the best parameters.
"""

import argparse
import asyncio
import io
import contextlib
import json
import subprocess
import os
import httpx
from tabulate import tabulate

from benchmark_continuous_batching import benchmark_batched_http, PROMPTS, manage_model, open_mactop_window

def get_telemetry():
    """Grab real-time Apple Silicon metrics from mactop --headless."""
    try:
        if os.system("command -v mactop >/dev/null 2>&1") == 0:
            res = subprocess.run(["mactop", "--headless", "--count", "1"],
                                 capture_output=True, text=True, timeout=2.0)
            if res.returncode == 0:
                data = json.loads(res.stdout)
                if isinstance(data, list) and len(data) > 0:
                    sm   = data[0].get("soc_metrics", {})
                    mem  = data[0].get("memory", {})
                    ru   = mem.get("used",   0) / (1024**3)
                    rt   = mem.get("total",  0) / (1024**3)
                    s_p  = sm.get("system_power", 0)
                    c_p  = sm.get("cpu_power",    0)
                    g_p  = sm.get("gpu_power",    0)
                    freq = sm.get("gpu_freq_mhz", 0)
                    temp = sm.get("gpu_temp",     0)
                    cpu_pct = data[0].get("cpu_usage", 0)
                    gpu_pct = data[0].get("gpu_usage", 0)
                    return (f"RAM {ru:.1f}/{rt:.0f}GB "
                            f"| CPU {cpu_pct:.0f}% GPU {gpu_pct:.0f}% @ {freq}MHz {temp:.0f}°C "
                            f"| Pwr {s_p:.1f}W (CPU {c_p:.1f}W GPU {g_p:.1f}W)")
    except Exception:
        pass
    return None

async def run_sweep(base_url: str, model: str):
    max_tokens = 128
    
    configs = [(8, 2), (8, 4), (8, 8), (16, 4), (16, 8), (16, 16), (32, 8), (32, 16)]
    mixed_prompts = PROMPTS
    same_prompt = [PROMPTS[6]]
    
    results_table = []
    
    print(f"\nStarting CB Configuration Sweep on M1 Max via HTTP ({base_url})...")
    print(f"Model: {model}")
    print(f"Max generation tokens per request: {max_tokens}\n")
        
    for scenario_name, prompt_list in [("Mixed Queries", mixed_prompts), ("Same Query (Prefix Cache Test)", same_prompt)]:
        print(f"=== Scenario: {scenario_name} ===")
        for concurrency, prefill_batch in configs:
            print(f"  Running Concurrency={concurrency}, PrefillBatch={prefill_batch}...", end=" ", flush=True)
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                summary = await benchmark_batched_http(
                    base_url=base_url,
                    model_path=model,
                    prompts=prompt_list,
                    concurrency=concurrency,
                    max_tokens=max_tokens,
                    cb_max_num_seqs=64, 
                    cb_prefill_batch_size=prefill_batch,
                    cb_chunked_prefill_tokens=2048,
                )
            
            # Extract and print any warnings that might have been swallowed
            captured_out = f.getvalue()
            if "Note: Continuous batching for 'multimodal' models is coming soon" in captured_out:
                print("\n  [!] Note: Continuous batching for 'multimodal' models is coming soon to Bodega.\n"
                      "      The engine currently falls back to sequential execution for vision models.", flush=True)

            telemetry = get_telemetry()
            telemetry_str = f"  [{telemetry}]" if telemetry else ""
            print(f"Done.{telemetry_str}", flush=True)

            results_table.append([
                scenario_name,
                concurrency,
                prefill_batch,
                f"{summary.mean_ttft_ms:.0f}",
                f"{summary.p95_ttft_ms:.0f}",
                f"{summary.mean_tps:.1f}",
                f"{summary.throughput_tps:.1f}"
            ])

    print("\n\n" + "=" * 85)
    print("  CONTINUOUS BATCHING SWEEP RESULTS")
    print("=" * 85)
    headers = ["Scenario", "Concurrency", "Prefill Batch", "Mean TTFT (ms)", "p95 TTFT (ms)", "Per-Req TPS", "System Throughput"]
    print(tabulate(results_table, headers=headers, tablefmt="github"))
    
    try:
        mixed_results = [r for r in results_table if r[0] == "Mixed Queries"]
        if mixed_results:
            best = max(mixed_results, key=lambda x: (float(x[6]), -float(x[3])))
            print("\n" + "=" * 85)
            print("  🏆 BEST REAL-WORLD CONFIGURATION (Mixed Queries)")
            print("=" * 85)
            print(f"  Concurrency:   {best[1]}")
            print(f"  Prefill Batch: {best[2]}")
            print(f"  → Yields highest System Throughput: {best[6]} tok/s  (Mean TTFT: {best[3]} ms)")
            print("=" * 85)
    except Exception:
        pass


async def run_sequential_multimodal(base_url: str, model: str):
    """Sequential mode for multimodal models — unload, reload with max_concurrency=3, then 3 requests one-by-one."""
    import httpx as _httpx
    import time as _time
    max_tokens = 128
    sample_prompts = [
        "What is the capital of France?",
        "Write a one-line Python function to reverse a string.",
        "Explain E=mc² in one sentence.",
    ]
    print(f"\nStarting SEQUENTIAL Throughput Test (Multimodal Model) via HTTP ({base_url})...")
    print(f"Model: {model}  |  Mode: Sequential (max_concurrency=3)")
    print(f"Requests: {len(sample_prompts)}  |  Running one-by-one\n")
    print("⚠  Continuous batching for multimodal is coming soon to Bodega. Running sequentially.\n")

    async with _httpx.AsyncClient() as client:
        # 1. Unload existing instance
        print(f"  [~] Unloading {model}...", end=" ", flush=True)
        try:
            r = await client.delete(f"{base_url}/v1/admin/unload-model/{model}", timeout=30)
            print("done." if r.status_code in [200, 204, 404] else f"status={r.status_code}", flush=True)
        except Exception as e:
            print(f"warn: {e}", flush=True)

        # 2. Reload with max_concurrency=3
        print(f"  [+] Reloading {model} with max_concurrency=3...", end=" ", flush=True)
        try:
            r = await client.post(f"{base_url}/v1/admin/load-model", json={
                "model_path": model, "model_id": model,
                "model_type": "multimodal",
                "max_concurrency": 3,
                "context_length": 8192,
            }, timeout=120)
            if r.status_code in [200, 201, 409]:
                print("ready.\n", flush=True)
            else:
                print(f"failed ({r.status_code}). Proceeding anyway.\n", flush=True)
        except Exception as e:
            print(f"error: {e}. Proceeding anyway.\n", flush=True)

        # 3. Run requests one by one using non-streaming for accurate TPS
        results_table = []
        url = f"{base_url}/v1/chat/completions"
        for i, prompt in enumerate(sample_prompts, 1):
            print(f"  Request #{i}: {prompt[:55]}... ", end="", flush=True)
            t0 = _time.perf_counter()
            try:
                resp = await client.post(url, json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "max_tokens": max_tokens,
                }, timeout=120)
                total_time = _time.perf_counter() - t0
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage", {})
                    completion_tokens = usage.get("completion_tokens", 0)
                    # TPS = completion tokens / total generation time
                    tps = completion_tokens / total_time if total_time > 0 else 0
                    print(f"Done. Time: {total_time*1000:.0f}ms | Tokens: {completion_tokens} | TPS: {tps:.1f} tok/s", flush=True)
                    results_table.append([f"Req #{i}", prompt[:38]+"...", f"{total_time*1000:.0f}ms", completion_tokens, f"{tps:.1f}"])
                else:
                    print(f"Error {resp.status_code}", flush=True)
                    results_table.append([f"Req #{i}", prompt[:38]+"...", "Error", 0, "0"])
            except Exception as e:
                print(f"Error: {e}", flush=True)
                results_table.append([f"Req #{i}", prompt[:38]+"...", "Error", 0, "0"])

    print("\n" + "=" * 75)
    print("  MULTIMODAL SEQUENTIAL RESULTS (3 requests, max_concurrency=3)")
    print("=" * 75)
    headers = ["Request", "Prompt", "Total Time", "Tokens Out", "TPS"]
    print(tabulate(results_table, headers=headers, tablefmt="github"))
    print("=" * 75)
    print("\n  ℹ  For full continuous batching, use an LM (language model) adapter.")
    print("     Multimodal continuous batching is coming soon to Bodega Inference Engine.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:44468", help="Server base URL")
    parser.add_argument("--model", default="srswti/bodega-raptor-90m", help="Model to use for sweep")
    parser.add_argument("--multimodal-sequential", action="store_true",
                        help="Run 3 sequential requests for multimodal models instead of CB sweep")
    args = parser.parse_args()
    print("  [Telemetry] Opening mactop in a new Terminal window...")
    open_mactop_window()
    if args.multimodal_sequential:
        asyncio.run(run_sequential_multimodal(args.base_url, args.model))
    else:
        asyncio.run(run_sweep(args.base_url, args.model))
