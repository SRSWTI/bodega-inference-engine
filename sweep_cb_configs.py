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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:44468", help="Server base URL")
    parser.add_argument("--model", default="srswti/bodega-raptor-90m", help="Model to use for sweep")
    args = parser.parse_args()
    print("  [Telemetry] Opening mactop in a new Terminal window...")
    open_mactop_window()
    asyncio.run(run_sweep(args.base_url, args.model))
