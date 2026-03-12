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
import httpx
from tabulate import tabulate

from benchmark_continuous_batching import benchmark_batched_http, PROMPTS, manage_model

async def run_sweep(base_url: str):
    model = "srswti/bodega-raptor-90m"
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
            
            results_table.append([
                scenario_name,
                concurrency,
                prefill_batch,
                f"{summary.mean_ttft_ms:.0f}",
                f"{summary.p95_ttft_ms:.0f}",
                f"{summary.mean_tps:.1f}",
                f"{summary.throughput_tps:.1f}"
            ])
            print("Done.")

    print("\n\n" + "=" * 85)
    print("  CONTINUOUS BATCHING SWEEP RESULTS")
    print("=" * 85)
    headers = ["Scenario", "Concurrency", "Prefill Batch", "Mean TTFT (ms)", "p95 TTFT (ms)", "Per-Req TPS", "System Throughput"]
    print(tabulate(results_table, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:44468", help="Server base URL")
    args = parser.parse_args()
    asyncio.run(run_sweep(args.base_url))
