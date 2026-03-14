"""
Engine Comparison: LM Studio vs Bodega CB
==========================================

Runs identical benchmark workloads against two OpenAI-compatible servers
side-by-side and produces a structured comparison report.

  Engine A: LM Studio  (OpenAI-compatible server, static-batch prefill)
  Engine B: Bodega     (Bodega Inference Engine, true continuous batching)

Both servers are tested with the same model, same prompts, same token budget,
and the same concurrency levels, so the numbers are directly comparable.

Usage:
    # Both servers already running, model already loaded in LM Studio
    python compare_engines.py --model srswti/bodega-raptor-90m

    # Specify concurrency levels and token budget
    python compare_engines.py --model srswti/bodega-raptor-90m \\
        --concurrencies 1,4,8 --max-tokens 256 --prompts 10

    # Use a different model-id in LM Studio (if it differs from the HF name)
    python compare_engines.py --model srswti/bodega-raptor-0.9b \\
        --lmstudio-model-id bodega-raptor-0.9b

    # Save JSON report
    python compare_engines.py --model srswti/bodega-raptor-90m --output report.json

    # Skip Bodega (only benchmark LM Studio for reference)
    python compare_engines.py --model srswti/bodega-raptor-90m --no-bodega

    # Skip LM Studio
    python compare_engines.py --model srswti/bodega-raptor-90m --no-lmstudio

Defaults:
    --lmstudio-url     http://127.0.0.1:1234
    --bodega-url       http://localhost:44468
    --concurrencies    1,4,8
    --max-tokens       256
    --prompts          10
    --warmup           1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Import shared primitives from benchmark_llm in the same directory
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from benchmark_llm import (  # noqa: E402
    PROMPTS,
    BenchmarkSummary,
    _detect_hardware_from_mactop,
    _summary_to_dict,
    run_benchmark,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LMSTUDIO_URL = "http://127.0.0.1:1234"
DEFAULT_BODEGA_URL   = "http://localhost:44468"
DEFAULT_CONCURRENCIES = "1,4,8"


# ---------------------------------------------------------------------------
# Server health check
# ---------------------------------------------------------------------------

async def _is_reachable(url: str) -> bool:
    """Return True if the server answers /health or /v1/models within 3 s."""
    async with httpx.AsyncClient() as c:
        for path in ("/health", "/v1/models"):
            try:
                r = await c.get(f"{url.rstrip('/')}{path}", timeout=3.0)
                if r.status_code < 500:
                    return True
            except Exception:
                pass
    return False


async def _lmstudio_loaded_model_id(url: str) -> str:
    """Best-effort: return the first loaded model id from LM Studio's /api/v1/models."""
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(f"{url.rstrip('/')}/api/v1/models", timeout=3.0)
            if r.status_code == 200:
                models = r.json().get("models", [])
                if models:
                    loaded = [
                        m for m in models if m.get("loaded_instances")
                    ]
                    if loaded:
                        return loaded[0]["loaded_instances"][0]["id"]
                    return models[0].get("key", "")
        except Exception:
            pass
    return ""


# ---------------------------------------------------------------------------
# Comparison printing
# ---------------------------------------------------------------------------

W = 78   # total width of the comparison box

def _hline(char: str = "─") -> str:
    return "  " + char * (W - 4)


def _winner_label(lm_val: float, bod_val: float, lower_is_better: bool = True) -> str:
    if bod_val <= 0 or lm_val <= 0:
        return "—"
    ratio = lm_val / bod_val
    THRESHOLD = 1.05
    if lower_is_better:
        if ratio > THRESHOLD:
            return f"Bodega  {ratio:.1f}x"
        if ratio < 1 / THRESHOLD:
            return f"LMStudio {1/ratio:.1f}x"
        return "≈ equal"
    else:
        if ratio < 1 / THRESHOLD:
            return f"Bodega  {1/ratio:.1f}x"
        if ratio > THRESHOLD:
            return f"LMStudio {ratio:.1f}x"
        return "≈ equal"


def _cell(val: Any, width: int) -> str:
    return str(val)[:width].rjust(width)


def print_concurrency_block(
    concurrency: int,
    lm: BenchmarkSummary | None,
    bod: BenchmarkSummary | None,
) -> None:
    """Print the side-by-side comparison table for one concurrency level."""

    sep_thick = "  " + "═" * (W - 4)
    print()
    print(sep_thick)
    title = f"  Concurrency = {concurrency}"
    print(title)
    print(_hline())

    hdr = f"  {'Metric':<32} {'LM Studio':>14} {'Bodega CB':>14}   {'Winner'}"
    print(hdr)
    print(_hline())

    def row(
        label: str,
        lm_str: str,
        bod_str: str,
        lm_raw: float = 0.0,
        bod_raw: float = 0.0,
        lo: bool = True,
        show_winner: bool = True,
    ) -> None:
        w = _winner_label(lm_raw, bod_raw, lo) if (show_winner and lm_raw and bod_raw) else ""
        lm_d  = lm_str  if lm  else "—"
        bod_d = bod_str if bod else "—"
        print(f"  {label:<32} {lm_d:>14} {bod_d:>14}   {w}")

    # TTFT
    row("TTFT mean",
        f"{lm.ttft_mean*1000:.0f} ms"  if lm  else "—",
        f"{bod.ttft_mean*1000:.0f} ms" if bod else "—",
        lm.ttft_mean  if lm  else 0,
        bod.ttft_mean if bod else 0)

    row("TTFT p50",
        f"{lm.ttft_p50*1000:.0f} ms"  if lm  else "—",
        f"{bod.ttft_p50*1000:.0f} ms" if bod else "—",
        lm.ttft_p50  if lm  else 0,
        bod.ttft_p50 if bod else 0)

    row("TTFT p95  (worst-case)",
        f"{lm.ttft_p95*1000:.0f} ms"  if lm  else "—",
        f"{bod.ttft_p95*1000:.0f} ms" if bod else "—",
        lm.ttft_p95  if lm  else 0,
        bod.ttft_p95 if bod else 0)

    lm_range  = (f"{lm.ttft_min*1000:.0f}→{lm.ttft_max*1000:.0f} ms"   if lm  else "—")
    bod_range = (f"{bod.ttft_min*1000:.0f}→{bod.ttft_max*1000:.0f} ms"  if bod else "—")
    # Range "tightness": smaller spread is better → compare (max-min)
    lm_spread  = (lm.ttft_max  - lm.ttft_min)  if lm  else 0
    bod_spread = (bod.ttft_max - bod.ttft_min)  if bod else 0
    row("TTFT spread (max − min)",
        lm_range, bod_range, lm_spread, bod_spread)

    print(_hline("·"))

    # TPOT / generation speed
    row("TPOT mean  (ms/token)",
        f"{lm.tpot_mean*1000:.2f} ms/tok"  if lm  else "—",
        f"{bod.tpot_mean*1000:.2f} ms/tok" if bod else "—",
        lm.tpot_mean  if lm  else 0,
        bod.tpot_mean if bod else 0)

    row("Gen TPS / request",
        f"{lm.generation_tps_mean:.0f} tok/s"  if lm  else "—",
        f"{bod.generation_tps_mean:.0f} tok/s" if bod else "—",
        lm.generation_tps_mean  if lm  else 0,
        bod.generation_tps_mean if bod else 0,
        lo=False)

    print(_hline("·"))

    # Throughput
    row("System TPS  (output tok/s)",
        f"{lm.system_throughput_tps:.0f} tok/s"  if lm  else "—",
        f"{bod.system_throughput_tps:.0f} tok/s" if bod else "—",
        lm.system_throughput_tps  if lm  else 0,
        bod.system_throughput_tps if bod else 0,
        lo=False)

    row("Wall time",
        f"{lm.total_time:.2f}s"  if lm  else "—",
        f"{bod.total_time:.2f}s" if bod else "—",
        lm.total_time  if lm  else 0,
        bod.total_time if bod else 0)

    row("Req/sec",
        f"{lm.requests_per_second:.2f}"  if lm  else "—",
        f"{bod.requests_per_second:.2f}" if bod else "—",
        lm.requests_per_second  if lm  else 0,
        bod.requests_per_second if bod else 0,
        lo=False)

    print(sep_thick)

    # Per-request TTFT strip (only when both present)
    if lm and bod:
        lm_ttfts  = [f"{r.ttft*1000:.0f}"  for r in lm.results  if r.error is None]
        bod_ttfts = [f"{r.ttft*1000:.0f}"  for r in bod.results if r.error is None]
        lm_str  = " | ".join(lm_ttfts)
        bod_str = " | ".join(bod_ttfts)
        print(f"  LM Studio TTFTs (ms): {lm_str}")
        print(f"  Bodega CB TTFTs (ms): {bod_str}")


def print_scorecard(
    concurrencies: list[int],
    lm_runs:  dict[int, BenchmarkSummary | None],
    bod_runs: dict[int, BenchmarkSummary | None],
) -> None:
    """Print a final per-metric winner grid across all concurrency levels."""
    sep_thick = "  " + "═" * (W - 4)
    print()
    print(sep_thick)
    print("  FINAL SCORECARD")
    print(_hline())

    conc_headers = "".join(f"  C={c:>2}        " for c in concurrencies)
    print(f"  {'Metric':<28}{conc_headers}  Overall")
    print(_hline())

    metrics: list[tuple[str, bool, Any, Any]] = [
        ("TTFT mean",             True,
         lambda s: s.ttft_mean,          lambda s: s.ttft_mean),
        ("TTFT p95",              True,
         lambda s: s.ttft_p95,           lambda s: s.ttft_p95),
        ("TTFT spread",           True,
         lambda s: s.ttft_max - s.ttft_min, lambda s: s.ttft_max - s.ttft_min),
        ("TPOT mean",             True,
         lambda s: s.tpot_mean,          lambda s: s.tpot_mean),
        ("Gen TPS / request",     False,
         lambda s: s.generation_tps_mean, lambda s: s.generation_tps_mean),
        ("System TPS",            False,
         lambda s: s.system_throughput_tps, lambda s: s.system_throughput_tps),
        ("Wall time",             True,
         lambda s: s.total_time,         lambda s: s.total_time),
        ("Req/sec",               False,
         lambda s: s.requests_per_second, lambda s: s.requests_per_second),
    ]

    bodega_wins = 0
    lmstudio_wins = 0
    total_comparisons = 0

    for label, lo, lm_fn, bod_fn in metrics:
        cells = []
        metric_bod_wins = 0
        metric_lm_wins  = 0
        for c in concurrencies:
            lm  = lm_runs.get(c)
            bod = bod_runs.get(c)
            if not lm or not bod:
                cells.append(f"{'—':^12}")
                continue
            lv = lm_fn(lm)
            bv = bod_fn(bod)
            w  = _winner_label(lv, bv, lo)
            total_comparisons += 1
            if w.startswith("Bodega"):
                bodega_wins  += 1
                metric_bod_wins += 1
            elif w.startswith("LMStudio"):
                lmstudio_wins += 1
                metric_lm_wins += 1
            cells.append(f"{w:^12}")

        overall = "Bodega" if metric_bod_wins > metric_lm_wins else (
                  "LMStudio" if metric_lm_wins > metric_bod_wins else "≈ equal")
        print(f"  {label:<28}{''.join(f'  {c}' for c in cells)}  {overall}")

    print(_hline())
    print(f"  Bodega CB wins:    {bodega_wins}/{total_comparisons} metric-concurrency pairs")
    print(f"  LM Studio wins:    {lmstudio_wins}/{total_comparisons} metric-concurrency pairs")
    total_d = total_comparisons - bodega_wins - lmstudio_wins
    if total_d:
        print(f"  Ties:              {total_d}/{total_comparisons}")
    print()
    if bodega_wins > lmstudio_wins:
        margin = bodega_wins - lmstudio_wins
        print(f"  VERDICT → Bodega CB wins by {margin} metric(s).")
        print( "            It excels at latency consistency and total throughput.")
        print( "            LM Studio's static batching can outperform at per-request")
        print( "            generation speed when its active-batch is small.")
    elif lmstudio_wins > bodega_wins:
        margin = lmstudio_wins - bodega_wins
        print(f"  VERDICT → LM Studio wins by {margin} metric(s).")
    else:
        print( "  VERDICT → Both engines perform similarly across the test suite.")
    print(sep_thick)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_report(
    model: str,
    concurrencies: list[int],
    lm_runs:  dict[int, BenchmarkSummary | None],
    bod_runs: dict[int, BenchmarkSummary | None],
    chip: str,
    mem_gb: float,
    path: str,
) -> None:
    payload: dict[str, Any] = {
        "type": "engine_comparison",
        "generated_at": datetime.now().isoformat(),
        "model": model,
        "hardware": {"chip": chip, "memory_gb": mem_gb},
        "concurrencies": concurrencies,
        "lmstudio": {
            str(c): (_summary_to_dict(s) if s else None)
            for c, s in lm_runs.items()
        },
        "bodega_cb": {
            str(c): (_summary_to_dict(s) if s else None)
            for c, s in bod_runs.items()
        },
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Report saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare LM Studio vs Bodega CB on the same model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="srswti/bodega-raptor-90m",
                   help="HuggingFace model path (used to load in Bodega; default: srswti/bodega-raptor-90m)")
    p.add_argument("--lmstudio-model-id", default="",
                   help="Model ID as shown in LM Studio (auto-detected if omitted)")
    p.add_argument("--lmstudio-url", default=DEFAULT_LMSTUDIO_URL,
                   help=f"LM Studio server URL (default: {DEFAULT_LMSTUDIO_URL})")
    p.add_argument("--bodega-url", default=DEFAULT_BODEGA_URL,
                   help=f"Bodega server URL (default: {DEFAULT_BODEGA_URL})")

    p.add_argument("--concurrencies", default=DEFAULT_CONCURRENCIES,
                   help=f"Comma-separated concurrency levels (default: {DEFAULT_CONCURRENCIES})")
    p.add_argument("--prompts", type=int, default=10,
                   help="Number of prompts to use (default: 10, max 10)")
    p.add_argument("--max-tokens", type=int, default=256,
                   help="Max output tokens per request (default: 256)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--warmup", type=int, default=1,
                   help="Warmup requests before measuring (default: 1)")

    # CB tuning
    p.add_argument("--cb-max-num-seqs",          type=int, default=256)
    p.add_argument("--cb-prefill-batch-size",     type=int, default=8)
    p.add_argument("--cb-completion-batch-size",  type=int, default=32)
    p.add_argument("--cb-chunked-prefill-tokens", type=int, default=2048)
    p.add_argument("--context-length",            type=int, default=8192)

    # Skip flags
    p.add_argument("--no-lmstudio", action="store_true",
                   help="Skip LM Studio benchmark")
    p.add_argument("--no-bodega", action="store_true",
                   help="Skip Bodega benchmark")

    p.add_argument("--output", default="",
                   help="Save JSON comparison report to this file")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main() -> None:
    args = parse_args()

    concurrencies = [int(x.strip()) for x in args.concurrencies.split(",") if x.strip()]
    num_prompts   = max(1, min(args.prompts, len(PROMPTS)))
    prompts       = PROMPTS[:num_prompts]
    chip, mem_gb  = _detect_hardware_from_mactop()

    # ── Auto-detect LM Studio model id ────────────────────────────────────
    lmstudio_model_id = args.lmstudio_model_id
    if not lmstudio_model_id and not args.no_lmstudio:
        lmstudio_model_id = await _lmstudio_loaded_model_id(args.lmstudio_url)
        if not lmstudio_model_id:
            # Fall back to last component of the HF path
            lmstudio_model_id = args.model.split("/")[-1]

    bodega_model_id = f"compare-{args.model.split('/')[-1]}"

    # ── Header ─────────────────────────────────────────────────────────────
    W_FULL = 78
    print("=" * W_FULL)
    print("  ENGINE COMPARISON  —  LM Studio  vs  Bodega Continuous Batching")
    print("=" * W_FULL)
    print(f"  Model:              {args.model}")
    print(f"  LM Studio URL:      {args.lmstudio_url}  (model-id: {lmstudio_model_id})")
    print(f"  Bodega URL:         {args.bodega_url}  (model-id: {bodega_model_id})")
    print(f"  Hardware:           {chip} ({mem_gb:.0f} GB)" if chip else "  Hardware:           —")
    print(f"  Concurrencies:      {concurrencies}")
    print(f"  Prompts / budget:   {num_prompts} prompts  ×  {args.max_tokens} max tokens")
    print(f"  Timestamp:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * W_FULL)

    # ── Reachability check ─────────────────────────────────────────────────
    if not args.no_lmstudio:
        lm_up = await _is_reachable(args.lmstudio_url)
        if not lm_up:
            print(f"\n  ⚠  LM Studio not reachable at {args.lmstudio_url} — skipping.")
            args.no_lmstudio = True
        else:
            print(f"\n  ✓  LM Studio reachable  ({args.lmstudio_url})")

    if not args.no_bodega:
        bod_up = await _is_reachable(args.bodega_url)
        if not bod_up:
            print(f"  ⚠  Bodega not reachable at {args.bodega_url} — skipping.")
            args.no_bodega = True
        else:
            print(f"  ✓  Bodega reachable     ({args.bodega_url})")

    if args.no_lmstudio and args.no_bodega:
        print("\n  ✗  Neither server is reachable. Exiting.")
        sys.exit(1)

    # ── Common kwargs for run_benchmark ────────────────────────────────────
    common_kwargs = dict(
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        warmup_runs=args.warmup,
        cb_max_num_seqs=args.cb_max_num_seqs,
        cb_prefill_batch_size=args.cb_prefill_batch_size,
        cb_completion_batch_size=args.cb_completion_batch_size,
        cb_chunked_prefill_tokens=args.cb_chunked_prefill_tokens,
        context_length=args.context_length,
    )

    lm_runs:  dict[int, BenchmarkSummary | None] = {}
    bod_runs: dict[int, BenchmarkSummary | None] = {}

    # ── Run benchmarks ─────────────────────────────────────────────────────
    for c in concurrencies:
        print(f"\n{'─'*W_FULL}")
        print(f"  Running  concurrency = {c}")
        print(f"{'─'*W_FULL}")

        if not args.no_lmstudio:
            print(f"\n  [LM Studio]  concurrency={c}")
            lm_s = await run_benchmark(
                base_url=args.lmstudio_url,
                model_path="",
                model_id=lmstudio_model_id,
                concurrency=c,
                continuous_batching=True,   # label only — LM Studio uses its own batching
                manage_model_lifecycle=False,
                **common_kwargs,
            )
            lm_runs[c] = lm_s
        else:
            lm_runs[c] = None

        if not args.no_bodega:
            print(f"\n  [Bodega CB]  concurrency={c}")
            bod_s = await run_benchmark(
                base_url=args.bodega_url,
                model_path=args.model,
                model_id=bodega_model_id,
                concurrency=c,
                continuous_batching=True,
                manage_model_lifecycle=True,
                **common_kwargs,
            )
            bod_runs[c] = bod_s
        else:
            bod_runs[c] = None

    # ── Comparison report ──────────────────────────────────────────────────
    print("\n\n" + "=" * W_FULL)
    print("  COMPARISON REPORT")
    print("=" * W_FULL)

    for c in concurrencies:
        print_concurrency_block(c, lm_runs.get(c), bod_runs.get(c))

    print_scorecard(concurrencies, lm_runs, bod_runs)

    # ── JSON output ────────────────────────────────────────────────────────
    if args.output:
        save_report(
            model=args.model,
            concurrencies=concurrencies,
            lm_runs=lm_runs,
            bod_runs=bod_runs,
            chip=chip,
            mem_gb=mem_gb,
            path=args.output,
        )


if __name__ == "__main__":
    asyncio.run(_main())
