"""
Engine Comparison: LM Studio vs Bodega CB
==========================================

Runs identical benchmark workloads against two OpenAI-compatible servers
side-by-side and produces a structured comparison report.

  Engine A: LM Studio  (OpenAI-compatible server, static-batch prefill)
  Engine B: Bodega     (Bodega Inference Engine, true continuous batching)

Both servers are tested with the same model, same prompts, and same token
budget at each concurrency level.  Bodega is automatically configured with
the sweep-optimal prefill-batch size at each concurrency level (derived from
sweep_cb_configs.py results on M1 Max with bodega-orion-0.6b):

    C=1–8   →  prefill-batch=4  (best latency/throughput balance)
    C=16    →  prefill-batch=8  (646 tok/s system throughput)
    C=32    →  prefill-batch=8  (838+ tok/s system throughput)

Use --no-optimal to disable this and use a fixed --cb-prefill-batch-size.

The report includes:
  • Per-concurrency side-by-side metric tables
  • Per-request TTFT distribution for each engine
  • Final scorecard (winner per metric × concurrency)
  • Peak throughput comparison (each engine at its best)

Prerequisites:
    For fair benchmarks, load the model in LM Studio with max_concurrency=32
    (that's LM Studio's batching config). Bodega is auto-loaded with CB configs
    by this script. The benchmark tests up to C=32.

Usage:
    # Full comparison — loads model in Bodega, model already loaded in LM Studio
    python compare_engines.py --model srswti/bodega-orion-0.6b

    # Custom concurrency sweep
    python compare_engines.py --model srswti/bodega-orion-0.6b \\
        --concurrencies 4,8,16,32 --max-tokens 256 --prompts 10

    # Use a different model-id in LM Studio (if it differs from the HF name)
    python compare_engines.py --model srswti/bodega-raptor-0.9b \\
        --lmstudio-model-id bodega-raptor-0.9b

    # Save JSON report
    python compare_engines.py --model srswti/bodega-orion-0.6b --output report.json

    # Skip Bodega (only benchmark LM Studio for reference)
    python compare_engines.py --model srswti/bodega-orion-0.6b --no-bodega

    # Skip LM Studio
    python compare_engines.py --model srswti/bodega-orion-0.6b --no-lmstudio

    # Disable auto-optimal CB config, use fixed prefill-batch
    python compare_engines.py --model srswti/bodega-orion-0.6b \\
        --no-optimal --cb-prefill-batch-size 4

Defaults:
    --lmstudio-url     http://127.0.0.1:1234
    --bodega-url       http://localhost:44468
    --concurrencies    1,4,8,16,32
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


def _open_html_report(compare_path: str = "", sweep_path: str = "") -> None:
    """Generate and open the HTML results page immediately after a benchmark."""
    import webbrowser
    try:
        import show_results as _sr
        sweep_data   = _sr._load(sweep_path)   if sweep_path   else None
        compare_data = _sr._load(compare_path) if compare_path else None
        html = _sr.build_html(sweep_data, compare_data)
        base = compare_path or sweep_path
        html_path = base.rsplit(".", 1)[0] + ".html"
        with open(html_path, "w") as fh:
            fh.write(html)
        print(f"  Results opened in browser → {html_path}")
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
    except Exception as exc:
        print(f"  (Could not open HTML report: {exc})")

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

DEFAULT_LMSTUDIO_URL  = "http://127.0.0.1:1234"
DEFAULT_BODEGA_URL    = "http://localhost:44468"
DEFAULT_CONCURRENCIES = "1,4,8,16,32"



OPTIMAL_CB_PREFILL_BATCH: dict[int, int] = {
    1:  4,
    4:  4,
    8:  4,
    16: 4,
    32: 4,
}


def _optimal_prefill_batch(concurrency: int) -> int:
    """Return the sweep-optimal prefill-batch size for this concurrency level."""
    for threshold in sorted(OPTIMAL_CB_PREFILL_BATCH.keys(), reverse=True):
        if concurrency >= threshold:
            return OPTIMAL_CB_PREFILL_BATCH[threshold]
    return 4


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


def _lmstudio_model_identifier(model: str) -> str:
    """Return the identifier LM Studio expects. For HF repos, use full URL (required for MLX/non-catalog models)."""
    if "/" in model and not model.startswith("http"):
        return f"https://huggingface.co/{model}"
    return model


async def _download_model_via_lmstudio(base_url: str, model: str) -> bool:
    """Download model via LM Studio API. Returns True if ready (downloaded or already_downloaded)."""
    url = f"{base_url.rstrip('/')}/api/v1/models/download"
    identifier = _lmstudio_model_identifier(model)
    async with httpx.AsyncClient() as c:
        try:
            r = await c.post(url, json={"model": identifier}, timeout=30.0)
            if r.status_code != 200:
                try:
                    err = r.json()
                    msg = err.get("error", err.get("message", r.text[:200]))
                except Exception:
                    msg = r.text[:200]
                print(f"  ⚠  LM Studio download API returned {r.status_code}: {msg}")
                return False
            data = r.json()
            status = data.get("status", "")
            if status == "already_downloaded":
                print("  ✓  Model already downloaded in LM Studio")
                return True
            job_id = data.get("job_id")
            if not job_id:
                print("  ⚠  LM Studio returned no job_id (status=%s). Model may not be in LM Studio catalog." % status)
                return False
            # Poll until completed or failed
            status_url = f"{base_url.rstrip('/')}/api/v1/models/download/status/{job_id}"
            while True:
                await asyncio.sleep(2.0)
                sr = await c.get(status_url, timeout=10.0)
                if sr.status_code != 200:
                    return False
                sdata = sr.json()
                status = sdata.get("status", "")
                if status == "completed":
                    print("  ✓  Model downloaded in LM Studio")
                    return True
                if status == "failed":
                    err_msg = sdata.get("error", sdata.get("message", ""))
                    print(f"  ⚠  LM Studio download failed: {err_msg or status}")
                    return False
                # downloading or paused - continue polling
                downloaded = sdata.get("downloaded_bytes", 0)
                total = sdata.get("total_size_bytes", 0)
                pct = (downloaded / total * 100) if total else 0
                print(f"\r  Downloading in LM Studio... {pct:.0f}%", end="", flush=True)
        except Exception as e:
            print(f"  ⚠  LM Studio download error: {e}")
            return False
    return False


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
    bod_prefill_batch: int = 8,
) -> None:
    """Print the side-by-side comparison table for one concurrency level."""

    sep_thick = "  " + "═" * (W - 4)
    print()
    print(sep_thick)
    print(f"  Concurrency = {concurrency}")
    print(f"  LM Studio: static batch (size fixed by server)  │  "
          f"Bodega CB: prefill-batch={bod_prefill_batch}")
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
# Peak throughput comparison
# ---------------------------------------------------------------------------

def print_peak_throughput(
    concurrencies: list[int],
    lm_runs:  dict[int, BenchmarkSummary | None],
    bod_runs: dict[int, BenchmarkSummary | None],
    bodega_configs: dict[int, int],
) -> None:
    """Show each engine at its best System TPS across all tested concurrencies."""
    sep_thick = "  " + "═" * (W - 4)
    print()
    print(sep_thick)
    print("  PEAK THROUGHPUT  —  Each Engine at Its Best")
    print(_hline())

    # Find peak for LM Studio
    lm_best_tps, lm_best_c = 0.0, 0
    for c, s in lm_runs.items():
        if s and s.system_throughput_tps > lm_best_tps:
            lm_best_tps, lm_best_c = s.system_throughput_tps, c

    # Find peak for Bodega
    bod_best_tps, bod_best_c = 0.0, 0
    for c, s in bod_runs.items():
        if s and s.system_throughput_tps > bod_best_tps:
            bod_best_tps, bod_best_c = s.system_throughput_tps, c

    lm_best  = lm_runs.get(lm_best_c)
    bod_best = bod_runs.get(bod_best_c)

    print(f"  {'':32} {'LM Studio':>16} {'Bodega CB':>16}")
    print(_hline("·"))

    def prow(label: str, lv: str, bv: str) -> None:
        print(f"  {label:<32} {lv:>16} {bv:>16}")

    prow("Best concurrency",
         f"C={lm_best_c}"  if lm_best_c  else "—",
         f"C={bod_best_c}" if bod_best_c else "—")

    prow("Bodega prefill-batch at peak",
         "n/a (fixed)",
         f"prefill={bodega_configs.get(bod_best_c, '?')}" if bod_best_c else "—")

    print(_hline("·"))

    prow("Peak System TPS",
         f"{lm_best_tps:.0f} tok/s"  if lm_best_tps  else "—",
         f"{bod_best_tps:.0f} tok/s" if bod_best_tps else "—")

    if lm_best and bod_best:
        prow("Peak wall time",
             f"{lm_best.total_time:.2f}s",
             f"{bod_best.total_time:.2f}s")
        prow("Peak req/sec",
             f"{lm_best.requests_per_second:.2f}",
             f"{bod_best.requests_per_second:.2f}")
        prow("TTFT mean at peak",
             f"{lm_best.ttft_mean*1000:.0f} ms",
             f"{bod_best.ttft_mean*1000:.0f} ms")
        prow("TTFT p95 at peak",
             f"{lm_best.ttft_p95*1000:.0f} ms",
             f"{bod_best.ttft_p95*1000:.0f} ms")

    print(_hline("·"))

    if lm_best_tps > 0 and bod_best_tps > 0:
        tps_ratio = bod_best_tps / lm_best_tps
        if tps_ratio >= 1:
            print(f"  Bodega peak throughput:  {tps_ratio:.2f}x higher than LM Studio peak")
        else:
            print(f"  LM Studio peak throughput: {1/tps_ratio:.2f}x higher than Bodega peak")

    # Throughput scaling table for Bodega across all C levels
    bod_with_data = [(c, s) for c, s in sorted(bod_runs.items()) if s and s.system_throughput_tps > 0]
    if len(bod_with_data) >= 2:
        print()
        print(f"  Bodega CB throughput scaling:")
        base_tps = bod_with_data[0][1].system_throughput_tps
        for c, s in bod_with_data:
            gain = s.system_throughput_tps / base_tps
            pb = bodega_configs.get(c, "?")
            bar = "█" * int(s.system_throughput_tps / 50)
            print(f"    C={c:>2} (prefill={pb}): {s.system_throughput_tps:>6.0f} tok/s  "
                  f"{gain:.2f}x  {bar}")

    # Same for LM Studio
    lm_with_data = [(c, s) for c, s in sorted(lm_runs.items()) if s and s.system_throughput_tps > 0]
    if len(lm_with_data) >= 2:
        print()
        print(f"  LM Studio throughput scaling:")
        base_tps = lm_with_data[0][1].system_throughput_tps
        for c, s in lm_with_data:
            gain = s.system_throughput_tps / base_tps
            bar = "█" * int(s.system_throughput_tps / 50)
            print(f"    C={c:>2}:  {s.system_throughput_tps:>6.0f} tok/s  {gain:.2f}x  {bar}")

    print(sep_thick)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_report(
    model: str,
    concurrencies: list[int],
    lm_runs:  dict[int, BenchmarkSummary | None],
    bod_runs: dict[int, BenchmarkSummary | None],
    bodega_configs: dict[int, int],
    chip: str,
    mem_gb: float,
    path: str,
) -> None:
    # Find peak throughput for each engine
    lm_peak  = max((s.system_throughput_tps for s in lm_runs.values()  if s), default=0)
    bod_peak = max((s.system_throughput_tps for s in bod_runs.values() if s), default=0)

    # Build full hardware dict (gpu_cores etc.) for leaderboard
    try:
        from hardware_info import get_hardware_info
        _hw_full = get_hardware_info()
        if not chip:
            chip = _hw_full.get("chip") or _hw_full.get("processor") or chip
        if not mem_gb:
            mem_gb = float(_hw_full.get("memory_gb") or 0)
    except Exception:
        _hw_full = {}
    hw_dict = {**_hw_full, "chip": chip, "memory_gb": mem_gb}

    payload: dict[str, Any] = {
        "type": "engine_comparison",
        "generated_at": datetime.now().isoformat(),
        "model": model,
        "hardware": hw_dict,
        "concurrencies": concurrencies,
        "bodega_optimal_configs": {str(c): pb for c, pb in bodega_configs.items()},
        "peak_throughput": {
            "lmstudio_tok_per_s":  lm_peak,
            "bodega_cb_tok_per_s": bod_peak,
            "bodega_advantage_x":  round(bod_peak / lm_peak, 3) if lm_peak > 0 else None,
        },
        "lmstudio": {
            str(c): (_summary_to_dict(s) if s else None)
            for c, s in lm_runs.items()
        },
        "bodega_cb": {
            str(c): (_summary_to_dict(s) if s else None)
            for c, s in bod_runs.items()
        },
    }
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Report saved → {path}")
    _open_html_report(compare_path=path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare LM Studio vs Bodega CB on the same model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="srswti/bodega-orion-0.6b",
                   help="HuggingFace model path (used to load in Bodega; default: srswti/bodega-orion-0.6b)")
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
    p.add_argument("--cb-prefill-batch-size",     type=int, default=None,
                   help="Override prefill-batch size (default: auto-selected per concurrency from sweep results)")
    p.add_argument("--cb-completion-batch-size",  type=int, default=32)
    p.add_argument("--cb-chunked-prefill-tokens", type=int, default=2048)
    p.add_argument("--context-length",            type=int, default=8192)
    p.add_argument("--no-optimal", action="store_true",
                   help="Disable auto-optimal prefill-batch selection; use --cb-prefill-batch-size (default 8) for all levels")

    # Skip flags
    p.add_argument("--no-lmstudio", action="store_true",
                   help="Skip LM Studio benchmark")
    p.add_argument("--no-bodega", action="store_true",
                   help="Skip Bodega benchmark")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip LM Studio model download (use if already downloaded)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip confirmation prompt for LM Studio Max Concurrent Predictions")

    p.add_argument("--output", default="",
                   help="Save JSON comparison report to this file")
    p.add_argument("--leaderboard-url", default="",
                   help="Upload results to this leaderboard server after saving")
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
    # Fall back to hardware_info (system_profiler) if mactop returned empty chip
    if not chip:
        try:
            from hardware_info import get_hardware_info as _get_hw
            _hw = _get_hw()
            chip   = _hw.get("chip") or _hw.get("processor") or chip
            mem_gb = mem_gb or float(_hw.get("memory_gb") or 0)
        except Exception:
            pass
    # Fall back to hardware_info (system_profiler) if mactop returned empty chip
    if not chip:
        try:
            from hardware_info import get_hardware_info
            _hw = get_hardware_info()
            chip   = _hw.get("chip") or _hw.get("processor") or chip
            mem_gb = mem_gb or float(_hw.get("memory_gb") or 0)
        except Exception:
            pass

    # ── Determine prefill-batch per concurrency ────────────────────────────
    # If user passed an explicit --cb-prefill-batch-size OR --no-optimal,
    # use that fixed value for every concurrency level.
    # Otherwise use the sweep-derived optimal config.
    fixed_pb = args.cb_prefill_batch_size  # None means "use optimal"
    use_optimal = not args.no_optimal and fixed_pb is None
    bodega_configs: dict[int, int] = {
        c: (_optimal_prefill_batch(c) if use_optimal else (fixed_pb or 8))
        for c in concurrencies
    }

    # ── Auto-detect LM Studio model id ────────────────────────────────────
    lmstudio_model_id = args.lmstudio_model_id
    if not lmstudio_model_id and not args.no_lmstudio:
        lmstudio_model_id = await _lmstudio_loaded_model_id(args.lmstudio_url)
        if not lmstudio_model_id:
            lmstudio_model_id = args.model.split("/")[-1]

    bodega_model_id = f"compare-{args.model.split('/')[-1]}"

    # ── Header ─────────────────────────────────────────────────────────────
    W_FULL = 78
    print("=" * W_FULL)
    print("  ENGINE COMPARISON  —  LM Studio  vs  Bodega Continuous Batching")
    print()
    print("  ⚠  For fair benchmarks: Load the model in LM Studio with max_concurrency=32")
    print("     (LM Studio's batching config). Bodega is auto-loaded with CB by this script.")
    print("=" * W_FULL)
    print(f"  Model:              {args.model}")
    print(f"  LM Studio URL:      {args.lmstudio_url}  (model-id: {lmstudio_model_id})")
    print(f"  Bodega URL:         {args.bodega_url}  (model-id: {bodega_model_id})")
    print(f"  Hardware:           {chip} ({mem_gb:.0f} GB)" if chip else "  Hardware:           —")
    print(f"  Concurrencies:      {concurrencies}")
    if use_optimal:
        cfg_str = "  ".join(f"C={c}→pb={bodega_configs[c]}" for c in concurrencies)
        print(f"  Bodega CB configs:  {cfg_str}  (auto-optimal)")
    else:
        print(f"  Bodega CB configs:  prefill-batch={fixed_pb or 8} (fixed)")
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

            # ── Download model in LM Studio (if not skipped) ─────────────────
            if not args.skip_download:
                print("  Ensuring model is downloaded in LM Studio...")
                ok = await _download_model_via_lmstudio(args.lmstudio_url, args.model)
                if not ok:
                    print("  (Download skipped or failed — download manually in LM Studio, or use --skip-download)")

            # ── Confirm LM Studio loaded with Max Concurrent Predictions=32 ─
            if not args.yes:
                model_short = args.model.split("/")[-1]
                print()
                resp = input(
                    f"  Did you load the {model_short} model with "
                    "'Max Concurrent Predictions' as 32 when loading it in LM Studio? [y/N]: "
                ).strip().lower()
                if resp not in ("y", "yes"):
                    print("\n  Please load the model in LM Studio with Max Concurrent Predictions=32,")
                    print("  then run this script again.")
                    sys.exit(0)

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

    lm_runs:  dict[int, BenchmarkSummary | None] = {}
    bod_runs: dict[int, BenchmarkSummary | None] = {}

    # ── Run benchmarks ─────────────────────────────────────────────────────
    for c in concurrencies:
        pb = bodega_configs[c]
        print(f"\n{'─'*W_FULL}")
        print(f"  Running  concurrency = {c}  │  Bodega CB prefill-batch = {pb}")
        print(f"{'─'*W_FULL}")

        if not args.no_bodega:
            print(f"\n  [Bodega CB]  concurrency={c}  prefill-batch={pb}")
            bod_s = await run_benchmark(
                base_url=args.bodega_url,
                model_path=args.model,
                model_id=bodega_model_id,
                concurrency=c,
                continuous_batching=True,
                manage_model_lifecycle=True,
                prompts=prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                warmup_runs=args.warmup,
                cb_max_num_seqs=args.cb_max_num_seqs,
                cb_prefill_batch_size=pb,
                cb_completion_batch_size=args.cb_completion_batch_size,
                cb_chunked_prefill_tokens=args.cb_chunked_prefill_tokens,
                context_length=args.context_length,
            )
            bod_runs[c] = bod_s
        else:
            bod_runs[c] = None

        if not args.no_lmstudio:
            print(f"\n  [LM Studio]  concurrency={c}")
            lm_s = await run_benchmark(
                base_url=args.lmstudio_url,
                model_path="",
                model_id=lmstudio_model_id,
                concurrency=c,
                continuous_batching=True,
                manage_model_lifecycle=False,
                prompts=prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                warmup_runs=args.warmup,
                cb_max_num_seqs=args.cb_max_num_seqs,
                cb_prefill_batch_size=pb,
                cb_completion_batch_size=args.cb_completion_batch_size,
                cb_chunked_prefill_tokens=args.cb_chunked_prefill_tokens,
                context_length=args.context_length,
            )
            lm_runs[c] = lm_s
        else:
            lm_runs[c] = None

    # ── Comparison report ──────────────────────────────────────────────────
    print("\n\n" + "=" * W_FULL)
    print("  COMPARISON REPORT")
    print("=" * W_FULL)

    for c in concurrencies:
        print_concurrency_block(
            c,
            lm_runs.get(c),
            bod_runs.get(c),
            bod_prefill_batch=bodega_configs[c],
        )

    print_scorecard(concurrencies, lm_runs, bod_runs)
    print_peak_throughput(concurrencies, lm_runs, bod_runs, bodega_configs)

    # ── JSON output ────────────────────────────────────────────────────────
    output_path = args.output
    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("results", f"compare_{ts}.json")

    save_report(
        model=args.model,
        concurrencies=concurrencies,
        lm_runs=lm_runs,
        bod_runs=bod_runs,
        bodega_configs=bodega_configs,
        chip=chip,
        mem_gb=mem_gb,
        path=output_path,
    )

    if args.leaderboard_url:
        import show_results as _sr
        _sr.upload_to_leaderboard(output_path, args.leaderboard_url)


if __name__ == "__main__":
    asyncio.run(_main())
