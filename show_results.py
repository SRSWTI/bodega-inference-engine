#!/usr/bin/env python3
"""
Benchmark Results Viewer
========================
Reads the saved sweep + compare JSON files and generates a self-contained
HTML report, then opens it in the default browser.

Usage:
    python show_results.py results/sweep_20260314_120000.json results/compare_20260314_120000.json
    python show_results.py          # auto-finds latest pair in ./results/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import webbrowser
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# JSON loaders
# ---------------------------------------------------------------------------

def _load(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def _find_latest(results_dir: str, prefix: str) -> str | None:
    pattern = os.path.join(results_dir, f"{prefix}_*.json")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_BADGE_COLORS = {
    "Bodega":   ("#00e5a0", "#003322"),
    "LMStudio": ("#f0a500", "#2a1a00"),
    "equal":    ("#8892a4", "#1a1e24"),
}


def _badge(text: str) -> str:
    for key, (bg, fg) in _BADGE_COLORS.items():
        if key in text:
            return f'<span class="badge" style="background:{bg};color:{fg}">{text}</span>'
    return f'<span class="badge">{text}</span>'


def _fmt(val: Any, suffix: str = "") -> str:
    if val is None or val == "":
        return "—"
    return f"{val}{suffix}"


def _winner_label(lm_val: float, bod_val: float, lower_is_better: bool = True) -> str:
    if bod_val <= 0 or lm_val <= 0:
        return "—"
    ratio = lm_val / bod_val
    THRESHOLD = 1.05
    if lower_is_better:
        if ratio > THRESHOLD:
            return f"Bodega {ratio:.1f}x"
        if ratio < 1 / THRESHOLD:
            return f"LMStudio {1/ratio:.1f}x"
        return "≈ equal"
    else:
        if ratio < 1 / THRESHOLD:
            return f"Bodega {1/ratio:.1f}x"
        if ratio > THRESHOLD:
            return f"LMStudio {ratio:.1f}x"
        return "≈ equal"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _hw_card(hw: dict[str, Any]) -> str:
    chip    = hw.get("chip") or hw.get("processor", "Unknown")
    mem_gb  = hw.get("memory_gb", "?")
    p_cores = hw.get("performance_cores")
    e_cores = hw.get("efficiency_cores")
    t_cores = hw.get("total_cores") or hw.get("cpu_count", "?")
    gpu     = hw.get("gpu_cores")
    model   = hw.get("model", "")

    cpu_str = f"{t_cores} cores"
    if p_cores and e_cores:
        cpu_str = f"{t_cores} cores ({p_cores}P + {e_cores}E)"

    rows = [
        ("Chip",       chip),
        ("Mac Model",  model or "—"),
        ("Memory",     f"{mem_gb} GB"),
        ("CPU",        cpu_str),
        ("GPU Cores",  f"{gpu}" if gpu else "—"),
    ]
    inner = "".join(
        f'<tr><td class="kv-key">{k}</td><td class="kv-val">{v}</td></tr>'
        for k, v in rows
    )
    return f'<div class="card"><h2>Device</h2><table class="kv">{inner}</table></div>'


def _meta_card(model: str, generated_at: str, source: str) -> str:
    try:
        dt = datetime.fromisoformat(generated_at)
        ts = dt.strftime("%b %d, %Y  %H:%M:%S")
    except Exception:
        ts = generated_at
    rows = [
        ("Model",    f'<code>{model}</code>'),
        ("Run at",   ts),
        ("Source",   source),
    ]
    inner = "".join(
        f'<tr><td class="kv-key">{k}</td><td class="kv-val">{v}</td></tr>'
        for k, v in rows
    )
    return f'<div class="card"><h2>Benchmark Run</h2><table class="kv">{inner}</table></div>'


def _sweep_section(data: dict[str, Any]) -> str:
    results  = data.get("results", [])
    best     = data.get("best_mixed")

    if not results:
        return '<div class="card"><h2>CB Sweep</h2><p>No results.</p></div>'

    # Find best row key for highlighting
    best_key = None
    if best:
        best_key = (best.get("concurrency"), best.get("prefill_batch"))

    # Group by scenario
    scenarios: dict[str, list[dict]] = {}
    for r in results:
        scenarios.setdefault(r["scenario"], []).append(r)

    html = '<div class="card"><h2>Continuous Batching Sweep</h2>'

    for scenario, rows in scenarios.items():
        html += f'<h3 class="scenario-title">{scenario}</h3>'
        html += '''<div class="table-wrap"><table>
        <thead><tr>
          <th>Concurrency</th>
          <th>Prefill Batch</th>
          <th>Mean TTFT (ms)</th>
          <th>P95 TTFT (ms)</th>
          <th>Per-Req TPS</th>
          <th>System Throughput (tok/s)</th>
        </tr></thead><tbody>'''

        for r in rows:
            is_best = (best_key == (r["concurrency"], r["prefill_batch"])
                       and scenario == "Mixed Queries")
            row_cls = ' class="best-row"' if is_best else ""
            star    = ' <span class="star" title="Best Mixed-Query Config">🏆</span>' if is_best else ""
            html += (
                f"<tr{row_cls}>"
                f"<td>{r['concurrency']}{star}</td>"
                f"<td>{r['prefill_batch']}</td>"
                f"<td>{r['mean_ttft_ms']}</td>"
                f"<td>{r['p95_ttft_ms']}</td>"
                f"<td>{r['per_req_tps']}</td>"
                f"<td>{r['system_throughput_tps']}</td>"
                f"</tr>"
            )
        html += "</tbody></table></div>"

    if best:
        html += (
            f'<div class="best-banner">'
            f'🏆 <strong>Best Mixed-Query Config:</strong> '
            f'Concurrency&nbsp;{best["concurrency"]}, Prefill-Batch&nbsp;{best["prefill_batch"]} '
            f'→ {best["system_throughput_tps"]} tok/s system throughput '
            f'(Mean TTFT: {best["mean_ttft_ms"]} ms)'
            f'</div>'
        )

    html += "</div>"
    return html


def _compare_section(data: dict[str, Any]) -> str:
    concurrencies  = data.get("concurrencies", [])
    lmstudio_runs  = data.get("lmstudio",   {})
    bodega_runs    = data.get("bodega_cb",  {})
    bod_configs    = data.get("bodega_optimal_configs", {})
    peak           = data.get("peak_throughput", {})

    if not concurrencies:
        return '<div class="card"><h2>Engine Comparison</h2><p>No results.</p></div>'

    METRICS = [
        ("TTFT mean (ms)",          "ttft_mean",           True,  1000),
        ("TTFT p50 (ms)",           "ttft_p50",            True,  1000),
        ("TTFT p95 (ms)",           "ttft_p95",            True,  1000),
        ("TPOT mean (ms/tok)",      "tpot_mean",           True,  1000),
        ("Gen TPS / request",       "generation_tps_mean", False, 1),
        ("System TPS (tok/s)",      "system_throughput_tps", False, 1),
        ("Wall time (s)",           "total_time",          True,  1),
        ("Req / sec",               "requests_per_second", False, 1),
    ]

    html = '<div class="card"><h2>Engine Comparison: LM Studio vs Bodega CB</h2>'

    # Per-concurrency tables
    for c in concurrencies:
        lm  = lmstudio_runs.get(str(c))
        bod = bodega_runs.get(str(c))
        pb  = bod_configs.get(str(c), "?")

        html += (
            f'<h3 class="scenario-title">'
            f'Concurrency = {c} &nbsp;│&nbsp; Bodega CB prefill-batch = {pb}'
            f'</h3>'
        )
        html += '''<div class="table-wrap"><table>
        <thead><tr>
          <th>Metric</th>
          <th>LM Studio</th>
          <th>Bodega CB</th>
          <th>Winner</th>
        </tr></thead><tbody>'''

        for label, key, lo, scale in METRICS:
            lv = (lm.get(key,  0) or 0) * scale if lm  else 0
            bv = (bod.get(key, 0) or 0) * scale if bod else 0
            lv_s  = f"{lv:.1f}"  if lv  else "—"
            bv_s  = f"{bv:.1f}"  if bv  else "—"
            win_s = _winner_label(lv, bv, lo) if (lv and bv) else "—"
            html += (
                f"<tr>"
                f"<td>{label}</td>"
                f"<td>{lv_s}</td>"
                f"<td>{bv_s}</td>"
                f"<td>{_badge(win_s)}</td>"
                f"</tr>"
            )

        html += "</tbody></table></div>"

    # Scorecard summary
    html += '<h3 class="scenario-title">Scorecard Summary</h3>'
    html += '''<div class="table-wrap"><table>
    <thead><tr><th>Metric</th>'''
    for c in concurrencies:
        html += f"<th>C={c}</th>"
    html += "<th>Overall</th></tr></thead><tbody>"

    for label, key, lo, scale in METRICS:
        html += f"<tr><td>{label}</td>"
        bod_wins = 0
        lm_wins  = 0
        for c in concurrencies:
            lm  = lmstudio_runs.get(str(c))
            bod = bodega_runs.get(str(c))
            lv  = (lm.get(key,  0) or 0) * scale if lm  else 0
            bv  = (bod.get(key, 0) or 0) * scale if bod else 0
            win = _winner_label(lv, bv, lo) if (lv and bv) else "—"
            if "Bodega" in win:
                bod_wins += 1
            elif "LMStudio" in win:
                lm_wins  += 1
            html += f"<td>{_badge(win)}</td>"
        if bod_wins > lm_wins:
            overall = "Bodega"
        elif lm_wins > bod_wins:
            overall = "LMStudio"
        else:
            overall = "≈ equal"
        html += f"<td>{_badge(overall)}</td></tr>"

    html += "</tbody></table></div>"

    # Peak throughput
    lm_peak  = peak.get("lmstudio_tok_per_s",  0)
    bod_peak = peak.get("bodega_cb_tok_per_s", 0)
    adv      = peak.get("bodega_advantage_x")
    if lm_peak or bod_peak:
        adv_str = ""
        if adv is not None:
            if adv >= 1:
                adv_str = f"Bodega is <strong>{adv:.2f}x</strong> higher throughput"
            else:
                adv_str = f"LM Studio is <strong>{1/adv:.2f}x</strong> higher throughput"
        html += (
            f'<div class="best-banner">'
            f'Peak throughput — LM Studio: <strong>{lm_peak:.0f} tok/s</strong> &nbsp;│&nbsp; '
            f'Bodega CB: <strong>{bod_peak:.0f} tok/s</strong>'
            f'{(" &nbsp;→&nbsp; " + adv_str) if adv_str else ""}'
            f'</div>'
        )

    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Full HTML page
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg:      #0d1117;
  --surface: #161b22;
  --border:  #30363d;
  --text:    #c9d1d9;
  --muted:   #8b949e;
  --accent:  #58a6ff;
  --green:   #3fb950;
  --gold:    #e3b341;
  --best-bg: #1a2a1a;
  --best-border: #3fb950;
  --radius:  8px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
  font-size: 14px;
  line-height: 1.6;
  padding: 32px 16px 64px;
}

h1 {
  font-size: 22px;
  font-weight: 600;
  color: #fff;
  margin-bottom: 6px;
  letter-spacing: -0.5px;
}

.subtitle {
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 28px;
}

.grid-top {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 16px;
  max-width: 860px;
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 24px;
  margin-bottom: 20px;
  max-width: 1100px;
}

.card h2 {
  font-size: 15px;
  font-weight: 600;
  color: #fff;
  margin-bottom: 14px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}

.scenario-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--accent);
  margin: 18px 0 8px;
  letter-spacing: 0.3px;
}

table.kv { width: 100%; border-collapse: collapse; }
table.kv td { padding: 4px 8px; }
.kv-key { color: var(--muted); width: 110px; }
.kv-val { color: var(--text); font-family: "SF Mono", monospace; font-size: 13px; }
.kv-val code { background: #1c2128; padding: 1px 5px; border-radius: 3px; font-size: 12px; }

.table-wrap { overflow-x: auto; margin-bottom: 4px; }

table:not(.kv) {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

table:not(.kv) thead tr {
  background: #1c2128;
}

table:not(.kv) th {
  padding: 8px 12px;
  text-align: left;
  color: var(--muted);
  font-weight: 600;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}

table:not(.kv) td {
  padding: 7px 12px;
  border-bottom: 1px solid #21262d;
  font-family: "SF Mono", monospace;
  white-space: nowrap;
}

table:not(.kv) tr:last-child td { border-bottom: none; }

table:not(.kv) tbody tr:hover { background: #1c2128; }

tr.best-row { background: var(--best-bg) !important; }
tr.best-row td { border-left: 2px solid var(--best-border); }
tr.best-row td:first-child { border-left: 2px solid var(--best-border); }

.star { font-size: 14px; margin-left: 4px; }

.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  background: #1c2128;
  color: var(--muted);
  white-space: nowrap;
}

.best-banner {
  margin-top: 14px;
  padding: 10px 14px;
  background: #1a2a1a;
  border: 1px solid var(--best-border);
  border-radius: 6px;
  font-size: 13px;
  color: var(--green);
}

@media (max-width: 680px) {
  .grid-top { grid-template-columns: 1fr; }
}
"""


def build_html(sweep: dict | None, compare: dict | None) -> str:
    # Extract common metadata
    model       = (sweep or compare or {}).get("model", "unknown")
    gen_at      = (sweep or compare or {}).get("generated_at", "")
    hw          = (sweep or compare or {}).get("hardware", {})
    source_tags = []
    if sweep:   source_tags.append("CB Sweep")
    if compare: source_tags.append("Engine Comparison")
    source = " + ".join(source_tags)

    hw_card   = _hw_card(hw)
    meta_card = _meta_card(model, gen_at, source)

    sweep_html   = _sweep_section(sweep)   if sweep   else ""
    compare_html = _compare_section(compare) if compare else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bodega Benchmark Results — {model}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Bodega Inference Engine — Benchmark Results</h1>
<p class="subtitle">Generated {datetime.now().strftime("%b %d, %Y at %H:%M:%S")}</p>

<div class="grid-top">
  {hw_card}
  {meta_card}
</div>

{sweep_html}
{compare_html}
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def upload_to_leaderboard(json_path: str, url: str) -> None:
    """POST a saved benchmark JSON to the leaderboard server."""
    import httpx
    target = url.rstrip("/") + "/api/upload"
    try:
        with open(json_path) as fh:
            payload = json.load(fh)
        r = httpx.post(target, json=payload, timeout=15.0)
        if r.status_code == 200:
            row_id = r.json().get("id", "?")
            print(f"  ✓ Uploaded to leaderboard (id={row_id}) → {url.rstrip('/')}")
        else:
            print(f"  ⚠ Leaderboard upload failed: {r.status_code} {r.text[:120]}")
    except Exception as exc:
        print(f"  ⚠ Leaderboard upload error: {exc}")


def main() -> None:
    p = argparse.ArgumentParser(description="Open benchmark results in the browser")
    p.add_argument("sweep",   nargs="?", default="", help="Path to sweep JSON")
    p.add_argument("compare", nargs="?", default="", help="Path to compare JSON")
    p.add_argument("--results-dir", default="results",
                   help="Directory to auto-find latest JSONs (default: results)")
    p.add_argument("--out", default="",
                   help="Save HTML to this file instead of a temp file")
    p.add_argument("--upload", default="",
                   help="Upload the result JSON(s) to this leaderboard URL")
    args = p.parse_args()

    sweep_path   = args.sweep
    compare_path = args.compare

    # Auto-find if not provided
    if not sweep_path:
        sweep_path = _find_latest(args.results_dir, "sweep") or ""
    if not compare_path:
        compare_path = _find_latest(args.results_dir, "compare") or ""

    if not sweep_path and not compare_path:
        print(f"No result JSONs found in '{args.results_dir}/'. "
              "Run a benchmark first, or pass paths as arguments.", file=sys.stderr)
        sys.exit(1)

    sweep_data   = _load(sweep_path)   if sweep_path   else None
    compare_data = _load(compare_path) if compare_path else None

    html = build_html(sweep_data, compare_data)

    if args.out:
        out_path = args.out
    else:
        # Write to results dir alongside the JSON files
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(results_dir, f"report_{ts}.html")

    with open(out_path, "w") as fh:
        fh.write(html)

    abs_path = os.path.abspath(out_path)
    print(f"\n  Report saved → {abs_path}")
    print("  Opening in browser...")
    webbrowser.open(f"file://{abs_path}")

    if args.upload:
        if sweep_path:
            upload_to_leaderboard(sweep_path, args.upload)
        if compare_path:
            upload_to_leaderboard(compare_path, args.upload)


if __name__ == "__main__":
    main()
