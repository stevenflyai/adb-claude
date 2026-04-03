#!/usr/bin/env python3
"""
Azure Databricks + Claude Opus 4.6 — TTFT Stress Test (100 runs)
=================================================================
Sends 100 streaming requests and records TTFT, total latency, and TPS.
Generates charts: time series, histogram, and boxplot.

Usage:
  pip install openai python-dotenv matplotlib
  python test_ttft_stress.py
"""

import os
import time
import json
import statistics
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================
# Configuration
# ============================================================
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
DATABRICKS_BASE_URL = os.environ["DATABRICKS_BASE_URL"]
MODEL = "databricks-claude-opus-4-6"

TOTAL_RUNS = 100
MAX_TOKENS = 100
TEMPERATURE = 0.1

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Answer concisely in 1-2 sentences."},
    {"role": "user", "content": "What is cloud computing?"},
]

client = OpenAI(api_key=DATABRICKS_TOKEN, base_url=DATABRICKS_BASE_URL)


# ============================================================
# Measurement
# ============================================================
def measure_once() -> dict:
    """Single streaming request — returns TTFT, total latency, token count, TPS."""
    t_start = time.perf_counter()
    t_first = None
    tokens = 0

    stream = client.chat.completions.create(
        model=MODEL,
        messages=MESSAGES,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if t_first is None:
                t_first = time.perf_counter()
            tokens += 1

    t_end = time.perf_counter()
    ttft = (t_first - t_start) if t_first else None
    total = t_end - t_start
    gen_time = (t_end - t_first) if t_first else 0
    tps = tokens / gen_time if gen_time > 0 else 0

    return {"ttft": ttft, "total": total, "tokens": tokens, "tps": tps}


def percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


# ============================================================
# Main
# ============================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"TTFT Stress Test — {TOTAL_RUNS} runs")
    print(f"Model: {MODEL}")
    print(f"Endpoint: {DATABRICKS_BASE_URL}")
    print(f"Max tokens: {MAX_TOKENS}\n")

    results = []
    failures = 0

    for i in range(TOTAL_RUNS):
        idx = i + 1
        try:
            r = measure_once()
            if r["ttft"] is None:
                print(f"  [{idx:3d}/{TOTAL_RUNS}] FAIL: no tokens")
                failures += 1
                continue
            results.append(r)
            print(f"  [{idx:3d}/{TOTAL_RUNS}] TTFT={r['ttft']:.3f}s  "
                  f"Total={r['total']:.2f}s  Tokens={r['tokens']}  "
                  f"TPS={r['tps']:.1f}")
        except Exception as e:
            print(f"  [{idx:3d}/{TOTAL_RUNS}] ERROR: {e}")
            failures += 1

    if len(results) < 2:
        print("\nNot enough successful runs. Aborting.")
        return

    # ── Statistics ──
    ttfts = [r["ttft"] for r in results]
    totals = [r["total"] for r in results]
    tps_list = [r["tps"] for r in results]

    print(f"\n{'=' * 60}")
    print(f"  Results: {len(results)} success / {failures} failed")
    print(f"{'=' * 60}")
    for label, data in [("TTFT (s)", ttfts), ("Total (s)", totals), ("TPS", tps_list)]:
        print(f"\n  {label}:")
        print(f"    Min:    {min(data):.3f}")
        print(f"    P5:     {percentile(data, 5):.3f}")
        print(f"    P25:    {percentile(data, 25):.3f}")
        print(f"    Median: {statistics.median(data):.3f}")
        print(f"    Mean:   {statistics.mean(data):.3f}")
        print(f"    P75:    {percentile(data, 75):.3f}")
        print(f"    P95:    {percentile(data, 95):.3f}")
        print(f"    P99:    {percentile(data, 99):.3f}")
        print(f"    Max:    {max(data):.3f}")
        print(f"    Stdev:  {statistics.stdev(data):.3f}")

    # ── Save raw data ──
    json_path = f"ttft_stress_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({"config": {"model": MODEL, "runs": TOTAL_RUNS, "max_tokens": MAX_TOKENS},
                    "results": results}, f, indent=2)
    print(f"\nRaw data saved to {json_path}")

    # ── Charts ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"TTFT Stress Test — {MODEL}  ({len(results)} runs, {timestamp})",
                 fontsize=13, fontweight="bold")

    run_idx = list(range(1, len(results) + 1))

    # 1) TTFT time series
    ax = axes[0, 0]
    ax.plot(run_idx, ttfts, linewidth=0.8, alpha=0.7, color="#2196F3")
    ax.axhline(statistics.median(ttfts), color="#FF5722", linestyle="--", linewidth=1,
               label=f"median={statistics.median(ttfts):.3f}s")
    ax.axhline(percentile(ttfts, 95), color="#F44336", linestyle=":", linewidth=1,
               label=f"P95={percentile(ttfts, 95):.3f}s")
    ax.set_xlabel("Run #")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("TTFT over time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2) TTFT histogram
    ax = axes[0, 1]
    ax.hist(ttfts, bins=25, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(statistics.median(ttfts), color="#FF5722", linestyle="--", linewidth=1.5,
               label=f"median={statistics.median(ttfts):.3f}s")
    ax.axvline(percentile(ttfts, 95), color="#F44336", linestyle=":", linewidth=1.5,
               label=f"P95={percentile(ttfts, 95):.3f}s")
    ax.set_xlabel("TTFT (s)")
    ax.set_ylabel("Count")
    ax.set_title("TTFT distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3) Boxplot: TTFT + Total latency
    ax = axes[1, 0]
    bp = ax.boxplot([ttfts, totals], labels=["TTFT", "Total latency"],
                    patch_artist=True, widths=0.5)
    colors = ["#2196F3", "#4CAF50"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Seconds")
    ax.set_title("Latency boxplot")
    ax.grid(True, alpha=0.3, axis="y")

    # 4) TPS time series
    ax = axes[1, 1]
    ax.plot(run_idx, tps_list, linewidth=0.8, alpha=0.7, color="#4CAF50")
    ax.axhline(statistics.median(tps_list), color="#FF5722", linestyle="--", linewidth=1,
               label=f"median={statistics.median(tps_list):.1f} tok/s")
    ax.set_xlabel("Run #")
    ax.set_ylabel("Tokens / sec")
    ax.set_title("Throughput (TPS) over time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = f"ttft_stress_{timestamp}.png"
    fig.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")
    plt.close()


if __name__ == "__main__":
    main()
