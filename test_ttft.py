#!/usr/bin/env python3
"""
Azure Databricks + Claude Opus 4.6 — TTFT (Time To First Token) Test
=====================================================================
Measures TTFT using the OpenAI-compatible streaming endpoint.

Also reports total latency, total tokens, and tokens/sec for comparison.
Runs multiple iterations to compute min/median/max/p95 statistics.

Usage:
  pip install openai python-dotenv
  python test_ttft.py
"""

import os
import time
import statistics

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================
# Configuration
# ============================================================
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
DATABRICKS_BASE_URL = os.environ["DATABRICKS_BASE_URL"]
MODEL = "databricks-claude-opus-4-6"

# Test parameters
ITERATIONS = 5
MAX_TOKENS = 256
PROMPTS = [
    # Short prompt — tests baseline TTFT
    {"label": "Short prompt", "messages": [
        {"role": "user", "content": "What is 2+2?"}
    ]},
    # Medium prompt — system + user
    {"label": "Medium prompt (system + user)", "messages": [
        {"role": "system", "content": "You are a helpful cloud architecture expert. "
         "Answer concisely in 2-3 sentences."},
        {"role": "user", "content": "Explain the difference between Azure Databricks "
         "and standard Apache Spark."}
    ]},
    # Long system prompt — tests TTFT under heavier input
    {"label": "Long system prompt (~2k tokens)", "messages": [
        {"role": "system", "content": "You are an expert. " + " ".join(
            [f"Context item {i}: Azure Databricks provides a unified analytics platform "
             f"with optimized Spark runtime, Delta Lake integration, and ML capabilities. "
             f"Item {i} includes networking, security, and governance details."
             for i in range(1, 30)]
        )},
        {"role": "user", "content": "Summarize the key points in one sentence."}
    ]},
]

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=DATABRICKS_BASE_URL,
)


def measure_ttft(messages: list[dict]) -> dict:
    """Send a streaming request and measure TTFT + total latency."""
    t_start = time.perf_counter()
    t_first_token = None
    token_count = 0
    full_text = []

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.1,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if t_first_token is None:
                t_first_token = time.perf_counter()
            token_count += 1
            full_text.append(chunk.choices[0].delta.content)

    t_end = time.perf_counter()

    ttft = (t_first_token - t_start) if t_first_token else None
    total = t_end - t_start
    tps = token_count / (t_end - t_first_token) if t_first_token and t_end > t_first_token else 0

    return {
        "ttft_s": ttft,
        "total_s": total,
        "tokens": token_count,
        "tokens_per_sec": tps,
        "text": "".join(full_text),
    }


def percentile(data: list[float], p: float) -> float:
    """Simple percentile calculation."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_test(label: str, messages: list[dict], iterations: int):
    """Run multiple iterations and print statistics."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Iterations: {iterations}")
    print(f"{'=' * 60}")

    ttfts = []
    totals = []
    tps_list = []

    for i in range(iterations):
        print(f"\n  Run {i + 1}/{iterations}...", end=" ", flush=True)
        try:
            result = measure_ttft(messages)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        if result["ttft_s"] is None:
            print("FAILED: no tokens received")
            continue

        ttfts.append(result["ttft_s"])
        totals.append(result["total_s"])
        tps_list.append(result["tokens_per_sec"])

        print(f"TTFT={result['ttft_s']:.3f}s | "
              f"Total={result['total_s']:.2f}s | "
              f"Tokens={result['tokens']} | "
              f"TPS={result['tokens_per_sec']:.1f}")

        if i == 0:
            preview = result["text"][:100].replace("\n", " ")
            print(f"  Preview: {preview}...")

    if len(ttfts) < 2:
        print("\n  Not enough successful runs for statistics.")
        return

    print(f"\n  {'─' * 50}")
    print(f"  TTFT Statistics ({len(ttfts)} runs):")
    print(f"    Min:    {min(ttfts):.3f}s")
    print(f"    Median: {statistics.median(ttfts):.3f}s")
    print(f"    Mean:   {statistics.mean(ttfts):.3f}s")
    print(f"    P95:    {percentile(ttfts, 95):.3f}s")
    print(f"    Max:    {max(ttfts):.3f}s")
    print(f"    Stdev:  {statistics.stdev(ttfts):.3f}s")
    print(f"  Throughput:")
    print(f"    Median TPS: {statistics.median(tps_list):.1f} tokens/sec")
    print(f"    Median Total: {statistics.median(totals):.2f}s")


def main():
    print(f"TTFT Test — Azure Databricks Claude")
    print(f"Model: {MODEL}")
    print(f"Endpoint: {DATABRICKS_BASE_URL}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Iterations per prompt: {ITERATIONS}")

    for prompt_config in PROMPTS:
        run_test(prompt_config["label"], prompt_config["messages"], ITERATIONS)

    print(f"\n{'=' * 60}")
    print(f"  Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
