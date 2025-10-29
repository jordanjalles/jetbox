#!/usr/bin/env python3
"""Benchmark gpt-oss:20b vs qwen3:14b on L4-L7 tasks."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from run_stress_tests import TESTS, run_test


def run_benchmark(model_name: str, iterations: int = 5) -> dict[str, Any]:
    """Run L4-L7 tests for a specific model."""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*80}\n")

    # Set model environment variable
    os.environ["OLLAMA_MODEL"] = model_name

    # Filter to L4-L7 tests
    l4_to_l7_tests = [t for t in TESTS if 4 <= t["level"] <= 7]

    print(f"Running {len(l4_to_l7_tests)} tests (L4-L7) x {iterations} iterations")
    print(f"Total runs: {len(l4_to_l7_tests) * iterations}")
    print()

    all_results = []
    start_time = time.time()

    for iteration in range(1, iterations + 1):
        print(f"\n{'-'*60}")
        print(f"ITERATION {iteration}/{iterations}")
        print(f"{'-'*60}\n")

        for test in l4_to_l7_tests:
            print(f"Running {test['id']}: {test['name']}...")

            test_start = time.time()
            result = run_test(test)
            test_duration = time.time() - test_start

            result["iteration"] = iteration
            result["model"] = model_name
            result["wall_time"] = test_duration
            all_results.append(result)

            status = "✓ PASS" if result["success"] else "✗ FAIL"
            print(f"  {status} ({result['rounds']} rounds, {result['duration']:.1f}s)")

    total_time = time.time() - start_time

    # Calculate statistics
    total_tests = len(all_results)
    passed = sum(1 for r in all_results if r["success"])
    pass_rate = (passed / total_tests * 100) if total_tests > 0 else 0

    avg_duration = sum(r["duration"] for r in all_results) / total_tests if total_tests > 0 else 0
    avg_rounds = sum(r["rounds"] for r in all_results) / total_tests if total_tests > 0 else 0

    # Calculate by level
    from collections import defaultdict
    by_level = defaultdict(lambda: {"passed": 0, "total": 0, "durations": [], "rounds": []})

    for r in all_results:
        level = r["level"]
        by_level[level]["total"] += 1
        if r["success"]:
            by_level[level]["passed"] += 1
        by_level[level]["durations"].append(r["duration"])
        by_level[level]["rounds"].append(r["rounds"])

    level_stats = {}
    for level in sorted(by_level.keys()):
        stats = by_level[level]
        level_stats[f"L{level}"] = {
            "pass_rate": stats["passed"] / stats["total"] * 100,
            "avg_duration": sum(stats["durations"]) / len(stats["durations"]),
            "avg_rounds": sum(stats["rounds"]) / len(stats["rounds"]),
            "passed": stats["passed"],
            "total": stats["total"]
        }

    summary = {
        "model": model_name,
        "iterations": iterations,
        "total_tests": total_tests,
        "passed": passed,
        "failed": total_tests - passed,
        "pass_rate": pass_rate,
        "avg_duration": avg_duration,
        "avg_rounds": avg_rounds,
        "total_wall_time": total_time,
        "level_stats": level_stats,
        "all_results": all_results
    }

    return summary


def compare_models(summary1: dict, summary2: dict) -> None:
    """Print comparison between two models."""
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}\n")

    model1 = summary1["model"]
    model2 = summary2["model"]

    print(f"{'Metric':<30} {model1:<20} {model2:<20} {'Winner':<15}")
    print(f"{'-'*85}")

    # Pass rate
    pr1 = summary1["pass_rate"]
    pr2 = summary2["pass_rate"]
    winner = model1 if pr1 > pr2 else (model2 if pr2 > pr1 else "TIE")
    print(f"{'Pass Rate':<30} {pr1:.1f}%{'':<15} {pr2:.1f}%{'':<15} {winner:<15}")

    # Average duration
    d1 = summary1["avg_duration"]
    d2 = summary2["avg_duration"]
    winner = model1 if d1 < d2 else (model2 if d2 < d1 else "TIE")
    speedup = d1 / d2 if d2 > 0 else 0
    print(f"{'Avg Duration (seconds)':<30} {d1:.1f}s{'':<15} {d2:.1f}s{'':<15} {winner:<15}")
    if speedup > 1:
        print(f"  → {model2} is {speedup:.2f}x faster")
    elif speedup < 1 and speedup > 0:
        print(f"  → {model1} is {1/speedup:.2f}x faster")

    # Average rounds
    r1 = summary1["avg_rounds"]
    r2 = summary2["avg_rounds"]
    winner = model1 if r1 < r2 else (model2 if r2 < r1 else "TIE")
    print(f"{'Avg Rounds':<30} {r1:.1f}{'':<16} {r2:.1f}{'':<16} {winner:<15}")

    # Total wall time
    t1 = summary1["total_wall_time"]
    t2 = summary2["total_wall_time"]
    winner = model1 if t1 < t2 else (model2 if t2 < t1 else "TIE")
    print(f"{'Total Wall Time':<30} {t1/60:.1f}m{'':<15} {t2/60:.1f}m{'':<15} {winner:<15}")

    print(f"\n{'-'*85}")
    print("BY LEVEL:\n")

    for level in sorted(summary1["level_stats"].keys()):
        stats1 = summary1["level_stats"][level]
        stats2 = summary2["level_stats"][level]

        print(f"{level}:")
        print(f"  Pass Rate:    {model1}: {stats1['pass_rate']:.0f}% | {model2}: {stats2['pass_rate']:.0f}%")
        print(f"  Avg Duration: {model1}: {stats1['avg_duration']:.1f}s | {model2}: {stats2['avg_duration']:.1f}s")
        print(f"  Avg Rounds:   {model1}: {stats1['avg_rounds']:.1f} | {model2}: {stats2['avg_rounds']:.1f}")
        print()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Benchmark gpt-oss:20b
    summary_gptoss = run_benchmark("gpt-oss:20b", iterations=5)

    # Save intermediate results
    with open(f"benchmark_gptoss_{timestamp}.json", "w") as f:
        json.dump(summary_gptoss, f, indent=2)

    print(f"\n{'='*80}")
    print(f"gpt-oss:20b RESULTS")
    print(f"{'='*80}")
    print(f"Pass Rate: {summary_gptoss['pass_rate']:.1f}%")
    print(f"Avg Duration: {summary_gptoss['avg_duration']:.1f}s")
    print(f"Avg Rounds: {summary_gptoss['avg_rounds']:.1f}")
    print(f"Total Time: {summary_gptoss['total_wall_time']/60:.1f} minutes")

    # Benchmark qwen3:14b
    summary_qwen3 = run_benchmark("qwen3:14b", iterations=5)

    # Save intermediate results
    with open(f"benchmark_qwen3_{timestamp}.json", "w") as f:
        json.dump(summary_qwen3, f, indent=2)

    print(f"\n{'='*80}")
    print(f"qwen3:14b RESULTS")
    print(f"{'='*80}")
    print(f"Pass Rate: {summary_qwen3['pass_rate']:.1f}%")
    print(f"Avg Duration: {summary_qwen3['avg_duration']:.1f}s")
    print(f"Avg Rounds: {summary_qwen3['avg_rounds']:.1f}")
    print(f"Total Time: {summary_qwen3['total_wall_time']/60:.1f} minutes")

    # Compare models
    compare_models(summary_gptoss, summary_qwen3)

    # Save comparison report
    comparison = {
        "timestamp": timestamp,
        "gpt-oss:20b": summary_gptoss,
        "qwen3:14b": summary_qwen3
    }

    with open(f"benchmark_comparison_{timestamp}.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to:")
    print(f"  - benchmark_gptoss_{timestamp}.json")
    print(f"  - benchmark_qwen3_{timestamp}.json")
    print(f"  - benchmark_comparison_{timestamp}.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
