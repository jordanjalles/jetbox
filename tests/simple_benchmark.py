#!/usr/bin/env python3
"""Simple direct benchmark comparing gpt-oss:20b vs qwen3:14b."""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# L3-L7 tasks
TASKS = [
    # Level 3
    {
        "id": "L3-refactor",
        "level": 3,
        "name": "Refactor Functions to Class",
        "task": "Create calculator.py with add, subtract, multiply functions as standalone functions. Then refactor to use a Calculator class with methods instead.",
        "timeout": 240,
    },
    {
        "id": "L3-bugfix",
        "level": 3,
        "name": "Fix Buggy Code",
        "task": "Create buggy.py with: a divide function without zero check, a list access without bounds check, and missing error handling. Then fix all bugs.",
        "timeout": 240,
    },
    # Level 4
    {
        "id": "L4-todo",
        "level": 4,
        "name": "TodoList with Persistence",
        "task": "Create a TodoList class in todo.py with methods: add_task, remove_task, mark_complete, list_pending. Use a simple list storage. Include tests.",
        "timeout": 300,
    },
    {
        "id": "L4-calc",
        "level": 4,
        "name": "Calculator with Tests",
        "task": "Create calculator.py with add, subtract, multiply, divide functions. Write test_calculator.py with comprehensive tests for all functions. Run tests to verify.",
        "timeout": 300,
    },
    # Level 5
    {
        "id": "L5-classes",
        "level": 5,
        "name": "Refactor to Classes",
        "task": "Create a todo_list.py with TodoList class that has add_task, complete_task, and list_tasks methods. Include tests in test_todo.py.",
        "timeout": 360,
    },
    {
        "id": "L5-data",
        "level": 5,
        "name": "Data Processing",
        "task": "Create data_processor.py that can read CSV and JSON files and convert between them. Include simple parsing and tests.",
        "timeout": 360,
    },
    # Level 6
    {
        "id": "L6-api",
        "level": 6,
        "name": "Simple API",
        "task": "Create a simple REST API with Flask that has GET and POST endpoints for items. Include basic error handling and a test file.",
        "timeout": 420,
    },
    {
        "id": "L6-plugin",
        "level": 6,
        "name": "Plugin System",
        "task": "Design a simple plugin system where plugins can be loaded from a plugins/ directory. Include a base plugin class, one example plugin, and tests.",
        "timeout": 420,
    },
    # Level 7
    {
        "id": "L7-queue",
        "level": 7,
        "name": "Task Queue System",
        "task": "Build a simple task queue with priority levels and basic worker thread support. Include tests for priority ordering and basic concurrency.",
        "timeout": 480,
    },
    {
        "id": "L7-parser",
        "level": 7,
        "name": "Expression Parser",
        "task": "Create a simple math expression parser that handles +, -, *, / operations and parentheses. Include an evaluator and tests.",
        "timeout": 480,
    },
]


def run_task(task: dict, model: str, iteration: int) -> dict:
    """Run a single task with the specified model."""
    print(f"\n{'='*60}")
    print(f"[{model}] Iteration {iteration} - {task['id']}: {task['name']}")
    print(f"{'='*60}")

    # Set model
    env = os.environ.copy()
    env["OLLAMA_MODEL"] = model

    # Build command
    cmd = ["python", "agent.py", task["task"]]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=task["timeout"],
        )

        duration = time.time() - start_time
        success = result.returncode == 0

        # Extract metrics from output
        rounds = 0
        if "Round" in result.stdout:
            import re
            round_matches = re.findall(r'Round (\d+)', result.stdout)
            if round_matches:
                rounds = max(int(r) for r in round_matches)

        print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Rounds: {rounds}")

        return {
            "task_id": task["id"],
            "task_name": task["name"],
            "level": task["level"],
            "model": model,
            "iteration": iteration,
            "success": success,
            "duration": duration,
            "rounds": rounds,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"  Status: ✗ TIMEOUT")
        print(f"  Duration: {duration:.1f}s")

        return {
            "task_id": task["id"],
            "task_name": task["name"],
            "level": task["level"],
            "model": model,
            "iteration": iteration,
            "success": False,
            "duration": duration,
            "rounds": 0,
            "returncode": -1,
            "timeout": True,
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"  Status: ✗ ERROR - {e}")

        return {
            "task_id": task["id"],
            "task_name": task["name"],
            "level": task["level"],
            "model": model,
            "iteration": iteration,
            "success": False,
            "duration": duration,
            "rounds": 0,
            "returncode": -2,
            "error": str(e),
        }


def benchmark_model(model: str, iterations: int = 5) -> list:
    """Run all tasks for a model multiple times."""
    print(f"\n{'#'*70}")
    print(f"# BENCHMARKING: {model}")
    print(f"# Tasks: {len(TASKS)}, Iterations: {iterations}")
    print(f"{'#'*70}")

    results = []

    for iteration in range(1, iterations + 1):
        print(f"\n{'-'*70}")
        print(f"ITERATION {iteration}/{iterations}")
        print(f"{'-'*70}")

        for task in TASKS:
            result = run_task(task, model, iteration)
            results.append(result)

            # Clean up workspace between tasks
            subprocess.run(
                ["rm", "-rf", ".agent_context", ".agent_workspace"],
                capture_output=True
            )

            # Give Ollama a moment between tasks
            time.sleep(2)

    return results


def analyze_results(results: list) -> dict:
    """Calculate statistics from results."""
    total = len(results)
    passed = sum(1 for r in results if r["success"])

    durations = [r["duration"] for r in results]
    rounds = [r["rounds"] for r in results if r["rounds"] > 0]

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total * 100) if total > 0 else 0,
        "avg_duration": sum(durations) / len(durations) if durations else 0,
        "avg_rounds": sum(rounds) / len(rounds) if rounds else 0,
        "min_duration": min(durations) if durations else 0,
        "max_duration": max(durations) if durations else 0,
    }


def compare_models(results1: list, results2: list, model1: str, model2: str):
    """Print comparison between two models."""
    stats1 = analyze_results(results1)
    stats2 = analyze_results(results2)

    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}\n")

    print(f"{'Metric':<25} {model1:<20} {model2:<20}")
    print(f"{'-'*70}")

    print(f"{'Pass Rate':<25} {stats1['pass_rate']:.1f}%{'':<15} {stats2['pass_rate']:.1f}%")
    print(f"{'Avg Duration':<25} {stats1['avg_duration']:.1f}s{'':<15} {stats2['avg_duration']:.1f}s")

    if stats2['avg_duration'] > 0:
        speedup = stats1['avg_duration'] / stats2['avg_duration']
        faster_model = model2 if speedup > 1 else model1
        factor = speedup if speedup > 1 else 1/speedup
        print(f"{'Speed Winner':<25} {faster_model} ({factor:.2f}x faster)")

    print(f"{'Avg Rounds':<25} {stats1['avg_rounds']:.1f}{'':<16} {stats2['avg_rounds']:.1f}")

    # By level breakdown
    print(f"\n{'='*70}")
    print("BY LEVEL:")
    print(f"{'='*70}\n")

    for level in sorted(set(r["level"] for r in results1)):
        level_results1 = [r for r in results1 if r["level"] == level]
        level_results2 = [r for r in results2 if r["level"] == level]

        stats_l1 = analyze_results(level_results1)
        stats_l2 = analyze_results(level_results2)

        print(f"L{level}:")
        print(f"  Pass Rate:    {model1}: {stats_l1['pass_rate']:.0f}%  |  {model2}: {stats_l2['pass_rate']:.0f}%")
        print(f"  Avg Duration: {model1}: {stats_l1['avg_duration']:.1f}s  |  {model2}: {stats_l2['avg_duration']:.1f}s")
        print()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Benchmark both models
    print("\n" + "="*70)
    print("STARTING BENCHMARK")
    print("="*70)

    results_gptoss = benchmark_model("gpt-oss:20b", iterations=5)
    results_qwen3 = benchmark_model("qwen3:14b", iterations=5)

    # Save results
    output = {
        "timestamp": timestamp,
        "gpt-oss:20b": {
            "results": results_gptoss,
            "stats": analyze_results(results_gptoss),
        },
        "qwen3:14b": {
            "results": results_qwen3,
            "stats": analyze_results(results_qwen3),
        }
    }

    output_file = f"simple_benchmark_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # Print comparison
    compare_models(results_gptoss, results_qwen3, "gpt-oss:20b", "qwen3:14b")

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
