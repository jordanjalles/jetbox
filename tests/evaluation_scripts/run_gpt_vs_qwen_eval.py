#!/usr/bin/env python3
"""
GPT-OSS vs Qwen3:8b Comparison: L3-L6 x5

Tests 5 different tasks at each complexity level (L3-L6) to compare:
- gpt-oss:20b (baseline, 13GB)
- qwen3:8b (winner from initial eval, 5.2GB)

Total: 40 tests (4 levels × 5 tasks × 2 models)
"""

import subprocess
import time
import json
import os
from pathlib import Path
from datetime import datetime

# Test definitions (5 variants per level)
TESTS = {
    "L3": [
        "Create a Python package 'mathx' with add, subtract, multiply, divide functions in separate files, with tests for all functions",
        "Create a 'string_utils' package with functions: reverse(s), capitalize_words(s), count_vowels(s), is_palindrome(s). Include tests.",
        "Build a 'data_structures' package with Stack and Queue classes in separate files. Include comprehensive tests.",
        "Create a 'validators' package with email, phone, url, and password validation functions. Include tests for all.",
        "Build a 'converters' package with temperature, distance, weight, and time conversion functions. Include tests.",
    ],
    "L4": [
        "Create a 'requests_wrapper' package that wraps HTTP requests with retry logic. Include tests.",
        "Build a 'json_validator' package that validates JSON schemas with detailed error messages. Include tests.",
        "Create a 'file_processor' package that reads/writes CSV, JSON, and XML files with error handling. Include tests.",
        "Build a 'cache_manager' package with TTL-based in-memory caching and LRU eviction. Include tests.",
        "Create a 'logger_wrapper' package with multiple output formats (JSON, plain text) and log levels. Include tests.",
    ],
    "L5": [
        "Create a Flask REST API with CRUD endpoints for a User model (fields: id, name, email). Use in-memory storage. Include POST /users, GET /users, GET /users/<id>, PUT /users/<id>, DELETE /users/<id>. Write pytest tests for all endpoints.",
        "Build a Flask API for a Book library with Book model (id, title, author, isbn, available). Use in-memory storage. Include CRUD endpoints and pytest tests.",
        "Create a Flask REST API for a Product catalog (id, name, price, stock, category). Use in-memory storage. Include CRUD operations and comprehensive tests.",
        "Build a Flask API for managing Tasks (id, title, description, status, priority). Use in-memory storage. Include all CRUD endpoints and pytest tests.",
        "Create a Flask REST API for Student records (id, name, grade, courses). Use in-memory storage. Include CRUD endpoints and full test coverage.",
    ],
    "L6": [
        "Build a Flask blog API with User and Post models. Include user registration/login with JWT tokens, post CRUD operations, and SQLite database. Write pytest tests for auth and posts.",
        "Create a Flask e-commerce API with Product and Order models. Include JWT authentication, product catalog, order management, SQLite storage, and unit tests.",
        "Build a Flask todo API with User and Task models. Include authentication with JWT, task CRUD with ownership, SQLite database, and comprehensive tests.",
        "Create a Flask notes API with User and Note models. Include JWT auth, note CRUD with sharing, SQLite storage, and pytest tests.",
        "Build a Flask inventory API with User and Item models. Include authentication, item tracking with quantity updates, SQLite database, and full tests.",
    ],
}

# Models to test
MODELS = [
    "gpt-oss:20b",    # Baseline
    "qwen3:8b",       # Winner
]

# Timeouts per level (in seconds)
TIMEOUTS = {
    "L3": 180,   # 3 minutes
    "L4": 240,   # 4 minutes
    "L5": 300,   # 5 minutes
    "L6": 420,   # 7 minutes
}

def run_test(level, model, run_num, goal, timeout, output_dir):
    """Run a single test with a specific model."""
    test_id = f"{level}_run{run_num}_{model.replace(':', '_')}"
    log_file = output_dir / f"{test_id}.log"

    print(f"\n{'='*70}")
    print(f"Running {test_id}")
    print(f"Model: {model}")
    print(f"Level: {level}")
    print(f"Goal: {goal[:80]}...")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        # Set model environment variable
        env = os.environ.copy()
        env["OLLAMA_MODEL"] = model

        # For L3-L4: Use TaskExecutor directly
        # For L5-L6: Use Orchestrator
        if level in ["L3", "L4"]:
            # Write goal to temp file to avoid quoting issues
            goal_file = output_dir / f"{test_id}_goal.txt"
            with open(goal_file, 'w') as f:
                f.write(goal)

            python_code = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from task_executor_agent import TaskExecutorAgent

with open('{goal_file}', 'r') as f:
    goal = f.read()

agent = TaskExecutorAgent(
    workspace=None,
    goal=goal
)
result = agent.run()
print(f"Status: {{result.get('status')}}")
sys.exit(0 if result.get('status') == 'success' else 1)
"""

            cmd = [
                "timeout", str(timeout),
                "python", "-c", python_code
            ]
        else:  # L5-L6
            cmd = [
                "timeout", str(timeout),
                "python", "orchestrator_main.py",
                goal,
                "--once"
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10,  # Extra buffer
            env=env
        )

        elapsed = time.time() - start_time

        # Save full log
        with open(log_file, 'w') as f:
            f.write(f"=== {test_id} ===\n")
            f.write(f"Model: {model}\n")
            f.write(f"Level: {level}\n")
            f.write(f"Run: {run_num}\n")
            f.write(f"Goal: {goal}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Elapsed: {elapsed:.1f}s\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)

        # Analyze result
        stdout = result.stdout.lower()
        stderr = result.stderr.lower()

        # Success indicators
        success_indicators = [
            "goal complete",
            "mark_complete",
            "completed with status: success",
            "status: success"
        ]

        # Failure indicators
        failure_indicators = [
            "error:",
            "exception:",
            "traceback",
            "failed",
            "timed out"
        ]

        has_success = any(ind in stdout for ind in success_indicators)
        has_failure = any(ind in stderr or ind in stdout for ind in failure_indicators)

        # Determine status
        if result.returncode == 124:  # timeout return code
            status = "TIMEOUT"
        elif result.returncode == 0 or (has_success and not has_failure):
            status = "SUCCESS"
        elif has_failure:
            status = "FAILED"
        else:
            status = "UNKNOWN"

        emoji = {"SUCCESS": "✅", "FAILED": "❌", "TIMEOUT": "⏱️", "UNKNOWN": "❓"}[status]
        print(f"{emoji} {test_id}: {status} ({elapsed:.1f}s)")

        return {
            "test_id": test_id,
            "model": model,
            "level": level,
            "run": run_num,
            "goal": goal,
            "status": status,
            "elapsed": elapsed,
            "returncode": result.returncode,
            "log_file": str(log_file)
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"⏱️ {test_id}: TIMEOUT ({elapsed:.1f}s)")

        return {
            "test_id": test_id,
            "model": model,
            "level": level,
            "run": run_num,
            "goal": goal,
            "status": "TIMEOUT",
            "elapsed": elapsed,
            "returncode": 124,
            "log_file": str(log_file)
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"💥 {test_id}: ERROR - {e}")

        return {
            "test_id": test_id,
            "model": model,
            "level": level,
            "run": run_num,
            "goal": goal,
            "status": "ERROR",
            "elapsed": elapsed,
            "error": str(e),
            "log_file": str(log_file)
        }

def main():
    """Run all tests and generate comparison report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("evaluation_results") / f"gpt_vs_qwen_l3_l6_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("GPT-OSS vs Qwen3:8b COMPARISON EVALUATION")
    print("Models: gpt-oss:20b vs qwen3:8b")
    print("Levels: L3-L6 (5 tasks each)")
    print("Total tests: 40 (4 levels × 5 tasks × 2 models)")
    print(f"Output directory: {output_dir}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    results = []
    start_time = time.time()

    # Run all tests (organized by level, then run, then model)
    for level in ["L3", "L4", "L5", "L6"]:
        goals = TESTS[level]
        timeout = TIMEOUTS[level]

        for run_num, goal in enumerate(goals, 1):
            for model in MODELS:
                result = run_test(level, model, run_num, goal, timeout, output_dir)
                results.append(result)

                # Small delay between tests
                time.sleep(2)

    total_elapsed = time.time() - start_time

    # Save results JSON
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "models": MODELS,
            "levels": list(TESTS.keys()),
            "total_elapsed": total_elapsed,
            "results": results
        }, f, indent=2)

    # Generate comparison report
    report_file = output_dir / "COMPARISON_REPORT.md"
    generate_report(results, total_elapsed, report_file)

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    print(f"Results: {results_file}")
    print(f"Report: {report_file}")
    print(f"{'='*70}\n")

    # Print summary
    with open(report_file) as f:
        print(f.read())

def generate_report(results, total_elapsed, report_file):
    """Generate markdown comparison report."""

    # Organize by model and level
    by_model = {"gpt-oss:20b": [], "qwen3:8b": []}
    by_level = {level: [] for level in TESTS.keys()}

    for r in results:
        by_model[r["model"]].append(r)
        by_level[r["level"]].append(r)

    # Calculate model stats
    model_stats = {}
    for model, model_results in by_model.items():
        total = len(model_results)
        success = sum(1 for r in model_results if r["status"] == "SUCCESS")
        failed = sum(1 for r in model_results if r["status"] == "FAILED")
        timeout = sum(1 for r in model_results if r["status"] == "TIMEOUT")
        unknown = sum(1 for r in model_results if r["status"] == "UNKNOWN")
        error = sum(1 for r in model_results if r["status"] == "ERROR")

        successful_results = [r for r in model_results if r["status"] == "SUCCESS"]
        avg_time = sum(r["elapsed"] for r in successful_results) / len(successful_results) if successful_results else 0

        model_stats[model] = {
            "total": total,
            "success": success,
            "failed": failed,
            "timeout": timeout,
            "unknown": unknown,
            "error": error,
            "avg_time": avg_time,
            "success_rate": (success / total * 100) if total > 0 else 0
        }

    # Calculate level stats (per model)
    level_stats = {}
    for level in TESTS.keys():
        level_stats[level] = {}
        for model in MODELS:
            level_results = [r for r in results if r["level"] == level and r["model"] == model]
            total = len(level_results)
            success = sum(1 for r in level_results if r["status"] == "SUCCESS")

            successful = [r for r in level_results if r["status"] == "SUCCESS"]
            avg_time = sum(r["elapsed"] for r in successful) / len(successful) if successful else 0

            level_stats[level][model] = {
                "total": total,
                "success": success,
                "avg_time": avg_time,
                "success_rate": (success / total * 100) if total > 0 else 0
            }

    # Write report
    with open(report_file, 'w') as f:
        f.write("# GPT-OSS vs Qwen3:8b Comparison Report\n\n")
        f.write(f"**Date**: {datetime.now().isoformat()}\n")
        f.write(f"**Total Time**: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)\n")
        f.write("**Models**: gpt-oss:20b (baseline) vs qwen3:8b (challenger)\n")
        f.write("**Levels**: L3-L6 (5 tasks per level)\n")
        f.write(f"**Total Tests**: {len(results)}\n\n")

        f.write("## Overall Performance\n\n")
        f.write("| Model | Success | Failed | Timeout | Unknown | Error | Success Rate | Avg Time (successful) |\n")
        f.write("|-------|---------|--------|---------|---------|-------|--------------|----------------------|\n")

        for model in MODELS:
            s = model_stats[model]
            f.write(f"| {model} | {s['success']}/{s['total']} | {s['failed']} | {s['timeout']} | {s['unknown']} | {s['error']} | {s['success_rate']:.1f}% | {s['avg_time']:.1f}s |\n")

        # Winner determination
        f.write("\n## Winner\n\n")
        gpt_rate = model_stats["gpt-oss:20b"]["success_rate"]
        qwen_rate = model_stats["qwen3:8b"]["success_rate"]

        if qwen_rate > gpt_rate:
            diff = qwen_rate - gpt_rate
            f.write(f"**🏆 qwen3:8b wins** with {qwen_rate:.1f}% success rate (+{diff:.1f}% vs baseline)\n\n")
        elif gpt_rate > qwen_rate:
            diff = gpt_rate - qwen_rate
            f.write(f"**🏆 gpt-oss:20b wins** with {gpt_rate:.1f}% success rate (+{diff:.1f}% vs qwen3:8b)\n\n")
        else:
            f.write(f"**🤝 TIE** - Both models achieved {gpt_rate:.1f}% success rate\n\n")

        # Speed comparison (for successful tests)
        gpt_time = model_stats["gpt-oss:20b"]["avg_time"]
        qwen_time = model_stats["qwen3:8b"]["avg_time"]

        if qwen_time > 0 and gpt_time > 0:
            speedup = gpt_time / qwen_time
            f.write(f"**Speed**: qwen3:8b is {speedup:.2f}x ")
            f.write("faster\n\n" if speedup > 1 else "slower\n\n")

        f.write("## Performance by Level\n\n")
        f.write("| Level | gpt-oss:20b | qwen3:8b | Winner |\n")
        f.write("|-------|-------------|----------|--------|\n")

        for level in ["L3", "L4", "L5", "L6"]:
            gpt_stats = level_stats[level]["gpt-oss:20b"]
            qwen_stats = level_stats[level]["qwen3:8b"]

            gpt_str = f"{gpt_stats['success']}/{gpt_stats['total']} ({gpt_stats['success_rate']:.0f}%)"
            qwen_str = f"{qwen_stats['success']}/{qwen_stats['total']} ({qwen_stats['success_rate']:.0f}%)"

            if qwen_stats['success_rate'] > gpt_stats['success_rate']:
                winner = "qwen3:8b 🏆"
            elif gpt_stats['success_rate'] > qwen_stats['success_rate']:
                winner = "gpt-oss:20b 🏆"
            else:
                winner = "Tie"

            f.write(f"| {level} | {gpt_str} | {qwen_str} | {winner} |\n")

        f.write("\n## Detailed Results by Level\n\n")

        for level in ["L3", "L4", "L5", "L6"]:
            f.write(f"### {level} Results\n\n")

            # Group by run number
            level_results = by_level[level]
            by_run = {}
            for r in level_results:
                run = r["run"]
                if run not in by_run:
                    by_run[run] = []
                by_run[run].append(r)

            f.write("| Run | Goal | gpt-oss:20b | qwen3:8b |\n")
            f.write("|-----|------|-------------|----------|\n")

            for run in sorted(by_run.keys()):
                run_results = by_run[run]
                goal = run_results[0]["goal"][:60] + "..."

                # Get results for each model
                gpt_result = next((r for r in run_results if r["model"] == "gpt-oss:20b"), None)
                qwen_result = next((r for r in run_results if r["model"] == "qwen3:8b"), None)

                def format_result(r):
                    if not r:
                        return "N/A"
                    emoji = {"SUCCESS": "✅", "FAILED": "❌", "TIMEOUT": "⏱️", "UNKNOWN": "❓", "ERROR": "💥"}
                    return f"{emoji.get(r['status'], '?')} {r['elapsed']:.1f}s"

                gpt_str = format_result(gpt_result)
                qwen_str = format_result(qwen_result)

                f.write(f"| {run} | {goal} | {gpt_str} | {qwen_str} |\n")

            f.write("\n")

        f.write("## Key Findings\n\n")

        # Identify patterns
        gpt_successes = model_stats["gpt-oss:20b"]["success"]
        qwen_successes = model_stats["qwen3:8b"]["success"]

        f.write(f"- **Total Successes**: gpt-oss:20b ({gpt_successes}), qwen3:8b ({qwen_successes})\n")

        if model_stats["gpt-oss:20b"]["timeout"] > 0:
            f.write(f"- gpt-oss:20b had {model_stats['gpt-oss:20b']['timeout']} timeouts\n")

        if model_stats["qwen3:8b"]["timeout"] > 0:
            f.write(f"- qwen3:8b had {model_stats['qwen3:8b']['timeout']} timeouts\n")

        if model_stats["gpt-oss:20b"]["unknown"] > 0:
            f.write(f"- gpt-oss:20b had {model_stats['gpt-oss:20b']['unknown']} UNKNOWN statuses (completion detection issue)\n")

        if model_stats["qwen3:8b"]["unknown"] > 0:
            f.write(f"- qwen3:8b had {model_stats['qwen3:8b']['unknown']} UNKNOWN statuses\n")

        f.write("\n")

if __name__ == "__main__":
    main()
