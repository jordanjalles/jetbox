#!/usr/bin/env python3
"""
Model Comparison Evaluation: L1-L7 x1 across multiple models

Tests each level once with multiple models to compare speed and quality:
- gpt-oss:20b (baseline)
- qwen3:4b (smallest, 32K context)
- qwen3:8b (medium, 128K context)
- qwen3:14b (largest, 128K context)
"""

import subprocess
import time
import json
import os
from pathlib import Path
from datetime import datetime

# Test definitions (one per level for speed)
TESTS = {
    "L1": "Create a Python file called greet.py with a function greet(name) that returns 'Hello, {name}!'",
    "L2": "Create calculator.py with an add(a, b) function and a test for it",
    "L3": "Create a Python package 'mathx' with add, subtract, multiply, divide functions in separate files, with tests for all functions",
    "L4": "Create a 'requests_wrapper' package that wraps HTTP requests with retry logic. Include tests.",
    "L5": "Create a Flask REST API with CRUD endpoints for a User model (fields: id, name, email). Use in-memory storage. Include POST /users, GET /users, GET /users/<id>, PUT /users/<id>, DELETE /users/<id>. Write pytest tests for all endpoints.",
    "L6": "Build a Flask blog API with User and Post models. Include user registration/login with JWT tokens, post CRUD operations, and SQLite database. Write pytest tests for auth and posts.",
    "L7": "Build a task management system with: 1) User authentication (register/login), 2) Projects with multiple tasks, 3) Task assignment to users, 4) Status tracking (todo/in-progress/done), 5) SQLite database, 6) Flask REST API, 7) Unit tests for all endpoints. Make it production-ready.",
}

# Models to test
MODELS = [
    "gpt-oss:20b",    # Baseline (128K context, 13GB)
    "qwen3:4b",       # Smallest (32K context, 2.5GB)
    "qwen3:8b",       # Medium (128K context, 5.2GB)
    "qwen3:14b",      # Largest (128K context, 9.3GB)
]

# Timeouts per level (in seconds)
TIMEOUTS = {
    "L1": 60,    # 1 minute
    "L2": 120,   # 2 minutes
    "L3": 180,   # 3 minutes
    "L4": 240,   # 4 minutes
    "L5": 300,   # 5 minutes
    "L6": 420,   # 7 minutes
    "L7": 600,   # 10 minutes
}

def run_test(level, model, goal, timeout, output_dir):
    """Run a single test with a specific model."""
    test_id = f"{level}_{model.replace(':', '_')}"
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

        # For L1-L4: Use TaskExecutor directly
        # For L5-L7: Use Orchestrator
        if level in ["L1", "L2", "L3", "L4"]:
            # Write goal to temp file to avoid quoting issues
            goal_file = output_dir / f"{test_id}_goal.txt"
            with open(goal_file, 'w') as f:
                f.write(goal)

            python_code = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from agents.task_executor_agent import TaskExecutorAgent

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
        else:  # L5-L7
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

        print(f"✓ {test_id}: {status} ({elapsed:.1f}s)")

        return {
            "test_id": test_id,
            "model": model,
            "level": level,
            "goal": goal,
            "status": status,
            "elapsed": elapsed,
            "returncode": result.returncode,
            "log_file": str(log_file)
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ {test_id}: TIMEOUT ({elapsed:.1f}s)")

        return {
            "test_id": test_id,
            "model": model,
            "level": level,
            "goal": goal,
            "status": "TIMEOUT",
            "elapsed": elapsed,
            "returncode": 124,
            "log_file": str(log_file)
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ {test_id}: ERROR - {e}")

        return {
            "test_id": test_id,
            "model": model,
            "level": level,
            "goal": goal,
            "status": "ERROR",
            "elapsed": elapsed,
            "error": str(e),
            "log_file": str(log_file)
        }

def main():
    """Run all tests and generate comparison report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("evaluation_results") / f"model_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("MODEL COMPARISON EVALUATION")
    print(f"Testing: {', '.join(MODELS)}")
    print("Levels: L1-L7 (1 test each)")
    print(f"Output directory: {output_dir}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    results = []
    start_time = time.time()

    # Run all tests (organized by level, then model for fairness)
    for level in ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]:
        goal = TESTS[level]
        timeout = TIMEOUTS[level]

        for model in MODELS:
            result = run_test(level, model, goal, timeout, output_dir)
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
    by_model = {model: [] for model in MODELS}
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
        error = sum(1 for r in model_results if r["status"] == "ERROR")
        avg_time = sum(r["elapsed"] for r in model_results) / total if total > 0 else 0

        model_stats[model] = {
            "total": total,
            "success": success,
            "failed": failed,
            "timeout": timeout,
            "error": error,
            "avg_time": avg_time,
            "success_rate": (success / total * 100) if total > 0 else 0
        }

    # Calculate level stats (across all models)
    level_stats = {}
    for level, level_results in by_level.items():
        total = len(level_results)
        success = sum(1 for r in level_results if r["status"] == "SUCCESS")
        avg_time = sum(r["elapsed"] for r in level_results) / total if total > 0 else 0

        level_stats[level] = {
            "total": total,
            "success": success,
            "avg_time": avg_time,
            "success_rate": (success / total * 100) if total > 0 else 0
        }

    # Write report
    with open(report_file, 'w') as f:
        f.write("# Model Comparison Evaluation Report\n\n")
        f.write(f"**Date**: {datetime.now().isoformat()}\n")
        f.write(f"**Total Time**: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)\n")
        f.write(f"**Models Tested**: {', '.join(MODELS)}\n")
        f.write("**Levels**: L1-L7 (1 test per level per model)\n")
        f.write(f"**Total Tests**: {len(results)}\n\n")

        f.write("## Performance Comparison by Model\n\n")
        f.write("| Model | Success | Failed | Timeout | Error | Success Rate | Avg Time | Size |\n")
        f.write("|-------|---------|--------|---------|-------|--------------|----------|------|\n")

        model_sizes = {
            "gpt-oss:20b": "13 GB",
            "qwen3:4b": "2.5 GB",
            "qwen3:8b": "5.2 GB",
            "qwen3:14b": "9.3 GB"
        }

        for model in MODELS:
            s = model_stats[model]
            size = model_sizes.get(model, "?")
            f.write(f"| {model} | {s['success']}/{s['total']} | {s['failed']} | {s['timeout']} | {s['error']} | {s['success_rate']:.1f}% | {s['avg_time']:.1f}s | {size} |\n")

        f.write("\n## Key Metrics\n\n")

        # Find fastest model
        fastest_model = min(MODELS, key=lambda m: model_stats[m]['avg_time'])
        f.write(f"**Fastest Model**: {fastest_model} ({model_stats[fastest_model]['avg_time']:.1f}s avg)\n")

        # Find most accurate model
        best_model = max(MODELS, key=lambda m: model_stats[m]['success_rate'])
        f.write(f"**Highest Success Rate**: {best_model} ({model_stats[best_model]['success_rate']:.1f}%)\n")

        # Speed comparison to baseline
        baseline = "gpt-oss:20b"
        baseline_time = model_stats[baseline]['avg_time']
        f.write(f"\n**Speed vs Baseline ({baseline})**:\n")
        for model in MODELS:
            if model == baseline:
                continue
            speedup = baseline_time / model_stats[model]['avg_time']
            f.write(f"- {model}: {speedup:.2f}x ")
            if speedup > 1:
                f.write("faster\n")
            else:
                f.write("slower\n")

        f.write("\n## Success Rate by Level\n\n")
        f.write("| Level | Success Rate | Avg Time |\n")
        f.write("|-------|--------------|----------|\n")

        for level in ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]:
            s = level_stats[level]
            f.write(f"| {level} | {s['success_rate']:.1f}% ({s['success']}/{s['total']}) | {s['avg_time']:.1f}s |\n")

        f.write("\n## Detailed Results by Level\n\n")

        for level in ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]:
            f.write(f"### {level} Results\n\n")
            f.write(f"**Goal**: {TESTS[level]}\n\n")
            f.write("| Model | Status | Time | Log |\n")
            f.write("|-------|--------|------|-----|\n")

            for r in by_level[level]:
                status_emoji = {
                    "SUCCESS": "✅",
                    "FAILED": "❌",
                    "TIMEOUT": "⏱️",
                    "ERROR": "💥"
                }.get(r["status"], "❓")

                f.write(f"| {r['model']} | {status_emoji} {r['status']} | {r['elapsed']:.1f}s | `{Path(r['log_file']).name}` |\n")

            f.write("\n")

        f.write("## Analysis\n\n")

        # Determine winner
        best_overall = max(MODELS, key=lambda m: (model_stats[m]['success_rate'], -model_stats[m]['avg_time']))
        f.write(f"**Recommended Model**: {best_overall}\n")
        f.write(f"- Success Rate: {model_stats[best_overall]['success_rate']:.1f}%\n")
        f.write(f"- Average Time: {model_stats[best_overall]['avg_time']:.1f}s\n")
        f.write(f"- Size: {model_sizes[best_overall]}\n\n")

        f.write("**Trade-offs**:\n")
        f.write("- **Speed**: Smaller models (qwen3:4b) may be faster but less accurate\n")
        f.write("- **Quality**: Larger models (gpt-oss:20b, qwen3:14b) may be more accurate but slower\n")
        f.write("- **Context**: qwen3:4b limited to 32K tokens, others support 128K\n\n")

        # Recommendations by use case
        f.write("**Use Case Recommendations**:\n")
        f.write("- **Development/Iteration**: Use fastest model with acceptable quality\n")
        f.write("- **Production/Complex Tasks**: Use most accurate model\n")
        f.write("- **Resource-Constrained**: Use smallest model that meets requirements\n\n")

if __name__ == "__main__":
    main()
