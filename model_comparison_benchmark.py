#!/usr/bin/env python3
"""
Model Comparison Benchmark: gpt-oss:20b vs qwen3:8b vs qwen3:14b

Runs L3-L6 level tasks with 3 variations each across three models.
Tests 12 tasks per model = 36 total evaluations.
"""
from __future__ import annotations
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))
from task_executor_agent import TaskExecutorAgent


@dataclass
class BenchmarkTask:
    """Single benchmark task configuration."""
    level: str
    variation: int
    description: str
    timeout: int
    expected_file_patterns: list[str] = None
    run_tests: bool = False


# Define benchmark tasks: L3-L6 with 3 variations each
BENCHMARK_TASKS = [
    # L3 Tasks: Multi-file packages
    BenchmarkTask(
        level="L3", variation=1,
        description="Create a geometry package with circle.py, rectangle.py, triangle.py. Each module calculates area. Include tests.",
        timeout=180,
        expected_file_patterns=["geometry/*.py", "tests/test_*.py"],
        run_tests=True
    ),
    BenchmarkTask(
        level="L3", variation=2,
        description="Create a string_utils package with reverse.py, capitalize.py, count_words.py. Each has one function. Include tests.",
        timeout=180,
        expected_file_patterns=["string_utils/*.py", "tests/test_*.py"],
        run_tests=True
    ),
    BenchmarkTask(
        level="L3", variation=3,
        description="Create a data_structures package with stack.py, queue.py, linked_list.py. Each implements one data structure. Include tests.",
        timeout=180,
        expected_file_patterns=["data_structures/*.py", "tests/test_*.py"],
        run_tests=True
    ),
    
    # L4 Tasks: Package with tests and linting
    BenchmarkTask(
        level="L4", variation=1,
        description="Create a text_processor module with word_count, char_count, line_count functions. Write pytest tests. Run ruff and fix issues.",
        timeout=240,
        run_tests=True
    ),
    BenchmarkTask(
        level="L4", variation=2,
        description="Create a temperature_converter with celsius_to_fahrenheit, fahrenheit_to_celsius. Write tests. Run ruff linting.",
        timeout=240,
        run_tests=True
    ),
    BenchmarkTask(
        level="L4", variation=3,
        description="Create a list_operations module with filter_evens, sum_list, max_value functions. Write tests and run ruff.",
        timeout=240,
        run_tests=True
    ),
    
    # L5 Tasks: More complex packages
    BenchmarkTask(
        level="L5", variation=1,
        description="Create a simple JSON config loader that reads/writes config files with validation. Include type checking and tests.",
        timeout=300,
        run_tests=True
    ),
    BenchmarkTask(
        level="L5", variation=2,
        description="Create a CSV parser that loads CSV files into dicts, handles headers, and supports filtering. Write comprehensive tests.",
        timeout=300,
        run_tests=True
    ),
    BenchmarkTask(
        level="L5", variation=3,
        description="Create a logging wrapper that adds timestamps, log levels, and file output. Include configuration options and tests.",
        timeout=300,
        run_tests=True
    ),
    
    # L6 Tasks: Complex systems
    BenchmarkTask(
        level="L6", variation=1,
        description="Create a simple task queue that supports add, process, and status tracking. Include priority levels. Write tests.",
        timeout=300,
        run_tests=True
    ),
    BenchmarkTask(
        level="L6", variation=2,
        description="Create a cache system with get/set/delete operations and TTL support. Include size limits. Write tests.",
        timeout=300,
        run_tests=True
    ),
    BenchmarkTask(
        level="L6", variation=3,
        description="Create a simple router that matches URL patterns and dispatches to handlers. Support params. Write tests.",
        timeout=300,
        run_tests=True
    ),
]


def run_task(task: BenchmarkTask, model: str) -> dict:
    """
    Run a single task with specified model.
    
    Returns:
        Result dict with status, time, files created, etc.
    """
    print(f"\n{'='*80}")
    print(f"Running {task.level}-V{task.variation} with {model}")
    print(f"Task: {task.description[:60]}...")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Set model via environment variable
        os.environ["OLLAMA_MODEL"] = model
        
        # Create agent
        agent = TaskExecutorAgent(
            workspace=None,  # New isolated workspace
            goal=task.description,
            timeout_seconds=task.timeout,
            exclude_behaviors=["ChatbotBehavior"]
        )
        
        workspace_path = str(agent.workspace)
        
        # Run agent
        print(f"[{task.level}-V{task.variation}] Starting execution with {model}...")
        result = agent.run(max_rounds=50)
        
        elapsed = time.time() - start_time
        
        # Collect files created
        files_created = []
        if agent.workspace and agent.workspace.exists():
            all_files = list(agent.workspace.rglob("*"))
            files_created = [
                str(f.relative_to(agent.workspace))
                for f in all_files
                if f.is_file() and not f.name.startswith(".")
            ]
        
        # Run tests if requested
        test_result = None
        if task.run_tests and result.get("status") == "success":
            import subprocess
            try:
                test_output = subprocess.run(
                    ["python", "-m", "pytest", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(agent.workspace),
                    env={**os.environ, "PYTHONPATH": str(agent.workspace)}
                )
                test_result = {
                    "passed": test_output.returncode == 0,
                    "output": test_output.stdout + test_output.stderr
                }
            except Exception as e:
                test_result = {"passed": False, "error": str(e)}
        
        return {
            "level": task.level,
            "variation": task.variation,
            "model": model,
            "status": result.get("status", "unknown"),
            "time": elapsed,
            "files_created": len(files_created),
            "file_list": files_created,
            "workspace": workspace_path,
            "test_result": test_result,
            "success": result.get("status") == "success" and (
                not task.run_tests or (test_result and test_result.get("passed", False))
            )
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{task.level}-V{task.variation}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "level": task.level,
            "variation": task.variation,
            "model": model,
            "status": "error",
            "time": elapsed,
            "error": str(e),
            "success": False
        }


def print_summary(results: list[dict]) -> None:
    """Print benchmark summary."""
    print("\n" + "="*100)
    print("MODEL COMPARISON BENCHMARK RESULTS")
    print("="*100)
    
    # Group by model
    models = sorted(set(r["model"] for r in results))
    
    for model in models:
        model_results = [r for r in results if r["model"] == model]
        total = len(model_results)
        successful = sum(1 for r in model_results if r["success"])
        avg_time = sum(r["time"] for r in model_results) / total if total > 0 else 0
        
        print(f"\n{'='*100}")
        print(f"MODEL: {model}")
        print(f"{'='*100}")
        print(f"Success Rate: {successful}/{total} ({100*successful//total if total > 0 else 0}%)")
        print(f"Average Time: {avg_time:.1f}s")
        print(f"Total Time: {sum(r['time'] for r in model_results):.1f}s")
        
        # Level breakdown
        print(f"\nBreakdown by Level:")
        for level in ["L3", "L4", "L5", "L6"]:
            level_results = [r for r in model_results if r["level"] == level]
            if level_results:
                level_success = sum(1 for r in level_results if r["success"])
                level_total = len(level_results)
                level_avg_time = sum(r["time"] for r in level_results) / level_total
                print(f"  {level}: {level_success}/{level_total} ({100*level_success//level_total if level_total > 0 else 0}%) - Avg: {level_avg_time:.1f}s")
    
    # Comparison table
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")
    print(f"{'Model':<20} {'Success Rate':<15} {'Avg Time':<12} {'Total Time':<12}")
    print("-"*100)
    
    for model in models:
        model_results = [r for r in results if r["model"] == model]
        total = len(model_results)
        successful = sum(1 for r in model_results if r["success"])
        avg_time = sum(r["time"] for r in model_results) / total if total > 0 else 0
        total_time = sum(r["time"] for r in model_results)
        
        success_pct = f"{successful}/{total} ({100*successful//total if total > 0 else 0}%)"
        print(f"{model:<20} {success_pct:<15} {avg_time:<12.1f} {total_time:<12.1f}")


def save_results(results: list[dict]) -> None:
    """Save benchmark results to JSON."""
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"model_comparison_{timestamp}.json"
    
    output = {
        "timestamp": timestamp,
        "models_tested": sorted(set(r["model"] for r in results)),
        "total_tasks": len(results),
        "results": results
    }
    
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")


def main():
    """Run model comparison benchmark."""
    models = ["gpt-oss:20b", "qwen3:8b", "qwen3:14b"]
    
    print("="*100)
    print("MODEL COMPARISON BENCHMARK")
    print("="*100)
    print(f"Models: {', '.join(models)}")
    print(f"Tasks per model: {len(BENCHMARK_TASKS)}")
    print(f"Total evaluations: {len(models) * len(BENCHMARK_TASKS)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    
    all_results = []
    
    # Run each task with each model
    for model in models:
        print(f"\n\n{'#'*100}")
        print(f"TESTING MODEL: {model}")
        print(f"{'#'*100}")
        
        for task in BENCHMARK_TASKS:
            result = run_task(task, model)
            all_results.append(result)
            
            # Brief status
            status_icon = "✅" if result["success"] else "❌"
            print(f"\n{status_icon} {task.level}-V{task.variation}: {result['status']} in {result['time']:.1f}s")
            
            # Small delay between tasks
            time.sleep(2)
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    save_results(all_results)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit code based on overall success
    total_success = sum(1 for r in all_results if r["success"])
    total_tasks = len(all_results)
    sys.exit(0 if total_success > total_tasks // 2 else 1)


if __name__ == "__main__":
    main()
