#!/usr/bin/env python3
"""Quick 3-task benchmark to test framework before running full suite."""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    level: str
    variation: int
    description: str
    timeout: int = 180

@dataclass
class BenchmarkResult:
    """Result of running one task with one model."""
    model: str
    level: str
    variation: int
    success: bool
    time_seconds: float
    rounds_used: int
    error: str = ""

# Quick test: Just L3-V1 across 3 models
QUICK_TEST_TASKS = [
    BenchmarkTask(
        level="L3",
        variation=1,
        description="Create a geometry package with circle.py, rectangle.py, triangle.py. Each module calculates area. Include tests.",
        timeout=180
    ),
]

MODELS = ["gpt-oss:20b", "qwen3:8b", "qwen3:14b"]

def run_agent(task: BenchmarkTask, model: str) -> BenchmarkResult:
    """Run agent on a task with a specific model."""
    print(f"\n{'='*80}")
    print(f"Running {task.level}-V{task.variation} with {model}")
    print(f"Task: {task.description[:60]}...")
    print(f"{'='*80}\n")

    # Set model via environment variable
    env = os.environ.copy()
    env["OLLAMA_MODEL"] = model

    # Create unique workspace for this task+model combo
    workspace_name = f"benchmark_{task.level.lower()}_v{task.variation}_{model.replace(':', '_')}"
    workspace_path = Path(f".agent_workspaces/{workspace_name}")

    # Clean workspace if it exists
    if workspace_path.exists():
        print(f"[BENCHMARK] Cleaning existing workspace: {workspace_path}")
        import shutil
        shutil.rmtree(workspace_path)

    print(f"[{task.level}-V{task.variation}] Starting execution with {model}...")
    start_time = time.time()

    try:
        # Run task_executor directly (not orchestrator)
        result = subprocess.run(
            [
                sys.executable,
                "run_task_executor.py",
                task.description,
                "--workspace", str(workspace_path),
                "--config", "task_executor_config.yaml"
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=task.timeout
        )

        elapsed = time.time() - start_time

        # Check if goal completed successfully (exit code 0 = success)
        agent_success = result.returncode == 0

        # Extract rounds used
        rounds_used = 0
        for line in result.stdout.splitlines():
            if "Round" in line and "/" in line:
                try:
                    rounds_used = int(line.split("Round ")[1].split("/")[0])
                except (IndexError, ValueError):
                    pass

        # Run tests if workspace has tests
        test_success = False
        if workspace_path.exists():
            test_result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            test_success = test_result.returncode == 0

        final_success = agent_success and test_success

        print(f"[{task.level}-V{task.variation}] {'✅ SUCCESS' if final_success else '❌ FAILED'} ({elapsed:.1f}s, {rounds_used} rounds)")

        return BenchmarkResult(
            model=model,
            level=task.level,
            variation=task.variation,
            success=final_success,
            time_seconds=elapsed,
            rounds_used=rounds_used,
            error="" if final_success else "Agent failed or tests failed"
        )

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"[{task.level}-V{task.variation}] ❌ TIMEOUT ({elapsed:.1f}s)")
        return BenchmarkResult(
            model=model,
            level=task.level,
            variation=task.variation,
            success=False,
            time_seconds=elapsed,
            rounds_used=0,
            error="Timeout"
        )
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{task.level}-V{task.variation}] ❌ ERROR: {e}")
        return BenchmarkResult(
            model=model,
            level=task.level,
            variation=task.variation,
            success=False,
            time_seconds=elapsed,
            rounds_used=0,
            error=str(e)
        )

def main():
    """Run quick benchmark."""
    print("="*100)
    print("QUICK 3-TASK BENCHMARK TEST")
    print("="*100)
    print(f"Models: {', '.join(MODELS)}")
    print(f"Tasks: 1 (L3-V1)")
    print(f"Total evaluations: 3")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    print()

    all_results = []

    # Test one task across all 3 models
    task = QUICK_TEST_TASKS[0]

    for model in MODELS:
        print(f"\n{'#'*100}")
        print(f"TESTING MODEL: {model}")
        print(f"{'#'*100}\n")

        result = run_agent(task, model)
        all_results.append(result)

        # Brief pause between models
        time.sleep(2)

    # Print summary
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)

    for model in MODELS:
        model_results = [r for r in all_results if r.model == model]
        successes = sum(1 for r in model_results if r.success)
        total = len(model_results)
        avg_time = sum(r.time_seconds for r in model_results) / total if total > 0 else 0

        print(f"\n{model}:")
        print(f"  Success rate: {successes}/{total} ({100*successes/total:.0f}%)")
        print(f"  Avg time:     {avg_time:.1f}s")
        print(f"  Total time:   {sum(r.time_seconds for r in model_results):.1f}s")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"evaluation_results/quick_benchmark_{timestamp}.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "models": MODELS,
            "total_evaluations": len(all_results),
            "results": [asdict(r) for r in all_results]
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
