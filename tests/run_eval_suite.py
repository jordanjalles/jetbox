#!/usr/bin/env python3
"""Run L3-L4-L5 evaluation suite with 5 iterations."""

import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def check_ollama_health() -> bool:
    """Check if Ollama is responsive."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def restart_ollama_if_needed() -> None:
    """Restart Ollama if it's not responding (helps prevent hangs)."""
    if not check_ollama_health():
        print("[warning] Ollama not responding, waiting and checking again...")
        time.sleep(5)

        # Check again after wait
        if check_ollama_health():
            print("[info] Ollama recovered after wait")
            return

        print("[warning] Ollama still not responding. Manual restart may be needed.")
        print("[info] To restart Ollama:")
        print("  Windows: Restart Ollama app or run: ollama serve")
        print("  Linux: systemctl restart ollama")
        print("[info] Continuing test anyway...")
        time.sleep(2)


def clean_agent_state():
    """Clean agent state between runs."""
    if Path(".agent_workspace").exists():
        shutil.rmtree(".agent_workspace")
    if Path(".agent_context").exists():
        shutil.rmtree(".agent_context")
    for log in ["agent_v2.log", "agent_ledger.log", "agent.log"]:
        if Path(log).exists():
            Path(log).unlink()


def run_full_suite(iteration: int) -> list[dict[str, Any]]:
    """Run full L3-L4-L5 test suite once."""
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration}/5")
    print(f"{'='*70}\n")

    # Clean state
    clean_agent_state()

    # Check Ollama health before starting iteration (prevents timeout issues)
    restart_ollama_if_needed()

    # Run stress tests
    try:
        subprocess.run(
            ["python", "tests/run_stress_tests.py", "3,4,5"],
            timeout=3600,  # 1 hour timeout for full suite
        )

        # Load results
        if Path("stress_test_results.json").exists():
            with open("stress_test_results.json") as f:
                results = json.load(f)
                # Add iteration number to each result
                for r in results:
                    r["iteration"] = iteration
                return results

    except subprocess.TimeoutExpired:
        print(f"Iteration {iteration} timed out!")
    except Exception as e:
        print(f"Iteration {iteration} error: {e}")

    return []


def main():
    """Run full evaluation suite 10 times."""
    all_results = []

    print("="*70)
    print("EVALUATION SUITE: L3-L4-L5 Tests (10 full iterations)")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    for iteration in range(1, 11):
        results = run_full_suite(iteration)
        all_results.extend(results)

        # Save incremental results
        with open("eval_suite_results_x10.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Quick summary
        if results:
            successes = sum(1 for r in results if r.get("success"))
            print(f"\nIteration {iteration} summary: {successes}/{len(results)} passed")

    # Final analysis
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("EVALUATION SUITE COMPLETE")
    print("="*70)
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Aggregate by test
    by_test = {}
    for result in all_results:
        test_id = result.get("id", "unknown")
        if test_id not in by_test:
            by_test[test_id] = {
                "name": result.get("name", ""),
                "level": result.get("level", 0),
                "iterations": [],
                "success_count": 0,
                "total_duration": 0,
                "total_rounds": 0,
            }

        by_test[test_id]["iterations"].append({
            "iteration": result.get("iteration", 0),
            "success": result.get("success", False),
            "duration": result.get("duration", 0),
            "rounds": result.get("rounds", 0),
            "failure_mode": result.get("failure_mode"),
        })

        if result.get("success"):
            by_test[test_id]["success_count"] += 1
        by_test[test_id]["total_duration"] += result.get("duration", 0)
        by_test[test_id]["total_rounds"] += result.get("rounds", 0)

    # Print summary table
    print("\nRESULTS BY TEST:")
    print("-" * 80)
    print(f"{'Test':<8} {'Name':<30} {'Pass Rate':<12} {'Avg Rounds':<12} {'Avg Time'}")
    print("-" * 80)

    for test_id in sorted(by_test.keys()):
        data = by_test[test_id]
        n_iterations = len(data["iterations"])
        pass_rate = (data["success_count"] / n_iterations * 100) if n_iterations > 0 else 0
        avg_rounds = data["total_rounds"] / n_iterations if n_iterations > 0 else 0
        avg_time = data["total_duration"] / n_iterations if n_iterations > 0 else 0

        print(f"{test_id:<8} {data['name'][:29]:<30} "
              f"{data['success_count']}/{n_iterations} ({pass_rate:>3.0f}%)  "
              f"{avg_rounds:>6.1f}       {avg_time:>6.1f}s")

    # By level
    print("\nRESULTS BY LEVEL:")
    print("-" * 80)
    by_level = {}
    for test_id, data in by_test.items():
        level = data["level"]
        if level not in by_level:
            by_level[level] = {"total": 0, "passed": 0}
        n_iter = len(data["iterations"])
        by_level[level]["total"] += n_iter
        by_level[level]["passed"] += data["success_count"]

    for level in sorted(by_level.keys()):
        stats = by_level[level]
        pct = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"Level {level}: {stats['passed']}/{stats['total']} passed ({pct:.0f}%)")

    # Failure modes
    failure_modes = {}
    for result in all_results:
        if not result.get("success") and result.get("failure_mode"):
            fm = result["failure_mode"]
            if fm not in failure_modes:
                failure_modes[fm] = []
            failure_modes[fm].append(f"{result['id']}-iter{result.get('iteration', 0)}")

    if failure_modes:
        print("\nFAILURE MODES:")
        print("-" * 80)
        for mode, instances in sorted(failure_modes.items()):
            print(f"  {mode}: {len(instances)} occurrences")
            print(f"    Examples: {', '.join(instances[:5])}")

    print(f"\nDetailed results saved to: eval_suite_results.json")


if __name__ == "__main__":
    main()
