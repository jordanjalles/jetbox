#!/usr/bin/env python3
"""Re-test the 4 tests that had timeout issues after applying the fix."""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Import test definitions and helper functions from run_stress_tests
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.run_stress_tests import TESTS, run_test, clean_workspace, restart_ollama_if_needed


# Tests that had timeout issues (L3-2, L3-3, L4-1, L5-2)
TIMEOUT_TEST_IDS = ["L3-2", "L3-3", "L4-1", "L5-2"]


def main():
    """Run timeout-prone tests 10 times each to verify fix."""
    print("="*80)
    print("TIMEOUT FIX VERIFICATION TEST")
    print("="*80)
    print(f"Testing: {', '.join(TIMEOUT_TEST_IDS)}")
    print(f"Iterations: 10 per test (40 total runs)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    all_results = []
    start_time = time.time()

    # Get test definitions
    tests_to_run = [t for t in TESTS if t["id"] in TIMEOUT_TEST_IDS]

    for iteration in range(1, 11):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}/10")
        print(f"{'='*80}\n")

        # Check Ollama health before iteration
        restart_ollama_if_needed()

        for test in tests_to_run:
            result = run_test(test)
            result["iteration"] = iteration
            all_results.append(result)

            # Save incremental results
            with open("timeout_retest_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

            # Brief pause between tests
            time.sleep(1)

    # Analysis
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("RETEST COMPLETE")
    print("="*80)
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Aggregate by test
    by_test = defaultdict(lambda: {"passed": 0, "total": 0, "timeouts": 0, "failures": []})

    for result in all_results:
        test_id = result["id"]
        by_test[test_id]["total"] += 1

        if result.get("success"):
            by_test[test_id]["passed"] += 1
        else:
            if result.get("failure_mode") == "timeout":
                by_test[test_id]["timeouts"] += 1
            by_test[test_id]["failures"].append({
                "iteration": result.get("iteration"),
                "mode": result.get("failure_mode"),
                "rounds": result.get("rounds", 0),
                "duration": result.get("duration", 0),
            })

    # Print results
    print("\n" + "="*80)
    print("RESULTS BY TEST")
    print("="*80)
    print(f"{'Test':<8} {'Name':<30} {'Pass Rate':<12} {'Timeouts':<10} {'Status'}")
    print("-"*80)

    for test_id in sorted(TIMEOUT_TEST_IDS):
        data = by_test[test_id]
        pass_rate = (data["passed"] / data["total"] * 100) if data["total"] > 0 else 0

        # Get test name
        test_name = next((t["name"] for t in tests_to_run if t["id"] == test_id), "Unknown")

        status_icon = "✓" if pass_rate == 100 else "⚠" if pass_rate >= 70 else "✗"

        print(f"{test_id:<8} {test_name[:29]:<30} "
              f"{data['passed']}/{data['total']} ({pass_rate:>3.0f}%)  "
              f"{data['timeouts']:<10} {status_icon}")

        if data["failures"]:
            print(f"  Failures:")
            for f in data["failures"]:
                print(f"    - Iter {f['iteration']}: {f['mode']} "
                      f"({f['duration']:.1f}s, {f['rounds']} rounds)")

    # Compare with previous results
    print("\n" + "="*80)
    print("COMPARISON WITH PREVIOUS RUN")
    print("="*80)

    previous_results = {
        "L3-2": {"passed": 7, "total": 10, "timeouts": 3},
        "L3-3": {"passed": 7, "total": 10, "timeouts": 1},
        "L4-1": {"passed": 7, "total": 10, "timeouts": 3},
        "L5-2": {"passed": 7, "total": 10, "timeouts": 2},
    }

    print(f"\n{'Test':<8} {'Previous':<15} {'New':<15} {'Timeout Change':<15} {'Improvement'}")
    print("-"*80)

    total_prev_pass = 0
    total_new_pass = 0
    total_prev_timeout = 0
    total_new_timeout = 0

    for test_id in sorted(TIMEOUT_TEST_IDS):
        prev = previous_results[test_id]
        curr = by_test[test_id]

        prev_rate = prev["passed"] / prev["total"] * 100
        curr_rate = curr["passed"] / curr["total"] * 100
        improvement = curr_rate - prev_rate

        total_prev_pass += prev["passed"]
        total_new_pass += curr["passed"]
        total_prev_timeout += prev["timeouts"]
        total_new_timeout += curr["timeouts"]

        timeout_change = curr["timeouts"] - prev["timeouts"]
        timeout_str = f"{prev['timeouts']} → {curr['timeouts']}"

        improvement_str = f"+{improvement:.0f}%" if improvement > 0 else f"{improvement:.0f}%"
        if improvement > 0:
            improvement_str = "✓ " + improvement_str
        elif improvement < 0:
            improvement_str = "✗ " + improvement_str
        else:
            improvement_str = "= " + improvement_str

        print(f"{test_id:<8} {prev['passed']}/10 ({prev_rate:.0f}%){'':<2} "
              f"{curr['passed']}/10 ({curr_rate:.0f}%){'':<2} "
              f"{timeout_str:<15} {improvement_str}")

    print("-"*80)
    prev_total_rate = total_prev_pass / 40 * 100
    curr_total_rate = total_new_pass / 40 * 100
    total_improvement = curr_total_rate - prev_total_rate

    print(f"{'TOTAL':<8} {total_prev_pass}/40 ({prev_total_rate:.1f}%){'':<2} "
          f"{total_new_pass}/40 ({curr_total_rate:.1f}%){'':<2} "
          f"{total_prev_timeout} → {total_new_timeout}{'':<6} "
          f"{'+' if total_improvement > 0 else ''}{total_improvement:.1f}%")

    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    if total_new_timeout < total_prev_timeout:
        print(f"✓ SUCCESS: Timeouts reduced from {total_prev_timeout} to {total_new_timeout}")
        print(f"  ({(total_prev_timeout - total_new_timeout) / total_prev_timeout * 100:.0f}% reduction)")

    if total_new_pass > total_prev_pass:
        print(f"✓ IMPROVEMENT: Pass rate improved from {prev_total_rate:.1f}% to {curr_total_rate:.1f}%")
        print(f"  (+{total_new_pass - total_prev_pass} successful tests)")
    elif total_new_pass == total_prev_pass:
        print(f"= STABLE: Pass rate maintained at {curr_total_rate:.1f}%")
    else:
        print(f"✗ REGRESSION: Pass rate decreased from {prev_total_rate:.1f}% to {curr_total_rate:.1f}%")

    print(f"\nDetailed results saved to: timeout_retest_results.json")


if __name__ == "__main__":
    main()
