#!/usr/bin/env python3
"""
L1-L6 Comprehensive Evaluation Test Suite
Tests the lifecycle API migration to ensure no regressions.

This script runs 6 levels of increasing complexity:
L1: Simple file creation
L2: File with function + tests
L3: Multi-file package
L4: Package with tests and linting
L5: Data validator with schema
L6: Event bus system

Each level has a timeout and verifies:
- Success/failure status
- Files created correctly
- Tests pass (if applicable)
- Time taken
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import shutil
import json

# Add workspace to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from task_executor_agent import TaskExecutorAgent


class EvaluationTest:
    """Single test case for evaluation."""

    def __init__(
        self,
        level: str,
        goal: str,
        timeout_seconds: int,
        expected_files: list[str] | None = None,
        run_tests: bool = False,
        run_linting: bool = False,
    ):
        self.level = level
        self.goal = goal
        self.timeout_seconds = timeout_seconds
        self.expected_files = expected_files or []
        self.run_tests = run_tests
        self.run_linting = run_linting

        # Results
        self.passed = False
        self.status = None
        self.time_taken = 0.0
        self.files_created = []
        self.errors = []
        self.workspace = None
        self.result_dict = {}


def run_test(test: EvaluationTest) -> EvaluationTest:
    """
    Run a single evaluation test.

    Args:
        test: EvaluationTest instance

    Returns:
        Updated EvaluationTest with results
    """
    print(f"\n{'='*80}")
    print(f"Running {test.level}: {test.goal[:60]}")
    print(f"Timeout: {test.timeout_seconds}s")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        # Create agent (exclude ChatbotBehavior for autonomous execution)
        agent = TaskExecutorAgent(
            workspace=None,  # Create new isolated workspace
            goal=test.goal,
            timeout_seconds=test.timeout_seconds,
            exclude_behaviors=["ChatbotBehavior"]
        )

        # Get workspace before running
        test.workspace = str(agent.workspace)

        # Run the agent
        print(f"[{test.level}] Starting agent execution...")
        result = agent.run(max_rounds=50)  # Reasonable limit for these simple tasks

        # Record time taken
        test.time_taken = time.time() - start_time
        test.result_dict = result

        # Check result status
        test.status = result.get("status", "unknown")

        if test.status == "success":
            print(f"[{test.level}] ✅ Agent reported SUCCESS")

            # Verify files were created
            if agent.workspace and agent.workspace.exists():
                all_files = list(agent.workspace.rglob("*"))
                test.files_created = [
                    str(f.relative_to(agent.workspace))
                    for f in all_files
                    if f.is_file() and not f.name.startswith(".")
                ]

                print(f"[{test.level}] Files created ({len(test.files_created)}):")
                for f in sorted(test.files_created):
                    print(f"  - {f}")

                # Check expected files
                if test.expected_files:
                    missing_files = []
                    for expected in test.expected_files:
                        if expected not in test.files_created:
                            missing_files.append(expected)

                    if missing_files:
                        print(f"[{test.level}] ⚠️  Missing expected files:")
                        for f in missing_files:
                            print(f"  - {f}")
                        test.errors.append(f"Missing files: {', '.join(missing_files)}")

                # Run tests if requested
                if test.run_tests:
                    print(f"[{test.level}] Running tests...")
                    import subprocess
                    try:
                        # Run pytest with python -m to ensure workspace is in sys.path
                        result = subprocess.run(
                            ["python", "-m", "pytest", "-q"],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            cwd=str(agent.workspace),  # Run from workspace directory
                            env={**os.environ, "PYTHONPATH": str(agent.workspace)}  # Ensure workspace in path
                        )
                        if result.returncode == 0:
                            print(f"[{test.level}] ✅ Tests passed")
                        else:
                            print(f"[{test.level}] ❌ Tests failed")
                            print(result.stdout)
                            print(result.stderr)
                            test.errors.append("Tests failed")
                    except Exception as e:
                        print(f"[{test.level}] ⚠️  Could not run tests: {e}")
                        test.errors.append(f"Test execution error: {e}")

                # Run linting if requested
                if test.run_linting:
                    print(f"[{test.level}] Running linting...")
                    import subprocess
                    try:
                        result = subprocess.run(
                            ["ruff", "check", str(agent.workspace)],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            print(f"[{test.level}] ✅ Linting passed")
                        else:
                            # Ruff returns non-zero if there are warnings/errors
                            # Check if there are actual errors vs just warnings
                            output = result.stdout + result.stderr
                            if "error" in output.lower():
                                print(f"[{test.level}] ⚠️  Linting errors found")
                                print(output)
                                test.errors.append("Linting errors")
                            else:
                                print(f"[{test.level}] ℹ️  Linting warnings (non-blocking)")
                    except Exception as e:
                        print(f"[{test.level}] ⚠️  Could not run linting: {e}")

            # Mark as passed if no errors
            test.passed = len(test.errors) == 0

        elif test.status == "failure":
            print(f"[{test.level}] ❌ Agent reported FAILURE")
            reason = result.get("reason", "unknown")
            print(f"Reason: {reason}")
            test.errors.append(f"Agent failed: {reason}")
            test.passed = False

        else:
            print(f"[{test.level}] ⚠️  Unknown status: {test.status}")
            test.errors.append(f"Unknown status: {test.status}")
            test.passed = False

    except Exception as e:
        test.time_taken = time.time() - start_time
        test.status = "error"
        test.passed = False
        test.errors.append(f"Exception: {str(e)}")
        print(f"[{test.level}] ❌ Exception: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print(f"\n{test.level} Result: {'✅ PASS' if test.passed else '❌ FAIL'}")
    print(f"Time: {test.time_taken:.1f}s")
    if test.errors:
        print("Errors:")
        for error in test.errors:
            print(f"  - {error}")

    return test


def print_summary(tests: list[EvaluationTest]) -> None:
    """Print overall test summary."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    total = len(tests)
    passed = sum(1 for t in tests if t.passed)

    print(f"\nOverall: {passed}/{total} tests passed ({100*passed//total if total > 0 else 0}%)\n")

    # Table header
    print(f"{'Level':<8} {'Status':<12} {'Time':<10} {'Files':<8} {'Errors':<40}")
    print("-"*80)

    # Test results
    for test in tests:
        status_icon = "✅ PASS" if test.passed else "❌ FAIL"
        time_str = f"{test.time_taken:.1f}s"
        files_str = str(len(test.files_created))
        error_str = "; ".join(test.errors[:2]) if test.errors else ""
        if len(error_str) > 38:
            error_str = error_str[:35] + "..."

        print(f"{test.level:<8} {status_icon:<12} {time_str:<10} {files_str:<8} {error_str:<40}")

    print("\n" + "="*80)
    print(f"Total time: {sum(t.time_taken for t in tests):.1f}s")
    print(f"Pass rate: {passed}/{total} ({100*passed//total if total > 0 else 0}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Lifecycle API migration successful.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")


def save_results(tests: list[EvaluationTest]) -> None:
    """Save test results to JSON file."""
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"lifecycle_api_l1_l6_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "total_tests": len(tests),
        "passed": sum(1 for t in tests if t.passed),
        "failed": sum(1 for t in tests if not t.passed),
        "total_time": sum(t.time_taken for t in tests),
        "tests": [
            {
                "level": t.level,
                "goal": t.goal,
                "passed": t.passed,
                "status": t.status,
                "time_taken": t.time_taken,
                "files_created": t.files_created,
                "errors": t.errors,
                "workspace": t.workspace,
            }
            for t in tests
        ]
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved to: {results_file}")


def main():
    """Run all L1-L6 tests."""
    # Unset OLLAMA_MODEL to ensure tests use config file settings
    if "OLLAMA_MODEL" in os.environ:
        print(f"[ENV] Unsetting OLLAMA_MODEL (was: {os.environ['OLLAMA_MODEL']})")
        del os.environ["OLLAMA_MODEL"]

    print("="*80)
    print("L1-L6 Lifecycle API Evaluation Test Suite")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define test cases
    tests = [
        # L1: Simple file creation (2 min timeout)
        EvaluationTest(
            level="L1",
            goal="Create a file hello.py with the code: print('Hello World')",
            timeout_seconds=120,
            expected_files=["hello.py"],
        ),

        # L2: File with function (3 min timeout)
        EvaluationTest(
            level="L2",
            goal="Create calculator.py with an add(a, b) function that returns a + b. Create test_calculator.py that tests add() with pytest.",
            timeout_seconds=180,
            expected_files=["calculator.py", "test_calculator.py"],
            run_tests=True,
        ),

        # L3: Multi-file package (4 min timeout)
        EvaluationTest(
            level="L3",
            goal="Create a mathx package with __init__.py, add.py, subtract.py, multiply.py, and divide.py. Each module should have one function. Create tests/test_mathx.py that tests all functions.",
            timeout_seconds=240,
            expected_files=[
                "mathx/__init__.py",
                "mathx/add.py",
                "mathx/subtract.py",
                "mathx/multiply.py",
                "mathx/divide.py",
                "tests/test_mathx.py"
            ],
            run_tests=True,
        ),

        # L4: Package with tests and linting (4 min timeout)
        EvaluationTest(
            level="L4",
            goal="Create a calculator package with basic operations (add, subtract, multiply, divide). Write tests with pytest. Run ruff linting and fix any issues.",
            timeout_seconds=240,
            run_tests=True,
            run_linting=True,
        ),

        # L5: Data validator (5 min timeout)
        EvaluationTest(
            level="L5",
            goal="Create a data validator package that validates dictionaries against schemas. Include type checking, required fields, and range validation. Write comprehensive tests.",
            timeout_seconds=300,
            run_tests=True,
        ),

        # L6: Event system (5 min timeout)
        EvaluationTest(
            level="L6",
            goal="Create an event bus system that allows subscribing to events and publishing them. Support multiple subscribers per event. Write tests for subscribe, publish, and unsubscribe.",
            timeout_seconds=300,
            run_tests=True,
        ),
    ]

    # Run all tests
    results = []
    for test in tests:
        result = run_test(test)
        results.append(result)

        # Brief pause between tests
        time.sleep(2)

    # Print summary
    print_summary(results)

    # Save results
    save_results(results)

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Exit with appropriate code
    sys.exit(0 if all(t.passed for t in results) else 1)


if __name__ == "__main__":
    main()
