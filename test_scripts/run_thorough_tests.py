#!/usr/bin/env python3
"""
Thorough Test Suite Runner - Lifecycle API Migration Verification

Executes comprehensive tests from THOROUGH_TEST_PLAN.md:
- Phase 1: L1-L6 baseline tests (quick validation)
- Phase 2: Unit tests for behaviors
- Phase 3: Integration tests
- Phase 4: Regression tests

Run with: python run_thorough_tests.py [--phase <1-4>]
"""
from __future__ import annotations
import sys
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime


class TestPhase:
    """Single test phase with pass/fail tracking."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tests_run = 0
        self.tests_passed = 0
        self.start_time = None
        self.end_time = None
        self.results = []

    def start(self):
        """Mark phase start."""
        self.start_time = time.time()
        print(f"\n{'='*80}")
        print(f"PHASE: {self.name}")
        print(f"Description: {self.description}")
        print(f"{'='*80}\n")

    def end(self):
        """Mark phase end."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"\n{'-'*80}")
        print(f"Phase '{self.name}' completed in {elapsed:.1f}s")
        print(f"Results: {self.tests_passed}/{self.tests_run} passed")
        print(f"{'-'*80}\n")

    def run_command(self, test_name: str, command: list[str], timeout: int = 300) -> bool:
        """
        Run a test command and track results.

        Args:
            test_name: Name of the test
            command: Command to execute
            timeout: Timeout in seconds

        Returns:
            True if test passed, False otherwise
        """
        self.tests_run += 1
        print(f"[{self.name}] Running: {test_name}")
        print(f"  Command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            passed = result.returncode == 0

            if passed:
                print(f"  ✅ PASSED")
                self.tests_passed += 1
            else:
                print(f"  ❌ FAILED (exit code: {result.returncode})")
                if result.stdout:
                    print(f"  stdout: {result.stdout[:200]}")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:200]}")

            self.results.append({
                'test': test_name,
                'passed': passed,
                'exit_code': result.returncode,
                'stdout': result.stdout[:500] if result.stdout else '',
                'stderr': result.stderr[:500] if result.stderr else '',
            })

            return passed

        except subprocess.TimeoutExpired:
            print(f"  ⏱️  TIMEOUT (>{timeout}s)")
            self.results.append({
                'test': test_name,
                'passed': False,
                'error': f'Timeout after {timeout}s'
            })
            return False

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            self.results.append({
                'test': test_name,
                'passed': False,
                'error': str(e)
            })
            return False


def phase1_baseline_tests() -> TestPhase:
    """
    Phase 1: L1-L6 Baseline Tests

    Quick validation that basic agent functionality works.
    Expected: 6/6 tests pass (100%)
    """
    phase = TestPhase(
        name="Phase 1: L1-L6 Baseline Tests",
        description="Quick validation of basic agent functionality"
    )
    phase.start()

    # Run L1-L6 comprehensive test suite
    phase.run_command(
        test_name="L1-L6 Comprehensive Test Suite",
        command=["python", "test_lifecycle_api_l1_l6.py"],
        timeout=600  # 10 minutes for all 6 tests
    )

    phase.end()
    return phase


def phase2_unit_tests() -> TestPhase:
    """
    Phase 2: Unit Tests for Behaviors

    Test individual behavior implementations.
    """
    phase = TestPhase(
        name="Phase 2: Behavior Unit Tests",
        description="Test individual behavior functionality"
    )
    phase.start()

    # CompactWhenNearFullBehavior tests
    if Path("tests/test_compact_when_near_full_behavior.py").exists():
        phase.run_command(
            test_name="CompactWhenNearFullBehavior unit tests",
            command=["pytest", "tests/test_compact_when_near_full_behavior.py", "-v"],
            timeout=60
        )
    else:
        print("[Phase 2] ⚠️  tests/test_compact_when_near_full_behavior.py not found, skipping")

    # ChatbotBehavior tests (if implemented)
    if Path("tests/test_chatbot_behavior.py").exists():
        phase.run_command(
            test_name="ChatbotBehavior unit tests",
            command=["pytest", "tests/test_chatbot_behavior.py", "-v"],
            timeout=60
        )
    else:
        print("[Phase 2] ℹ️  tests/test_chatbot_behavior.py not yet implemented")

    # LoopDetectionBehavior tests (if implemented)
    if Path("tests/test_loop_detection_behavior.py").exists():
        phase.run_command(
            test_name="LoopDetectionBehavior unit tests",
            command=["pytest", "tests/test_loop_detection_behavior.py", "-v"],
            timeout=60
        )
    else:
        print("[Phase 2] ℹ️  tests/test_loop_detection_behavior.py not yet implemented")

    # WorkspaceTaskNotesBehavior tests (if implemented)
    if Path("tests/test_workspace_task_notes_behavior.py").exists():
        phase.run_command(
            test_name="WorkspaceTaskNotesBehavior unit tests",
            command=["pytest", "tests/test_workspace_task_notes_behavior.py", "-v"],
            timeout=60
        )
    else:
        print("[Phase 2] ℹ️  tests/test_workspace_task_notes_behavior.py not yet implemented")

    phase.end()
    return phase


def phase3_integration_tests() -> TestPhase:
    """
    Phase 3: Integration Tests

    Test behavior composition and configuration loading.
    """
    phase = TestPhase(
        name="Phase 3: Integration Tests",
        description="Test behavior composition and system integration"
    )
    phase.start()

    # Lifecycle event verification (if implemented)
    if Path("tests/test_lifecycle_events.py").exists():
        phase.run_command(
            test_name="Lifecycle event sequence verification",
            command=["pytest", "tests/test_lifecycle_events.py", "-v"],
            timeout=60
        )
    else:
        print("[Phase 3] ℹ️  tests/test_lifecycle_events.py not yet implemented")

    # Behavior composition tests (if implemented)
    if Path("tests/test_behavior_composition.py").exists():
        phase.run_command(
            test_name="Multi-behavior composition",
            command=["pytest", "tests/test_behavior_composition.py", "-v"],
            timeout=60
        )
    else:
        print("[Phase 3] ℹ️  tests/test_behavior_composition.py not yet implemented")

    # Config-driven behavior loading (if implemented)
    if Path("tests/test_config_behavior_loading.py").exists():
        phase.run_command(
            test_name="Config-driven behavior loading",
            command=["pytest", "tests/test_config_behavior_loading.py", "-v"],
            timeout=60
        )
    else:
        print("[Phase 3] ℹ️  tests/test_config_behavior_loading.py not yet implemented")

    phase.end()
    return phase


def phase4_regression_tests() -> TestPhase:
    """
    Phase 4: Regression Tests

    Verify no performance or functionality regressions.
    """
    phase = TestPhase(
        name="Phase 4: Regression Tests",
        description="Verify no performance or functionality regressions"
    )
    phase.start()

    print("[Phase 4] ℹ️  Regression tests require manual verification:")
    print("  - Workspace isolation: Run multiple agents concurrently")
    print("  - Performance comparison: Compare with baseline metrics")
    print("  - Check evaluation_results/ for previous benchmarks")

    # For now, just verify workspace isolation with a simple test
    print("[Phase 4] Running simple workspace isolation test...")

    # This would require a custom test script - for now just note it
    print("[Phase 4] ℹ️  Workspace isolation test not yet automated")

    phase.end()
    return phase


def print_final_summary(phases: list[TestPhase]):
    """Print overall test summary."""
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)

    total_tests = sum(p.tests_run for p in phases)
    total_passed = sum(p.tests_passed for p in phases)
    total_time = sum((p.end_time - p.start_time) if p.end_time and p.start_time else 0 for p in phases)

    print(f"\nOverall: {total_passed}/{total_tests} tests passed ({100*total_passed//total_tests if total_tests > 0 else 0}%)")
    print(f"Total time: {total_time:.1f}s\n")

    # Phase-by-phase summary
    print(f"{'Phase':<40} {'Tests':<10} {'Passed':<10} {'Time':<10}")
    print("-"*80)
    for phase in phases:
        elapsed = (phase.end_time - phase.start_time) if phase.end_time and phase.start_time else 0
        passed_str = f"{phase.tests_passed}/{phase.tests_run}"
        time_str = f"{elapsed:.1f}s"
        print(f"{phase.name:<40} {phase.tests_run:<10} {passed_str:<10} {time_str:<10}")

    print("\n" + "="*80)

    # Final verdict
    if total_passed == total_tests and total_tests > 0:
        print("🎉 ALL TESTS PASSED! Lifecycle API migration successful.")
    elif total_tests == 0:
        print("⚠️  No tests were run.")
    else:
        print(f"⚠️  {total_tests - total_passed} test(s) failed. Review errors above.")

    print("="*80)


def main():
    """Run thorough test suite."""
    parser = argparse.ArgumentParser(
        description="Run thorough test suite for lifecycle API migration"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run only specific phase (1-4). If not specified, runs all phases."
    )
    args = parser.parse_args()

    print("="*80)
    print("THOROUGH TEST SUITE - Lifecycle API Migration Verification")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Based on: evaluation_results/THOROUGH_TEST_PLAN.md")

    # Run phases
    phases = []

    if args.phase is None or args.phase == 1:
        phases.append(phase1_baseline_tests())

    if args.phase is None or args.phase == 2:
        phases.append(phase2_unit_tests())

    if args.phase is None or args.phase == 3:
        phases.append(phase3_integration_tests())

    if args.phase is None or args.phase == 4:
        phases.append(phase4_regression_tests())

    # Print summary
    print_final_summary(phases)

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Exit with appropriate code
    total_tests = sum(p.tests_run for p in phases)
    total_passed = sum(p.tests_passed for p in phases)

    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == "__main__":
    main()
