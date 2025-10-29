"""
Test framework for running test functions and reporting results.

The TestRunner class provides a simple interface to register test functions,
run them, and collect results. It is intentionally lightweight and does
not depend on external testing libraries.

Example usage:

    def test_add():
        assert 1 + 1 == 2

    runner = TestRunner()
    runner.add_test(test_add)
    runner.run()
    runner.report()

The report prints a summary of passed/failed tests and details for failures.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class TestResult:
    name: str
    passed: bool
    exception: Exception | None = None
    traceback: str | None = None

class TestRunner:
    """Simple test runner.

    The runner keeps a list of test callables. Each test is expected to
    raise an exception (typically AssertionError) on failure. The runner
    captures the exception and traceback, marking the test as failed.
    """

    def __init__(self) -> None:
        self._tests: List[Callable[[], None]] = []
        self._results: List[TestResult] = []

    def add_test(self, test: Callable[[], None]) -> None:
        """Register a test function.

        Parameters
        ----------
        test: Callable[[], None]
            A zero‑argument function that performs assertions.
        """
        self._tests.append(test)

    def run(self) -> None:
        """Execute all registered tests and store results."""
        self._results.clear()
        for test in self._tests:
            name = getattr(test, "__name__", str(test))
            try:
                test()
                self._results.append(TestResult(name=name, passed=True))
            except Exception as exc:  # noqa: BLE001 - intentional broad except
                tb = traceback.format_exc()
                self._results.append(
                    TestResult(name=name, passed=False, exception=exc, traceback=tb)
                )

    def report(self) -> None:
        """Print a summary of test results to stdout."""
        passed = sum(1 for r in self._results if r.passed)
        failed = len(self._results) - passed
        total = len(self._results)
        print(f"\nTest run: {total} test{'s' if total != 1 else ''}\n")
        print(f"Passed: {passed}\nFailed: {failed}\n")
        if failed:
            print("Failed tests details:\n")
            for r in self._results:
                if not r.passed:
                    print(f"- {r.name}: {r.exception}")
                    print(r.traceback)

    def results(self) -> List[TestResult]:
        """Return the list of TestResult objects for programmatic access."""
        return list(self._results)

# Demo when run directly
if __name__ == "__main__":
    def sample_pass():
        assert 2 + 2 == 4

    def sample_fail():
        assert 2 + 2 == 5

    runner = TestRunner()
    runner.add_test(sample_pass)
    runner.add_test(sample_fail)
    runner.run()
    runner.report()
