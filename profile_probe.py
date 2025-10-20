#!/usr/bin/env python3
"""Detailed profiling of probe_state bottleneck."""
import time
import subprocess
from pathlib import Path

def timeit(name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"  {name:40s}: {elapsed:7.2f}ms")
            return result
        return wrapper
    return decorator

@timeit("Path.exists() - mathx/__init__.py")
def check_pkg():
    return Path("mathx/__init__.py").exists()

@timeit("Path.exists() - tests/test_mathx.py")
def check_tests():
    return Path("tests/test_mathx.py").exists()

@timeit("Path.exists() - pyproject.toml")
def check_pyproject():
    return Path("pyproject.toml").exists()

@timeit("ruff check . (full)")
def run_ruff_full():
    return subprocess.run(["ruff", "check", "."], capture_output=True, text=True, timeout=5)

@timeit("ruff check . --exit-zero")
def run_ruff_exit_zero():
    return subprocess.run(["ruff", "check", ".", "--exit-zero"], capture_output=True, text=True, timeout=5)

@timeit("ruff check --select E,F (minimal)")
def run_ruff_minimal():
    return subprocess.run(["ruff", "check", ".", "--select", "E,F"], capture_output=True, text=True, timeout=5)

@timeit("pytest tests/ -q")
def run_pytest_quiet():
    return subprocess.run(["pytest", "tests/", "-q"], capture_output=True, text=True, timeout=10)

@timeit("pytest tests/ -q --collect-only")
def run_pytest_collect():
    return subprocess.run(["pytest", "tests/", "-q", "--collect-only"], capture_output=True, text=True, timeout=5)

@timeit("pytest tests/ -q -x (fail fast)")
def run_pytest_failfast():
    return subprocess.run(["pytest", "tests/", "-q", "-x"], capture_output=True, text=True, timeout=10)

@timeit("pytest --version")
def run_pytest_version():
    return subprocess.run(["pytest", "--version"], capture_output=True, text=True, timeout=5)

print("="*60)
print("PROBE_STATE DETAILED BREAKDOWN")
print("="*60)

print("\n--- File existence checks ---")
check_pkg()
check_tests()
check_pyproject()

print("\n--- Ruff variations ---")
try:
    run_ruff_full()
    run_ruff_exit_zero()
    run_ruff_minimal()
except Exception as e:
    print(f"  Error: {e}")

print("\n--- Pytest variations ---")
try:
    run_pytest_version()
    run_pytest_collect()
    run_pytest_quiet()
    run_pytest_failfast()
except Exception as e:
    print(f"  Error: {e}")

print("\n--- Combined (typical probe_state) ---")
start = time.perf_counter()
exists_pkg = Path("mathx/__init__.py").exists()
exists_tests = Path("tests/test_mathx.py").exists()
exists_pyproject = Path("pyproject.toml").exists()
ruff_result = subprocess.run(["ruff", "check", "."], capture_output=True, text=True, timeout=5)
pytest_result = subprocess.run(["pytest", "tests/", "-q"], capture_output=True, text=True, timeout=10)
total = (time.perf_counter() - start) * 1000
print(f"  Total probe_state time                  : {total:7.2f}ms")

print("\n--- Analysis ---")
print(f"  File checks negligible: ~0.5ms total")
print(f"  Ruff is primary bottleneck: ~20-30ms")
print(f"  Pytest is major bottleneck: ~200-250ms")
print(f"  Combined: ~250-280ms per probe")

print("\n--- Optimization Ideas ---")
print("  1. Cache probe results (avoid re-running on every round)")
print("  2. Skip pytest if tests/ doesn't exist yet")
print("  3. Run ruff + pytest in parallel (concurrent.futures)")
print("  4. Use --exit-zero for ruff to avoid error handling overhead")
print("  5. Probe less frequently (every N rounds, not every round)")
print("  6. Use pytest --collect-only for existence check")
print("  7. Skip probes if no files were written since last probe")
