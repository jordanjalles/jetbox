#!/usr/bin/env python3
"""Run systematic stress tests on the agent and log results."""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def slugify(text: str, max_length: int = 50) -> str:
    """Convert text to URL-friendly slug (matches workspace_manager.py logic)."""
    slug = text.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')[:max_length].rstrip('-')
    return slug or 'workspace'


def get_workspace_path(task_description: str) -> Path:
    """Get the agent's workspace path for a given task."""
    workspace_slug = slugify(task_description)
    return Path(".agent_workspace") / workspace_slug


# Test suite definition
TESTS = [
    # Level 1: Basic
    {
        "id": "L1-1",
        "level": 1,
        "name": "Hello World",
        "task": "Write a hello world script",
        "expected_files": ["hello_world.py"],
        "timeout": 60,
    },
    {
        "id": "L1-2",
        "level": 1,
        "name": "Simple Math Function",
        "task": "Create a function that adds two numbers and save it to math_utils.py",
        "expected_files": ["math_utils.py"],
        "timeout": 60,
    },
    {
        "id": "L1-3",
        "level": 1,
        "name": "Basic Test File",
        "task": "Create math_utils.py with an add(a,b) function, then write test_math_utils.py with tests for it",
        "expected_files": ["math_utils.py", "test_math_utils.py"],
        "timeout": 120,
    },

    # Level 2: Intermediate
    {
        "id": "L2-1",
        "level": 2,
        "name": "Calculator with Tests",
        "task": "Create calculator.py with add, subtract, multiply, and divide functions. Write test_calculator.py with tests for all functions.",
        "expected_files": ["calculator.py", "test_calculator.py"],
        "timeout": 180,
        "verify_cmd": ["python", "-m", "pytest", "test_calculator.py", "-q"],
    },
    {
        "id": "L2-2",
        "level": 2,
        "name": "Rock Paper Scissors",
        "task": "Create a rock-paper-scissors game in rps.py that can be played in the terminal",
        "expected_files": ["rps.py"],
        "timeout": 180,
    },
    {
        "id": "L2-3",
        "level": 2,
        "name": "Package with Modules",
        "task": "Create a mathx package with mathx/basic.py (add, subtract), mathx/advanced.py (multiply, divide), mathx/__init__.py, and tests/test_mathx.py testing all functions",
        "expected_files": ["mathx/__init__.py", "mathx/basic.py", "mathx/advanced.py", "tests/test_mathx.py"],
        "timeout": 240,
    },

    # Level 3: Advanced
    {
        "id": "L3-1",
        "level": 3,
        "name": "Refactor to Class",
        "task": "Create calculator.py with add, subtract, multiply functions. Then refactor it to use a Calculator class with methods instead of standalone functions.",
        "expected_files": ["calculator.py"],
        "timeout": 240,
        "setup": lambda task: (get_workspace_path(task).mkdir(parents=True, exist_ok=True),
                              (get_workspace_path(task) / "calculator.py").write_text(
            "def add(a, b):\n    return a + b\n\n"
            "def subtract(a, b):\n    return a - b\n\n"
            "def multiply(a, b):\n    return a * b\n"
        ))[1],
    },
    {
        "id": "L3-2",
        "level": 3,
        "name": "Fix Buggy Code",
        "task": "Fix all the bugs in buggy.py and make sure it runs without errors",
        "expected_files": ["buggy.py"],
        "timeout": 240,
        "setup": lambda task: (get_workspace_path(task).mkdir(parents=True, exist_ok=True),
                              (get_workspace_path(task) / "buggy.py").write_text(
            "def divide(a, b):\n"
            "    return a / b  # Bug: no zero check\n\n"
            "def get_item(lst, idx):\n"
            "    return lst[idx]  # Bug: no bounds check\n\n"
            "def parse_int(s):\n"
            "    return int(s)  # Bug: no error handling\n\n"
            "# Bug: infinite loop\n"
            "def count_to_ten():\n"
            "    i = 0\n"
            "    while i < 10:\n"
            "        print(i)\n"
            "        # Missing: i += 1\n"
        ))[1],
    },
    {
        "id": "L3-3",
        "level": 3,
        "name": "Add Feature to Package",
        "task": "Add a square_root function to mathx/advanced.py and add tests for it in tests/test_mathx.py. Make sure all existing tests still pass.",
        "expected_files": ["mathx/advanced.py", "tests/test_mathx.py"],
        "timeout": 240,
        "setup": lambda task: setup_mathx_package(task),
    },

    # Level 4: Expert
    {
        "id": "L4-1",
        "level": 4,
        "name": "TodoList with Persistence",
        "task": "Create a TodoList class in todo.py with methods: add_task, remove_task, mark_complete, list_pending, save_to_file, and load_from_file. Use JSON for persistence. Include tests.",
        "expected_files": ["todo.py", "test_todo.py"],
        "timeout": 300,
    },
    {
        "id": "L4-2",
        "level": 4,
        "name": "Debug Failing Tests",
        "task": "The tests in test_broken.py are failing. Debug the code in broken.py and fix all issues so tests pass.",
        "expected_files": ["broken.py", "test_broken.py"],
        "timeout": 300,
        "setup": lambda task: setup_failing_tests(task),
    },
    {
        "id": "L4-3",
        "level": 4,
        "name": "Optimize Slow Code",
        "task": "The fibonacci function in slow_fib.py is very slow. Optimize it using memoization or dynamic programming to make it faster.",
        "expected_files": ["slow_fib.py"],
        "timeout": 300,
        "setup": lambda task: (get_workspace_path(task).mkdir(parents=True, exist_ok=True),
                              (get_workspace_path(task) / "slow_fib.py").write_text(
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n-1) + fibonacci(n-2)\n"
        ))[1],
    },

    # Level 5: Extreme
    {
        "id": "L5-1",
        "level": 5,
        "name": "Multi-Format Data Pipeline",
        "task": "Create a data processing module that can read CSV, JSON, and XML files and convert between formats. Include a unified interface.",
        "expected_files": ["data_pipeline.py"],
        "timeout": 360,
    },
    {
        "id": "L5-2",
        "level": 5,
        "name": "Large-Scale Refactoring",
        "task": "Refactor the entire mathx package to use a unified MathOperation base class that all operations inherit from. Maintain all existing functionality and tests.",
        "expected_files": ["mathx/base.py", "mathx/basic.py", "mathx/advanced.py"],
        "timeout": 360,
        "setup": lambda task: setup_mathx_package(task),
    },
    {
        "id": "L5-3",
        "level": 5,
        "name": "Ambiguous Requirements",
        "task": "Create a useful utility for working with text files",
        "expected_files": [],  # Unclear what files should be created
        "timeout": 300,
    },
]


def setup_mathx_package(task_description: str) -> None:
    """Set up a basic mathx package for testing IN the agent's workspace."""
    workspace = get_workspace_path(task_description)
    workspace.mkdir(parents=True, exist_ok=True)

    (workspace / "mathx").mkdir(exist_ok=True)
    (workspace / "tests").mkdir(exist_ok=True)

    (workspace / "mathx" / "__init__.py").write_text(
        "from mathx.basic import add, subtract\n"
        "from mathx.advanced import multiply, divide\n"
    )

    (workspace / "mathx" / "basic.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def subtract(a, b):\n"
        "    return a - b\n"
    )

    (workspace / "mathx" / "advanced.py").write_text(
        "def multiply(a, b):\n"
        "    return a * b\n\n"
        "def divide(a, b):\n"
        "    if b == 0:\n"
        "        raise ValueError('Division by zero')\n"
        "    return a / b\n"
    )

    (workspace / "tests" / "test_mathx.py").write_text(
        "from mathx import add, subtract, multiply, divide\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_subtract():\n"
        "    assert subtract(5, 3) == 2\n\n"
        "def test_multiply():\n"
        "    assert multiply(3, 4) == 12\n\n"
        "def test_divide():\n"
        "    assert divide(10, 2) == 5\n"
    )


def setup_failing_tests(task_description: str) -> None:
    """Set up broken code with failing tests IN the agent's workspace."""
    workspace = get_workspace_path(task_description)
    workspace.mkdir(parents=True, exist_ok=True)

    (workspace / "broken.py").write_text(
        "def reverse_string(s):\n"
        "    # Bug: returns None instead of reversed string\n"
        "    s[::-1]\n\n"
        "def sum_list(numbers):\n"
        "    # Bug: returns 0 for empty list, should work\n"
        "    total = 0\n"
        "    for n in numbers:\n"
        "        total += numbers  # Bug: should be 'n' not 'numbers'\n"
        "    return total\n\n"
        "def is_even(n):\n"
        "    # Bug: logic is inverted\n"
        "    return n % 2 == 1\n"
    )

    (workspace / "test_broken.py").write_text(
        "from broken import reverse_string, sum_list, is_even\n\n"
        "def test_reverse_string():\n"
        "    assert reverse_string('hello') == 'olleh'\n\n"
        "def test_sum_list():\n"
        "    assert sum_list([1, 2, 3]) == 6\n"
        "    assert sum_list([]) == 0\n\n"
        "def test_is_even():\n"
        "    assert is_even(2) == True\n"
        "    assert is_even(3) == False\n"
    )


def clean_workspace() -> None:
    """Clean up workspace before test."""
    # Remove entire .agent_workspace
    if Path(".agent_workspace").exists():
        shutil.rmtree(".agent_workspace")

    # Remove agent state
    if Path(".agent_context").exists():
        shutil.rmtree(".agent_context")

    # Remove log files
    for log_file in ["agent_v2.log", "agent_ledger.log", "agent.log"]:
        if Path(log_file).exists():
            Path(log_file).unlink()

    # Remove __pycache__ directories recursively
    for pycache in Path(".").rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)

    # Define files to preserve
    preserve_files = {
        "agent.py", "context_manager.py", "status_display.py",
        "workspace_manager.py", "completion_detector.py", "agent_config.py",
        "test_status_display.py", "run_stress_tests.py", "diag_speed.py",
        "pyproject.toml", "README.md", "CLAUDE.md",
        "AGENT_ARCHITECTURE.md", "STATUS_DISPLAY.md",
        "TEST_IMPROVEMENTS_PROPOSAL.md", "PHASE2_REVISED_PROPOSAL.md",
        "stress_test_results.json", "agent_stress_tests.md",
        ".gitignore", ".git"
    }

    # REMOVED: DO NOT delete files in root - too dangerous!
    # The agent creates files in .agent_workspace which gets cleaned above
    # Test artifacts stay in workspace, not root


def run_test(test: dict[str, Any]) -> dict[str, Any]:
    """Run a single test and collect results."""
    print(f"\n{'='*70}")
    print(f"Running {test['id']}: {test['name']}")
    print(f"Task: {test['task']}")
    print(f"{'='*70}\n")

    result = {
        "id": test["id"],
        "level": test["level"],
        "name": test["name"],
        "task": test["task"],
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "rounds": 0,
        "duration": 0,
        "output": "",
        "error": None,
        "files_created": [],
        "failure_mode": None,
    }

    # Setup
    clean_workspace()
    if "setup" in test:
        test["setup"](test["task"])

    # Run agent
    start_time = time.time()
    try:
        proc = subprocess.run(
            ["python", "agent.py", test["task"]],
            capture_output=True,
            text=True,
            timeout=test["timeout"],
        )
        result["duration"] = time.time() - start_time
        result["output"] = proc.stdout + proc.stderr

        # Check if agent completed - expanded success patterns
        success_patterns = [
            "Goal achieved",
            "All tasks finished",
            "goal_complete",
            "Successfully created",
            r"✓.*complete",
            "Task completed successfully",
            "marked complete",
        ]

        for pattern in success_patterns:
            if "\\." in pattern or "*" in pattern:  # Regex pattern
                import re
                if re.search(pattern, result["output"], re.IGNORECASE):
                    result["success"] = True
                    break
            else:  # Simple substring match
                if pattern in result["output"]:
                    result["success"] = True
                    break

        # Check failure modes if not successful
        if not result["success"]:
            if "Hit MAX_ROUNDS" in result["output"]:
                result["failure_mode"] = "max_rounds_exceeded"
            elif "loop" in result["output"].lower() and "detect" in result["output"].lower():
                result["failure_mode"] = "infinite_loop"
            else:
                result["failure_mode"] = "unknown_failure"

    except subprocess.TimeoutExpired:
        result["duration"] = time.time() - start_time
        result["failure_mode"] = "timeout"
        result["error"] = f"Timeout after {test['timeout']}s"
    except Exception as e:
        result["duration"] = time.time() - start_time
        result["failure_mode"] = "execution_error"
        result["error"] = str(e)

    # Check expected files
    result["files_created"] = [
        str(f) for f in test.get("expected_files", [])
        if Path(f).exists()
    ]

    # Verify with command if provided
    if "verify_cmd" in test and result["success"]:
        try:
            verify = subprocess.run(
                test["verify_cmd"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if verify.returncode != 0:
                result["success"] = False
                result["failure_mode"] = "verification_failed"
                result["error"] = verify.stderr
        except Exception as e:
            result["success"] = False
            result["failure_mode"] = "verification_error"
            result["error"] = str(e)

    # Count rounds from output
    for line in result["output"].split("\n"):
        if "ROUND" in line and ":" in line:
            try:
                round_no = int(line.split("ROUND")[1].split(":")[0].strip())
                result["rounds"] = max(result["rounds"], round_no)
            except:
                pass

    # Print summary
    status = "✓ PASS" if result["success"] else "✗ FAIL"
    print(f"\n{status} - {result['duration']:.1f}s - {result['rounds']} rounds")
    if result["failure_mode"]:
        print(f"Failure mode: {result['failure_mode']}")
    if result["error"]:
        print(f"Error: {result['error'][:200]}")

    return result


def main() -> None:
    """Run all stress tests."""
    print("Agent Stress Test Suite")
    print("="*70)

    # Allow running specific levels
    levels_to_run = None
    if len(sys.argv) > 1:
        levels_to_run = [int(x) for x in sys.argv[1].split(",")]
        print(f"Running only levels: {levels_to_run}")

    results = []

    for test in TESTS:
        if levels_to_run and test["level"] not in levels_to_run:
            continue

        result = run_test(test)
        results.append(result)

        # Save incremental results
        with open("stress_test_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Generate summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    by_level = {}
    for r in results:
        level = r["level"]
        if level not in by_level:
            by_level[level] = {"total": 0, "passed": 0, "failed": 0}
        by_level[level]["total"] += 1
        if r["success"]:
            by_level[level]["passed"] += 1
        else:
            by_level[level]["failed"] += 1

    for level in sorted(by_level.keys()):
        stats = by_level[level]
        pct = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"\nLevel {level}: {stats['passed']}/{stats['total']} passed ({pct:.0f}%)")

    # Failure modes
    failure_modes = {}
    for r in results:
        if not r["success"] and r["failure_mode"]:
            fm = r["failure_mode"]
            if fm not in failure_modes:
                failure_modes[fm] = []
            failure_modes[fm].append(r["id"])

    if failure_modes:
        print("\nFailure Modes:")
        for mode, tests in failure_modes.items():
            print(f"  {mode}: {', '.join(tests)}")

    print(f"\nDetailed results saved to: stress_test_results.json")


if __name__ == "__main__":
    main()
