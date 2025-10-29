#!/usr/bin/env python3
"""Quick sanity check for L1 tasks after bug fixes."""
import sys
import tempfile
from pathlib import Path
import subprocess
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_executor_agent import TaskExecutorAgent


# L1 tasks from evaluation suite
L1_TASKS = [
    {
        "name": "simple_function",
        "goal": "Create a Python file called greet.py with a function greet(name) that returns 'Hello, {name}!'",
        "expected_files": ["greet.py"],
        "validation": ["python", "-c", "from greet import greet; assert greet('World') == 'Hello, World!'"],
    },
    {
        "name": "simple_math",
        "goal": "Create math_utils.py with functions: add(a,b), subtract(a,b), multiply(a,b), divide(a,b). Include proper error handling for division by zero.",
        "expected_files": ["math_utils.py"],
        "validation": ["python", "-c", "from math_utils import add, divide; assert add(2,3)==5; assert divide(10,2)==5"],
    },
    {
        "name": "list_operations",
        "goal": "Create list_utils.py with functions: get_max(lst), get_min(lst), get_average(lst), remove_duplicates(lst)",
        "expected_files": ["list_utils.py"],
        "validation": ["python", "-c", "from list_utils import get_max, remove_duplicates; assert get_max([1,5,3])==5; assert remove_duplicates([1,2,2,3])==[1,2,3]"],
    },
]


def run_task(task_def, workspace):
    """Run a single task and validate results."""
    print(f"\n{'='*70}")
    print(f"TASK: {task_def['name']}")
    print(f"Goal: {task_def['goal']}")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        # Create and run agent
        agent = TaskExecutorAgent(
            workspace=workspace,
            goal=task_def['goal'],
            max_rounds=20,
            model="gpt-oss:20b"
        )

        result = agent.run()
        duration = time.time() - start_time

        workspace_dir = agent.workspace_manager.workspace_dir
        print(f"\nAgent finished in {duration:.1f}s")
        print(f"Status: {result.get('status')}")
        print(f"Workspace: {workspace_dir}")

        # Check files exist
        files_created = []
        for expected_file in task_def['expected_files']:
            file_path = workspace_dir / expected_file
            if file_path.exists():
                files_created.append(expected_file)
                print(f"  ✓ {expected_file} exists")
            else:
                print(f"  ✗ {expected_file} missing")

        all_files_exist = len(files_created) == len(task_def['expected_files'])

        # Run validation
        validation_passed = False
        if all_files_exist:
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(workspace_dir)
                validation_result = subprocess.run(
                    task_def['validation'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if validation_result.returncode == 0:
                    validation_passed = True
                    print(f"  ✓ Validation passed")
                else:
                    print(f"  ✗ Validation failed")
                    if validation_result.stderr:
                        print(f"    Error: {validation_result.stderr[:200]}")
            except Exception as e:
                print(f"  ✗ Validation error: {e}")
            finally:
                os.chdir(original_cwd)

        success = all_files_exist and validation_passed

        return {
            "name": task_def['name'],
            "success": success,
            "duration": duration,
            "files_ok": all_files_exist,
            "validation_ok": validation_passed,
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"  ✗ Exception: {e}")
        return {
            "name": task_def['name'],
            "success": False,
            "duration": duration,
            "files_ok": False,
            "validation_ok": False,
            "error": str(e)
        }


def main():
    """Run L1 sanity check."""
    print("="*70)
    print("L1 TASKS SANITY CHECK")
    print("Testing basic single-file tasks after bug fixes")
    print("="*70)

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        for task_def in L1_TASKS:
            result = run_task(task_def, workspace)
            results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for r in results if r['success'])
    total = len(results)
    pass_rate = (passed / total * 100) if total > 0 else 0

    for r in results:
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        print(f"{status:8} {r['name']:25} ({r['duration']:.1f}s)")

    print(f"\nPass rate: {passed}/{total} ({pass_rate:.1f}%)")

    if pass_rate >= 66:
        print("✓ Good! L1 tasks are passing at acceptable rate")
    else:
        print("✗ Poor pass rate - more debugging needed")

    return 0 if pass_rate >= 66 else 1


if __name__ == "__main__":
    sys.exit(main())
