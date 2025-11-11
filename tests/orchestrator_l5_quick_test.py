#!/usr/bin/env python3
"""
Quick orchestrator test with 1-2 L5 tasks to validate mode system.
"""
import sys
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.evaluation_suite_extended import get_extended_tasks


def run_single_task(task_name, timeout_minutes=10):
    """Run a single task by name."""
    all_tasks = get_extended_tasks()
    task = next((t for t in all_tasks if t.name == task_name), None)

    if not task:
        print(f"Task '{task_name}' not found")
        return False

    workspace = tempfile.mkdtemp(prefix=f"quick_test_{task.name}_")
    workspace_path = Path(workspace)

    print(f"\n{'='*70}")
    print(f"Testing: L{task.level} - {task.name}")
    print(f"Description: {task.description}")
    print(f"Goal: {task.goal}")
    print(f"Workspace: {workspace}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    try:
        cmd = [
            sys.executable, "/workspace/agent.py",
            "--team", "default",
            task.goal
        ]

        print(f"Running: {' '.join(cmd[:3])} '{cmd[3]}'")
        print(f"Timeout: {timeout_minutes} minutes\n")

        result = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n{'='*70}")
        print(f"RESULT: L{task.level} - {task.name}")
        print(f"{'='*70}")
        print(f"Duration: {duration:.1f}s")
        print(f"Exit code: {result.returncode}")

        # Check files
        print(f"\nExpected files:")
        for expected_file in task.expected_files:
            file_path = workspace_path / expected_file
            exists = file_path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {expected_file}")

        # Show workspace contents
        print(f"\nWorkspace contents:")
        for item in sorted(workspace_path.rglob("*")):
            if item.is_file() and not item.name.startswith('.'):
                rel_path = item.relative_to(workspace_path)
                size = item.stat().st_size
                print(f"  - {rel_path} ({size} bytes)")

        # Run validation
        print(f"\nValidation:")
        validation_passed = True
        for val_cmd in task.validation_commands:
            try:
                val_result = subprocess.run(
                    val_cmd,
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                passed = val_result.returncode == 0
                status = "✓" if passed else "✗"
                print(f"  {status} {' '.join(val_cmd)}")
                if not passed:
                    print(f"     Error: {val_result.stderr[:200]}")
                    validation_passed = False
            except Exception as e:
                print(f"  ✗ {' '.join(val_cmd)}")
                print(f"     Error: {e}")
                validation_passed = False

        # Show stdout (last 50 lines)
        print(f"\nAgent output (last 50 lines):")
        for line in result.stdout.splitlines()[-50:]:
            print(f"  {line}")

        if result.stderr:
            print(f"\nAgent errors:")
            for line in result.stderr.splitlines()[-20:]:
                print(f"  {line}")

        success = result.returncode == 0 and validation_passed

        print(f"\n{'='*70}")
        if success:
            print(f"✅ SUCCESS: {task.name}")
        else:
            print(f"❌ FAILED: {task.name}")
        print(f"{'='*70}\n")

        return success

    except subprocess.TimeoutExpired:
        print(f"\n❌ TIMEOUT after {timeout_minutes} minutes")
        return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run quick test."""
    print("="*70)
    print("ORCHESTRATOR QUICK TEST (L5 Tasks)")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Test tasks (starting with simpler L5 tasks)
    test_tasks = [
        "url_shortener",  # Simpler L5 task
        # "blog_system",   # Can add more if first one works
    ]

    results = []
    for task_name in test_tasks:
        success = run_single_task(task_name, timeout_minutes=15)
        results.append((task_name, success))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for task_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {task_name}")

    success_count = sum(1 for _, success in results if success)
    print(f"\nTotal: {success_count}/{len(results)} successful")
    print("="*70)

    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
