"""
Quick L5 evaluation with flexible validation.

Simpler version that just runs the 5 L5 tasks.
"""
import sys
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.evaluation_suite_extended import get_extended_tasks
from tests.flexible_validation import VALIDATORS


def run_task(task, run_number):
    """Run a single task and validate."""

    workspace = tempfile.mkdtemp(prefix=f"eval_L5_{task.name}_run{run_number}_")
    workspace_path = Path(workspace)

    print(f"\n{'='*70}")
    print(f"Running: L5 - {task.name} (Run {run_number})")
    print(f"Workspace: {workspace}")
    print(f"{'='*70}")

    try:
        cmd = [
            sys.executable, "/workspace/agent.py",
            "--team", "eval_with_inspection",
            "--workspace", workspace,
            task.goal
        ]

        result = subprocess.run(
            cmd,
            cwd="/workspace",
            capture_output=True,
            text=True,
            timeout=600
        )

        success = result.returncode == 0

        # Use flexible validation
        if task.name in VALIDATORS:
            flex_success, flex_message = VALIDATORS[task.name](workspace_path)
            validation_passed = flex_success
            validation_msg = flex_message
        else:
            # Fallback to rigid validation
            files_exist = all((workspace_path / f).exists() for f in task.expected_files)
            validation_passed = files_exist
            validation_msg = f"Files exist: {files_exist}"

        overall_success = success and validation_passed

        # Print result
        status = "✓ SUCCESS" if overall_success else "✗ FAILED"
        print(f"\n{status} - L5 {task.name} Run {run_number}")
        print(f"Validation: {validation_msg}")

        return overall_success

    except subprocess.TimeoutExpired:
        print(f"\n✗ TIMEOUT - L5 {task.name} Run {run_number}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - L5 {task.name} Run {run_number}: {e}")
        return False


def main():
    """Run L5 evaluation."""

    print("="*70)
    print("L5 RE-EVALUATION WITH FLEXIBLE VALIDATION")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print()

    # Get L5 tasks
    all_tasks = get_extended_tasks()
    l5_tasks = [t for t in all_tasks if t.level == 5]

    print(f"L5 Tasks: {len(l5_tasks)}")
    for task in l5_tasks:
        print(f"  - {task.name}")
    print()

    results = []
    for task in l5_tasks:
        for run in [1, 2]:
            success = run_task(task, run)
            results.append((task.name, run, success))

    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    successful = sum(1 for _, _, s in results if s)
    total = len(results)

    for name, run, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name} run{run}")

    print()
    print(f"Total: {successful}/{total} ({100*successful//total}%)")
    print()
    print("COMPARISON:")
    print(f"  Previous (rigid validation): 0/10 (0%)")
    print(f"  Current (flexible validation): {successful}/10 ({100*successful//10}%)")
    print(f"  Improvement: +{successful} successes")
    print("="*70)


if __name__ == "__main__":
    main()
