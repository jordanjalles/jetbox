#!/usr/bin/env python3
"""
Quick validation of the evaluation suite without running full tests.

Tests that:
1. Test definitions are valid
2. Validation logic works
3. Result tracking works
"""
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent))

from tests.test_project_evaluation import (
    L5_TASKS, L6_TASKS, L7_TASKS, L8_TASKS,
    validate_task_result, find_workspace_for_task, save_test_result
)


def test_task_definitions():
    """Verify all task definitions are valid."""
    print("Testing task definitions...")

    all_tasks = L5_TASKS + L6_TASKS + L7_TASKS + L8_TASKS

    print(f"  Found {len(all_tasks)} tasks:")
    print(f"    L5: {len(L5_TASKS)} tasks")
    print(f"    L6: {len(L6_TASKS)} tasks")
    print(f"    L7: {len(L7_TASKS)} tasks")
    print(f"    L8: {len(L8_TASKS)} tasks")

    for task in all_tasks:
        assert "id" in task, f"Task missing 'id': {task}"
        assert "level" in task, f"Task {task['id']} missing 'level'"
        assert "name" in task, f"Task {task['id']} missing 'name'"
        assert "goal" in task, f"Task {task['id']} missing 'goal'"
        assert "validation" in task or task["level"] == "L8", f"Task {task['id']} missing 'validation'"

        print(f"  ✓ {task['id']}: {task['name']}")

    print()


def test_validation_logic():
    """Test validation logic with mock workspace."""
    print("Testing validation logic...")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create mock files
        (workspace / "calculator.py").write_text("""
class Calculator:
    def evaluate(self, expression):
        return eval(expression)
""")

        (workspace / "test_calculator.py").write_text("""
def test_evaluate():
    calc = Calculator()
    assert calc.evaluate("2 + 2") == 4
""")

        # Test with L5 calculator task
        task = {
            "id": "test_task",
            "level": "L5",
            "validation": {
                "classes": ["Calculator"],
                "functions": ["evaluate"],
            }
        }

        result = validate_task_result(workspace, task)

        print(f"  Files created: {result['files_created']}")
        print(f"  Symbols found: {result['details'].get('symbols_found', False)}")
        print(f"  Tests passed: {result['tests_passed']}")
        print(f"  Code quality: {result['code_quality']}")

        assert len(result['files_created']) == 2, "Should find 2 files"
        assert result['details'].get('symbols_found', False), "Should find required symbols"

        print("  ✓ Validation logic works")

    print()


def test_workspace_finding():
    """Test workspace finding logic."""
    print("Testing workspace finding...")

    task = {
        "id": "test_task",
        "goal": "Create a simple web API for managing todo items",
    }

    # This should find the existing workspace
    workspace = find_workspace_for_task(task)
    print(f"  Found workspace: {workspace}")

    if workspace.exists():
        print(f"  ✓ Workspace exists: {workspace}")
        files = list(workspace.glob("*.py"))
        print(f"    Python files: {len(files)}")
    else:
        print(f"  ℹ Workspace doesn't exist (expected if test hasn't run): {workspace}")

    print()


def test_result_tracking():
    """Test result saving."""
    print("Testing result tracking...")

    task = {
        "id": "test_task",
        "level": "L5",
        "name": "Test Task",
        "description": "Test description",
    }

    run_result = {
        "success": True,
        "duration": 123.45,
    }

    validation_result = {
        "success": True,
        "files_created": ["test.py"],
        "tests_passed": True,
        "code_quality": True,
        "details": {},
    }

    # Save result
    save_test_result(task, run_result, validation_result)

    # Check file exists
    results_file = Path("evaluation_results/project_eval_results.jsonl")
    assert results_file.exists(), "Results file should be created"

    # Read last line
    with open(results_file) as f:
        lines = f.readlines()
        last_line = lines[-1]
        import json
        result = json.loads(last_line)

        assert result["task_id"] == "test_task"
        assert result["level"] == "L5"
        print(f"  ✓ Result saved: {result['task_id']}")

    print()


def main():
    """Run all quick tests."""
    print("=" * 80)
    print("EVALUATION SUITE QUICK VALIDATION")
    print("=" * 80)
    print()

    try:
        test_task_definitions()
        test_validation_logic()
        test_workspace_finding()
        test_result_tracking()

        print("=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        print()
        print("The evaluation suite is ready to use!")
        print()
        print("To run a single task test:")
        print("  pytest tests/test_project_evaluation.py::test_l5_task[L5_cli_calculator] -v")
        print()
        print("To run all tests:")
        print("  python run_project_evaluation.py")
        print()

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
