#!/usr/bin/env python3
"""
End-to-end test for Orchestrator with real TaskExecutor delegation.

Tests:
1. Simple delegation: Orchestrator → TaskExecutor
2. Workspace iteration: Multiple executors on same workspace
3. Jetbox notes: Context continuity between runs
"""
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator_agent import OrchestratorAgent
from task_executor_agent import TaskExecutorAgent
from agent_config import config


def test_simple_delegation():
    """Test executor can complete a simple task."""
    print("\n" + "="*70)
    print("TEST 1: Simple Delegation")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create executor directly
        executor = TaskExecutorAgent(
            workspace=workspace,
            goal="Create a Python file called hello.py that prints 'Hello World'",
            max_rounds=10
        )

        result = executor.run()
        print(f"Result: {result.get('status')}")

        # Check if file was created
        hello_py = executor.workspace_manager.workspace_dir / "hello.py"
        success = False
        if hello_py.exists():
            print(f"✓ File created: {hello_py}")
            print(f"  Content: {hello_py.read_text()}")
            success = True
        else:
            print(f"✗ File not created")

        return {"test": "simple_delegation", "success": success, "result": result}


def test_workspace_iteration():
    """Test multiple executors can iterate on same workspace."""
    print("\n" + "="*70)
    print("TEST 2: Workspace Iteration")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Phase 1: Create initial code
        print("\nPhase 1: Create calculator")
        executor1 = TaskExecutorAgent(
            workspace=workspace,
            goal="Create calculator.py with an add(a, b) function",
            max_rounds=15
        )

        result1 = executor1.run()
        print(f"Phase 1 result: {result1.get('status')}")

        calc_workspace = executor1.workspace_manager.workspace_dir

        # Check jetbox notes were created
        notes_file = calc_workspace / "jetboxnotes.md"
        if notes_file.exists():
            print(f"✓ Jetbox notes created: {notes_file}")
            notes_content = notes_file.read_text()
            print(f"  Notes preview: {notes_content[:200]}...")
        else:
            print(f"✗ Jetbox notes not found")
            return {"test": "workspace_iteration", "success": False}

        # Phase 2: Iterate on same workspace
        print("\nPhase 2: Add multiply function")
        executor2 = TaskExecutorAgent(
            workspace=workspace,  # Base directory
            workspace_path=calc_workspace,  # Reuse existing workspace!
            goal="Add a multiply(a, b) function to calculator.py",
            max_rounds=15
        )

        result2 = executor2.run()
        print(f"Phase 2 result: {result2.get('status')}")

        # Verify both functions exist
        calc_file = calc_workspace / "calculator.py"
        if calc_file.exists():
            content = calc_file.read_text()
            has_add = "def add" in content
            has_multiply = "def multiply" in content

            print(f"✓ calculator.py exists")
            print(f"  {'✓' if has_add else '✗'} Has add function")
            print(f"  {'✓' if has_multiply else '✗'} Has multiply function")

            # Check if notes were updated
            notes_content_after = notes_file.read_text()
            notes_updated = len(notes_content_after) > len(notes_content)
            print(f"  {'✓' if notes_updated else '✗'} Jetbox notes updated")

            success = has_add and has_multiply and notes_updated
        else:
            print(f"✗ calculator.py not found")
            success = False

        return {
            "test": "workspace_iteration",
            "success": success,
            "workspace": str(calc_workspace)
        }


def test_jetbox_notes_continuity():
    """Test jetbox notes provide context across workspace iterations."""
    print("\n" + "="*70)
    print("TEST 3: Jetbox Notes Continuity")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create initial work
        print("\nStep 1: Create Todo class")
        executor1 = TaskExecutorAgent(
            workspace=workspace,
            goal="Create todo.py with a Todo class. Todo should have: title, completed (bool), and a toggle_complete() method.",
            max_rounds=15
        )

        result1 = executor1.run()
        todo_workspace = executor1.workspace_manager.workspace_dir

        # Capture initial notes
        notes_file = todo_workspace / "jetboxnotes.md"
        initial_notes = ""
        if notes_file.exists():
            initial_notes = notes_file.read_text()
            print(f"✓ Initial notes: {len(initial_notes)} chars")

        # Wait a moment to ensure different timestamps
        time.sleep(1)

        # Continue work - this should load previous notes
        print("\nStep 2: Add list management")
        executor2 = TaskExecutorAgent(
            workspace=workspace,  # Base directory
            workspace_path=todo_workspace,  # Reuse existing workspace!
            goal="Add a TodoList class that can add/remove/list todos",
            max_rounds=15
        )

        result2 = executor2.run()

        # Check if notes reflect both tasks
        final_notes = notes_file.read_text() if notes_file.exists() else ""
        print(f"✓ Final notes: {len(final_notes)} chars")

        # Verify continuity
        has_todo_info = "Todo" in final_notes or "todo" in final_notes
        has_list_info = "TodoList" in final_notes or "list" in final_notes
        notes_grew = len(final_notes) > len(initial_notes)

        print(f"  {'✓' if has_todo_info else '✗'} Notes mention Todo class")
        print(f"  {'✓' if has_list_info else '✗'} Notes mention TodoList")
        print(f"  {'✓' if notes_grew else '✗'} Notes grew from iteration")

        # Check files
        todo_py = todo_workspace / "todo.py"
        has_both_classes = False
        if todo_py.exists():
            content = todo_py.read_text()
            has_both_classes = "class Todo" in content and "class TodoList" in content
            print(f"  {'✓' if has_both_classes else '✗'} Both classes in todo.py")

        success = has_todo_info and has_list_info and notes_grew and has_both_classes

        return {
            "test": "jetbox_notes_continuity",
            "success": success,
            "initial_notes_size": len(initial_notes),
            "final_notes_size": len(final_notes),
        }


def main():
    """Run all orchestrator end-to-end tests."""
    print("="*70)
    print("ORCHESTRATOR END-TO-END TESTS")
    print("="*70)
    print("\nTesting orchestrator delegation, workspace iteration, and jetbox notes.")
    print()

    results = []

    # Run tests
    try:
        results.append(test_simple_delegation())
    except Exception as e:
        print(f"✗ Test 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append({"test": "simple_delegation", "success": False, "error": str(e)})

    try:
        results.append(test_workspace_iteration())
    except Exception as e:
        print(f"✗ Test 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append({"test": "workspace_iteration", "success": False, "error": str(e)})

    try:
        results.append(test_jetbox_notes_continuity())
    except Exception as e:
        print(f"✗ Test 3 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append({"test": "jetbox_notes_continuity", "success": False, "error": str(e)})

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for r in results if r.get("success"))
    total = len(results)

    for result in results:
        status = "✓ PASS" if result.get("success") else "✗ FAIL"
        print(f"{status} - {result['test']}")

    print(f"\nTotal: {passed}/{total} tests passed")

    # Save results
    output_file = Path("orchestrator_e2e_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {"passed": passed, "total": total}
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
