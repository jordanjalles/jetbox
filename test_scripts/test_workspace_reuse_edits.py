#!/usr/bin/env python3
"""
Test workspace reuse and multiple edits on same files.

Verifies:
1. Agent can create initial workspace with files
2. Agent can reuse workspace and edit existing files
3. Multiple sequential edits work correctly
4. Workspace state persists between runs
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.task_executor_agent import TaskExecutorAgent


def test_workspace_reuse_and_edits():
    """Test creating workspace, then reusing it for edits."""
    print("\n" + "="*80)
    print("TEST: Workspace Reuse and Multiple Edits")
    print("="*80)

    # Phase 1: Create initial calculator
    print("\n--- PHASE 1: Create Initial Calculator ---")
    agent1 = TaskExecutorAgent(
        workspace=None,  # Create new workspace
        goal="Create calculator.py with add(a,b) and subtract(a,b) functions. Create test_calculator.py with tests.",
        timeout_seconds=120,
        exclude_behaviors=["ChatbotBehavior"]
    )

    start = time.time()
    result1 = agent1.run(max_rounds=20)
    elapsed1 = time.time() - start

    workspace_path = agent1.workspace
    print(f"\n[Phase 1] Status: {result1['status']}")
    print(f"[Phase 1] Time: {elapsed1:.1f}s")
    print(f"[Phase 1] Workspace: {workspace_path}")

    if result1['status'] != 'success':
        print(f"❌ Phase 1 FAILED: {result1.get('reason')}")
        return False

    # Verify files exist
    calc_file = workspace_path / "calculator.py"
    test_file = workspace_path / "test_calculator.py"

    if not calc_file.exists():
        print(f"❌ calculator.py not found in {workspace_path}")
        return False

    if not test_file.exists():
        print(f"❌ test_calculator.py not found in {workspace_path}")
        return False

    # Read initial content
    initial_calc_content = calc_file.read_text()
    print(f"\n[Phase 1] calculator.py created ({len(initial_calc_content)} bytes)")
    print(f"[Phase 1] Functions: add, subtract")

    # Phase 2: Reuse workspace and add multiply function
    print("\n--- PHASE 2: Add Multiply Function (Same Workspace) ---")
    agent2 = TaskExecutorAgent(
        workspace=workspace_path,  # Reuse existing workspace
        goal="Add a multiply(a,b) function to calculator.py and add tests for it in test_calculator.py.",
        timeout_seconds=120,
        exclude_behaviors=["ChatbotBehavior"]
    )

    start = time.time()
    result2 = agent2.run(max_rounds=20)
    elapsed2 = time.time() - start

    print(f"\n[Phase 2] Status: {result2['status']}")
    print(f"[Phase 2] Time: {elapsed2:.1f}s")

    if result2['status'] != 'success':
        print(f"❌ Phase 2 FAILED: {result2.get('reason')}")
        return False

    # Verify multiply was added
    updated_calc_content = calc_file.read_text()

    if 'multiply' not in updated_calc_content.lower():
        print(f"❌ multiply function not found in calculator.py")
        print(f"Content length: {len(updated_calc_content)} bytes")
        return False

    print(f"[Phase 2] calculator.py updated ({len(updated_calc_content)} bytes)")
    print(f"[Phase 2] Functions: add, subtract, multiply")

    # Phase 3: Reuse workspace again and add divide function
    print("\n--- PHASE 3: Add Divide Function (Same Workspace) ---")
    agent3 = TaskExecutorAgent(
        workspace=workspace_path,  # Reuse again
        goal="Add a divide(a,b) function to calculator.py with zero division handling. Add tests for it.",
        timeout_seconds=120,
        exclude_behaviors=["ChatbotBehavior"]
    )

    start = time.time()
    result3 = agent3.run(max_rounds=20)
    elapsed3 = time.time() - start

    print(f"\n[Phase 3] Status: {result3['status']}")
    print(f"[Phase 3] Time: {elapsed3:.1f}s")

    if result3['status'] != 'success':
        print(f"❌ Phase 3 FAILED: {result3.get('reason')}")
        return False

    # Verify divide was added
    final_calc_content = calc_file.read_text()

    if 'divide' not in final_calc_content.lower():
        print(f"❌ divide function not found in calculator.py")
        return False

    print(f"[Phase 3] calculator.py updated ({len(final_calc_content)} bytes)")
    print(f"[Phase 3] Functions: add, subtract, multiply, divide")

    # Phase 4: Run final tests to verify all functions work
    print("\n--- PHASE 4: Verify All Functions Work ---")
    import subprocess

    result = subprocess.run(
        ["python", "-m", "pytest", "-v"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(workspace_path),
        env={**dict(os.environ), "PYTHONPATH": str(workspace_path)}
    )

    if result.returncode == 0:
        print(f"✅ All tests passed!")
        # Count test functions
        test_content = test_file.read_text()
        test_count = test_content.count('def test_')
        print(f"[Phase 4] {test_count} test functions executed successfully")
    else:
        print(f"❌ Tests failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Phase 1: Initial creation - {elapsed1:.1f}s")
    print(f"✅ Phase 2: Add multiply - {elapsed2:.1f}s")
    print(f"✅ Phase 3: Add divide - {elapsed3:.1f}s")
    print(f"✅ Phase 4: All tests pass")
    print(f"\nTotal time: {elapsed1 + elapsed2 + elapsed3:.1f}s")
    print(f"Workspace: {workspace_path}")
    print(f"Final functions: add, subtract, multiply, divide")
    print("\n✅ WORKSPACE REUSE AND EDITS TEST PASSED!")

    return True


if __name__ == "__main__":
    import os

    try:
        success = test_workspace_reuse_and_edits()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
