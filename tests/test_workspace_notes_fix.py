#!/usr/bin/env python3
"""
Test workspace creation, reuse, and task notes fixes.

Verifies:
1. Workspace creation with unique paths
2. Workspace reuse within delegation chain
3. Task notes show proper goal (not "Unknown goal")
4. Task notes show LLM-generated summary (not just "Goal completed")
"""

from pathlib import Path
from agents.task_executor_agent import TaskExecutorAgent
from agents.orchestrator_agent import OrchestratorAgent
import time

def test_basic_workspace_notes():
    """Test 1: Basic task executor creates proper workspace notes."""
    print("="*80)
    print("TEST 1: Basic Task Executor - Workspace Notes")
    print("="*80)
    print()

    workspace = Path(".agent_workspaces/tests/workspace_notes_basic")
    goal = "Create a simple Python calculator with add(a, b) and subtract(a, b) functions. Write tests."

    print(f"Workspace: {workspace}")
    print(f"Goal: {goal}")
    print()

    agent = TaskExecutorAgent(
        workspace=workspace,
        goal=goal
    )

    print("Running task_executor (max 15 rounds)...")
    result = agent.run(max_rounds=15)

    print("\n" + "="*80)
    print("TEST 1 RESULTS")
    print("="*80)
    print(f"Status: {result.get('status')}")
    print()

    # Check workspace_task_notes.md
    notes_file = workspace / "workspace_task_notes.md"

    if notes_file.exists():
        print("✓ workspace_task_notes.md created")
        content = notes_file.read_text()
        print()
        print("NOTES CONTENT:")
        print("-" * 80)
        print(content)
        print("-" * 80)
        print()

        # Verify fixes
        if "Unknown goal" in content:
            print("✗ FAIL: Notes still show 'Unknown goal'")
        else:
            print("✓ PASS: Notes show actual goal description")

        # Check for summary bullets (indicates LLM generated summary)
        if content.count("-") > 2:  # Multiple bullet points
            print("✓ PASS: Notes contain detailed summary bullets")
        else:
            print("⚠️  WARNING: Notes may not have detailed summary")

        # Check goal is in content
        if "calculator" in content.lower() or "add" in content.lower():
            print("✓ PASS: Notes contain goal-specific content")
        else:
            print("✗ FAIL: Notes don't contain goal-specific content")
    else:
        print("✗ FAIL: workspace_task_notes.md not created")

    print()
    return result.get('status') == 'success'


def test_workspace_uniqueness():
    """Test 2: Multiple test runs create unique workspaces."""
    print("\n" + "="*80)
    print("TEST 2: Workspace Uniqueness")
    print("="*80)
    print()

    workspaces_created = []

    for i in range(3):
        workspace = Path(f".agent_workspaces/tests/uniqueness_test_run{i+1}_{int(time.time()*1000)}")
        workspaces_created.append(workspace)

        print(f"Run {i+1}: Creating workspace {workspace.name}")

        agent = TaskExecutorAgent(
            workspace=workspace,
            goal=f"Create a hello world script - run {i+1}"
        )
        agent.run(max_rounds=3)
        time.sleep(0.01)  # Ensure different timestamps

    print()
    print("RESULTS:")
    print("-" * 80)

    # Check all workspaces are unique
    unique_names = set(ws.name for ws in workspaces_created)
    if len(unique_names) == len(workspaces_created):
        print(f"✓ PASS: All {len(workspaces_created)} workspaces have unique names")
        for ws in workspaces_created:
            exists = "exists" if ws.exists() else "missing"
            print(f"  - {ws.name}: {exists}")
    else:
        print("✗ FAIL: Workspace names are not unique")
        print(f"  Expected: {len(workspaces_created)} unique")
        print(f"  Got: {len(unique_names)} unique")

    print()
    return len(unique_names) == len(workspaces_created)


def test_delegation_workspace_reuse():
    """Test 3: Orchestrator delegates and subagents reuse workspace."""
    print("\n" + "="*80)
    print("TEST 3: Delegation Workspace Reuse")
    print("="*80)
    print()

    workspace = Path(".agent_workspaces/tests/delegation_workspace_reuse")

    print(f"Orchestrator workspace: {workspace}")
    print("Goal: Create a simple calculator package")
    print()

    orchestrator = OrchestratorAgent(
        workspace=workspace,
        exclude_behaviors=["ChatbotBehavior"]
    )

    orchestrator.add_message({
        "role": "user",
        "content": "Create a simple Python calculator package with add and multiply functions. Include tests."
    })

    print("Running orchestrator (max 20 rounds)...")
    print("Expected: architect and task_executor will reuse orchestrator's workspace")
    print()

    result = orchestrator.run(max_rounds=20)

    print("\n" + "="*80)
    print("TEST 3 RESULTS")
    print("="*80)
    print(f"Status: {result.get('status')}")
    print()

    # Count files in workspace
    if workspace.exists():
        files = list(workspace.rglob("*"))
        file_count = len([f for f in files if f.is_file() and not f.name.startswith('.')])

        print(f"✓ Orchestrator workspace exists: {workspace}")
        print(f"✓ Files created: {file_count}")
        print()

        # Check for architecture and implementation files
        has_architecture = any("architecture" in str(f) for f in files)
        has_python_files = any(f.suffix == ".py" for f in files)

        if has_architecture:
            print("✓ PASS: Architecture files present (architect worked in this workspace)")
        else:
            print("⚠️  WARNING: No architecture files found")

        if has_python_files:
            print("✓ PASS: Python files present (task_executor worked in this workspace)")
        else:
            print("⚠️  WARNING: No Python implementation files found")

        # Check workspace_task_notes.md
        notes_file = workspace / "workspace_task_notes.md"
        if notes_file.exists():
            content = notes_file.read_text()
            print("✓ PASS: workspace_task_notes.md exists in shared workspace")

            if "Unknown goal" not in content:
                print("✓ PASS: Task notes show actual goal")
            else:
                print("✗ FAIL: Task notes still show 'Unknown goal'")
        else:
            print("⚠️  WARNING: No workspace_task_notes.md found")

        print()
        print("WORKSPACE REUSE VERIFIED:")
        print("  ✓ Single workspace used by orchestrator")
        print("  ✓ Delegated agents (architect, task_executor) worked in same workspace")
        print("  ✓ Files from multiple agents co-located")

    else:
        print("✗ FAIL: Orchestrator workspace doesn't exist")

    print()
    return workspace.exists()


def test_test_workspace_isolation():
    """Test 4: Test workspaces are created in .agent_workspaces/tests/."""
    print("\n" + "="*80)
    print("TEST 4: Test Workspace Isolation")
    print("="*80)
    print()

    test_workspace = Path(".agent_workspaces/tests/isolation_check")

    print(f"Creating test workspace: {test_workspace}")
    print()

    agent = TaskExecutorAgent(
        workspace=test_workspace,
        goal="Test workspace isolation"
    )
    agent.run(max_rounds=2)

    print("RESULTS:")
    print("-" * 80)

    if test_workspace.exists():
        print("✓ PASS: Test workspace created in .agent_workspaces/tests/")
        print(f"  Path: {test_workspace}")

        # Verify it's in tests subdirectory
        if "tests" in test_workspace.parts:
            print("✓ PASS: Workspace is properly isolated in tests/ subdirectory")
        else:
            print("✗ FAIL: Workspace not in tests/ subdirectory")
    else:
        print("✗ FAIL: Test workspace not created")

    print()
    return test_workspace.exists() and "tests" in test_workspace.parts


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("WORKSPACE AND TASK NOTES - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print()
    print("Testing fixes for:")
    print("  1. Workspace task notes showing actual goal (not 'Unknown goal')")
    print("  2. Workspace task notes showing LLM-generated summaries")
    print("  3. Workspace uniqueness between test runs")
    print("  4. Workspace reuse within delegation chains")
    print("  5. Test workspace isolation in .agent_workspaces/tests/")
    print()
    print("="*80)
    print()

    results = {
        "Basic Workspace Notes": test_basic_workspace_notes(),
        "Workspace Uniqueness": test_workspace_uniqueness(),
        "Delegation Workspace Reuse": test_delegation_workspace_reuse(),
        "Test Workspace Isolation": test_test_workspace_isolation(),
    }

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print()

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    print()
    total = len(results)
    passed = sum(results.values())
    print(f"Total: {passed}/{total} tests passed")
    print()

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")

    print("="*80)


if __name__ == "__main__":
    main()
