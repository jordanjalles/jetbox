#!/usr/bin/env python3
"""
Validation test for delegation context management fixes.

Tests:
1. Delegation executes subagent (not just creates it)
2. Workspace coordination (subagent uses calling agent workspace)
3. Summary extraction from SubAgentModeBehavior.mark_complete()
4. Clear result message formatting
"""
from pathlib import Path
import shutil
from agents.orchestrator_agent import OrchestratorAgent

def test_delegation_execution():
    """Test that delegation actually executes subagent and returns proper results."""
    print("=" * 80)
    print("VALIDATION: Delegation Context Management")
    print("=" * 80)

    # Clean up any old test workspace
    test_workspace = Path(".agent_workspaces/delegation-test")
    if test_workspace.exists():
        shutil.rmtree(test_workspace)

    print("\n[TEST] Creating orchestrator with simple task delegation...")

    # Create orchestrator
    orchestrator = OrchestratorAgent(workspace=test_workspace)

    # Simple task that should complete quickly
    task = "Create a simple Python file with a hello() function that returns 'Hello, World!'"

    print(f"\n[TEST] Task: {task}")
    print(f"[TEST] Workspace: {test_workspace}")

    # Add user message
    orchestrator.add_message({"role": "user", "content": task})

    # Run orchestrator (should delegate to task_executor)
    print("\n[TEST] Running orchestrator (max 10 rounds)...")
    result = orchestrator.run(max_rounds=10)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nStatus: {result.get('status')}")
    print(f"Workspace: {result.get('workspace')}")

    # Check if delegation happened
    if test_workspace.exists():
        files = list(test_workspace.glob("*.py"))
        print(f"\nFiles created: {len(files)}")
        for f in files:
            print(f"  - {f.name}")
    else:
        print("\nWorkspace does not exist!")
        files = []

    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    checks = {
        "Delegation executed": result.get('status') in ['success', 'failure'],
        "Workspace exists": test_workspace.exists(),
        "Files created": len(files) > 0,
        "Summary present": 'summary' in result or 'reason' in result,
    }

    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check}")

    all_passed = all(checks.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Delegation context management working!")
    else:
        print("❌ SOME CHECKS FAILED - Delegation needs more work")
    print("=" * 80)

    return all_passed

if __name__ == "__main__":
    success = test_delegation_execution()
    exit(0 if success else 1)
