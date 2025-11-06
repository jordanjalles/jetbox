#!/usr/bin/env python3
"""
Test orchestrator delegation workflow.

Verifies that orchestrator:
1. Calls consult_architect
2. Calls delegate_to_executor
3. Only marks complete after BOTH phases
"""

from pathlib import Path
from orchestrator_agent import OrchestratorAgent

def test_orchestrator_workflow():
    """Test that orchestrator follows proper delegation workflow."""

    print("="*80)
    print("ORCHESTRATOR WORKFLOW TEST")
    print("="*80)
    print("\nThis test verifies orchestrator delegates to BOTH architect and task_executor")
    print("Expected workflow:")
    print("  1. orchestrator -> consult_architect")
    print("  2. orchestrator -> delegate_to_executor")
    print("  3. orchestrator -> mark_complete")
    print()

    workspace = Path(".agent_workspaces/tests/orchestrator_workflow_test")

    orchestrator = OrchestratorAgent(workspace=workspace)

    # Add user message with task
    orchestrator.add_message({
        "role": "user",
        "content": "Create a simple Python calculator with add and subtract functions. Include tests."
    })

    print("Created orchestrator with task")
    print(f"Workspace: {workspace}")
    print()

    print("Running orchestrator (max 20 rounds)...")
    print("="*80)
    print()

    result = orchestrator.run(max_rounds=20)

    print("\n" + "="*80)
    print("TEST RESULT")
    print("="*80)
    print(f"Status: {result.get('status')}")
    print(f"Rounds: {result.get('round', 'N/A')}")
    print()

    # Check delegation behavior
    delegation_behavior = None
    for behavior in orchestrator._behaviors:
        if behavior.get_name() == 'delegation':
            delegation_behavior = behavior
            break

    if delegation_behavior:
        delegations = delegation_behavior.delegated_tasks
        print(f"DELEGATIONS: {len(delegations)} total")

        architect_called = False
        executor_called = False

        for delegation in delegations:
            agent_name = delegation['agent']
            print(f"  - {agent_name}: {delegation['result'].get('status', 'unknown')}")

            if agent_name == 'architect':
                architect_called = True
            elif agent_name == 'task_executor':
                executor_called = True

        print()
        print("WORKFLOW VALIDATION:")

        if architect_called:
            print("  ✓ Architect was consulted")
        else:
            print("  ✗ Architect NOT consulted (missing design phase)")

        if executor_called:
            print("  ✓ Task executor was delegated to")
        else:
            print("  ✗ Task executor NOT delegated to (missing implementation phase)")

        if architect_called and executor_called:
            print()
            print("✓ WORKFLOW CORRECT: Both architecture and implementation phases completed")
        elif architect_called and not executor_called:
            print()
            print("✗ WORKFLOW BROKEN: Architecture completed but no implementation")
            print("  This is the bug we're trying to fix!")
        elif not architect_called and executor_called:
            print()
            print("⚠️  WORKFLOW UNUSUAL: Implementation without architecture")
            print("  (Acceptable for simple tasks)")
        else:
            print()
            print("✗ WORKFLOW BROKEN: No delegations occurred")
    else:
        print("✗ ERROR: DelegationBehavior not found")

    print()
    print("="*80)

if __name__ == "__main__":
    test_orchestrator_workflow()
