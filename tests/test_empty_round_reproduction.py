#!/usr/bin/env python3
"""
Reproduce the empty round issue.

Test if unknown tool calls are being tracked correctly.
"""

from pathlib import Path
from architect_agent import ArchitectAgent

def main():
    """Test architect with a simple task."""

    print("="*80)
    print("EMPTY ROUND REPRODUCTION TEST")
    print("="*80)

    # Create architect WITH GOAL (simulating delegation)
    workspace = Path(".agent_workspaces/tests/empty_round_test")
    goal = "Create architecture for a simple blog API."
    architect = ArchitectAgent(workspace=workspace, goal=goal)

    # Note: goal is set via constructor parameter (like delegation does)

    print("\n" + "="*80)
    print("RUNNING ARCHITECT (max 3 rounds)")
    print("="*80)

    # Run for just 3 rounds to see the pattern
    result = architect.run(max_rounds=3)

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Status: {result.get('status')}")
    print(f"Reason: {result.get('reason', result.get('summary', 'N/A'))}")

    # Check loop detection state
    print("\n" + "="*80)
    print("LOOP DETECTION STATE")
    print("="*80)

    for behavior in architect._behaviors:
        if behavior.get_name() == "loop_detection":
            print(f"Actions tracked: {len(behavior.action_history)}")
            print("Last 5 actions:")
            for action in behavior.action_history[-5:]:
                print(f"  - {action['tool_name']}: success={action['success']}")
            print(f"Consecutive empty rounds: {behavior.consecutive_empty_rounds}")
            break

if __name__ == "__main__":
    main()
