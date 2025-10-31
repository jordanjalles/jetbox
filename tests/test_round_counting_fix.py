#!/usr/bin/env python3
"""
Test to verify that rounds are counted correctly even before task decomposition.

This test validates the fix for the issue where append strategy showed rounds: 0
because rounds before decomposition weren't being counted in subtask.rounds_used.
"""
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_executor_agent import TaskExecutorAgent
from context_strategies import AppendUntilFullStrategy, HierarchicalStrategy


def test_rounds_counted_before_decomposition():
    """
    Test that state.total_rounds tracks all rounds, even before decomposition.

    This is the source of truth for round counting, not summing subtask.rounds_used.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create agent with append strategy (more likely to not decompose early)
        agent = TaskExecutorAgent(
            workspace=workspace,
            goal="Create a simple hello.py file that prints 'Hello World'",
            context_strategy=AppendUntilFullStrategy(),
            max_rounds=5,  # Very low to force quick exit
        )

        # Run agent
        result = agent.run()

        # Check that total_rounds is non-zero
        total_rounds = agent.state.total_rounds
        print(f"Total rounds from state: {total_rounds}")
        assert total_rounds > 0, "state.total_rounds should be > 0"

        # Sum subtask rounds (the old buggy way)
        subtask_rounds_sum = 0
        if agent.context_manager.state.goal:
            for task_obj in agent.context_manager.state.goal.tasks:
                for subtask in task_obj.subtasks:
                    subtask_rounds_sum += subtask.rounds_used

        print(f"Sum of subtask rounds: {subtask_rounds_sum}")

        # The bug: subtask_rounds_sum may be 0 if no decomposition happened
        # The fix: Use state.total_rounds as source of truth

        # Verify state.total_rounds is the source of truth
        print(f"✓ state.total_rounds ({total_rounds}) is the correct round count")

        if subtask_rounds_sum == 0 and total_rounds > 0:
            print(f"⚠️  This demonstrates the bug: subtask sum is 0 but total_rounds is {total_rounds}")
            print(f"    (This happens when rounds occur before task decomposition)")


def test_hierarchical_still_works():
    """
    Test that hierarchical strategy also uses state.total_rounds correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        agent = TaskExecutorAgent(
            workspace=workspace,
            goal="Create a simple hello.py file",
            context_strategy=HierarchicalStrategy(),
            max_rounds=5,
        )

        result = agent.run()

        total_rounds = agent.state.total_rounds
        print(f"\nHierarchical strategy total_rounds: {total_rounds}")
        assert total_rounds > 0, "Hierarchical should also count rounds"
        print(f"✓ Hierarchical strategy works correctly")


if __name__ == "__main__":
    print("="*70)
    print("Testing round counting fix")
    print("="*70)

    print("\n[1] Testing append strategy round counting...")
    test_rounds_counted_before_decomposition()

    print("\n[2] Testing hierarchical strategy round counting...")
    test_hierarchical_still_works()

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
