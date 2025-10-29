#!/usr/bin/env python3
"""
Test completion detection after tool result visibility fix.

This should complete in < 10 rounds (not timeout at 20).
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task_executor_agent import TaskExecutorAgent


def test_simple_task():
    """Test a simple L1 task to verify completion detection works."""
    print("=" * 70)
    print("TESTING COMPLETION DETECTION FIX")
    print("=" * 70)

    goal = "Create hello.py with a function greet(name) that returns 'Hello, {name}!'"
    print(f"\nGoal: {goal}")
    print(f"Expected: Agent should complete in < 10 rounds and call mark_subtask_complete\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        agent = TaskExecutorAgent(
            workspace=workspace,
            goal=goal,
            max_rounds=20,
            model="gpt-oss:20b",
            temperature=0.2
        )

        # Run the agent
        try:
            result = agent.run()

            # Check results
            print("\n" + "=" * 70)
            print("RESULTS")
            print("=" * 70)

            # Agent uses workspace isolation - check the actual workspace
            actual_workspace = agent.workspace_manager.workspace_dir
            hello_file = actual_workspace / "hello.py"
            file_exists = hello_file.exists()

            print(f"Actual workspace: {actual_workspace}")
            print(f"File created: {'✓' if file_exists else '✗'} {hello_file}")

            if file_exists:
                with open(hello_file) as f:
                    content = f.read()
                    has_function = 'def greet' in content
                    has_return = 'Hello' in content

                    print(f"Has greet() function: {'✓' if has_function else '✗'}")
                    print(f"Returns greeting: {'✓' if has_return else '✗'}")

            rounds = agent.state.total_rounds
            print(f"\nRounds used: {rounds}/20")

            if rounds >= 20:
                print("⚠️  TIMEOUT - Agent did not signal completion!")
                print("    This indicates completion detection is still not working.")
                return False
            else:
                print(f"✓ Completed in {rounds} rounds (good!)")
                return True

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = test_simple_task()
    sys.exit(0 if success else 1)
