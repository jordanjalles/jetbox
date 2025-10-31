"""
Simple test to debug AppendUntilFullStrategy.
"""
import tempfile
from pathlib import Path

from task_executor_agent import TaskExecutorAgent
from context_strategies import AppendUntilFullStrategy

def test_simple():
    """Test append strategy with simplest possible task."""
    print("\n" + "="*70)
    print("TESTING APPEND STRATEGY - SIMPLE FILE CREATION")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        agent = TaskExecutorAgent(
            workspace=workspace,
            goal="Create hello.py with print('Hello, World!')",
            max_rounds=10,
            context_strategy=AppendUntilFullStrategy(),
        )

        # Check context before run
        print("\n[DEBUG] Checking initial context...")
        context = agent.build_context()
        print(f"[DEBUG] Context messages: {len(context)}")
        for i, msg in enumerate(context):
            print(f"[DEBUG] Message {i} ({msg['role']}): {msg['content'][:200]}...")

        print("\n[DEBUG] Running agent...")
        result = agent.run()

        print(f"\n[RESULT] Status: {result.get('status')}")
        print(f"[RESULT] Rounds: {agent.state.total_rounds}")

        workspace_dir = agent.workspace_manager.workspace_dir
        hello_file = workspace_dir / "hello.py"

        print(f"[RESULT] File created: {hello_file.exists()}")
        if hello_file.exists():
            print(f"[RESULT] File content: {hello_file.read_text()}")

if __name__ == "__main__":
    test_simple()
