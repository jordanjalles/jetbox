#!/usr/bin/env python3
"""
Quick smoke test: verify agent can use bash for file operations.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task_executor_agent import TaskExecutorAgent


def test_bash_file_operations():
    """Test that agent uses run_bash for file write/read."""
    print("\n" + "="*70)
    print("SMOKE TEST: Bash File Operations")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Simple task that requires write + read
        agent = TaskExecutorAgent(
            workspace=workspace,
            goal="Create hello.py with 'print(\"Hello, World!\")'. Then read it back and verify the content.",
            max_rounds=15,
        )

        result = agent.run()

        print(f"\n{'='*70}")
        print(f"Result: {result.get('status')}")
        print(f"{'='*70}")

        # Check if hello.py was created
        workspace_dir = agent.workspace_manager.workspace_dir
        hello_file = workspace_dir / "hello.py"

        if hello_file.exists():
            content = hello_file.read_text()
            print(f"\n✓ hello.py created with content:\n{content}")

            if "Hello, World!" in content:
                print("✓ Content verified correct")
                return True
            else:
                print("✗ Content incorrect")
                return False
        else:
            print("✗ hello.py not created")
            return False


if __name__ == "__main__":
    success = test_bash_file_operations()
    sys.exit(0 if success else 1)
