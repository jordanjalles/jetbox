"""Quick L1 test to verify fixes."""
import tempfile
from pathlib import Path
from agents.task_executor_agent import TaskExecutorAgent

# Test 1: Simple File
print("=" * 80)
print("TEST: Create hello.py")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmp:
    ws = Path(tmp) / "test1"
    ws.mkdir()

    agent = TaskExecutorAgent(
        workspace=ws,
        goal='Create a file hello.py that prints Hello World'
    )

    result = agent.run(max_rounds=5)

    print(f"\nResult: {result.get('status')}")
    print(f"Reason: {result.get('reason', 'N/A')}")
    print(f"Rounds: {result.get('rounds_used')}")

    # Check file
    hello_file = ws / "hello.py"
    if hello_file.exists():
        print(f"\n✅ File created: {hello_file.name}")
        print(f"Content: {hello_file.read_text()}")
    else:
        print("\n❌ File NOT created")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
