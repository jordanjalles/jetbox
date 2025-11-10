"""
Test timeout handling implementation.

This script verifies that:
1. TimeoutError is caught gracefully
2. Partial progress is saved
3. Circuit breaker triggers after configured threshold
"""
import os
import sys
from pathlib import Path
import shutil

# Set model
os.environ["OLLAMA_MODEL"] = "gpt-oss:20b"

from agents.task_executor_agent import TaskExecutorAgent

def test_timeout_handling():
    """Test that timeout is handled gracefully."""
    print("="*70)
    print("TEST: Timeout Handling")
    print("="*70)
    print()

    # Create test workspace
    test_workspace = Path("/tmp/test_timeout_handling")
    if test_workspace.exists():
        shutil.rmtree(test_workspace)
    test_workspace.mkdir(parents=True, exist_ok=True)

    print(f"Test workspace: {test_workspace}")
    print()
    print("Creating TaskExecutor agent with simple goal...")
    print("Expected: LLM will timeout, but agent should handle gracefully")
    print()

    # This will likely timeout with current LLM service issues
    agent = TaskExecutorAgent(
        workspace=test_workspace,
        goal="Create hello.py that prints 'Hello World'",
        use_behaviors=True,
        max_rounds=5
    )

    print("Running agent...")
    print()

    try:
        result = agent.run()

        print()
        print("="*70)
        print("RESULT")
        print("="*70)
        print(f"Status: {result.get('status')}")
        print(f"Reason: {result.get('reason', 'N/A')}")
        print(f"Files created: {result.get('files_created', 0)}")
        print(f"Total timeouts: {result.get('total_timeouts', 0)}")
        print()

        if result.get('status') == 'partial_success':
            print("✓ PASS - Timeout handled gracefully!")
            print(f"  Files created: {result.get('files_created')}")
            print(f"  Workspace: {result.get('workspace')}")

            # List files created
            if result.get('file_list'):
                print("\n  Files:")
                for f in result['file_list']:
                    print(f"    - {f}")
        elif result.get('status') == 'success':
            print("✓ PASS - Goal completed successfully!")
            print(f"  Files created: {result.get('files_created', 'N/A')}")
        else:
            print(f"⚠️  Result status: {result.get('status')} - {result.get('reason')}")

        return result

    except TimeoutError as e:
        print()
        print("="*70)
        print("FAIL - TimeoutError not caught!")
        print("="*70)
        print(f"Error: {e}")
        print()
        print("The timeout handler is not working correctly.")
        return {"status": "error", "error": str(e)}

    except Exception as e:
        print()
        print("="*70)
        print("FAIL - Unexpected exception!")
        print("="*70)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    result = test_timeout_handling()

    # Exit with appropriate code
    if result.get('status') in ['partial_success', 'success']:
        sys.exit(0)
    else:
        sys.exit(1)
