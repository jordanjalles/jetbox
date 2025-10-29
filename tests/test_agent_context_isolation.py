"""
Test that orchestrator and task executor maintain separate contexts.

Verifies:
1. Each agent has its own state file
2. Orchestrator messages don't leak to task executor
3. Task executor messages don't leak to orchestrator
4. Context switching properly clears and loads correct state
"""
import sys
from pathlib import Path
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator_agent import OrchestratorAgent
from task_executor_agent import TaskExecutorAgent


def test_agents_have_separate_state_files():
    """Test that orchestrator and task executor use different state files."""
    print("\n=== Testing Agents Have Separate State Files ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create orchestrator
        orchestrator = OrchestratorAgent(workspace=tmppath)
        orch_state_file = orchestrator.state_file

        # Create task executor
        executor = TaskExecutorAgent(
            workspace=tmppath,
            goal="Test goal",
            max_rounds=1
        )
        exec_state_file = executor.state_file

        # Verify different state files
        assert orch_state_file != exec_state_file
        assert "orchestrator" in str(orch_state_file)
        assert "task_executor" in str(exec_state_file)

        print(f"✓ Orchestrator state: {orch_state_file.name}")
        print(f"✓ Task executor state: {exec_state_file.name}")
        print("✓ State files are separate")


def test_orchestrator_context_does_not_leak_to_executor():
    """Test that orchestrator messages don't appear in task executor context."""
    print("\n=== Testing Orchestrator Context Does Not Leak ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Step 1: Create orchestrator and add messages
        orchestrator = OrchestratorAgent(workspace=tmppath)
        orchestrator.add_user_message("User message 1 from orchestrator")
        orchestrator.add_message({
            "role": "assistant",
            "content": "Assistant response 1 from orchestrator"
        })
        orchestrator.add_user_message("User message 2 from orchestrator")
        orchestrator.persist_state()

        orch_message_count = len(orchestrator.state.messages)
        print(f"✓ Orchestrator has {orch_message_count} messages")

        # Step 2: Create task executor (should have empty context)
        executor = TaskExecutorAgent(
            workspace=tmppath,
            goal="Test goal for executor",
            max_rounds=1
        )

        exec_message_count = len(executor.state.messages)
        print(f"✓ Task executor has {exec_message_count} messages")

        # Verify executor doesn't have orchestrator's messages
        assert exec_message_count == 0, "Task executor should start with empty messages"

        # Verify orchestrator messages not in executor
        for msg in orchestrator.state.messages:
            assert msg not in executor.state.messages, \
                f"Orchestrator message leaked to executor: {msg['content'][:50]}"

        print("✓ No message leakage from orchestrator to executor")


def test_executor_context_does_not_leak_to_orchestrator():
    """Test that task executor messages don't appear in orchestrator context."""
    print("\n=== Testing Task Executor Context Does Not Leak ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Step 1: Create task executor and add messages
        executor = TaskExecutorAgent(
            workspace=tmppath,
            goal="Test goal for executor",
            max_rounds=1
        )
        executor.add_message({
            "role": "user",
            "content": "User message 1 from executor"
        })
        executor.add_message({
            "role": "assistant",
            "content": "Assistant response 1 from executor"
        })
        executor.persist_state()

        exec_message_count = len(executor.state.messages)
        print(f"✓ Task executor has {exec_message_count} messages")

        # Step 2: Create orchestrator (should have empty context)
        orchestrator = OrchestratorAgent(workspace=tmppath)

        orch_message_count = len(orchestrator.state.messages)
        print(f"✓ Orchestrator has {orch_message_count} messages")

        # Verify orchestrator doesn't have executor's messages
        assert orch_message_count == 0, "Orchestrator should start with empty messages"

        # Verify executor messages not in orchestrator
        for msg in executor.state.messages:
            assert msg not in orchestrator.state.messages, \
                f"Executor message leaked to orchestrator: {msg['content'][:50]}"

        print("✓ No message leakage from executor to orchestrator")


def test_context_switching_loads_correct_state():
    """Test that switching between agents loads the correct previous context."""
    print("\n=== Testing Context Switching Loads Correct State ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Scenario: Simulate orchestrator -> executor -> orchestrator flow

        # Phase 1: Orchestrator session 1
        print("\n1. Orchestrator Session 1")
        orch1 = OrchestratorAgent(workspace=tmppath)
        orch1.add_user_message("Create a calculator")
        orch1.add_message({"role": "assistant", "content": "I'll delegate that"})
        orch1.persist_state()
        orch1_messages = len(orch1.state.messages)
        print(f"   ✓ Orchestrator saved {orch1_messages} messages")

        # Phase 2: Task executor session (delegated work)
        print("\n2. Task Executor Session (delegated)")
        executor1 = TaskExecutorAgent(
            workspace=tmppath,
            goal="Create calculator",
            max_rounds=1
        )
        executor1.add_message({"role": "user", "content": "Building calculator..."})
        executor1.add_message({"role": "assistant", "content": "Calculator built"})
        executor1.persist_state()
        exec1_messages = len(executor1.state.messages)
        print(f"   ✓ Task executor saved {exec1_messages} messages")

        # Phase 3: Orchestrator session 2 (return from delegation)
        print("\n3. Orchestrator Session 2 (after delegation)")
        orch2 = OrchestratorAgent(workspace=tmppath)

        # Should load previous orchestrator state, NOT executor state
        assert len(orch2.state.messages) == orch1_messages, \
            f"Expected {orch1_messages} messages, got {len(orch2.state.messages)}"

        # Verify it's the SAME messages from orch1
        assert orch2.state.messages[0]["content"] == "Create a calculator"
        assert orch2.state.messages[1]["content"] == "I'll delegate that"

        # Verify executor messages NOT present
        for msg in executor1.state.messages:
            assert msg not in orch2.state.messages, \
                f"Executor message leaked to orchestrator session 2: {msg}"

        print("   ✓ Orchestrator correctly loaded its own previous state")

        # Phase 4: Task executor session 2 (resume work)
        print("\n4. Task Executor Session 2 (resume)")
        executor2 = TaskExecutorAgent(
            workspace=tmppath,
            goal="Create calculator",  # Same goal = should load previous state
            max_rounds=1
        )

        # Should load previous executor state, NOT orchestrator state
        assert len(executor2.state.messages) == exec1_messages, \
            f"Expected {exec1_messages} messages, got {len(executor2.state.messages)}"

        # Verify it's the SAME messages from executor1
        assert executor2.state.messages[0]["content"] == "Building calculator..."
        assert executor2.state.messages[1]["content"] == "Calculator built"

        # Verify orchestrator messages NOT present
        for msg in orch1.state.messages:
            assert msg not in executor2.state.messages, \
                f"Orchestrator message leaked to executor session 2: {msg}"

        print("   ✓ Task executor correctly loaded its own previous state")


def test_state_files_persist_correctly():
    """Test that state files are written correctly and can be re-loaded."""
    print("\n=== Testing State Files Persist Correctly ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        agent_context_dir = tmppath / ".agent_context"

        # Create orchestrator and save state
        orch = OrchestratorAgent(workspace=tmppath)
        orch.add_user_message("Test message")
        orch.persist_state()

        # Verify state file exists
        orch_state_file = agent_context_dir / "orchestrator_state.json"
        assert orch_state_file.exists(), "Orchestrator state file should exist"

        # Load state file and verify contents
        with open(orch_state_file) as f:
            state_data = json.load(f)

        assert state_data["name"] == "orchestrator"
        assert len(state_data["messages"]) == 1
        assert state_data["messages"][0]["content"] == "Test message"

        print("✓ Orchestrator state persisted correctly")

        # Create task executor and save state
        executor = TaskExecutorAgent(
            workspace=tmppath,
            goal="Test",
            max_rounds=1
        )
        executor.add_message({"role": "user", "content": "Executor message"})
        executor.persist_state()

        # Verify state file exists
        exec_state_file = agent_context_dir / "task_executor_state.json"
        assert exec_state_file.exists(), "Task executor state file should exist"

        # Load state file and verify contents
        with open(exec_state_file) as f:
            state_data = json.load(f)

        assert state_data["name"] == "task_executor"
        assert len(state_data["messages"]) == 1
        assert state_data["messages"][0]["content"] == "Executor message"

        print("✓ Task executor state persisted correctly")

        # Verify both state files exist simultaneously
        assert orch_state_file.exists()
        assert exec_state_file.exists()
        print("✓ Both state files coexist without conflict")


if __name__ == "__main__":
    try:
        test_agents_have_separate_state_files()
        test_orchestrator_context_does_not_leak_to_executor()
        test_executor_context_does_not_leak_to_orchestrator()
        test_context_switching_loads_correct_state()
        test_state_files_persist_correctly()

        print("\n" + "=" * 60)
        print("✅ ALL CONTEXT ISOLATION TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
