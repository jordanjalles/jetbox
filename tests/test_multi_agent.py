"""
Basic tests for multi-agent architecture.

Tests agent instantiation, registry, and basic operations.
"""
from pathlib import Path
import tempfile
import shutil

from agent_registry import AgentRegistry
from orchestrator_agent import OrchestratorAgent
from task_executor_agent import TaskExecutorAgent


def test_agent_registry():
    """Test agent registry creation and lookup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = AgentRegistry(workspace=Path(tmpdir))

        # List agents
        agents = registry.list_agents()
        assert "orchestrator" in agents
        assert "task_executor" in agents

        # Get delegation graph
        graph = registry.get_delegation_graph()
        assert graph["orchestrator"] == ["task_executor"]
        assert graph["task_executor"] == []

        print("✓ Agent registry works")


def test_orchestrator_creation():
    """Test orchestrator agent creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = AgentRegistry(workspace=Path(tmpdir))
        orchestrator = registry.get_agent("orchestrator")

        assert orchestrator.name == "orchestrator"
        assert orchestrator.get_context_strategy() == "append_until_full"

        # Check tools
        tools = orchestrator.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "delegate_to_executor" in tool_names
        assert "clarify_with_user" in tool_names

        print("✓ Orchestrator agent works")


def test_task_executor_creation():
    """Test task executor agent creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = AgentRegistry(workspace=Path(tmpdir))
        executor = registry.get_agent("task_executor")

        assert executor.name == "task_executor"
        assert executor.get_context_strategy() == "hierarchical"

        # Check tools
        tools = executor.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "write_file" in tool_names
        assert "read_file" in tool_names
        assert "mark_subtask_complete" in tool_names

        print("✓ TaskExecutor agent works")


def test_delegation():
    """Test delegation from orchestrator to task executor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = AgentRegistry(workspace=Path(tmpdir))

        # Test allowed delegation
        assert registry.can_delegate("orchestrator", "task_executor")

        # Test disallowed delegation
        assert not registry.can_delegate("task_executor", "orchestrator")

        # Test actual delegation
        result = registry.delegate_task(
            from_agent="orchestrator",
            to_agent="task_executor",
            task_description="Create a test file",
        )

        assert result["success"] == True
        assert "task_executor" in result["agent"]

        print("✓ Delegation works")


def test_orchestrator_context_building():
    """Test orchestrator context building."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = OrchestratorAgent(workspace=Path(tmpdir))

        # Add some messages
        orch.add_user_message("Hello")
        orch.add_message({"role": "assistant", "content": "Hi there!"})

        # Build context
        context = orch.build_context()

        # Should have: system prompt + messages
        assert len(context) >= 3
        assert context[0]["role"] == "system"
        assert any(m["content"] == "Hello" for m in context)

        print("✓ Orchestrator context building works")


def test_task_executor_context_building():
    """Test task executor context building."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = TaskExecutorAgent(workspace=Path(tmpdir), goal="Test goal")

        # Build context
        context = executor.build_context()

        # Should have: system prompt + task info
        assert len(context) >= 2
        assert context[0]["role"] == "system"
        assert any("GOAL: Test goal" in str(m.get("content", "")) for m in context)

        print("✓ TaskExecutor context building works")


def test_orchestrator_token_estimation():
    """Test token estimation for compaction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = OrchestratorAgent(workspace=Path(tmpdir))

        # Add messages
        messages = [
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Response"},
        ]

        tokens = orch._estimate_tokens(messages)
        assert tokens > 0

        print("✓ Token estimation works")


def test_orchestrator_compaction():
    """Test message compaction when near token limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = OrchestratorAgent(workspace=Path(tmpdir))

        # Add many messages
        for i in range(30):
            orch.add_user_message(f"Message {i}")
            orch.add_message({"role": "assistant", "content": f"Response {i}"})

        # Compact
        compacted = orch._compact_messages(orch.state.messages)

        # Should have summary + recent messages
        assert len(compacted) < len(orch.state.messages)
        assert compacted[0]["role"] == "user"
        assert "summary" in compacted[0]["content"].lower()

        print("✓ Message compaction works")


if __name__ == "__main__":
    test_agent_registry()
    test_orchestrator_creation()
    test_task_executor_creation()
    test_delegation()
    test_orchestrator_context_building()
    test_task_executor_context_building()
    test_orchestrator_token_estimation()
    test_orchestrator_compaction()

    print("\n✅ All tests passed!")
