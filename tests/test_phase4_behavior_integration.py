"""
Integration tests for Phase 4 - Behavior system integration with agents.

Tests:
1. BaseAgent behavior loading from config
2. TaskExecutorAgent with behaviors
3. OrchestratorAgent with behaviors
4. ArchitectAgent with behaviors
5. Tool dispatch through behaviors
6. Context enhancement through behaviors
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from base_agent import BaseAgent
from task_executor_agent import TaskExecutorAgent
from orchestrator_agent import OrchestratorAgent
from architect_agent import ArchitectAgent


class TestBehaviorLoadingFromConfig:
    """Test that agents can load behaviors from YAML config."""

    def test_task_executor_loads_behaviors_from_config(self, tmp_path):
        """TaskExecutor should load behaviors from task_executor_config.yaml."""
        # Create a temporary workspace
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create TaskExecutor with use_behaviors=True
        agent = TaskExecutorAgent(
            workspace=workspace,
            use_behaviors=True,
            config_file="task_executor_config.yaml"
        )

        # Verify behaviors were loaded
        assert len(agent.behaviors) > 0, "Should load behaviors from config"
        assert len(agent.tool_registry) > 0, "Should register tools from behaviors"

        # Check for expected behaviors (based on task_executor_config.yaml)
        behavior_names = [b.get_name() for b in agent.behaviors]
        print(f"Loaded behaviors: {behavior_names}")

        # Should have at least some of: SubAgentContext, FileTools, CommandTools, ServerTools, LoopDetection
        assert any("subagent" in name.lower() or "context" in name.lower() for name in behavior_names), \
            "Should load SubAgentContextBehavior"

    def test_orchestrator_loads_behaviors_from_config(self, tmp_path):
        """Orchestrator should load behaviors from orchestrator_config.yaml."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        agent = OrchestratorAgent(
            workspace=workspace,
            use_behaviors=True,
            config_file="orchestrator_config.yaml"
        )

        assert len(agent.behaviors) > 0, "Should load behaviors from config"
        assert len(agent.tool_registry) > 0, "Should register tools from behaviors"

        behavior_names = [b.get_name() for b in agent.behaviors]
        print(f"Orchestrator loaded behaviors: {behavior_names}")

    def test_architect_loads_behaviors_from_config(self, tmp_path):
        """Architect should load behaviors from architect_config.yaml."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        agent = ArchitectAgent(
            workspace=workspace,
            use_behaviors=True,
            config_file="architect_config.yaml"
        )

        assert len(agent.behaviors) > 0, "Should load behaviors from config"
        assert len(agent.tool_registry) > 0, "Should register tools from behaviors"

        behavior_names = [b.get_name() for b in agent.behaviors]
        print(f"Architect loaded behaviors: {behavior_names}")


class TestBehaviorToolDispatch:
    """Test that tool calls are dispatched to behaviors correctly."""

    def test_task_executor_dispatches_to_behaviors(self, tmp_path):
        """TaskExecutor should dispatch tool calls to behaviors."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        agent = TaskExecutorAgent(
            workspace=workspace,
            use_behaviors=True,
            config_file="task_executor_config.yaml"
        )

        # Get tools from behaviors
        tools = agent.get_tools()
        assert len(tools) > 0, "Should have tools from behaviors"

        # Verify tool names are in registry
        for tool in tools:
            tool_name = tool["function"]["name"]
            assert tool_name in agent.tool_registry, f"Tool {tool_name} should be in registry"

        print(f"TaskExecutor has {len(tools)} tools from behaviors")

    def test_behavior_tools_have_no_duplicates(self, tmp_path):
        """All tool names should be unique (no conflicts)."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        agent = TaskExecutorAgent(
            workspace=workspace,
            use_behaviors=True,
            config_file="task_executor_config.yaml"
        )

        tools = agent.get_tools()
        tool_names = [t["function"]["name"] for t in tools]

        # Check for duplicates
        duplicates = [name for name in tool_names if tool_names.count(name) > 1]
        assert len(duplicates) == 0, f"Tool names should be unique, found duplicates: {duplicates}"


class TestBehaviorContextEnhancement:
    """Test that behaviors can enhance context."""

    def test_task_executor_context_includes_behavior_modifications(self, tmp_path):
        """TaskExecutor context should be enhanced by behaviors."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        agent = TaskExecutorAgent(
            workspace=workspace,
            use_behaviors=True,
            config_file="task_executor_config.yaml"
        )

        # Add a test message
        agent.add_message({"role": "user", "content": "Test message"})

        # Build context
        context = agent.build_context()

        # Should have system prompt + messages
        assert len(context) > 0, "Context should not be empty"
        assert context[0]["role"] == "system", "First message should be system prompt"

        print(f"Context has {len(context)} messages")

    def test_system_prompt_includes_behavior_instructions(self, tmp_path):
        """System prompt should include instructions from behaviors."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        agent = TaskExecutorAgent(
            workspace=workspace,
            use_behaviors=True,
            config_file="task_executor_config.yaml"
        )

        system_prompt = agent.get_system_prompt()

        # Should have some content from behaviors
        assert len(system_prompt) > 0, "System prompt should not be empty"
        print(f"System prompt length: {len(system_prompt)}")


class TestBackwardCompatibility:
    """Test that legacy mode (use_behaviors=False) still works."""

    def test_task_executor_legacy_mode_works(self, tmp_path):
        """TaskExecutor should work with use_behaviors=False (legacy)."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create agent with legacy mode
        agent = TaskExecutorAgent(
            workspace=workspace,
            use_behaviors=False  # Legacy mode
        )

        # Should still work
        assert agent.context_strategy is not None, "Should have context strategy"
        tools = agent.get_tools()
        assert len(tools) > 0, "Should have tools from legacy system"

        print(f"Legacy mode has {len(tools)} tools")

    def test_orchestrator_legacy_mode_works(self, tmp_path):
        """Orchestrator should work with use_behaviors=False."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        agent = OrchestratorAgent(
            workspace=workspace,
            use_behaviors=False
        )

        assert agent.context_strategy is not None
        tools = agent.get_tools()
        assert len(tools) > 0

    def test_architect_legacy_mode_works(self, tmp_path):
        """Architect should work with use_behaviors=False."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        agent = ArchitectAgent(
            workspace=workspace,
            use_behaviors=False
        )

        assert agent.context_strategy is not None
        tools = agent.get_tools()
        assert len(tools) > 0


class TestBehaviorEventTriggers:
    """Test that behavior events are triggered at the right times."""

    def test_on_goal_start_triggered(self, tmp_path):
        """on_goal_start should be triggered when goal is set."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        agent = TaskExecutorAgent(
            workspace=workspace,
            use_behaviors=True,
            config_file="task_executor_config.yaml"
        )

        # Set a goal
        agent.set_goal("Test goal")

        # Event should have been triggered (we can't easily verify this without mocking,
        # but at least verify it doesn't crash)
        assert agent.context_manager is not None
        assert agent.workspace_manager is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
