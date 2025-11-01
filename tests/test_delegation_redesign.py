"""
Test suite for delegation redesign.

This test verifies that:
1. All agents can be delegated to (via CLI or tool call)
2. All agents support chat mode when no goal provided
3. Orchestrator can delegate to any agent (including other orchestrators)
4. DelegationBehavior is truly generic and works with any agent
5. SubAgentModeBehavior and ChatbotBehavior compose correctly
"""

import pytest
from pathlib import Path
import tempfile
import shutil


class TestSubAgentModeBehavior:
    """Test SubAgentModeBehavior makes agents delegatable."""

    def test_subagent_mode_behavior_imports(self):
        """Test that SubAgentModeBehavior can be imported."""
        from behaviors.subagent_mode import SubAgentModeBehavior

        behavior = SubAgentModeBehavior(is_subagent=True)
        assert behavior.get_name() == "subagent_mode"

    def test_backward_compatibility_alias(self):
        """Test that SubAgentContextBehavior alias still works."""
        from behaviors.subagent_mode import SubAgentContextBehavior

        behavior = SubAgentContextBehavior()
        # Should be same as SubAgentModeBehavior
        assert behavior.get_name() == "subagent_mode"

    def test_completion_tools_provided(self):
        """Test that mark_complete and mark_failed tools are provided."""
        from behaviors.subagent_mode import SubAgentModeBehavior

        behavior = SubAgentModeBehavior()
        tools = behavior.get_tools()

        tool_names = [tool["function"]["name"] for tool in tools]
        assert "mark_complete" in tool_names
        assert "mark_failed" in tool_names

    def test_goal_context_injection(self):
        """Test that goal is injected into context."""
        from behaviors.subagent_mode import SubAgentModeBehavior
        from context_manager import ContextManager

        behavior = SubAgentModeBehavior(is_subagent=True)
        ctx = ContextManager()
        ctx.load_or_init("Test goal")

        context = [{"role": "system", "content": "System prompt"}]
        enhanced = behavior.enhance_context(context, context_manager=ctx)

        # Should have inserted goal message after system prompt
        assert len(enhanced) == 2
        assert "DELEGATED GOAL: Test goal" in enhanced[1]["content"]


class TestChatbotBehavior:
    """Test ChatbotBehavior enables interactive chat mode."""

    def test_chatbot_behavior_imports(self):
        """Test that ChatbotBehavior can be imported."""
        from behaviors.chatbot import ChatbotBehavior

        behavior = ChatbotBehavior()
        assert behavior.get_name() == "chatbot"

    def test_set_goal_tool_provided(self):
        """Test that set_goal tool is provided."""
        from behaviors.chatbot import ChatbotBehavior

        behavior = ChatbotBehavior()
        tools = behavior.get_tools()

        tool_names = [tool["function"]["name"] for tool in tools]
        assert "set_goal" in tool_names

    def test_set_goal_tool_dispatch(self):
        """Test that set_goal tool can be dispatched."""
        from behaviors.chatbot import ChatbotBehavior

        behavior = ChatbotBehavior()

        # Mock agent
        class MockAgent:
            def __init__(self):
                self.behaviors = [behavior]

        agent = MockAgent()

        result = behavior.dispatch_tool(
            "set_goal",
            {"goal": "Create a calculator", "requirements": "Python, pytest"},
            agent=agent
        )

        assert result["success"] is True
        assert result["goal"] == "Create a calculator"
        assert result["mode_transition"] == "chat → execution"

    def test_chat_mode_activation(self):
        """Test that chat mode activates when no goal present."""
        from behaviors.chatbot import ChatbotBehavior

        behavior = ChatbotBehavior()

        # Mock agent with no goal
        class MockAgent:
            def __init__(self):
                self.context_manager = None

        agent = MockAgent()

        behavior.on_agent_start(agent)
        assert behavior.chat_mode_active is True


class TestDelegationBehavior:
    """Test DelegationBehavior supports delegating to any agent."""

    def test_delegation_behavior_imports(self):
        """Test that DelegationBehavior can be imported."""
        from behaviors.delegation import DelegationBehavior

        # Create with minimal relationships
        relationships = {
            "can_delegate_to": ["task_executor"],
            "task_executor": {
                "class": "TaskExecutorAgent",
                "blurb": "Handles coding tasks"
            }
        }

        behavior = DelegationBehavior(relationships)
        assert behavior.get_name() == "delegation"

    def test_delegation_tools_created_from_config(self):
        """Test that delegation tools are auto-generated from config."""
        from behaviors.delegation import DelegationBehavior
        import yaml

        # Load actual config
        with open("task_executor_config.yaml") as f:
            executor_config = yaml.safe_load(f)

        relationships = {
            "can_delegate_to": ["task_executor"],
            "task_executor": {
                "class": "TaskExecutorAgent",
                "blurb": executor_config.get("blurb", ""),
                "delegation_tool": executor_config.get("delegation_tool", {})
            }
        }

        behavior = DelegationBehavior(relationships)
        tools = behavior.get_tools()

        # Should have delegate_to_executor tool
        tool_names = [tool["function"]["name"] for tool in tools]
        assert "delegate_to_executor" in tool_names

    def test_orchestrator_delegation_tool_created(self):
        """Test that orchestrator can be delegated to."""
        from behaviors.delegation import DelegationBehavior
        import yaml

        # Load orchestrator config
        with open("orchestrator_config.yaml") as f:
            orch_config = yaml.safe_load(f)

        # Simulate parent orchestrator delegating to child orchestrator
        relationships = {
            "can_delegate_to": ["orchestrator"],
            "orchestrator": {
                "class": "OrchestratorAgent",
                "blurb": orch_config.get("blurb", ""),
                "delegation_tool": orch_config.get("delegation_tool", {})
            }
        }

        behavior = DelegationBehavior(relationships)
        tools = behavior.get_tools()

        # Should have delegate_to_orchestrator tool
        tool_names = [tool["function"]["name"] for tool in tools]
        assert "delegate_to_orchestrator" in tool_names

    def test_generic_delegation_dispatch(self):
        """Test that delegation dispatch works for any agent."""
        from behaviors.delegation import DelegationBehavior

        relationships = {
            "can_delegate_to": ["task_executor"],
            "task_executor": {
                "class": "TaskExecutorAgent",
                "blurb": "Handles coding tasks"
            }
        }

        behavior = DelegationBehavior(relationships)

        # Mock agent
        class MockAgent:
            pass

        agent = MockAgent()

        # Call delegation tool (won't actually run, just setup)
        result = behavior.dispatch_tool(
            "delegate_to_executor",
            {
                "task_description": "Create a calculator",
                "workspace_mode": "new"
            },
            agent=agent
        )

        # Should succeed (even though it's just setup)
        assert result["success"] is True
        assert result["target_agent"] == "task_executor"
        assert result["goal"] == "Create a calculator"


class TestAgentConfigs:
    """Test that all agent configs are updated correctly."""

    def test_task_executor_config_has_chatbot(self):
        """Test TaskExecutor config includes ChatbotBehavior."""
        import yaml

        with open("task_executor_config.yaml") as f:
            config = yaml.safe_load(f)

        behavior_types = [b["type"] for b in config.get("behaviors", [])]
        assert "ChatbotBehavior" in behavior_types

    def test_orchestrator_config_has_chatbot(self):
        """Test Orchestrator config includes ChatbotBehavior."""
        import yaml

        with open("orchestrator_config.yaml") as f:
            config = yaml.safe_load(f)

        behavior_types = [b["type"] for b in config.get("behaviors", [])]
        assert "ChatbotBehavior" in behavior_types

    def test_orchestrator_config_has_delegation_tool(self):
        """Test Orchestrator config defines delegation_tool."""
        import yaml

        with open("orchestrator_config.yaml") as f:
            config = yaml.safe_load(f)

        assert "delegation_tool" in config
        assert config["delegation_tool"]["name"] == "delegate_to_orchestrator"

    def test_architect_config_has_chatbot(self):
        """Test Architect config includes ChatbotBehavior."""
        import yaml

        with open("architect_config.yaml") as f:
            config = yaml.safe_load(f)

        behavior_types = [b["type"] for b in config.get("behaviors", [])]
        assert "ChatbotBehavior" in behavior_types


class TestEndToEndDelegation:
    """End-to-end tests for delegation scenarios."""

    def test_task_executor_can_be_instantiated_with_new_behaviors(self):
        """Test TaskExecutor instantiates correctly with new behaviors."""
        from task_executor_agent import TaskExecutorAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = TaskExecutorAgent(
                workspace=Path(tmpdir),
                goal="Test goal",
                use_behaviors=True
            )

            # Check behaviors loaded
            behavior_names = [b.get_name() for b in agent.behaviors]

            # Should have SubAgentModeBehavior (auto-added) or ChatbotBehavior
            assert "subagent_mode" in behavior_names or "chatbot" in behavior_names

    def test_orchestrator_can_be_instantiated_with_new_behaviors(self):
        """Test Orchestrator instantiates correctly with new behaviors."""
        from orchestrator_agent import OrchestratorAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = OrchestratorAgent(
                workspace=Path(tmpdir),
                use_behaviors=True
            )

            # Check behaviors loaded
            behavior_names = [b.get_name() for b in agent.behaviors]

            # Should have ChatbotBehavior and DelegationBehavior
            assert "chatbot" in behavior_names
            assert "delegation" in behavior_names

    def test_architect_can_be_instantiated_with_new_behaviors(self):
        """Test Architect instantiates correctly with new behaviors."""
        from architect_agent import ArchitectAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = ArchitectAgent(
                workspace=Path(tmpdir),
                use_behaviors=True
            )

            # Check behaviors loaded
            behavior_names = [b.get_name() for b in agent.behaviors]

            # Should have SubAgentModeBehavior (auto-added) and ChatbotBehavior
            assert "subagent_mode" in behavior_names or "chatbot" in behavior_names


class TestBehaviorComposition:
    """Test that behaviors compose correctly without conflicts."""

    def test_no_tool_conflicts_in_task_executor(self):
        """Test TaskExecutor has no tool name conflicts."""
        from task_executor_agent import TaskExecutorAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = TaskExecutorAgent(
                workspace=Path(tmpdir),
                goal="Test",
                use_behaviors=True
            )

            tools = agent.get_tools()
            tool_names = [t["function"]["name"] for t in tools]

            # Check for duplicates
            assert len(tool_names) == len(set(tool_names)), f"Duplicate tools found: {tool_names}"

    def test_no_tool_conflicts_in_orchestrator(self):
        """Test Orchestrator has no tool name conflicts."""
        from orchestrator_agent import OrchestratorAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = OrchestratorAgent(
                workspace=Path(tmpdir),
                use_behaviors=True
            )

            tools = agent.get_tools()
            tool_names = [t["function"]["name"] for t in tools]

            # Check for duplicates
            assert len(tool_names) == len(set(tool_names)), f"Duplicate tools found: {tool_names}"

    def test_subagent_mode_and_chatbot_compose(self):
        """Test SubAgentModeBehavior and ChatbotBehavior work together."""
        from behaviors.subagent_mode import SubAgentModeBehavior
        from behaviors.chatbot import ChatbotBehavior

        subagent_mode = SubAgentModeBehavior(is_subagent=True)
        chatbot = ChatbotBehavior()

        # Get all tools
        all_tools = subagent_mode.get_tools() + chatbot.get_tools()
        tool_names = [t["function"]["name"] for t in all_tools]

        # Should have no conflicts
        assert len(tool_names) == len(set(tool_names))

        # Should have both completion and goal-setting tools
        assert "mark_complete" in tool_names
        assert "mark_failed" in tool_names
        assert "set_goal" in tool_names


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
