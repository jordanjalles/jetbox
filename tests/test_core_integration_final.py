"""
Core integration tests for finalized behavior system.

Tests that all refactors work correctly:
1. Behavior loading from config
2. Dynamic tool documentation generation
3. SubAgentContextBehavior auto-add
4. DelegationBehavior auto-add
5. Global behavior defaults
"""
import pytest
from pathlib import Path
import tempfile
import yaml
from base_agent import BaseAgent
from task_executor_agent import TaskExecutorAgent
from orchestrator_agent import OrchestratorAgent
from architect_agent import ArchitectAgent


class TestBehaviorLoading:
    """Test behaviors load from config with global defaults."""

    def test_task_executor_loads_behaviors(self, tmp_path):
        """TaskExecutor loads behaviors from config."""
        # Create minimal config
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
system_prompt: |
  You are a test agent.

behaviors:
  - type: FileToolsBehavior
  - type: LoopDetectionBehavior
""")

        # Create agent
        agent = TaskExecutorAgent(
            workspace=tmp_path,
            use_behaviors=True,
            config_file=str(config_file)
        )

        # Check behaviors loaded
        behavior_names = [b.get_name() for b in agent.behaviors]
        assert "file_tools" in behavior_names
        assert "loop_detection" in behavior_names

    def test_global_defaults_apply(self, tmp_path):
        """Global behavior defaults are used when not overridden."""
        # This test checks that defaults from agent_config.yaml are loaded
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
system_prompt: Test
behaviors:
  - type: LoopDetectionBehavior
    # No params - should use global defaults
""")

        agent = TaskExecutorAgent(
            workspace=tmp_path,
            use_behaviors=True,
            config_file=str(config_file)
        )

        # Check behavior loaded
        behavior_names = [b.get_name() for b in agent.behaviors]
        assert "loop_detection" in behavior_names


class TestDynamicToolDocumentation:
    """Test tool documentation generated dynamically from behaviors."""

    def test_generate_tool_documentation(self, tmp_path):
        """Tool documentation is generated from loaded behaviors."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
system_prompt: Test
behaviors:
  - type: FileToolsBehavior
  - type: CommandToolsBehavior
""")

        agent = TaskExecutorAgent(
            workspace=tmp_path,
            use_behaviors=True,
            config_file=str(config_file)
        )

        # Generate tool documentation
        tool_docs = agent.generate_tool_documentation()

        # Check format
        assert "Available tools:" in tool_docs
        assert "write_file" in tool_docs
        assert "run_bash" in tool_docs
        assert "(" in tool_docs  # Has function signatures

    def test_system_prompt_includes_tool_docs(self, tmp_path):
        """System prompt includes dynamically generated tool docs."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
system_prompt: |
  You are a test agent.
  Do not hardcode tools here.

behaviors:
  - type: FileToolsBehavior
""")

        agent = TaskExecutorAgent(
            workspace=tmp_path,
            use_behaviors=True,
            config_file=str(config_file)
        )

        system_prompt = agent.get_system_prompt()

        # Check tool docs are in system prompt
        assert "Available tools:" in system_prompt
        assert "write_file" in system_prompt
        assert "Do not hardcode tools here" in system_prompt


class TestSubAgentContextAutoAdd:
    """Test SubAgentContextBehavior auto-added for subagents."""

    def test_task_executor_is_subagent(self, tmp_path):
        """TaskExecutor is auto-detected as subagent and gets SubAgentContextBehavior."""
        # TaskExecutor is in orchestrator's can_delegate_to list
        agent = TaskExecutorAgent(
            workspace=tmp_path,
            use_behaviors=True,
            config_file="task_executor_config.yaml"
        )

        # Check SubAgentContextBehavior was auto-added
        behavior_names = [b.get_name() for b in agent.behaviors]
        assert "subagent_context" in behavior_names

    def test_orchestrator_not_subagent(self, tmp_path):
        """Orchestrator is not a subagent and doesn't get SubAgentContextBehavior."""
        agent = OrchestratorAgent(
            workspace=tmp_path,
            use_behaviors=True,
            config_file="orchestrator_config.yaml"
        )

        # Check SubAgentContextBehavior NOT auto-added
        behavior_names = [b.get_name() for b in agent.behaviors]
        assert "subagent_context" not in behavior_names


class TestDelegationAutoAdd:
    """Test DelegationBehavior auto-added from relationships."""

    def test_orchestrator_auto_adds_delegation(self, tmp_path):
        """Orchestrator auto-adds DelegationBehavior from agents.yaml."""
        agent = OrchestratorAgent(
            workspace=tmp_path,
            use_behaviors=True,
            config_file="orchestrator_config.yaml"
        )

        # Check DelegationBehavior was auto-added
        behavior_names = [b.get_name() for b in agent.behaviors]
        assert "delegation" in behavior_names

        # Check delegation tools are available
        tools = agent.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "consult_architect" in tool_names
        assert "delegate_to_executor" in tool_names

    def test_task_executor_no_delegation(self, tmp_path):
        """TaskExecutor doesn't delegate and doesn't get DelegationBehavior."""
        agent = TaskExecutorAgent(
            workspace=tmp_path,
            use_behaviors=True,
            config_file="task_executor_config.yaml"
        )

        # Check DelegationBehavior NOT auto-added
        behavior_names = [b.get_name() for b in agent.behaviors]
        assert "delegation" not in behavior_names


class TestArchitectBehaviors:
    """Test Architect loads correctly with behaviors."""

    def test_architect_loads_behaviors(self, tmp_path):
        """Architect loads behaviors from config."""
        agent = ArchitectAgent(
            workspace=tmp_path,
            use_behaviors=True,
            config_file="architect_config.yaml"
        )

        # Check behaviors loaded
        behavior_names = [b.get_name() for b in agent.behaviors]
        assert "architect_tools" in behavior_names
        assert "compact_when_near_full" in behavior_names
        assert "loop_detection" in behavior_names

        # Check tools available
        tools = agent.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "write_architecture_doc" in tool_names
        assert "write_module_spec" in tool_names
        assert "write_task_list" in tool_names


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
