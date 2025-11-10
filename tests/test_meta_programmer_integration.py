"""
Integration tests for MetaProgrammer Agent (Phase 5).

Test coverage:
- Agent Configuration (3 tests)
- Team Configuration (3 tests)
- Behavior Loading (4 tests)
- Tool Availability (5 tests)
- End-to-End Workflows (2 tests)

Total: 17 tests
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import yaml
import tempfile
from agents.task_executor_agent import TaskExecutorAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.architect_agent import ArchitectAgent


class TestMetaProgrammerAgentConfiguration:
    """Test suite for MetaProgrammer agent configuration."""

    # ============================================================
    # Agent Configuration (3 tests)
    # ============================================================

    def test_meta_programmer_config_exists(self):
        """MetaProgrammer config file exists and is valid YAML."""
        config_path = Path("config/agents/meta_programmer.yaml")
        assert config_path.exists(), "meta_programmer.yaml should exist"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "role" in config
        assert "blurb" in config
        assert "system_prompt" in config
        assert "behaviors" in config
        assert "delegation_tool" in config

    def test_meta_programmer_has_required_behaviors(self):
        """MetaProgrammer config includes all required behaviors."""
        config_path = Path("config/agents/meta_programmer.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        behavior_types = [b['type'] for b in config['behaviors']]

        # Required behaviors from spec
        required_behaviors = [
            "ChatbotBehavior",
            "CompactWhenNearFullBehavior",
            "CreateBehaviorBehavior",
            "CreateAgentBehavior",
            "ValidationBehavior",
            "SandboxTestBehavior",
            "ReadFileToolsBehavior",
            "WriteFileToolsBehavior",
            "DirectoryToolsBehavior",
            "CommandToolsBehavior"
        ]

        for required in required_behaviors:
            assert required in behavior_types, f"Missing required behavior: {required}"

    def test_meta_programmer_delegation_tool_complete(self):
        """MetaProgrammer delegation tool has complete specification."""
        config_path = Path("config/agents/meta_programmer.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        delegation_tool = config['delegation_tool']

        assert delegation_tool['name'] == "delegate_to_meta_programmer"
        assert 'description' in delegation_tool
        assert 'parameters' in delegation_tool

        params = delegation_tool['parameters']
        assert 'task_description' in params
        assert 'task_type' in params
        assert 'safety_mode' in params

        # Check task_type enum
        assert params['task_type']['enum'] == ["create_behavior", "create_agent"]

        # Check safety_mode enum
        assert params['safety_mode']['enum'] == ["dryrun", "review", "auto", "strict"]

    # ============================================================
    # Team Configuration (3 tests)
    # ============================================================

    def test_meta_team_config_exists(self):
        """Meta team config file exists and is valid YAML."""
        config_path = Path("config/teams/meta.yaml")
        assert config_path.exists(), "meta.yaml team config should exist"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "name" in config
        assert "description" in config
        assert "agents" in config

    def test_meta_team_has_all_agents(self):
        """Meta team includes orchestrator, meta_programmer, task_executor, architect."""
        config_path = Path("config/teams/meta.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        agents = config['agents']

        assert 'orchestrator' in agents
        assert 'meta_programmer' in agents
        assert 'task_executor' in agents
        assert 'architect' in agents

        # Check agent classes
        assert agents['orchestrator']['class'] == "OrchestratorAgent"
        assert agents['meta_programmer']['class'] == "TaskExecutorAgent"
        assert agents['task_executor']['class'] == "TaskExecutorAgent"
        assert agents['architect']['class'] == "ArchitectAgent"

    def test_meta_team_no_delegation_cycles(self):
        """Meta team has no delegation cycles."""
        config_path = Path("config/teams/meta.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        agents = config['agents']

        # Build delegation graph
        graph = {}
        for agent_name, agent_config in agents.items():
            graph[agent_name] = agent_config.get('can_delegate_to', [])

        # Check for cycles using DFS
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited = set()
        for agent in graph:
            if agent not in visited:
                assert not has_cycle(agent, visited, set()), f"Delegation cycle detected involving {agent}"

    # ============================================================
    # Behavior Loading (4 tests)
    # ============================================================

    def test_create_behavior_behavior_loads(self):
        """CreateBehaviorBehavior loads successfully."""
        from behaviors.create_behavior import CreateBehaviorBehavior

        behavior = CreateBehaviorBehavior()
        assert behavior is not None
        assert behavior.get_name() == "create_behavior"

        tools = behavior.get_tools()
        assert len(tools) > 0
        assert any(t['function']['name'] == 'create_behavior' for t in tools)

    def test_create_agent_behavior_loads(self):
        """CreateAgentBehavior loads successfully."""
        from behaviors.create_agent import CreateAgentBehavior

        behavior = CreateAgentBehavior()
        assert behavior is not None
        assert behavior.get_name() == "create_agent"

        tools = behavior.get_tools()
        assert len(tools) > 0
        assert any(t['function']['name'] == 'create_agent' for t in tools)

    def test_validation_behavior_loads(self):
        """ValidationBehavior loads successfully."""
        from behaviors.validation import ValidationBehavior

        behavior = ValidationBehavior()
        assert behavior is not None
        assert behavior.get_name() == "validation"

        tools = behavior.get_tools()
        assert len(tools) > 0

    def test_sandbox_test_behavior_loads(self):
        """SandboxTestBehavior loads successfully."""
        from behaviors.sandbox_test import SandboxTestBehavior

        behavior = SandboxTestBehavior()
        assert behavior is not None
        assert behavior.get_name() == "sandbox_test"

        tools = behavior.get_tools()
        assert len(tools) > 0

    # ============================================================
    # Tool Availability (5 tests)
    # ============================================================

    def test_meta_programmer_has_create_behavior_tool(self):
        """MetaProgrammer has create_behavior tool available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            agent = TaskExecutorAgent(
                workspace=workspace,
                goal="test",
                config_file="config/agents/meta_programmer.yaml"
            )

            tool_names = [tool['function']['name'] for tool in agent.get_tools()]
            assert 'create_behavior' in tool_names

    def test_meta_programmer_has_create_agent_tool(self):
        """MetaProgrammer has create_agent tool available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            agent = TaskExecutorAgent(
                workspace=workspace,
                goal="test",
                config_file="config/agents/meta_programmer.yaml"
            )

            tool_names = [tool['function']['name'] for tool in agent.get_tools()]
            assert 'create_agent' in tool_names

    def test_meta_programmer_has_validation_tools(self):
        """MetaProgrammer has validation tools available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            agent = TaskExecutorAgent(
                workspace=workspace,
                goal="test",
                config_file="config/agents/meta_programmer.yaml"
            )

            tool_names = [tool['function']['name'] for tool in agent.get_tools()]

            # Check for validation tools
            validation_tools = [t for t in tool_names if 'validate' in t]
            assert len(validation_tools) > 0, "Should have validation tools"

    def test_meta_programmer_has_test_tools(self):
        """MetaProgrammer has test execution tools available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            agent = TaskExecutorAgent(
                workspace=workspace,
                goal="test",
                config_file="config/agents/meta_programmer.yaml"
            )

            tool_names = [tool['function']['name'] for tool in agent.get_tools()]

            # Check for test tools
            test_tools = [t for t in tool_names if 'test' in t.lower()]
            assert len(test_tools) > 0, "Should have test tools"

    def test_meta_programmer_has_file_tools(self):
        """MetaProgrammer has file operation tools available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            agent = TaskExecutorAgent(
                workspace=workspace,
                goal="test",
                config_file="config/agents/meta_programmer.yaml"
            )

            tool_names = [tool['function']['name'] for tool in agent.get_tools()]

            # Check for file tools
            assert 'read_file' in tool_names
            assert 'write_file' in tool_names
            assert 'list_dir' in tool_names

    # ============================================================
    # End-to-End Workflows (2 tests)
    # ============================================================

    def test_meta_programmer_can_instantiate(self):
        """MetaProgrammer agent can instantiate without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Should not raise any exceptions
            agent = TaskExecutorAgent(
                workspace=workspace,
                goal="Create a simple test behavior",
                config_file="config/agents/meta_programmer.yaml"
            )

            assert agent is not None
            assert agent.workspace == workspace
            assert len(agent.behaviors) > 0
            assert len(agent.get_tools()) > 0

    def test_orchestrator_can_delegate_to_meta_programmer(self):
        """Orchestrator in meta team can delegate to meta_programmer."""
        config_path = Path("config/teams/meta.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        orchestrator = config['agents']['orchestrator']
        can_delegate_to = orchestrator.get('can_delegate_to', [])

        assert 'meta_programmer' in can_delegate_to, "Orchestrator should be able to delegate to meta_programmer"


class TestMetaProgrammerSystemPrompt:
    """Test that MetaProgrammer system prompt contains key guidance."""

    def test_system_prompt_has_critical_principles(self):
        """System prompt includes critical principles."""
        config_path = Path("config/agents/meta_programmer.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        system_prompt = config['system_prompt']

        assert "Single Responsibility" in system_prompt
        assert "Zero Dependencies" in system_prompt
        assert "Composability" in system_prompt
        assert "Safety First" in system_prompt

    def test_system_prompt_has_workflow_guidance(self):
        """System prompt includes workflow guidance."""
        config_path = Path("config/agents/meta_programmer.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        system_prompt = config['system_prompt']

        assert "WORKFLOW" in system_prompt
        assert "create_behavior" in system_prompt
        assert "create_agent" in system_prompt
        assert "safety_mode" in system_prompt

    def test_system_prompt_emphasizes_review_mode(self):
        """System prompt emphasizes review mode by default."""
        config_path = Path("config/agents/meta_programmer.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        system_prompt = config['system_prompt']

        # Should mention review mode as default
        assert "review" in system_prompt.lower()


class TestOrchestratorIntegration:
    """Test that orchestrator is updated with meta_programmer guidance."""

    def test_orchestrator_mentions_meta_programmer(self):
        """Orchestrator config mentions meta_programmer."""
        config_path = Path("config/agents/orchestrator.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        system_prompt = config['system_prompt']

        assert "MetaProgrammer" in system_prompt or "meta_programmer" in system_prompt

    def test_orchestrator_has_delegation_guidance(self):
        """Orchestrator config has clear delegation guidance."""
        config_path = Path("config/agents/orchestrator.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        system_prompt = config['system_prompt']

        # Should have section about when to delegate
        assert "delegate" in system_prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
