"""
Tests for agent validation utilities.

Comprehensive test suite covering:
- validate_yaml_syntax
- validate_agent_config_structure
- validate_behavior_references
- validate_agent_dag
- validate_delegation_tool_schema

Target: >90% code coverage
"""

import pytest
from utils.agent_validator import (
    validate_yaml_syntax,
    validate_agent_config_structure,
    validate_behavior_references,
    validate_agent_dag,
    validate_delegation_tool_schema
)


class TestValidateYamlSyntax:
    """Test validate_yaml_syntax function."""

    def test_valid_yaml(self):
        """Valid YAML passes validation."""
        yaml_content = """
key: value
nested:
  inner: data
"""
        result = validate_yaml_syntax(yaml_content)
        assert result["valid"] is True
        assert result["data"] == {
            "key": "value",
            "nested": {"inner": "data"}
        }

    def test_valid_list(self):
        """YAML with lists passes."""
        yaml_content = """
items:
  - one
  - two
  - three
"""
        result = validate_yaml_syntax(yaml_content)
        assert result["valid"] is True
        assert result["data"]["items"] == ["one", "two", "three"]

    def test_empty_yaml(self):
        """Empty YAML string is treated as no content."""
        yaml_content = ""
        result = validate_yaml_syntax(yaml_content)
        # Empty string is treated as no content provided
        assert result["valid"] is False
        assert "No YAML content" in result["error"]

    def test_yaml_with_whitespace_only(self):
        """YAML with only whitespace is valid (returns None)."""
        yaml_content = "   \n  \n"
        result = validate_yaml_syntax(yaml_content)
        # Whitespace-only YAML is valid, returns None
        assert result["valid"] is True
        assert result.get("data") is None

    def test_invalid_yaml_syntax(self):
        """Invalid YAML syntax fails."""
        # Use truly invalid YAML (unclosed bracket)
        yaml_content = "key: [value"
        result = validate_yaml_syntax(yaml_content)
        assert result["valid"] is False
        assert "error" in result

    def test_no_content_provided(self):
        """No content or file path fails."""
        result = validate_yaml_syntax()
        assert result["valid"] is False
        assert "No YAML content" in result["error"]


class TestValidateAgentConfigStructure:
    """Test validate_agent_config_structure function."""

    def test_valid_minimal_config(self):
        """Minimal valid config passes."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing. Works well. Best for tests.",
            "system_prompt": "You are a test agent. Use tools.",
            "behaviors": []
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is True

    def test_valid_config_with_behaviors(self):
        """Config with behaviors passes."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing. Works well. Best for tests.",
            "system_prompt": "You are a test agent. Use tools.",
            "behaviors": [
                {"type": "ChatbotBehavior"},
                {"type": "DirectoryToolsBehavior"}
            ]
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is True

    def test_missing_role_fails(self):
        """Config missing role fails."""
        config = {
            "blurb": "Does testing.",
            "system_prompt": "You are a test agent.",
            "behaviors": []
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is False
        assert "role" in result["error"].lower()

    def test_missing_blurb_fails(self):
        """Config missing blurb fails."""
        config = {
            "role": "Test Agent",
            "system_prompt": "You are a test agent.",
            "behaviors": []
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is False
        assert "blurb" in result["error"].lower()

    def test_missing_system_prompt_fails(self):
        """Config missing system_prompt fails."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing.",
            "behaviors": []
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is False
        assert "system_prompt" in result["error"].lower()

    def test_missing_behaviors_fails(self):
        """Config missing behaviors fails."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing.",
            "system_prompt": "You are a test agent."
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is False
        assert "behaviors" in result["error"].lower()

    def test_empty_role_fails(self):
        """Empty role string fails."""
        config = {
            "role": "",
            "blurb": "Does testing.",
            "system_prompt": "You are a test agent.",
            "behaviors": []
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is False
        assert "role" in result["error"].lower()

    def test_empty_blurb_fails(self):
        """Empty blurb fails."""
        config = {
            "role": "Test Agent",
            "blurb": "",
            "system_prompt": "You are a test agent.",
            "behaviors": []
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is False
        assert "blurb" in result["error"].lower()

    def test_empty_system_prompt_fails(self):
        """Empty system prompt fails."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing.",
            "system_prompt": "",
            "behaviors": []
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is False
        assert "system prompt" in result["error"].lower()  # Note: error uses "system prompt" not "system_prompt"

    def test_behaviors_not_list_fails(self):
        """Behaviors must be a list."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing.",
            "system_prompt": "You are a test agent.",
            "behaviors": "not a list"
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is False
        assert "list" in result["error"].lower()

    def test_empty_behaviors_warns(self):
        """Empty behaviors list generates warning."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing. Works well. Best for tests.",
            "system_prompt": "You are a test agent.",
            "behaviors": []
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is True
        assert "warnings" in result
        assert any("no behaviors" in w.lower() for w in result["warnings"])

    def test_short_blurb_warns(self):
        """Very short blurb generates warning."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing",  # Only 1 sentence
            "system_prompt": "You are a test agent.",
            "behaviors": []
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is True
        assert "warnings" in result
        assert any("short" in w.lower() for w in result["warnings"])

    def test_long_blurb_warns(self):
        """Very long blurb generates warning."""
        config = {
            "role": "Test Agent",
            "blurb": "." * 15,  # 15 sentences
            "system_prompt": "You are a test agent.",
            "behaviors": []
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is True
        assert "warnings" in result
        assert any("long" in w.lower() for w in result["warnings"])

    def test_conversational_prompt_warns(self):
        """Conversational system prompt generates warning."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing. Works well. Best for tests.",
            "system_prompt": "Hi! I'm here to help you with your tasks!",
            "behaviors": []
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is True
        assert "warnings" in result
        assert any("conversational" in w.lower() for w in result["warnings"])

    def test_behavior_missing_type_fails(self):
        """Behavior without type key fails."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing.",
            "system_prompt": "You are a test agent.",
            "behaviors": [
                {"params": {}}  # Missing type
            ]
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is False
        assert "type" in result["error"].lower()

    def test_behavior_not_dict_fails(self):
        """Behavior that's not a dict fails."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing.",
            "system_prompt": "You are a test agent.",
            "behaviors": ["not a dict"]
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is False
        assert "dictionary" in result["error"].lower()

    def test_valid_delegation_tool(self):
        """Config with valid delegation_tool passes."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing. Works well. Best for tests.",
            "system_prompt": "You are a test agent.",
            "behaviors": [],
            "delegation_tool": {
                "name": "delegate_to_test",
                "description": "Delegate to test agent",
                "parameters": {
                    "task": {
                        "type": "string",
                        "description": "Task description"
                    }
                }
            }
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is True

    def test_delegation_tool_not_dict_fails(self):
        """delegation_tool must be a dict."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing.",
            "system_prompt": "You are a test agent.",
            "behaviors": [],
            "delegation_tool": "not a dict"
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is False
        assert "dictionary" in result["error"].lower()

    def test_delegation_tool_missing_keys_fails(self):
        """delegation_tool missing required keys fails."""
        config = {
            "role": "Test Agent",
            "blurb": "Does testing.",
            "system_prompt": "You are a test agent.",
            "behaviors": [],
            "delegation_tool": {
                "name": "delegate_to_test"
                # Missing description and parameters
            }
        }
        result = validate_agent_config_structure(config)
        assert result["valid"] is False

    def test_non_dict_config_fails(self):
        """Config that's not a dict fails."""
        config = "not a dict"
        result = validate_agent_config_structure(config)
        assert result["valid"] is False
        assert "dictionary" in result["error"].lower()


class TestValidateBehaviorReferences:
    """Test validate_behavior_references function."""

    def test_all_known_behaviors(self):
        """Config with all known behaviors passes."""
        config = {
            "behaviors": [
                {"type": "ChatbotBehavior"},
                {"type": "DirectoryToolsBehavior"},
                {"type": "ReadFileToolsBehavior"}
            ]
        }
        result = validate_behavior_references(config)
        assert result["valid"] is True

    def test_unknown_behavior_fails(self):
        """Config with unknown behavior fails."""
        config = {
            "behaviors": [
                {"type": "ChatbotBehavior"},
                {"type": "UnknownBehavior"}
            ]
        }
        result = validate_behavior_references(config)
        assert result["valid"] is False
        assert "unknown_behaviors" in result
        assert "UnknownBehavior" in result["unknown_behaviors"]

    def test_multiple_unknown_behaviors(self):
        """Multiple unknown behaviors are all detected."""
        config = {
            "behaviors": [
                {"type": "UnknownBehavior1"},
                {"type": "UnknownBehavior2"}
            ]
        }
        result = validate_behavior_references(config)
        assert result["valid"] is False
        assert len(result["unknown_behaviors"]) == 2

    def test_missing_behaviors_key_fails(self):
        """Config without behaviors key fails."""
        config = {}
        result = validate_behavior_references(config)
        assert result["valid"] is False
        assert "behaviors" in result["error"].lower()

    def test_behaviors_not_list_fails(self):
        """Behaviors that's not a list fails."""
        config = {
            "behaviors": "not a list"
        }
        result = validate_behavior_references(config)
        assert result["valid"] is False

    def test_empty_behaviors_list(self):
        """Empty behaviors list passes (no references to validate)."""
        config = {
            "behaviors": []
        }
        result = validate_behavior_references(config)
        assert result["valid"] is True


class TestValidateAgentDag:
    """Test validate_agent_dag function."""

    def test_no_cycles_simple(self):
        """Simple DAG with no cycles passes."""
        config = {
            "agents": {
                "a": {"can_delegate_to": ["b"]},
                "b": {"can_delegate_to": []},
            }
        }
        result = validate_agent_dag(config)
        assert result["valid"] is True

    def test_no_cycles_complex(self):
        """Complex DAG with no cycles passes."""
        config = {
            "agents": {
                "orchestrator": {"can_delegate_to": ["architect", "executor"]},
                "architect": {"can_delegate_to": []},
                "executor": {"can_delegate_to": []},
            }
        }
        result = validate_agent_dag(config)
        assert result["valid"] is True

    def test_simple_cycle_detected(self):
        """Simple 2-node cycle is detected."""
        config = {
            "agents": {
                "a": {"can_delegate_to": ["b"]},
                "b": {"can_delegate_to": ["a"]},
            }
        }
        result = validate_agent_dag(config)
        assert result["valid"] is False
        assert "cycle" in result
        assert len(result["cycle"]) >= 2

    def test_self_cycle_detected(self):
        """Self-referencing cycle is detected."""
        config = {
            "agents": {
                "a": {"can_delegate_to": ["a"]},
            }
        }
        result = validate_agent_dag(config)
        assert result["valid"] is False
        assert "cycle" in result

    def test_three_node_cycle_detected(self):
        """3-node cycle is detected."""
        config = {
            "agents": {
                "a": {"can_delegate_to": ["b"]},
                "b": {"can_delegate_to": ["c"]},
                "c": {"can_delegate_to": ["a"]},
            }
        }
        result = validate_agent_dag(config)
        assert result["valid"] is False
        assert "cycle" in result

    def test_complex_dag_with_cycle(self):
        """Complex graph with one cycle is detected."""
        config = {
            "agents": {
                "orchestrator": {"can_delegate_to": ["architect", "executor"]},
                "architect": {"can_delegate_to": ["executor"]},
                "executor": {"can_delegate_to": ["architect"]},  # Cycle
            }
        }
        result = validate_agent_dag(config)
        assert result["valid"] is False
        assert "cycle" in result

    def test_no_agents_key_fails(self):
        """Config without agents key fails."""
        config = {}
        result = validate_agent_dag(config)
        assert result["valid"] is False
        assert "agents" in result["error"].lower()

    def test_agents_not_dict_fails(self):
        """Agents that's not a dict fails."""
        config = {
            "agents": "not a dict"
        }
        result = validate_agent_dag(config)
        assert result["valid"] is False
        assert "dictionary" in result["error"].lower()

    def test_empty_agents_dict(self):
        """Empty agents dict passes (no cycles possible)."""
        config = {
            "agents": {}
        }
        result = validate_agent_dag(config)
        assert result["valid"] is True

    def test_agents_without_can_delegate_to(self):
        """Agents without can_delegate_to key are treated as terminal."""
        config = {
            "agents": {
                "a": {},  # No can_delegate_to
                "b": {"can_delegate_to": ["a"]},
            }
        }
        result = validate_agent_dag(config)
        assert result["valid"] is True


class TestValidateDelegationToolSchema:
    """Test validate_delegation_tool_schema function."""

    def test_valid_delegation_tool(self):
        """Valid delegation tool passes."""
        tool = {
            "name": "delegate_to_executor",
            "description": "Delegate task to executor",
            "parameters": {
                "task_description": {
                    "type": "string",
                    "description": "Task to execute",
                    "required": True
                }
            }
        }
        result = validate_delegation_tool_schema(tool)
        assert result["valid"] is True

    def test_valid_multiple_parameters(self):
        """Tool with multiple parameters passes."""
        tool = {
            "name": "delegate_to_agent",
            "description": "Delegate",
            "parameters": {
                "task": {
                    "type": "string",
                    "description": "Task"
                },
                "context": {
                    "type": "string",
                    "description": "Context"
                }
            }
        }
        result = validate_delegation_tool_schema(tool)
        assert result["valid"] is True

    def test_missing_name_fails(self):
        """Tool missing name fails."""
        tool = {
            "description": "Delegate",
            "parameters": {}
        }
        result = validate_delegation_tool_schema(tool)
        assert result["valid"] is False
        assert "name" in result["error"].lower()

    def test_missing_description_fails(self):
        """Tool missing description fails."""
        tool = {
            "name": "delegate_to_agent",
            "parameters": {}
        }
        result = validate_delegation_tool_schema(tool)
        assert result["valid"] is False
        assert "description" in result["error"].lower()

    def test_missing_parameters_fails(self):
        """Tool missing parameters fails."""
        tool = {
            "name": "delegate_to_agent",
            "description": "Delegate"
        }
        result = validate_delegation_tool_schema(tool)
        assert result["valid"] is False
        assert "parameters" in result["error"].lower()

    def test_name_not_starting_with_delegate_fails(self):
        """Tool name not starting with 'delegate_' fails."""
        tool = {
            "name": "execute_task",
            "description": "Execute task",
            "parameters": {
                "task": {"type": "string", "description": "Task"}
            }
        }
        result = validate_delegation_tool_schema(tool)
        assert result["valid"] is False
        assert "delegate_" in result["error"]

    def test_empty_name_fails(self):
        """Empty name fails."""
        tool = {
            "name": "",
            "description": "Delegate",
            "parameters": {}
        }
        result = validate_delegation_tool_schema(tool)
        assert result["valid"] is False

    def test_empty_description_fails(self):
        """Empty description fails."""
        tool = {
            "name": "delegate_to_agent",
            "description": "",
            "parameters": {}
        }
        result = validate_delegation_tool_schema(tool)
        assert result["valid"] is False

    def test_parameters_not_dict_fails(self):
        """Parameters must be a dict."""
        tool = {
            "name": "delegate_to_agent",
            "description": "Delegate",
            "parameters": "not a dict"
        }
        result = validate_delegation_tool_schema(tool)
        assert result["valid"] is False
        assert "dictionary" in result["error"].lower()

    def test_empty_parameters_fails(self):
        """Empty parameters dict fails (must have at least one param)."""
        tool = {
            "name": "delegate_to_agent",
            "description": "Delegate",
            "parameters": {}
        }
        result = validate_delegation_tool_schema(tool)
        assert result["valid"] is False
        assert "at least one parameter" in result["error"]

    def test_parameter_missing_type_fails(self):
        """Parameter without type fails."""
        tool = {
            "name": "delegate_to_agent",
            "description": "Delegate",
            "parameters": {
                "task": {
                    "description": "Task"
                    # Missing type
                }
            }
        }
        result = validate_delegation_tool_schema(tool)
        assert result["valid"] is False
        assert "type" in result["error"].lower()

    def test_parameter_missing_description_fails(self):
        """Parameter without description fails."""
        tool = {
            "name": "delegate_to_agent",
            "description": "Delegate",
            "parameters": {
                "task": {
                    "type": "string"
                    # Missing description
                }
            }
        }
        result = validate_delegation_tool_schema(tool)
        assert result["valid"] is False
        assert "description" in result["error"].lower()

    def test_non_dict_tool_fails(self):
        """Non-dict tool fails."""
        tool = "not a dict"
        result = validate_delegation_tool_schema(tool)
        assert result["valid"] is False
        assert "dictionary" in result["error"].lower()


class TestIntegration:
    """Integration tests combining multiple validators."""

    def test_complete_agent_config_validation(self):
        """Complete agent config passes all validations."""
        yaml_content = """
role: "Test Executor"
blurb: |
  TestExecutor handles test execution tasks.
  Works well with pytest and other frameworks.
  Best for automated testing and verification.

delegation_tool:
  name: "delegate_to_test_executor"
  description: "Delegate testing task to executor"
  parameters:
    task_description:
      type: string
      description: "Description of test task"
      required: true

system_prompt: |
  You are a test execution agent.

  Guidelines:
  - ALWAYS use tools - never just respond with text
  - Run tests using run_bash tool
  - Report results clearly

behaviors:
  - type: ChatbotBehavior
  - type: DirectoryToolsBehavior
  - type: CommandToolsBehavior
"""
        # Validate YAML syntax
        yaml_result = validate_yaml_syntax(yaml_content)
        assert yaml_result["valid"] is True

        config = yaml_result["data"]

        # Validate structure
        struct_result = validate_agent_config_structure(config)
        assert struct_result["valid"] is True

        # Validate behavior references
        behavior_result = validate_behavior_references(config)
        assert behavior_result["valid"] is True

        # Validate delegation tool
        delegation_result = validate_delegation_tool_schema(config["delegation_tool"])
        assert delegation_result["valid"] is True

    def test_agents_yaml_validation(self):
        """Validate agents.yaml style config."""
        yaml_content = """
agents:
  orchestrator:
    class: OrchestratorAgent
    can_delegate_to:
      - architect
      - executor

  architect:
    class: ArchitectAgent
    can_delegate_to: []

  executor:
    class: TaskExecutorAgent
    can_delegate_to: []
"""
        # Parse YAML
        yaml_result = validate_yaml_syntax(yaml_content)
        assert yaml_result["valid"] is True

        # Validate DAG
        dag_result = validate_agent_dag(yaml_result["data"])
        assert dag_result["valid"] is True
