"""
Tests for ValidationBehavior.

Comprehensive test suite covering:
- get_name() returns "validation"
- All 7 validation tools are provided
- validate_python_syntax tool execution
- validate_tool_schema tool execution
- Error handling and edge cases

Target: >90% code coverage
"""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
from behaviors.validation import ValidationBehavior


class TestValidationBehaviorBasics:
    """Test basic behavior interface."""

    def test_get_name(self):
        """get_name() returns 'validation'."""
        behavior = ValidationBehavior()
        assert behavior.get_name() == "validation"

    def test_initialization(self):
        """Behavior initializes without errors."""
        behavior = ValidationBehavior()
        assert behavior is not None

    def test_initialization_with_kwargs(self):
        """Behavior accepts and ignores kwargs."""
        behavior = ValidationBehavior(some_param="value", another=42)
        assert behavior is not None


class TestValidationBehaviorTools:
    """Test tool definitions."""

    def test_get_tools_returns_list(self):
        """get_tools() returns a list."""
        behavior = ValidationBehavior()
        tools = behavior.get_tools()
        assert isinstance(tools, list)

    def test_get_tools_count(self):
        """get_tools() returns exactly 7 tools."""
        behavior = ValidationBehavior()
        tools = behavior.get_tools()
        assert len(tools) == 7

    def test_all_tools_present(self):
        """All 7 expected validation tools are present."""
        behavior = ValidationBehavior()
        tools = behavior.get_tools()

        tool_names = [tool["function"]["name"] for tool in tools]

        expected_tools = [
            "validate_python_syntax",
            "validate_behavior_independence",
            "validate_tool_schema",
            "validate_behavior_class",
            "validate_yaml_syntax",
            "validate_agent_dag",
            "validate_agent_config"
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Tool {expected} not found"

    def test_tool_structure(self):
        """Each tool has proper OpenAI function schema structure."""
        behavior = ValidationBehavior()
        tools = behavior.get_tools()

        for tool in tools:
            assert "type" in tool
            assert tool["type"] == "function"
            assert "function" in tool

            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

            params = func["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params

    def test_validate_python_syntax_tool_schema(self):
        """validate_python_syntax tool has correct schema."""
        behavior = ValidationBehavior()
        tools = behavior.get_tools()

        tool = next(t for t in tools if t["function"]["name"] == "validate_python_syntax")
        func = tool["function"]

        assert func["description"]
        assert "code" in func["parameters"]["properties"]
        assert "code" in func["parameters"]["required"]

    def test_validate_tool_schema_tool_schema(self):
        """validate_tool_schema tool has correct schema."""
        behavior = ValidationBehavior()
        tools = behavior.get_tools()

        tool = next(t for t in tools if t["function"]["name"] == "validate_tool_schema")
        func = tool["function"]

        assert func["description"]
        assert "tool" in func["parameters"]["properties"]
        assert "tool" in func["parameters"]["required"]


class TestValidatePythonSyntaxTool:
    """Test validate_python_syntax tool execution."""

    def test_valid_python_code(self):
        """Valid Python code returns success."""
        behavior = ValidationBehavior()
        agent = Mock()

        args = {"code": "print('hello world')"}
        result = behavior.dispatch_tool(agent, "validate_python_syntax", args)

        assert "result" in result
        assert result["result"]["valid"] is True

    def test_invalid_python_code(self):
        """Invalid Python code returns error."""
        behavior = ValidationBehavior()
        agent = Mock()

        args = {"code": "print('unclosed string"}
        result = behavior.dispatch_tool(agent, "validate_python_syntax", args)

        assert "result" in result
        assert result["result"]["valid"] is False
        assert "error" in result["result"]

    def test_empty_code(self):
        """Empty code is valid Python."""
        behavior = ValidationBehavior()
        agent = Mock()

        args = {"code": ""}
        result = behavior.dispatch_tool(agent, "validate_python_syntax", args)

        assert "result" in result
        assert result["result"]["valid"] is True

    def test_multiline_valid_code(self):
        """Valid multiline code passes."""
        behavior = ValidationBehavior()
        agent = Mock()

        code = """
def foo():
    return 42

class Bar:
    pass
"""
        args = {"code": code}
        result = behavior.dispatch_tool(agent, "validate_python_syntax", args)

        assert "result" in result
        assert result["result"]["valid"] is True

    def test_missing_code_argument(self):
        """Missing code argument is handled gracefully."""
        behavior = ValidationBehavior()
        agent = Mock()

        args = {}
        result = behavior.dispatch_tool(agent, "validate_python_syntax", args)

        # Empty string should be treated as valid Python
        assert "result" in result
        assert result["result"]["valid"] is True


class TestValidateToolSchemaTool:
    """Test validate_tool_schema tool execution."""

    def test_valid_minimal_tool(self):
        """Valid minimal tool schema passes."""
        behavior = ValidationBehavior()
        agent = Mock()

        tool = {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Does something",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }

        args = {"tool": tool}
        result = behavior.dispatch_tool(agent, "validate_tool_schema", args)

        assert "result" in result
        assert result["result"]["valid"] is True

    def test_valid_tool_with_parameters(self):
        """Valid tool with parameters passes."""
        behavior = ValidationBehavior()
        agent = Mock()

        tool = {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path"
                        }
                    },
                    "required": ["path"]
                }
            }
        }

        args = {"tool": tool}
        result = behavior.dispatch_tool(agent, "validate_tool_schema", args)

        assert "result" in result
        assert result["result"]["valid"] is True

    def test_invalid_tool_missing_type(self):
        """Tool missing type key fails."""
        behavior = ValidationBehavior()
        agent = Mock()

        tool = {
            "function": {
                "name": "my_tool",
                "description": "Does something",
                "parameters": {"type": "object", "properties": {}}
            }
        }

        args = {"tool": tool}
        result = behavior.dispatch_tool(agent, "validate_tool_schema", args)

        assert "result" in result
        assert result["result"]["valid"] is False
        assert "error" in result["result"]

    def test_invalid_tool_missing_function(self):
        """Tool missing function key fails."""
        behavior = ValidationBehavior()
        agent = Mock()

        tool = {"type": "function"}

        args = {"tool": tool}
        result = behavior.dispatch_tool(agent, "validate_tool_schema", args)

        assert "result" in result
        assert result["result"]["valid"] is False

    def test_invalid_tool_wrong_type_value(self):
        """Tool with wrong type value fails."""
        behavior = ValidationBehavior()
        agent = Mock()

        tool = {
            "type": "not_function",
            "function": {
                "name": "my_tool",
                "description": "Does something",
                "parameters": {"type": "object", "properties": {}}
            }
        }

        args = {"tool": tool}
        result = behavior.dispatch_tool(agent, "validate_tool_schema", args)

        assert "result" in result
        assert result["result"]["valid"] is False

    def test_non_dict_tool(self):
        """Non-dict tool fails validation."""
        behavior = ValidationBehavior()
        agent = Mock()

        args = {"tool": "not a dict"}
        result = behavior.dispatch_tool(agent, "validate_tool_schema", args)

        assert "result" in result
        assert result["result"]["valid"] is False

    def test_missing_tool_argument(self):
        """Missing tool argument is handled gracefully."""
        behavior = ValidationBehavior()
        agent = Mock()

        args = {}
        result = behavior.dispatch_tool(agent, "validate_tool_schema", args)

        assert "result" in result
        assert result["result"]["valid"] is False


class TestValidateBehaviorIndependenceTool:
    """Test validate_behavior_independence tool execution."""

    def test_behavior_independence_with_valid_file(self, tmp_path):
        """Valid behavior file passes independence check."""
        behavior = ValidationBehavior()

        # Create agent mock with workspace
        agent = Mock()
        agent.workspace = tmp_path

        # Create a valid behavior file
        behavior_file = tmp_path / "my_behavior.py"
        behavior_file.write_text("""
from behaviors.base import AgentBehavior

class MyBehavior(AgentBehavior):
    def get_name(self):
        return "my_behavior"
""")

        args = {"file_path": "my_behavior.py"}
        result = behavior.dispatch_tool(agent, "validate_behavior_independence", args)

        assert "result" in result
        assert result["result"]["valid"] is True

    def test_behavior_independence_with_cross_import(self, tmp_path):
        """Behavior with cross-import fails check."""
        behavior = ValidationBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        # Create behavior file with cross-import
        behavior_file = tmp_path / "my_behavior.py"
        behavior_file.write_text("""
from behaviors.command_tools import CommandToolsBehavior

class MyBehavior:
    pass
""")

        args = {"file_path": "my_behavior.py"}
        result = behavior.dispatch_tool(agent, "validate_behavior_independence", args)

        assert "result" in result
        assert result["result"]["valid"] is False

    def test_behavior_independence_file_not_found(self):
        """Non-existent file returns error."""
        behavior = ValidationBehavior()

        agent = Mock()
        agent.workspace = Path("/tmp/nonexistent")

        args = {"file_path": "nonexistent.py"}
        result = behavior.dispatch_tool(agent, "validate_behavior_independence", args)

        assert "error" in result
        assert "not found" in result["error"].lower()


class TestValidateBehaviorClassTool:
    """Test validate_behavior_class tool execution."""

    def test_valid_behavior_class(self):
        """Valid behavior class passes validation."""
        behavior = ValidationBehavior()
        agent = Mock()

        code = """
from behaviors.base import AgentBehavior

class MyBehavior(AgentBehavior):
    '''My behavior.'''

    def get_name(self):
        return "my_behavior"
"""

        args = {"code": code, "expected_name": "MyBehavior"}
        result = behavior.dispatch_tool(agent, "validate_behavior_class", args)

        assert "result" in result
        assert result["result"]["valid"] is True

    def test_invalid_behavior_class_no_inheritance(self):
        """Behavior without AgentBehavior inheritance fails."""
        behavior = ValidationBehavior()
        agent = Mock()

        code = """
class MyBehavior:
    '''My behavior.'''

    def get_name(self):
        return "my_behavior"
"""

        args = {"code": code, "expected_name": "MyBehavior"}
        result = behavior.dispatch_tool(agent, "validate_behavior_class", args)

        assert "result" in result
        assert result["result"]["valid"] is False

    def test_invalid_behavior_class_wrong_name(self):
        """Behavior with wrong class name fails."""
        behavior = ValidationBehavior()
        agent = Mock()

        code = """
from behaviors.base import AgentBehavior

class WrongName(AgentBehavior):
    '''My behavior.'''

    def get_name(self):
        return "my_behavior"
"""

        args = {"code": code, "expected_name": "MyBehavior"}
        result = behavior.dispatch_tool(agent, "validate_behavior_class", args)

        assert "result" in result
        assert result["result"]["valid"] is False


class TestValidateYamlSyntaxTool:
    """Test validate_yaml_syntax tool execution."""

    def test_valid_yaml(self):
        """Valid YAML passes validation."""
        behavior = ValidationBehavior()
        agent = Mock()

        yaml_str = """
key: value
nested:
  inner: data
"""

        args = {"yaml_str": yaml_str}
        result = behavior.dispatch_tool(agent, "validate_yaml_syntax", args)

        assert "result" in result
        assert result["result"]["valid"] is True

    def test_invalid_yaml(self):
        """Invalid YAML fails validation."""
        behavior = ValidationBehavior()
        agent = Mock()

        yaml_str = "key: [unclosed"

        args = {"yaml_str": yaml_str}
        result = behavior.dispatch_tool(agent, "validate_yaml_syntax", args)

        assert "result" in result
        assert result["result"]["valid"] is False


class TestValidateAgentDagTool:
    """Test validate_agent_dag tool execution."""

    def test_valid_dag_no_cycles(self):
        """Valid DAG passes validation."""
        behavior = ValidationBehavior()
        agent = Mock()

        agents_config = {
            "agents": {
                "orchestrator": {"can_delegate_to": ["executor"]},
                "executor": {"can_delegate_to": []}
            }
        }

        args = {"agents_config": agents_config}
        result = behavior.dispatch_tool(agent, "validate_agent_dag", args)

        assert "result" in result
        assert result["result"]["valid"] is True

    def test_invalid_dag_with_cycle(self):
        """DAG with cycle fails validation."""
        behavior = ValidationBehavior()
        agent = Mock()

        agents_config = {
            "agents": {
                "a": {"can_delegate_to": ["b"]},
                "b": {"can_delegate_to": ["a"]}
            }
        }

        args = {"agents_config": agents_config}
        result = behavior.dispatch_tool(agent, "validate_agent_dag", args)

        assert "result" in result
        assert result["result"]["valid"] is False


class TestValidateAgentConfigTool:
    """Test validate_agent_config tool execution."""

    def test_valid_agent_config(self):
        """Valid agent config passes validation."""
        behavior = ValidationBehavior()
        agent = Mock()

        config = {
            "role": "Test Agent",
            "blurb": "Does testing. Works well. Best for tests.",
            "system_prompt": "You are a test agent. Use tools.",
            "behaviors": []
        }

        args = {"config": config}
        result = behavior.dispatch_tool(agent, "validate_agent_config", args)

        assert "result" in result
        assert result["result"]["valid"] is True

    def test_invalid_agent_config_missing_field(self):
        """Agent config missing required field fails."""
        behavior = ValidationBehavior()
        agent = Mock()

        config = {
            "role": "Test Agent",
            "behaviors": []
            # Missing system_prompt (required field)
        }

        args = {"config": config}
        result = behavior.dispatch_tool(agent, "validate_agent_config", args)

        assert "result" in result
        assert result["result"]["valid"] is False


class TestDispatchToolErrorHandling:
    """Test error handling in dispatch_tool."""

    def test_unknown_tool_name(self):
        """Unknown tool name raises NotImplementedError."""
        behavior = ValidationBehavior()
        agent = Mock()

        # This should call super().dispatch_tool() which raises NotImplementedError
        with pytest.raises(NotImplementedError):
            behavior.dispatch_tool(agent, "unknown_tool", {})

    def test_exception_in_validation_function(self):
        """Exception in validation function is caught and returned."""
        behavior = ValidationBehavior()
        agent = Mock()

        # Pass invalid data that will cause an exception
        args = {"code": None}  # None will cause issues in validation
        result = behavior.dispatch_tool(agent, "validate_python_syntax", args)

        # Should return error dict instead of raising
        assert "error" in result or "result" in result


class TestIntegration:
    """Integration tests combining multiple validation tools."""

    def test_complete_behavior_validation_workflow(self, tmp_path):
        """Complete workflow: syntax -> independence -> class structure."""
        behavior = ValidationBehavior()
        agent = Mock()
        agent.workspace = tmp_path

        # Valid behavior code
        code = """
from behaviors.base import AgentBehavior

class TestBehavior(AgentBehavior):
    '''Test behavior for validation.'''

    def get_name(self):
        return "test_behavior"
"""

        # 1. Validate syntax
        syntax_result = behavior.dispatch_tool(
            agent, "validate_python_syntax", {"code": code}
        )
        assert syntax_result["result"]["valid"] is True

        # 2. Write to file for independence check
        behavior_file = tmp_path / "test_behavior.py"
        behavior_file.write_text(code)

        indep_result = behavior.dispatch_tool(
            agent, "validate_behavior_independence", {"file_path": "test_behavior.py"}
        )
        assert indep_result["result"]["valid"] is True

        # 3. Validate class structure
        class_result = behavior.dispatch_tool(
            agent, "validate_behavior_class",
            {"code": code, "expected_name": "TestBehavior"}
        )
        assert class_result["result"]["valid"] is True

    def test_complete_agent_config_validation_workflow(self):
        """Complete workflow: YAML -> config structure -> DAG."""
        behavior = ValidationBehavior()
        agent = Mock()

        # 1. Validate YAML
        yaml_str = """
role: "Test Agent"
blurb: "Does testing. Works well. Best for tests."
system_prompt: "You are a test agent."
behaviors: []
"""

        yaml_result = behavior.dispatch_tool(
            agent, "validate_yaml_syntax", {"yaml_str": yaml_str}
        )
        assert yaml_result["result"]["valid"] is True

        # 2. Validate config structure
        config = {
            "role": "Test Agent",
            "blurb": "Does testing. Works well. Best for tests.",
            "system_prompt": "You are a test agent.",
            "behaviors": []
        }

        config_result = behavior.dispatch_tool(
            agent, "validate_agent_config", {"config": config}
        )
        assert config_result["result"]["valid"] is True

        # 3. Validate DAG (separate agents config)
        agents_config = {
            "agents": {
                "test_agent": {"can_delegate_to": []}
            }
        }

        dag_result = behavior.dispatch_tool(
            agent, "validate_agent_dag", {"agents_config": agents_config}
        )
        assert dag_result["result"]["valid"] is True
