"""
Tests for behavior validation utilities.

Comprehensive test suite covering:
- validate_python_syntax
- validate_behavior_independence
- validate_tool_schema
- validate_behavior_class_structure

Target: >90% code coverage
"""

import pytest
from utils.behavior_validator import (
    validate_python_syntax,
    validate_behavior_independence,
    validate_tool_schema,
    validate_behavior_class_structure
)


class TestValidatePythonSyntax:
    """Test validate_python_syntax function."""

    def test_valid_syntax(self):
        """Valid Python code passes validation."""
        code = "print('hello world')"
        result = validate_python_syntax(code)
        assert result["valid"] is True
        assert "ast" in result

    def test_valid_multiline_code(self):
        """Valid multiline Python code passes."""
        code = """
def foo():
    return 42

class Bar:
    pass
"""
        result = validate_python_syntax(code)
        assert result["valid"] is True

    def test_syntax_error(self):
        """Code with syntax error fails validation."""
        code = "print('unclosed string"
        result = validate_python_syntax(code)
        assert result["valid"] is False
        assert "error" in result
        assert "Syntax error" in result["error"] or "Parse error" in result["error"]

    def test_invalid_indentation(self):
        """Code with indentation error fails."""
        code = """
def foo():
return 42
"""
        result = validate_python_syntax(code)
        assert result["valid"] is False
        assert "error" in result

    def test_empty_code(self):
        """Empty code is valid Python."""
        code = ""
        result = validate_python_syntax(code)
        assert result["valid"] is True

    def test_comment_only_code(self):
        """Comment-only code is valid."""
        code = "# This is a comment"
        result = validate_python_syntax(code)
        assert result["valid"] is True


class TestValidateBehaviorIndependence:
    """Test validate_behavior_independence function."""

    def test_no_imports(self):
        """Code with no imports passes."""
        code = """
class MyBehavior:
    def get_name(self):
        return "my_behavior"
"""
        result = validate_behavior_independence(code)
        assert result["valid"] is True

    def test_base_import_allowed(self):
        """Importing from behaviors.base is allowed."""
        code = """
from behaviors.base import AgentBehavior

class MyBehavior(AgentBehavior):
    pass
"""
        result = validate_behavior_independence(code)
        assert result["valid"] is True

    def test_base_agent_import_allowed(self):
        """Importing from behaviors.base_agent is allowed."""
        code = """
from behaviors.base_agent import BaseAgent

class MyBehavior:
    pass
"""
        result = validate_behavior_independence(code)
        assert result["valid"] is True

    def test_cross_behavior_import_fails(self):
        """Importing another behavior fails validation."""
        code = """
from behaviors.command_tools import CommandToolsBehavior

class MyBehavior:
    def __init__(self):
        self.cmd_tools = CommandToolsBehavior()
"""
        result = validate_behavior_independence(code)
        assert result["valid"] is False
        assert "error" in result
        assert "violations" in result
        assert len(result["violations"]) > 0

    def test_multiple_violations(self):
        """Multiple cross-imports are all detected."""
        code = """
from behaviors.command_tools import CommandToolsBehavior
from behaviors.read_file_tools import ReadFileToolsBehavior

class MyBehavior:
    pass
"""
        result = validate_behavior_independence(code)
        assert result["valid"] is False
        assert len(result["violations"]) == 2

    def test_import_statement_violation(self):
        """Import statement (not from) is detected."""
        code = """
import behaviors.command_tools

class MyBehavior:
    pass
"""
        result = validate_behavior_independence(code)
        assert result["valid"] is False
        assert "violations" in result

    def test_stdlib_imports_allowed(self):
        """Standard library imports are allowed."""
        code = """
import os
import sys
from pathlib import Path
from typing import Any

class MyBehavior:
    pass
"""
        result = validate_behavior_independence(code)
        assert result["valid"] is True

    def test_third_party_imports_allowed(self):
        """Third-party imports are allowed."""
        code = """
import pytest
from unittest.mock import Mock

class MyBehavior:
    pass
"""
        result = validate_behavior_independence(code)
        assert result["valid"] is True

    def test_syntax_error_handled(self):
        """Syntax errors are reported clearly."""
        code = "from behaviors.base import ("
        result = validate_behavior_independence(code)
        assert result["valid"] is False
        assert "error" in result
        assert "parse" in result["error"].lower()


class TestValidateToolSchema:
    """Test validate_tool_schema function."""

    def test_valid_minimal_tool(self):
        """Minimal valid tool schema passes."""
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
        result = validate_tool_schema(tool)
        assert result["valid"] is True

    def test_valid_tool_with_parameters(self):
        """Tool with parameters passes validation."""
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
        result = validate_tool_schema(tool)
        assert result["valid"] is True

    def test_missing_type_key(self):
        """Tool missing 'type' key fails."""
        tool = {
            "function": {
                "name": "my_tool",
                "description": "Does something",
                "parameters": {"type": "object", "properties": {}}
            }
        }
        result = validate_tool_schema(tool)
        assert result["valid"] is False
        assert "type" in result["error"].lower()

    def test_missing_function_key(self):
        """Tool missing 'function' key fails."""
        tool = {
            "type": "function"
        }
        result = validate_tool_schema(tool)
        assert result["valid"] is False
        assert "function" in result["error"].lower()

    def test_wrong_type_value(self):
        """Tool with wrong type value fails."""
        tool = {
            "type": "not_function",
            "function": {
                "name": "my_tool",
                "description": "Does something",
                "parameters": {"type": "object", "properties": {}}
            }
        }
        result = validate_tool_schema(tool)
        assert result["valid"] is False
        assert "function" in result["error"].lower()

    def test_missing_function_name(self):
        """Tool missing function name fails."""
        tool = {
            "type": "function",
            "function": {
                "description": "Does something",
                "parameters": {"type": "object", "properties": {}}
            }
        }
        result = validate_tool_schema(tool)
        assert result["valid"] is False
        assert "name" in result["error"].lower()

    def test_missing_function_description(self):
        """Tool missing description fails."""
        tool = {
            "type": "function",
            "function": {
                "name": "my_tool",
                "parameters": {"type": "object", "properties": {}}
            }
        }
        result = validate_tool_schema(tool)
        assert result["valid"] is False
        assert "description" in result["error"].lower()

    def test_missing_parameters(self):
        """Tool missing parameters fails."""
        tool = {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Does something"
            }
        }
        result = validate_tool_schema(tool)
        assert result["valid"] is False
        assert "parameters" in result["error"].lower()

    def test_parameters_not_object_type(self):
        """Parameters must have type 'object'."""
        tool = {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Does something",
                "parameters": {
                    "type": "string",
                    "properties": {}
                }
            }
        }
        result = validate_tool_schema(tool)
        assert result["valid"] is False
        assert "object" in result["error"].lower()

    def test_parameters_missing_properties(self):
        """Parameters must have 'properties' dict."""
        tool = {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Does something",
                "parameters": {
                    "type": "object"
                }
            }
        }
        result = validate_tool_schema(tool)
        assert result["valid"] is False
        assert "properties" in result["error"].lower()

    def test_non_dict_tool(self):
        """Non-dict tool fails validation."""
        tool = "not a dict"
        result = validate_tool_schema(tool)
        assert result["valid"] is False
        assert "dictionary" in result["error"].lower()

    def test_function_not_dict(self):
        """Function value must be dict."""
        tool = {
            "type": "function",
            "function": "not a dict"
        }
        result = validate_tool_schema(tool)
        assert result["valid"] is False
        assert "dictionary" in result["error"].lower()


class TestValidateBehaviorClassStructure:
    """Test validate_behavior_class_structure function."""

    def test_valid_minimal_behavior(self):
        """Minimal valid behavior passes."""
        code = """
from behaviors.base import AgentBehavior

class MyBehavior(AgentBehavior):
    '''My behavior.'''

    def get_name(self):
        return "my_behavior"
"""
        result = validate_behavior_class_structure(code)
        assert result["valid"] is True
        assert result["class_name"] == "MyBehavior"

    def test_valid_with_triple_quotes_docstring(self):
        """Behavior with triple-quote docstring passes."""
        code = '''
from behaviors.base import AgentBehavior

class MyBehavior(AgentBehavior):
    """
    My behavior with multiline docstring.

    Does something useful.
    """

    def get_name(self):
        return "my_behavior"
'''
        result = validate_behavior_class_structure(code)
        assert result["valid"] is True

    def test_no_class_fails(self):
        """Code without class definition fails."""
        code = """
def foo():
    pass
"""
        result = validate_behavior_class_structure(code)
        assert result["valid"] is False
        assert "No class" in result["error"]

    def test_class_not_inheriting_agentbehavior_fails(self):
        """Class not inheriting AgentBehavior fails."""
        code = """
class MyBehavior:
    '''My behavior.'''

    def get_name(self):
        return "my_behavior"
"""
        result = validate_behavior_class_structure(code)
        assert result["valid"] is False
        assert "AgentBehavior" in result["error"]

    def test_missing_docstring_fails(self):
        """Class without docstring fails."""
        code = """
from behaviors.base import AgentBehavior

class MyBehavior(AgentBehavior):
    def get_name(self):
        return "my_behavior"
"""
        result = validate_behavior_class_structure(code)
        assert result["valid"] is False
        assert "docstring" in result["error"].lower()

    def test_missing_get_name_fails(self):
        """Class without get_name method fails."""
        code = """
from behaviors.base import AgentBehavior

class MyBehavior(AgentBehavior):
    '''My behavior.'''

    def some_other_method(self):
        pass
"""
        result = validate_behavior_class_structure(code)
        assert result["valid"] is False
        assert "get_name" in result["error"]

    def test_dangerous_eval_detected(self):
        """Code with eval() is rejected."""
        code = """
from behaviors.base import AgentBehavior

class MyBehavior(AgentBehavior):
    '''My behavior.'''

    def get_name(self):
        return "my_behavior"

    def dangerous_method(self):
        eval("malicious code")
"""
        result = validate_behavior_class_structure(code)
        assert result["valid"] is False
        assert "dangerous" in result["error"].lower()
        assert "eval" in result["error"].lower()

    def test_dangerous_exec_detected(self):
        """Code with exec() is rejected."""
        code = """
from behaviors.base import AgentBehavior

class MyBehavior(AgentBehavior):
    '''My behavior.'''

    def get_name(self):
        return "my_behavior"

    def bad_method(self):
        exec("code")
"""
        result = validate_behavior_class_structure(code)
        assert result["valid"] is False
        assert "exec" in result["error"].lower()

    def test_dangerous_compile_detected(self):
        """Code with compile() is rejected."""
        code = """
from behaviors.base import AgentBehavior

class MyBehavior(AgentBehavior):
    '''My behavior.'''

    def get_name(self):
        return "my_behavior"

    def bad_method(self):
        compile("code", "<string>", "exec")
"""
        result = validate_behavior_class_structure(code)
        assert result["valid"] is False
        assert "compile" in result["error"].lower()

    def test_syntax_error_handled(self):
        """Syntax errors are reported."""
        code = "class MyBehavior("
        result = validate_behavior_class_structure(code)
        assert result["valid"] is False
        assert "parse" in result["error"].lower()

    def test_attribute_style_inheritance(self):
        """Handles behaviors.base.AgentBehavior style inheritance."""
        code = """
import behaviors.base

class MyBehavior(behaviors.base.AgentBehavior):
    '''My behavior.'''

    def get_name(self):
        return "my_behavior"
"""
        result = validate_behavior_class_structure(code)
        assert result["valid"] is True


class TestIntegration:
    """Integration tests combining multiple validators."""

    def test_complete_valid_behavior(self):
        """Complete valid behavior passes all validators."""
        code = """
from behaviors.base import AgentBehavior

class MyToolBehavior(AgentBehavior):
    '''Provides my custom tool.'''

    def get_name(self):
        return "my_tool"

    def get_tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Does something",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg": {"type": "string"}
                    }
                }
            }
        }]

    def dispatch_tool(self, agent, tool_name, args):
        if tool_name == "my_tool":
            return {"result": "ok"}
        return super().dispatch_tool(agent, tool_name, args)
"""
        # Test syntax
        syntax_result = validate_python_syntax(code)
        assert syntax_result["valid"] is True

        # Test independence
        indep_result = validate_behavior_independence(code)
        assert indep_result["valid"] is True

        # Test structure
        struct_result = validate_behavior_class_structure(code)
        assert struct_result["valid"] is True

    def test_validate_tool_from_behavior(self):
        """Extract and validate tool from behavior code."""
        code = """
from behaviors.base import AgentBehavior

class MyBehavior(AgentBehavior):
    '''My behavior.'''

    def get_name(self):
        return "my_behavior"

    def get_tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Does something",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }]
"""
        # Parse and execute to get tool
        import ast
        tree = ast.parse(code)
        # Note: In real usage, you'd execute get_tools() to get the tool dict
        # Here we just validate the code structure

        syntax_result = validate_python_syntax(code)
        assert syntax_result["valid"] is True

        struct_result = validate_behavior_class_structure(code)
        assert struct_result["valid"] is True

        # Manually extract tool for validation
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
        tool_result = validate_tool_schema(tool)
        assert tool_result["valid"] is True
