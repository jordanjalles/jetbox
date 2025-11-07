"""
Behavior validation utilities for self-extensibility system.

Provides validation functions for generated behavior code to ensure:
- Valid Python syntax
- Independence from other behaviors
- Well-formed tool schemas
- Adherence to composability principles
"""

from __future__ import annotations
import ast
from pathlib import Path
from typing import Any


# Known behaviors that shouldn't be imported by other behaviors
KNOWN_BEHAVIORS = {
    "chatbot",
    "compact_when_near_full",
    "directory_tools",
    "read_file_tools",
    "write_file_tools",
    "command_tools",
    "server_tools",
    "loop_detection",
    "workspace_task_notes",
    "delegation",
    "architect_tools",
    "hierarchical_context",
    "subagent_context",
    "architect_context",
}


def validate_python_syntax(code: str) -> dict[str, Any]:
    """
    Validate Python syntax.

    Args:
        code: Python source code to validate

    Returns:
        dict with keys:
        - valid (bool): True if syntax is valid
        - error (str): Error message if invalid (optional)
    """
    try:
        ast.parse(code)
        return {"valid": True}
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"Syntax error at line {e.lineno}: {e.msg}"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Unexpected error: {str(e)}"
        }


def validate_behavior_independence(file_path: str | Path) -> dict[str, Any]:
    """
    Check for cross-behavior imports.

    Behaviors must be independent and not import other behaviors
    (except behaviors.base for the AgentBehavior base class).

    Args:
        file_path: Path to behavior Python file

    Returns:
        dict with keys:
        - valid (bool): True if independent
        - error (str): Error message if dependent (optional)
        - imports (list): List of problematic imports (optional)
    """
    try:
        with open(file_path) as f:
            tree = ast.parse(f.read())

        problematic_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('behaviors.'):
                        # Check if importing another behavior (not base)
                        module = alias.name.split('.')[-1]
                        if module != 'base' and module in KNOWN_BEHAVIORS:
                            problematic_imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('behaviors.'):
                    # Check if importing from another behavior
                    module = node.module.split('.')[-1]
                    if module != 'base' and module in KNOWN_BEHAVIORS:
                        for alias in node.names:
                            problematic_imports.append(f"{node.module}.{alias.name}")

        if problematic_imports:
            return {
                "valid": False,
                "error": f"Cross-behavior imports detected",
                "imports": problematic_imports
            }

        return {"valid": True}

    except FileNotFoundError:
        return {
            "valid": False,
            "error": f"File not found: {file_path}"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Unexpected error: {str(e)}"
        }


def validate_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """
    Validate tool follows OpenAI function calling spec.

    Args:
        tool: Tool schema dict

    Returns:
        dict with keys:
        - valid (bool): True if schema is valid
        - error (str): Error message if invalid (optional)
    """
    # Required top-level keys
    required_keys = ["type", "function"]
    if not all(k in tool for k in required_keys):
        missing = [k for k in required_keys if k not in tool]
        return {
            "valid": False,
            "error": f"Missing required keys: {', '.join(missing)}"
        }

    # Type must be "function"
    if tool.get("type") != "function":
        return {
            "valid": False,
            "error": f"Type must be 'function', got: {tool.get('type')}"
        }

    # Required function keys
    function_keys = ["name", "description", "parameters"]
    function = tool.get("function", {})
    if not all(k in function for k in function_keys):
        missing = [k for k in function_keys if k not in function]
        return {
            "valid": False,
            "error": f"Missing required function keys: {', '.join(missing)}"
        }

    # Validate parameters structure
    params = function.get("parameters", {})
    if not isinstance(params, dict):
        return {
            "valid": False,
            "error": "Parameters must be a dict"
        }

    if params.get("type") != "object":
        return {
            "valid": False,
            "error": "Parameters type must be 'object'"
        }

    # Check for properties
    if "properties" not in params:
        return {
            "valid": False,
            "error": "Parameters must have 'properties' key"
        }

    if not isinstance(params["properties"], dict):
        return {
            "valid": False,
            "error": "Properties must be a dict"
        }

    return {"valid": True}


def validate_behavior_class(code: str, expected_name: str) -> dict[str, Any]:
    """
    Validate behavior class structure.

    Checks:
    - Class inherits from AgentBehavior
    - get_name() method is implemented
    - Class name matches expected pattern

    Args:
        code: Python source code
        expected_name: Expected class name (e.g., "MyBehavior")

    Returns:
        dict with keys:
        - valid (bool): True if valid
        - error (str): Error message if invalid (optional)
    """
    try:
        tree = ast.parse(code)

        # Find class definition
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        if not classes:
            return {
                "valid": False,
                "error": "No class definition found"
            }

        # Find the behavior class (should inherit from AgentBehavior)
        behavior_class = None
        for cls in classes:
            # Check if inherits from AgentBehavior
            for base in cls.bases:
                if isinstance(base, ast.Name) and base.id == "AgentBehavior":
                    behavior_class = cls
                    break

        if not behavior_class:
            return {
                "valid": False,
                "error": "No class inheriting from AgentBehavior found"
            }

        # Check class name
        if expected_name and behavior_class.name != expected_name:
            return {
                "valid": False,
                "error": f"Expected class name '{expected_name}', got '{behavior_class.name}'"
            }

        # Check for get_name method
        methods = [node.name for node in behavior_class.body if isinstance(node, ast.FunctionDef)]
        if "get_name" not in methods:
            return {
                "valid": False,
                "error": "Class must implement get_name() method"
            }

        return {"valid": True}

    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"Syntax error: {e.msg}"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Unexpected error: {str(e)}"
        }


def validate_no_super_in_dispatch(code: str) -> dict[str, Any]:
    """
    Validate that dispatch_tool does not call super().dispatch_tool().

    Calling super().dispatch_tool() causes double-dispatch bugs because
    AgentBehavior.dispatch_tool() is a final implementation that doesn't
    chain to other behaviors.

    Args:
        code: Python source code

    Returns:
        dict with keys:
        - valid (bool): True if no super() calls in dispatch_tool
        - error (str): Error message if invalid (optional)
        - line (int): Line number of problematic super() call (optional)
    """
    try:
        tree = ast.parse(code)

        # Find dispatch_tool method in AgentBehavior subclasses
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if this is a behavior class
                is_behavior = any(
                    isinstance(base, ast.Name) and base.id == "AgentBehavior"
                    for base in node.bases
                )

                if not is_behavior:
                    continue

                # Find dispatch_tool method
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "dispatch_tool":
                        # Check for super().dispatch_tool() calls
                        for call_node in ast.walk(method):
                            if isinstance(call_node, ast.Call):
                                # Check for super().dispatch_tool()
                                if isinstance(call_node.func, ast.Attribute):
                                    # call_node.func is Attribute(value=Call(func=Name(id='super')))
                                    if (call_node.func.attr == "dispatch_tool" and
                                        isinstance(call_node.func.value, ast.Call) and
                                        isinstance(call_node.func.value.func, ast.Name) and
                                        call_node.func.value.func.id == "super"):
                                        return {
                                            "valid": False,
                                            "error": (
                                                "dispatch_tool must not call super().dispatch_tool() - "
                                                "this causes double-dispatch. Return {\"error\": ...} "
                                                "for unknown tools instead."
                                            ),
                                            "line": call_node.lineno
                                        }

        return {"valid": True}

    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"Syntax error: {e.msg}"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Unexpected error: {str(e)}"
        }
