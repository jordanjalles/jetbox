"""
Validation utilities for behavior code generation.

This module provides functions to validate generated behavior code:
- validate_python_syntax: Check Python syntax
- validate_behavior_independence: Check no cross-behavior imports
- validate_tool_schema: Validate OpenAI function schema
- validate_behavior_class_structure: Check class structure

These validators are used by CreateBehaviorBehavior to ensure
generated behaviors follow Jetbox patterns and principles.
"""

import ast
from pathlib import Path
from typing import Any


# Known behavior module names (updated from behaviors directory)
KNOWN_BEHAVIORS = [
    "architect_tools",
    "chatbot",
    "command_tools",
    "compact_when_near_full",
    "delegation",
    "directory_tools",
    "loop_detection",
    "read_file_tools",
    "server_management",
    "server_tools",
    "status_display",
    "task_management",
    "workspace_management",
    "workspace_task_notes",
    "write_file_tools"
]


def validate_python_syntax(code: str) -> dict[str, Any]:
    """
    Validate Python syntax.

    Args:
        code: Python source code to validate

    Returns:
        Dictionary with:
        - valid: True if syntax is valid, False otherwise
        - error: Error message if invalid (optional)
        - ast: Parsed AST if valid (optional)

    Example:
        result = validate_python_syntax("print('hello')")
        assert result["valid"] is True
        result = validate_python_syntax("print('hello'")
        assert result["valid"] is False
        assert "error" in result
    """
    try:
        tree = ast.parse(code)
        return {"valid": True, "ast": tree}
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"Syntax error at line {e.lineno}: {e.msg}"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Parse error: {str(e)}"
        }


def validate_behavior_independence(code: str, file_path: str | None = None) -> dict[str, Any]:
    """
    Check for cross-behavior imports (violations of independence principle).

    Behaviors must be independent and not import other behaviors
    (except base.AgentBehavior).

    Args:
        code: Python source code to validate (optional if file_path provided)
        file_path: Path to source file (optional if code provided)

    Returns:
        Dictionary with:
        - valid: True if no cross-imports, False otherwise
        - error: Error message if invalid (optional)
        - violations: List of violating imports (optional)

    Example:
        code = "from behaviors.base import AgentBehavior"
        result = validate_behavior_independence(code)
        assert result["valid"] is True

        code = "from behaviors.file_tools import FileToolsBehavior"
        result = validate_behavior_independence(code)
        assert result["valid"] is False
    """
    # Read code from file if provided
    if file_path:
        try:
            code = Path(file_path).read_text()
        except Exception as e:
            return {
                "valid": False,
                "error": f"Failed to read file: {str(e)}"
            }

    # Parse code
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"Cannot parse code: {str(e)}"
        }

    violations = []

    # Check all imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # Check: import behaviors.X
            for alias in node.names:
                if alias.name.startswith('behaviors.'):
                    module = alias.name.split('.')[-1]
                    # Allow base, disallow other behaviors
                    if module != 'base' and module in KNOWN_BEHAVIORS:
                        violations.append(f"import {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            # Check: from behaviors.X import Y
            if node.module and node.module.startswith("behaviors."):
                module_name = node.module.split(".")[-1]
                # Allow base and base_agent, disallow other behaviors
                if module_name not in ['base', 'base_agent'] and module_name in KNOWN_BEHAVIORS:
                    imported_names = [alias.name for alias in node.names]
                    violations.append(f"from {node.module} import {', '.join(imported_names)}")

    if violations:
        return {
            "valid": False,
            "error": "Cross-behavior imports detected (violates independence principle)",
            "violations": violations
        }

    return {"valid": True}


def validate_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """
    Validate tool follows OpenAI function calling spec.

    Checks:
    - Has "type" and "function" keys
    - Function has "name", "description", "parameters"
    - Parameters type is "object"
    - Parameters has "properties" dict

    Args:
        tool: Tool definition dict

    Returns:
        Dictionary with:
        - valid: True if schema is valid, False otherwise
        - error: Error message if invalid (optional)

    Example:
        tool = {
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
        }
        result = validate_tool_schema(tool)
        assert result["valid"] is True
    """
    # Check top-level structure
    if not isinstance(tool, dict):
        return {"valid": False, "error": "Tool must be a dictionary"}

    required_keys = ["type", "function"]
    missing_keys = [k for k in required_keys if k not in tool]
    if missing_keys:
        return {
            "valid": False,
            "error": f"Missing required keys: {', '.join(missing_keys)}"
        }

    # Check type
    if tool.get("type") != "function":
        return {
            "valid": False,
            "error": f"Tool type must be 'function', got: {tool.get('type')}"
        }

    # Check function structure
    function = tool.get("function")
    if not isinstance(function, dict):
        return {"valid": False, "error": "Tool 'function' must be a dictionary"}

    function_keys = ["name", "description", "parameters"]
    missing_func_keys = [k for k in function_keys if k not in function]
    if missing_func_keys:
        return {
            "valid": False,
            "error": f"Function missing required keys: {', '.join(missing_func_keys)}"
        }

    # Check parameters structure
    params = function.get("parameters")
    if not isinstance(params, dict):
        return {"valid": False, "error": "Parameters must be a dictionary"}

    if params.get("type") != "object":
        return {
            "valid": False,
            "error": f"Parameters type must be 'object', got: {params.get('type')}"
        }

    if "properties" not in params:
        return {
            "valid": False,
            "error": "Parameters must have 'properties' dict"
        }

    if not isinstance(params["properties"], dict):
        return {
            "valid": False,
            "error": "Parameters 'properties' must be a dictionary"
        }

    return {"valid": True}


def validate_behavior_class_structure(code: str) -> dict[str, Any]:
    """
    Validate behavior class structure.

    Checks:
    - Has a class that inherits from AgentBehavior
    - Class implements get_name() method
    - Class has docstring
    - No obvious violations (eval, exec, etc.)

    Args:
        code: Python source code to validate

    Returns:
        Dictionary with:
        - valid: True if structure is valid, False otherwise
        - error: Error message if invalid (optional)
        - class_name: Name of behavior class if found (optional)

    Example:
        code = "from behaviors.base import AgentBehavior\\n\\n" + \\
               "class MyBehavior(AgentBehavior):\\n" + \\
               "    def get_name(self):\\n" + \\
               "        return 'my_behavior'\\n"
        result = validate_behavior_class_structure(code)
        # Returns {"valid": True, "class_name": "MyBehavior"}
    """
    # Parse code
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"Cannot parse code: {str(e)}"
        }

    # Find class definitions
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    if not classes:
        return {
            "valid": False,
            "error": "No class definition found"
        }

    # Find classes that inherit from AgentBehavior
    behavior_classes = []
    for cls in classes:
        for base in cls.bases:
            # Check if inherits from AgentBehavior
            if isinstance(base, ast.Name) and base.id == "AgentBehavior":
                behavior_classes.append(cls)
                break
            # Check if inherits from behaviors.base.AgentBehavior
            elif isinstance(base, ast.Attribute):
                if base.attr == "AgentBehavior":
                    behavior_classes.append(cls)
                    break

    if not behavior_classes:
        return {
            "valid": False,
            "error": "No class inheriting from AgentBehavior found"
        }

    # Validate first behavior class
    cls = behavior_classes[0]

    # Check docstring
    if not ast.get_docstring(cls):
        return {
            "valid": False,
            "error": f"Class {cls.name} missing docstring"
        }

    # Check get_name method exists
    has_get_name = False
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == "get_name":
            has_get_name = True
            break

    if not has_get_name:
        return {
            "valid": False,
            "error": f"Class {cls.name} missing get_name() method"
        }

    # Check for dangerous operations (eval, exec, compile, __import__)
    dangerous_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                    dangerous_calls.append(node.func.id)

    if dangerous_calls:
        return {
            "valid": False,
            "error": f"Dangerous operations detected: {', '.join(set(dangerous_calls))}"
        }

    return {
        "valid": True,
        "class_name": cls.name
    }
