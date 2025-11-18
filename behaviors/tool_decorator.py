"""
Tool decorator for automatic tool registration in behaviors.

This module provides a @tool decorator that eliminates manual tool registration
boilerplate. Instead of writing JSON schemas, dispatch routing, and documentation
separately, you just decorate your method:

    @tool(description="Write a file to disk")
    def write_file(self, path: str, content: str, append: bool = False) -> str:
        '''
        Write content to a file.

        Args:
            path: File path relative to workspace
            content: Content to write
            append: If true, append instead of overwriting

        Returns:
            Success message with file path
        '''
        # implementation

The decorator automatically:
- Generates OpenAI-compatible JSON schema from type hints
- Parses docstring for parameter descriptions
- Registers tool for auto-discovery by base behavior
- Handles tool dispatch automatically
- Injects 'agent' parameter (user methods don't need it in signature)

Usage:
    from behaviors.tool_decorator import tool

    class MyBehavior(AgentBehavior):
        @tool(description="Short tool description")
        def my_tool(self, param1: str, param2: int = 10) -> dict:
            '''Full docstring with Args: section'''
            # Access agent via self.agent (injected automatically)
            return {"result": "success"}
"""

import inspect
import re
from functools import wraps
from typing import Any, Callable, get_type_hints, get_origin, get_args


def tool(func_or_description=None):
    """
    Decorator for automatic tool registration.

    Converts a method into a registered tool by:
    1. Extracting parameter info from type hints
    2. Parsing docstring for descriptions (including tool description)
    3. Generating JSON schema
    4. Marking method as a tool for auto-discovery

    Can be used with or without arguments:
        @tool  # Uses docstring first line as description
        @tool()  # Same as above
        @tool(description="Custom description")  # Explicit override

    Args:
        func_or_description: Either the function being decorated (when used as @tool)
                            or a description string (when used as @tool(description="..."))

    Returns:
        Decorated method with tool metadata attached (or decorator function)

    Example:
        @tool  # Uses docstring first line
        def read_file(self, path: str, encoding: str = "utf-8") -> str:
            '''Read file contents.

            Args:
                path: File path to read
                encoding: Text encoding (default: utf-8)

            Returns:
                File contents as string
            '''
            # implementation

        @tool(description="Custom description")  # Explicit override
        def write_file(self, path: str, content: str) -> str:
            # implementation
    """
    def decorator(func: Callable, description: str | None = None) -> Callable:
        # Extract function signature info
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Determine tool description
        if description is None:
            # Use first line of docstring as description
            docstring = (func.__doc__ or "").strip()
            if docstring:
                # Get first non-empty line (summary line)
                tool_description = docstring.split('\n')[0].strip()
            else:
                # Fallback to function name if no docstring
                tool_description = func.__name__.replace('_', ' ').title()
        else:
            tool_description = description

        # Parse docstring for parameter descriptions
        param_docs = _parse_docstring_args(func.__doc__ or "")

        # Build JSON schema from type hints
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip 'self' and 'agent' parameters
            if param_name in ('self', 'agent'):
                continue

            # Get type hint
            param_type = type_hints.get(param_name, Any)

            # Convert type hint to JSON schema
            schema = _type_to_schema(param_type)

            # Add description from docstring
            if param_name in param_docs:
                schema['description'] = param_docs[param_name]

            properties[param_name] = schema

            # Required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        # Build complete tool definition
        tool_def = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

        # Wrap function to inject agent parameter
        @wraps(func)
        def wrapper(self, agent, **kwargs):
            # Store agent reference temporarily (for tool methods that need it)
            old_agent = getattr(self, 'agent', None)
            self.agent = agent
            try:
                # Call original function WITHOUT agent parameter
                # (user's decorated method doesn't include agent in signature)
                return func(self, **kwargs)
            finally:
                # Restore old agent reference
                if old_agent is None:
                    delattr(self, 'agent')
                else:
                    self.agent = old_agent

        # Attach metadata to wrapper
        wrapper._tool_definition = tool_def
        wrapper._is_tool = True
        wrapper._original_func = func

        return wrapper

    # Handle both @tool and @tool() / @tool(description="...") syntax
    if callable(func_or_description):
        # Called as @tool (no parentheses) - func_or_description is the function
        return decorator(func_or_description, description=None)
    else:
        # Called as @tool() or @tool(description="...")
        # func_or_description is None or a description string
        # Return a decorator that will be applied to the function
        return lambda func: decorator(func, description=func_or_description)


def _parse_docstring_args(docstring: str) -> dict[str, str]:
    """
    Parse Args: section from docstring.

    Supports both Google-style and NumPy-style docstrings:

    Google-style:
        Args:
            param1: Description here
            param2: Another description

    NumPy-style:
        Parameters
        ----------
        param1 : str
            Description here
        param2 : int
            Another description

    Args:
        docstring: Function docstring

    Returns:
        Dict mapping parameter name to description
    """
    if not docstring:
        return {}

    param_docs = {}

    # Try Google-style first (Args:)
    args_match = re.search(r'Args:\s*\n((?:\s+\w+.*\n)+)', docstring, re.MULTILINE)
    if args_match:
        args_section = args_match.group(1)
        # Match lines like "    param_name: Description here"
        for line_match in re.finditer(r'^\s+(\w+):\s*(.+)$', args_section, re.MULTILINE):
            param_name = line_match.group(1)
            param_desc = line_match.group(2).strip()
            param_docs[param_name] = param_desc
        return param_docs

    # Try NumPy-style (Parameters)
    params_match = re.search(r'Parameters\s*\n\s*-+\s*\n((?:.*\n)+?)(?:\n\w+\s*\n\s*-+|\Z)', docstring, re.MULTILINE)
    if params_match:
        params_section = params_match.group(1)
        # Match patterns like "param_name : type\n    Description"
        current_param = None
        for line in params_section.split('\n'):
            if line.strip():
                # Check if this is a parameter declaration line
                param_decl = re.match(r'^(\w+)\s*:', line)
                if param_decl:
                    current_param = param_decl.group(1)
                    param_docs[current_param] = ""
                elif current_param and line.startswith(' '):
                    # This is a description line
                    param_docs[current_param] += line.strip() + " "

        # Clean up descriptions
        param_docs = {k: v.strip() for k, v in param_docs.items() if v.strip()}
        return param_docs

    return {}


def _type_to_schema(python_type: Any) -> dict:
    """
    Convert Python type hint to JSON schema.

    Supports:
    - Primitives: str, int, float, bool
    - Optional: Optional[T], T | None
    - Collections: list[T], dict[K, V]
    - Literals: Literal['a', 'b', 'c']
    - Any: dict, list (without type args)

    Args:
        python_type: Python type hint

    Returns:
        JSON schema dict
    """
    # Handle string type annotations (forward references)
    if isinstance(python_type, str):
        return {"type": "string"}  # Default to string for unresolved types

    # Get origin type (list, dict, Union, etc.)
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle Optional[T] or T | None (Union types)
    if origin is type(None) | type:  # Python 3.10+ union syntax
        # Filter out None types
        non_none_types = [t for t in args if t is not type(None)]
        if len(non_none_types) == 1:
            return _type_to_schema(non_none_types[0])
        elif len(non_none_types) > 1:
            # Multiple non-None types - use first one (best effort)
            return _type_to_schema(non_none_types[0])
        else:
            return {"type": "null"}

    # Handle typing.Union (Python 3.9 Optional[T])
    try:
        from typing import Union
        if origin is Union:
            # Filter out None types
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                return _type_to_schema(non_none_types[0])
            elif len(non_none_types) > 1:
                # Multiple non-None types - use first one
                return _type_to_schema(non_none_types[0])
            else:
                return {"type": "null"}
    except ImportError:
        pass

    # Handle Literal types
    try:
        from typing import Literal
        # Check if it's a Literal (Python 3.8+)
        if hasattr(python_type, '__origin__') and python_type.__origin__ is Literal:
            # Extract literal values
            literal_values = list(get_args(python_type))
            # Infer type from first value
            if literal_values:
                first_val = literal_values[0]
                if isinstance(first_val, str):
                    return {"type": "string", "enum": literal_values}
                elif isinstance(first_val, int):
                    return {"type": "integer", "enum": literal_values}
                elif isinstance(first_val, bool):
                    return {"type": "boolean", "enum": literal_values}
            return {"type": "string", "enum": literal_values}
    except (ImportError, AttributeError):
        pass

    # Handle list[T]
    if origin is list:
        if args:
            item_schema = _type_to_schema(args[0])
            return {"type": "array", "items": item_schema}
        else:
            return {"type": "array"}

    # Handle dict[K, V]
    if origin is dict:
        # JSON schema doesn't support typed dicts well, just use object
        return {"type": "object"}

    # Handle basic types
    if python_type is str or python_type == str:
        return {"type": "string"}
    elif python_type is int or python_type == int:
        return {"type": "integer"}
    elif python_type is float or python_type == float:
        return {"type": "number"}
    elif python_type is bool or python_type == bool:
        return {"type": "boolean"}
    elif python_type is dict:
        return {"type": "object"}
    elif python_type is list:
        return {"type": "array"}
    elif python_type is Any:
        # Any type - allow any value
        return {}

    # Default to string for unknown types
    return {"type": "string"}
