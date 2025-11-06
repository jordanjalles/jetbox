"""
ValidationBehavior - Provides code validation tools for meta-programming.

Tools:
- validate_python_syntax: Check Python code syntax
- validate_behavior_independence: Check behavior has no cross-dependencies
- validate_tool_schema: Validate OpenAI function call schema
- validate_agent_dag: Check agent delegation graph for cycles

This behavior enables agents to validate generated code before installation.
"""

from typing import Any
from pathlib import Path
from behaviors.base import AgentBehavior

# Import validation utilities
from utils.behavior_validator import (
    validate_python_syntax,
    validate_behavior_independence,
    validate_tool_schema,
    validate_behavior_class
)
from utils.agent_validator import (
    validate_yaml_syntax,
    validate_agent_dag,
    validate_agent_config
)


class ValidationBehavior(AgentBehavior):
    """
    Provides validation tools for generated behaviors and agent configurations.

    This behavior wraps the validation utilities from utils/ and exposes them
    as tools that agents can use. All validations return structured results
    with "valid" boolean and optional "error" message.
    """

    def __init__(self, **kwargs):
        """Initialize validation behavior."""
        pass

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "validation"

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject tool documentation for validation tools.

        Called once during agent initialization to document available tools.

        Args:
            agent: Agent instance
            context: Initial context (system prompt only)

        Returns:
            Context with tool documentation injected
        """
        tools = self.get_tools()
        if not tools:
            return context

        # Build tool documentation
        tool_docs = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])

            # Build parameter signature
            param_strs = []
            for param_name, param_spec in params.items():
                param_type = param_spec.get("type", "any")
                is_required = param_name in required
                if is_required:
                    param_strs.append(f"{param_name}: {param_type}")
                else:
                    param_strs.append(f"{param_name}?: {param_type}")

            param_sig = ", ".join(param_strs) if param_strs else ""
            tool_docs.append(f"  - {name}({param_sig}): {desc}")

        if tool_docs:
            tool_message = f"\n{self.get_name()} tools:\n" + "\n".join(tool_docs)
            return self.inject_user_message_after_system(context, tool_message)

        return context

    def get_tools(self) -> list[dict[str, Any]]:
        """Return validation tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "validate_python_syntax",
                    "description": "Validate Python code syntax using AST parser",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to validate"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_behavior_independence",
                    "description": "Check that a behavior file has no cross-behavior dependencies",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to behavior Python file (relative to workspace)"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_tool_schema",
                    "description": "Validate that a tool follows OpenAI function calling specification",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool": {
                                "type": "object",
                                "description": "Tool schema dictionary to validate"
                            }
                        },
                        "required": ["tool"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_behavior_class",
                    "description": "Validate that code contains a proper behavior class",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code containing behavior class"
                            },
                            "expected_name": {
                                "type": "string",
                                "description": "Expected behavior class name (e.g., 'MyBehavior')"
                            }
                        },
                        "required": ["code", "expected_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_yaml_syntax",
                    "description": "Validate YAML syntax",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "yaml_str": {
                                "type": "string",
                                "description": "YAML string to validate"
                            }
                        },
                        "required": ["yaml_str"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_agent_dag",
                    "description": "Validate agent delegation graph has no cycles (DAG check)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agents_config": {
                                "type": "object",
                                "description": "Agents configuration dictionary with 'agents' key"
                            }
                        },
                        "required": ["agents_config"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_agent_config",
                    "description": "Validate complete agent configuration structure",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "config": {
                                "type": "object",
                                "description": "Agent config dictionary to validate (use this OR config_yaml)"
                            },
                            "config_yaml": {
                                "type": "string",
                                "description": "Agent config YAML string to validate (use this OR config)"
                            },
                            "behaviors_dir": {
                                "type": "string",
                                "description": "Optional path to behaviors directory for checking behavior existence"
                            }
                        },
                        "required": []
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        agent: Any,
        tool_name: str,
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle validation tool execution.

        Args:
            agent: Agent instance
            tool_name: Tool being called
            args: Tool arguments

        Returns:
            Validation result dict with "valid" and optional "error"/"issues"
        """
        if tool_name == "validate_python_syntax":
            return self._validate_python_syntax(agent, args)
        elif tool_name == "validate_behavior_independence":
            return self._validate_behavior_independence(agent, args)
        elif tool_name == "validate_tool_schema":
            return self._validate_tool_schema(agent, args)
        elif tool_name == "validate_behavior_class":
            return self._validate_behavior_class(agent, args)
        elif tool_name == "validate_yaml_syntax":
            return self._validate_yaml_syntax(agent, args)
        elif tool_name == "validate_agent_dag":
            return self._validate_agent_dag(agent, args)
        elif tool_name == "validate_agent_config":
            return self._validate_agent_config(agent, args)
        else:
            return super().dispatch_tool(agent, tool_name, args)

    def _validate_python_syntax(self, agent: Any, args: dict[str, Any]) -> dict[str, Any]:
        """Validate Python code syntax."""
        try:
            code = args.get("code", "")
            result = validate_python_syntax(code)
            return {"result": result}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    def _validate_behavior_independence(self, agent: Any, args: dict[str, Any]) -> dict[str, Any]:
        """Check behavior file for cross-dependencies."""
        try:
            file_path = args.get("file_path", "")

            # Resolve path relative to workspace
            workspace = agent.workspace if hasattr(agent, 'workspace') else Path.cwd()
            resolved_path = workspace / file_path

            if not resolved_path.exists():
                return {"error": f"File not found: {file_path}"}

            result = validate_behavior_independence(str(resolved_path))
            return {"result": result}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    def _validate_tool_schema(self, agent: Any, args: dict[str, Any]) -> dict[str, Any]:
        """Validate OpenAI function call schema."""
        try:
            tool = args.get("tool", {})
            result = validate_tool_schema(tool)
            return {"result": result}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    def _validate_behavior_class(self, agent: Any, args: dict[str, Any]) -> dict[str, Any]:
        """Validate behavior class structure."""
        try:
            code = args.get("code", "")
            expected_name = args.get("expected_name", "")
            result = validate_behavior_class(code, expected_name)
            return {"result": result}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    def _validate_yaml_syntax(self, agent: Any, args: dict[str, Any]) -> dict[str, Any]:
        """Validate YAML syntax."""
        try:
            yaml_str = args.get("yaml_str", "")
            result = validate_yaml_syntax(yaml_str)
            return {"result": result}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    def _validate_agent_dag(self, agent: Any, args: dict[str, Any]) -> dict[str, Any]:
        """Validate agent delegation graph for cycles."""
        try:
            agents_config = args.get("agents_config", {})
            result = validate_agent_dag(agents_config)
            return {"result": result}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    def _validate_agent_config(self, agent: Any, args: dict[str, Any]) -> dict[str, Any]:
        """Validate complete agent configuration."""
        try:
            import yaml

            # Support both dict and YAML string input
            config = args.get("config")
            config_yaml = args.get("config_yaml")

            if config_yaml:
                # Parse YAML string to dict
                config = yaml.safe_load(config_yaml)
            elif not config:
                return {"error": "Either 'config' or 'config_yaml' parameter required"}

            behaviors_dir = args.get("behaviors_dir")

            # Resolve behaviors_dir relative to workspace
            if behaviors_dir:
                workspace = agent.workspace if hasattr(agent, 'workspace') else Path.cwd()
                behaviors_dir = workspace / behaviors_dir

            result = validate_agent_config(config, behaviors_dir)
            return {"result": result}
        except yaml.YAMLError as e:
            return {"error": f"YAML parse error: {str(e)}"}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}
