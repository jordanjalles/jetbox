"""
ValidationBehavior - Provides code validation tools for meta-programming.

Tools:
- validate_python_syntax: Check Python code syntax
- validate_behavior_independence: Check behavior has no cross-dependencies
- validate_tool_schema: Validate OpenAI function call schema
- validate_agent_dag: Check agent delegation graph for cycles

This behavior enables agents to validate generated code before installation.

Now uses @tool decorator for automatic tool registration!"""

from typing import Any
from pathlib import Path
from behaviors.base import AgentBehavior
from behaviors.tool_decorator import tool
from behaviors.rule_of_two_types import RuleOfTwoProperty

# Import validation utilities
from utils.behavior_validator import (
    validate_python_syntax,
    validate_behavior_independence,
    validate_tool_schema,
    validate_behavior_class,
    validate_no_super_in_dispatch
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

    Security: [] None (validates agent-generated code, not user data)
    """

    # Rule of Two: Empty (utility behavior for meta-programming validation)
    rule_of_two_properties = set()

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
            # Use default role="system" for framework tool documentation
            return self.inject_message_after_system(context, tool_message)

        return context

    @tool
    def validate_python_syntax(self, code: str) -> dict[str, Any]:
        """Validate Python code syntax using AST parser.

        Args:
            code: Python code to validate

        Returns:
            Dict with "result" containing validation result
        """
        try:
            result = validate_python_syntax(code)
            return {"result": result}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    @tool
    def validate_behavior_independence(self, file_path: str) -> dict[str, Any]:
        """Check that a behavior file has no cross-behavior dependencies.

        Args:
            file_path: Path to behavior Python file (relative to workspace)

        Returns:
            Dict with "result" containing validation result
        """
        try:
            # Resolve path relative to workspace
            workspace = self.agent.workspace if hasattr(self.agent, 'workspace') else Path.cwd()
            resolved_path = workspace / file_path

            if not resolved_path.exists():
                return {"error": f"File not found: {file_path}"}

            result = validate_behavior_independence(str(resolved_path))
            return {"result": result}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    @tool
    def validate_tool_schema(self, tool: dict) -> dict[str, Any]:
        """Validate that a tool follows OpenAI function calling specification.

        Args:
            tool: Tool schema dictionary to validate

        Returns:
            Dict with "result" containing validation result
        """
        try:
            result = validate_tool_schema(tool)
            return {"result": result}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    @tool
    def validate_behavior_class(self, code: str, expected_name: str) -> dict[str, Any]:
        """Validate that code contains a proper behavior class.

        Args:
            code: Python code containing behavior class
            expected_name: Expected behavior class name (e.g., 'MyBehavior')

        Returns:
            Dict with "result" containing validation result
        """
        try:
            result = validate_behavior_class(code, expected_name)
            return {"result": result}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    @tool
    def validate_no_super_in_dispatch(self, code: str) -> dict[str, Any]:
        """Validate that dispatch_tool does not call super().dispatch_tool() (causes double-dispatch bug).

        Args:
            code: Python code to validate

        Returns:
            Dict with "result" containing validation result
        """
        try:
            result = validate_no_super_in_dispatch(code)
            return {"result": result}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    @tool
    def validate_yaml_syntax(self, yaml_str: str) -> dict[str, Any]:
        """Validate YAML syntax.

        Args:
            yaml_str: YAML string to validate

        Returns:
            Dict with "result" containing validation result
        """
        try:
            result = validate_yaml_syntax(yaml_str)
            return {"result": result}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    @tool
    def validate_agent_dag(self, agents_config: dict) -> dict[str, Any]:
        """Validate agent delegation graph has no cycles (DAG check).

        Args:
            agents_config: Agents configuration dictionary with 'agents' key

        Returns:
            Dict with "result" containing validation result
        """
        try:
            result = validate_agent_dag(agents_config)
            return {"result": result}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    @tool
    def validate_agent_config(
        self,
        config: dict = None,
        config_yaml: str = None,
        config_file: str = None,
        behaviors_dir: str = None
    ) -> dict[str, Any]:
        """Validate complete agent configuration structure.

        Args:
            config: Agent config dictionary to validate (use this OR config_yaml OR config_file)
            config_yaml: Agent config YAML string to validate (use this OR config OR config_file)
            config_file: Path to agent config YAML file (use this OR config OR config_yaml)
            behaviors_dir: Optional path to behaviors directory for checking behavior existence

        Returns:
            Dict with validation result
        """
        try:
            import yaml

            # Support dict, YAML string, or file path input
            if config_file:
                # Load from file
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif config_yaml:
                # Parse YAML string to dict
                config = yaml.safe_load(config_yaml)
            elif not config:
                return {"error": "Either 'config', 'config_yaml', or 'config_file' parameter required"}

            # Resolve behaviors_dir relative to workspace
            if behaviors_dir:
                workspace = self.agent.workspace if hasattr(self.agent, 'workspace') else Path.cwd()
                behaviors_dir = workspace / behaviors_dir

            result = validate_agent_config(config, behaviors_dir)
            # Return result directly (not wrapped) for consistency
            return result
        except yaml.YAMLError as e:
            return {"error": f"YAML parse error: {str(e)}"}
        except FileNotFoundError as e:
            return {"error": f"Config file not found: {str(e)}"}
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}
