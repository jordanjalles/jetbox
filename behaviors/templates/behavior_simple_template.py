"""
Simple behavior template showing actual AgentBehavior base class structure.

Use this template to understand the required interface.
"""

from typing import Any
from behaviors.base import AgentBehavior


class YourBehaviorName(AgentBehavior):
    """
    Brief description of what this behavior does.

    This behavior provides tools for [specific purpose].
    """

    def __init__(self, workspace_manager=None, **kwargs):
        """
        Initialize behavior.

        Args:
            workspace_manager: Optional WorkspaceManager for file path resolution
            **kwargs: Additional parameters (ignored for extensibility)
        """
        self.workspace_manager = workspace_manager

    def get_name(self) -> str:
        """
        Return unique behavior identifier.

        This name is used for logging and identification.

        Returns:
            Behavior name string (e.g., "http_request")
        """
        return "your_behavior_name"

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return tool definitions in OpenAI function format.

        Each tool must have:
        - name: Tool identifier
        - description: What the tool does
        - parameters: JSON Schema for tool arguments

        Returns:
            List of tool definition dicts
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "your_tool_name",
                    "description": "What this tool does",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param_name": {
                                "type": "string",
                                "description": "Parameter description"
                            },
                            "optional_param": {
                                "type": "integer",
                                "description": "Optional parameter description"
                            }
                        },
                        "required": ["param_name"]  # List required params here
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
        Execute tool by name.

        Args:
            agent: Agent instance (access workspace, state, etc.)
            tool_name: Name of tool to execute
            args: Tool arguments from LLM

        Returns:
            Result dict with either:
            - {"result": value, "success": True} on success
            - {"error": message} on failure
        """
        if tool_name == "your_tool_name":
            try:
                # Extract parameters
                param = args.get("param_name")
                optional = args.get("optional_param", default_value)

                # Do the work
                result = self._do_work(param, optional)

                return {"result": result, "success": True}

            except Exception as e:
                return {"error": str(e)}

        # Unknown tool - return error
        # IMPORTANT: Do NOT call super().dispatch_tool() as it causes double-dispatch
        return {"error": f"Unknown tool: {tool_name}"}

    def _do_work(self, param, optional):
        """
        Core functionality (separated for testability).

        Implement your actual tool logic here.
        """
        # Your implementation here
        pass

    # OPTIONAL: Override lifecycle hooks if needed

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Called once at agent startup.

        Use this to inject tool documentation into context.
        Standard pattern below handles tool documentation automatically.
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
