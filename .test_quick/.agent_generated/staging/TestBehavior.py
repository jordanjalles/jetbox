"""
Test behavior implementation for the TestBehavior identifier.
"""

from typing import Any
from behaviors.base import AgentBehavior


class TestbehaviorBehavior(AgentBehavior):
    """
    Test behavior providing the test_tool.
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

        Returns:
            Behavior name string.
        """
        return "TestBehavior"

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return tool definitions in OpenAI function format.

        Returns:
            List of tool definition dicts.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "Test",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param1": {
                                "type": "string",
                                "description": "Test param"
                            }
                        },
                        "required": ["param1"]
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
        if tool_name == "test_tool":
            try:
                param1 = args.get("param1")
                if param1 is None:
                    raise ValueError("param1 is required")
                result = self._do_work(param1)
                return {"result": result, "success": True}
            except Exception as e:
                return {"error": str(e)}

        return super().dispatch_tool(agent, tool_name, args)

    def _do_work(self, param1: str) -> str:
        """
        Core functionality for test_tool.

        Args:
            param1: The test parameter.

        Returns:
            A string indicating the tool was executed.
        """
        return f"Tool executed with param1: {param1}"
