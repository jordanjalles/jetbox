"""
{BEHAVIOR_NAME}Behavior - {ONE_SENTENCE_DESCRIPTION}

Provides tools:
- {TOOL_1_NAME}: {TOOL_1_DESCRIPTION}
- {TOOL_2_NAME}: {TOOL_2_DESCRIPTION}
"""

from typing import Any, TYPE_CHECKING
from behaviors.base import AgentBehavior

if TYPE_CHECKING:
    from behaviors.base_agent import BaseAgent


class {BEHAVIOR_CLASS_NAME}Behavior(AgentBehavior):
    """
    Provides {TOOL_CATEGORY} tools for agent use.
    """

    def __init__(self, workspace_manager=None, **kwargs):
        """
        Initialize behavior.

        Args:
            workspace_manager: Optional WorkspaceManager for path resolution
            **kwargs: Additional parameters (ignored)
        """
        self.workspace_manager = workspace_manager

    def get_name(self) -> str:
        return "{BEHAVIOR_NAME}"

    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions in OpenAI function call format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "{TOOL_NAME}",
                    "description": "{TOOL_DESCRIPTION}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "{PARAM_NAME}": {
                                "type": "{PARAM_TYPE}",
                                "description": "{PARAM_DESCRIPTION}"
                            }
                        },
                        "required": ["{REQUIRED_PARAM}"]
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        agent: "BaseAgent",
        tool_name: str,
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle tool execution.

        Args:
            agent: Agent instance (access agent.workspace, agent.state, etc.)
            tool_name: Tool being called
            args: Tool arguments from LLM

        Returns:
            Tool result dict ({"result": ..., "success": True} or {"error": ...})
        """
        if tool_name == "{TOOL_NAME}":
            return self._execute_{TOOL_NAME}(agent, args)
        else:
            # Fall through to parent for unknown tools
            return super().dispatch_tool(agent, tool_name, args)

    def _execute_{TOOL_NAME}(
        self,
        agent: "BaseAgent",
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute {TOOL_NAME} tool.

        Args:
            agent: Agent instance (for workspace access)
            args: Tool arguments

        Returns:
            Result dict
        """
        try:
            # Get parameters with defaults
            param1 = args.get("{PARAM_NAME}", "{DEFAULT_VALUE}")

            # Warn about unsupported parameters (parameter invention tolerance)
            supported = {"{PARAM_NAME}", "{OTHER_PARAM}"}
            unsupported = set(args.keys()) - supported
            if unsupported:
                print(f"[{self.get_name()}] Ignoring parameters: {unsupported}")

            # Access workspace if needed
            workspace = agent.workspace if hasattr(agent, 'workspace') else None

            # DO THE WORK HERE
            result = self._do_work(param1, workspace)

            return {"result": result, "success": True}

        except Exception as e:
            return {"error": str(e)}

    def _do_work(self, param1, workspace):
        """Core logic separated for testability."""
        # Implement actual functionality
        pass
