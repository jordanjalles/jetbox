"""
Composable agent behaviors system.

Behaviors extend agent capabilities through:
- Context injection (enhance_context)
- Tool registration (get_tools, dispatch_tool)
- Event handling (on_* methods)

Example:
    ```python
    from behaviors import AgentBehavior

    class MyBehavior(AgentBehavior):
        def get_name(self) -> str:
            return "my_behavior"

        def get_tools(self) -> list[dict[str, Any]]:
            return [...]  # Tool definitions

        def dispatch_tool(self, tool_name, args, **kwargs):
            # Handle tool calls
            return {"result": "..."}
    ```
"""

from behaviors.base import AgentBehavior

__all__ = ["AgentBehavior"]
