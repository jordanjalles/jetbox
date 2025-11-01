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
from behaviors.file_tools import FileToolsBehavior
from behaviors.command_tools import CommandToolsBehavior
from behaviors.server_tools import ServerToolsBehavior
from behaviors.architect_tools import ArchitectToolsBehavior
from behaviors.compact_when_near_full import CompactWhenNearFullBehavior
from behaviors.hierarchical_context import HierarchicalContextBehavior
from behaviors.subagent_context import SubAgentContextBehavior
from behaviors.architect_context import ArchitectContextBehavior
from behaviors.loop_detection import LoopDetectionBehavior

__all__ = [
    "AgentBehavior",
    "FileToolsBehavior",
    "CommandToolsBehavior",
    "ServerToolsBehavior",
    "ArchitectToolsBehavior",
    "CompactWhenNearFullBehavior",
    "HierarchicalContextBehavior",
    "SubAgentContextBehavior",
    "ArchitectContextBehavior",
    "LoopDetectionBehavior",
]
