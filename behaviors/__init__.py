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
from behaviors.directory_tools import DirectoryToolsBehavior
from behaviors.read_file_tools import ReadFileToolsBehavior
from behaviors.write_file_tools import WriteFileToolsBehavior
from behaviors.command_tools import CommandToolsBehavior
from behaviors.server_tools import ServerToolsBehavior
from behaviors.architect_tools import ArchitectToolsBehavior
from behaviors.compact_when_near_full import CompactWhenNearFullBehavior
from behaviors.subagent_context import SubAgentContextBehavior
from behaviors.loop_detection import LoopDetectionBehavior
from behaviors.workspace_task_notes import WorkspaceTaskNotesBehavior
from behaviors.status_display import StatusDisplayBehavior
from behaviors.delegation import DelegationBehavior

__all__ = [
    "AgentBehavior",
    "DirectoryToolsBehavior",
    "ReadFileToolsBehavior",
    "WriteFileToolsBehavior",
    "CommandToolsBehavior",
    "ServerToolsBehavior",
    "ArchitectToolsBehavior",
    "CompactWhenNearFullBehavior",
    "SubAgentContextBehavior",
    "LoopDetectionBehavior",
    "WorkspaceTaskNotesBehavior",
    "StatusDisplayBehavior",
    "DelegationBehavior",
]
