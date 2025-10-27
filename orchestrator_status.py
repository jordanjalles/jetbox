"""
Status display for Orchestrator agent.

Shows conversation flow and delegation status instead of task hierarchy.
"""
from __future__ import annotations
from typing import Any
from datetime import datetime


class OrchestratorStatusDisplay:
    """
    Status display adapted for Orchestrator agent.

    Shows:
    - Performance metrics (LLM timing, token usage)
    - Conversation summary (message count, compaction status)
    - Delegated tasks status
    - Recent activity
    """

    def __init__(self):
        """Initialize status display."""
        self.last_lines = 0

    def render(
        self,
        orchestrator: Any,
        registry: Any = None,
    ) -> str:
        """
        Render orchestrator status.

        Args:
            orchestrator: OrchestratorAgent instance
            registry: AgentRegistry instance (optional)

        Returns:
            Formatted status string
        """
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("ORCHESTRATOR STATUS")
        lines.append("=" * 60)
        lines.append("")

        # Conversation summary
        summary = orchestrator.get_conversation_summary()
        lines.append("CONVERSATION:")
        lines.append(f"  Messages: {summary['total_messages']}")
        lines.append(f"  Tokens: {summary['estimated_tokens']} / {summary['token_threshold']}")

        # Token usage bar
        token_pct = min(100, int(summary['estimated_tokens'] / summary['token_threshold'] * 100))
        bar = self._render_bar(token_pct, width=30)
        lines.append(f"  Usage: {bar} {token_pct}%")
        lines.append("")

        # Delegated tasks
        lines.append("DELEGATED TASKS:")
        if summary['delegated_tasks'] > 0:
            lines.append(f"  Total delegated: {summary['delegated_tasks']}")

            # Show executor status if available
            if registry:
                try:
                    executor_status = registry.get_agent_status("task_executor")
                    if executor_status.get("status") == "active":
                        lines.append(f"  TaskExecutor: {executor_status.get('subtask', 'working...')}")
                    else:
                        lines.append(f"  TaskExecutor: {executor_status.get('status')}")
                except:
                    lines.append("  TaskExecutor: unknown")
        else:
            lines.append("  (none)")
        lines.append("")

        # Performance
        lines.append("PERFORMANCE:")
        lines.append(f"  Rounds: {summary['rounds']}")
        lines.append("")

        # Recent messages preview
        messages = orchestrator.state.messages[-4:]  # Last 4 messages
        if messages:
            lines.append("RECENT ACTIVITY:")
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role == "user":
                    lines.append(f"  User: {self._truncate(content, 50)}")
                elif role == "assistant":
                    if "tool_calls" in msg:
                        tools = [tc["function"]["name"] for tc in msg["tool_calls"]]
                        lines.append(f"  Assistant: called {', '.join(tools)}")
                    elif content:
                        lines.append(f"  Assistant: {self._truncate(content, 50)}")
            lines.append("")

        return "\n".join(lines)

    def render_in_place(
        self,
        orchestrator: Any,
        registry: Any = None,
    ) -> None:
        """
        Render status in-place (updates previous output).

        Args:
            orchestrator: OrchestratorAgent instance
            registry: AgentRegistry instance (optional)
        """
        # Clear previous output
        if self.last_lines > 0:
            for _ in range(self.last_lines):
                print("\033[F\033[2K", end="")

        # Render new status
        status = self.render(orchestrator, registry)
        print(status, end="", flush=True)

        # Track line count
        self.last_lines = status.count("\n") + 1

    def _render_bar(self, percentage: int, width: int = 20) -> str:
        """
        Render a progress bar.

        Args:
            percentage: 0-100
            width: Bar width in characters

        Returns:
            Bar string like [████████░░░░░░░░░░░░]
        """
        filled = int(width * percentage / 100)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    def _truncate(self, text: str, max_len: int) -> str:
        """
        Truncate text to max length.

        Args:
            text: Text to truncate
            max_len: Maximum length

        Returns:
            Truncated text with "..." if needed
        """
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."
