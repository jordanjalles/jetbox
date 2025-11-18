"""
TUIDisplayBehavior - Integrates the TUI system with the behavior framework.

This behavior wraps the TUI display system (DisplayFactory, PlainDisplay, TextualDisplay)
to provide real-time progress visualization through the AgentBehavior interface.

Features:
- Event: on_goal_start - Initialize display (display.start())
- Event: on_round_start - Update status display (display.update_status())
- Event: on_tool_call - Log tool call events (display.log_event())
- Event: on_round_end - Check for pause (display.wait_if_paused())
- Event: on_goal_complete - Show completion and cleanup (display.show_completion(), display.stop())
- Event: on_timeout - Show timeout and cleanup (display.stop())
- No tools (utility behavior)

The behavior delegates to DisplayInterface implementations (PlainDisplay or TextualDisplay)
for actual rendering. Mode selection is automatic (detects TTY) or can be forced via
display_mode parameter.

Security: [] None (utility behavior - display only)
"""

import time
from typing import Any, Optional
from pathlib import Path
from behaviors.base import AgentBehavior
from tui import DisplayFactory, AgentEvent, EventType


class TUIDisplayBehavior(AgentBehavior):
    """
    Behavior that integrates the TUI display system with agent lifecycle.

    Automatically:
    - Initializes display on goal start (start())
    - Updates status at round start (update_status())
    - Logs tool calls and results (log_event())
    - Shows completion summary on goal complete (show_completion())
    - Cleans up display on completion/timeout (stop())
    - Supports pause/resume for interactive TUI mode (wait_if_paused())

    This is a utility behavior (no tools) that provides real-time
    progress visualization and event logging.

    Security: [] None (utility behavior - display only)
    """

    # Rule of Two: Empty (utility behavior for display only)
    rule_of_two_properties = set()

    def __init__(self, display_mode: str = "auto"):
        """
        Initialize TUI display behavior.

        Args:
            display_mode: Display mode to use:
                - "auto": Auto-detect (TUI if TTY, plain otherwise) [default]
                - "plain": Force plain text output
                - "textual": Force interactive TUI

        Note: CLI flags (--tui / --no-tui) override this parameter via JETBOX_TUI_MODE env var.
        """
        import os
        # Check for CLI override via environment variable
        env_mode = os.environ.get("JETBOX_TUI_MODE")
        self.display_mode = env_mode if env_mode else display_mode
        self.display = None
        self.start_time = None
        self.goal = None

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "tui_display"

    def on_goal_start(self, agent: Any, goal: str, workspace: Any = None) -> None:
        """
        Called when goal starts.

        Initializes the display and starts tracking time.

        Args:
            agent: Agent instance
            goal: The goal string
            workspace: Optional workspace path (unused)
        """
        # Store goal for later use
        self.goal = goal
        self.start_time = time.time()

        # Create display instance
        self.display = DisplayFactory.create(force_mode=self.display_mode)

        # Start display (enters alternate screen for TUI, no-op for plain)
        self.display.start()

    def on_round_start(
        self,
        agent: Any,
        round_number: int,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Called at start of each round.

        Updates the status display with current agent state.

        Args:
            agent: Agent instance
            round_number: Current round number
            context: Current context (passed through unchanged)

        Returns:
            Unmodified context
        """
        if not self.display:
            return context

        # Extract status information from agent
        goal = getattr(agent, 'goal', self.goal or "Unknown goal")
        agent_name = getattr(agent, 'name', agent.__class__.__name__)
        model = getattr(agent, 'model', "unknown")

        # Get max rounds (try multiple possible attribute names)
        max_rounds = (
            getattr(agent, 'max_rounds', None) or
            getattr(agent, 'MAX_ROUNDS', None) or
            50  # fallback default
        )

        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time if self.start_time else 0.0

        # Determine status
        status = "Running"

        # Get token usage if available
        tokens_used = None
        tokens_max = None
        if hasattr(agent, 'get_context_stats'):
            stats = agent.get_context_stats()
            if stats:
                tokens_used = stats.get('tokens_used')
                tokens_max = stats.get('tokens_max')

        # Update status display
        self.display.update_status(
            goal=goal,
            agent_name=agent_name,
            model=model,
            current_round=round_number,
            max_rounds=max_rounds,
            elapsed_time=elapsed_time,
            status=status,
            tokens_used=tokens_used,
            tokens_max=tokens_max,
        )

        return context

    def on_tool_call(
        self,
        agent: Any,
        tool_name: str,
        args: dict[str, Any],
        result: Any
    ) -> None:
        """
        Called after each tool execution.

        Logs tool call events and results to the display.

        Args:
            agent: Agent instance
            tool_name: Name of tool that was called
            args: Arguments passed to the tool
            result: Result returned by the tool
        """
        if not self.display:
            return

        # Format arguments for display (truncate long values)
        args_str = self._format_args(args)

        # Log tool call
        self.display.log_event(AgentEvent(
            type=EventType.TOOL_CALL,
            message=f"{tool_name}({args_str})",
        ))

        # Determine success and log result
        event_type, message, details = self._analyze_result(tool_name, result)

        self.display.log_event(AgentEvent(
            type=event_type,
            message=message,
            details=details,
        ))

    def on_round_end(self, agent: Any, round_number: int) -> None:
        """
        Called at end of each round.

        Checks for pause (if display supports it).

        Args:
            agent: Agent instance
            round_number: Current round number
        """
        if not self.display:
            return

        # Check for pause (TUI only - plain display returns False)
        if self.display.can_pause():
            self.display.wait_if_paused()

    def on_timeout(self, agent: Any, elapsed_seconds: float) -> None:
        """
        Called when goal times out.

        Shows timeout message and cleans up display.

        Args:
            agent: Agent instance
            elapsed_seconds: Time elapsed since goal start
        """
        if not self.display:
            return

        # Log timeout event
        self.display.log_event(AgentEvent(
            type=EventType.ERROR,
            message=f"Goal timed out after {elapsed_seconds:.1f}s",
        ))

        # Stop display (cleanup)
        self.display.stop()

    def on_goal_complete(self, agent: Any, success: bool, summary: str = "") -> None:
        """
        Called when goal completes.

        Shows completion summary and cleans up display.

        Args:
            agent: Agent instance
            success: True if goal succeeded, False if failed
            summary: Summary message (optional)
        """
        if not self.display:
            return

        # Calculate duration
        duration = time.time() - self.start_time if self.start_time else 0.0

        # Get files created (if workspace_manager available)
        files_created = []
        if hasattr(agent, 'workspace_manager'):
            workspace_manager = agent.workspace_manager
            if hasattr(workspace_manager, 'created_files'):
                files_created = list(workspace_manager.created_files)

        # If no files tracked, try to get workspace directory listing
        if not files_created and hasattr(agent, 'workspace'):
            workspace = agent.workspace
            if isinstance(workspace, (str, Path)):
                workspace_path = Path(workspace)
                if workspace_path.exists() and workspace_path.is_dir():
                    try:
                        # Get all files in workspace (exclude hidden dirs)
                        files_created = [
                            str(f.relative_to(workspace_path))
                            for f in workspace_path.rglob('*')
                            if f.is_file() and not any(
                                part.startswith('.') for part in f.parts
                            )
                        ][:10]  # Limit to 10 files
                    except Exception:
                        pass  # Ignore errors getting file list

        # Show completion screen
        self.display.show_completion(
            success=success,
            summary=summary or ("Task completed successfully" if success else "Task failed"),
            duration=duration,
            files_created=files_created,
        )

        # Stop display (cleanup)
        self.display.stop()

    def get_instructions(self) -> str:
        """
        Return TUI display instructions.

        Returns:
            Instructions about progress tracking
        """
        return """
TUI DISPLAY:
Your progress is tracked and displayed automatically in real-time.

The display shows:
- Current goal and agent status
- Round number and elapsed time
- Tool calls and their results
- Performance metrics (tokens, timing)

For interactive TUI mode:
- Press 'p' to pause execution
- Press 'r' to resume
- Press 'q' to quit

You don't need to manually report progress - the system handles it automatically.
"""

    # Helper methods

    def _format_args(self, args: dict[str, Any], max_length: int = 50) -> str:
        """
        Format tool arguments for display.

        Args:
            args: Tool arguments
            max_length: Maximum length for any single value

        Returns:
            Formatted arguments string
        """
        if not args:
            return ""

        formatted_parts = []
        for key, value in args.items():
            value_str = str(value)
            if len(value_str) > max_length:
                value_str = value_str[:max_length-3] + "..."
            formatted_parts.append(f"{key}={value_str}")

        return ", ".join(formatted_parts)

    def _analyze_result(
        self,
        tool_name: str,
        result: Any
    ) -> tuple[EventType, str, Optional[dict[str, Any]]]:
        """
        Analyze tool result and determine event type and message.

        Args:
            tool_name: Name of the tool
            result: Result from tool execution

        Returns:
            (event_type, message, details) tuple
        """
        # Handle dict results (structured tool responses)
        if isinstance(result, dict):
            # Check for explicit error
            if "error" in result:
                return (
                    EventType.ERROR,
                    f"Error: {result['error']}",
                    result if len(str(result)) < 200 else None
                )

            # Check for success flag
            if result.get("success") is False:
                return (
                    EventType.ERROR,
                    f"{tool_name} failed",
                    result if len(str(result)) < 200 else None
                )

            # Success with result
            if "result" in result:
                result_str = str(result["result"])
                if len(result_str) > 100:
                    result_str = result_str[:100] + "..."
                return (
                    EventType.SUCCESS,
                    f"Success: {result_str}",
                    None
                )

            # Generic success
            return (
                EventType.SUCCESS,
                f"{tool_name} completed",
                None
            )

        # Handle string results
        if isinstance(result, str):
            # Check for error keywords
            if any(keyword in result.lower() for keyword in ["error", "failed", "exception"]):
                return (
                    EventType.ERROR,
                    result[:100] + ("..." if len(result) > 100 else ""),
                    None
                )

            # Success
            return (
                EventType.SUCCESS,
                result[:100] + ("..." if len(result) > 100 else ""),
                None
            )

        # Handle other types
        result_str = str(result)
        return (
            EventType.TOOL_RESULT,
            result_str[:100] + ("..." if len(result_str) > 100 else ""),
            None
        )
