"""
StatusDisplayBehavior - Real-time progress and performance visualization.

DEPRECATED: This behavior is deprecated and will be removed in v2.0.
Status display is being redesigned for the behavior system.

This behavior wraps the StatusDisplay class to provide progress tracking
and performance monitoring through the AgentBehavior interface.

Features:
- Event: on_goal_start - Initialize display
- Event: on_tool_call - Record action statistics
- Event: on_round_end - Render status display
- Event: on_timeout - Update final stats
- Event: on_goal_complete - Show final summary
- No tools (utility behavior)

The behavior delegates to the StatusDisplay class for actual rendering.
"""

import warnings
from typing import Any
from behaviors.base import AgentBehavior
from status_display import StatusDisplay

# Emit deprecation warning when module is imported
warnings.warn(
    "StatusDisplayBehavior is deprecated and will be removed in v2.0. "
    "Status display is being redesigned for the behavior system.",
    DeprecationWarning,
    stacklevel=2
)


class StatusDisplayBehavior(AgentBehavior):
    """
    Behavior that provides real-time status display and performance tracking.

    Automatically:
    - Initializes status display on goal start
    - Records tool call statistics
    - Renders hierarchical status at round end
    - Tracks performance metrics (LLM timing, tokens, success rates)
    - Shows final summary on goal completion

    This is a utility behavior (no tools) that integrates the
    StatusDisplay system with the behavior framework.
    """

    def __init__(self, reset_stats: bool = False, show_hierarchical: bool = True):
        """
        Initialize status display behavior.

        Args:
            reset_stats: If True, reset performance stats on init (default: False)
            show_hierarchical: If True, show hierarchical task tree (default: True)
        """
        self.display = None
        self.reset_stats = reset_stats
        self.show_hierarchical = show_hierarchical
        self.context_manager = None

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "status_display"

    def on_goal_start(self, goal: str, **kwargs: Any) -> None:
        """
        Called when goal starts.

        Initializes the StatusDisplay with the context manager.

        Args:
            goal: The goal string
            **kwargs: Additional context (context_manager, etc.)
        """
        # Get context manager
        if "context_manager" in kwargs:
            self.context_manager = kwargs["context_manager"]

            # Initialize display
            if self.display is None and self.context_manager:
                self.display = StatusDisplay(
                    ctx=self.context_manager,
                    reset_stats=self.reset_stats
                )
                print(f"\n[status_display] Initialized for goal: {goal}\n")

    def on_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        **kwargs: Any
    ) -> None:
        """
        Called after each tool execution.

        Records action statistics for performance tracking.

        Args:
            tool_name: Name of tool that was called
            args: Arguments passed to the tool
            result: Result returned by the tool
            **kwargs: Additional context
        """
        if not self.display:
            return

        # Determine success
        if isinstance(result, dict):
            success = not ("error" in result or result.get("success") is False)
        elif isinstance(result, str):
            success = "error" not in result.lower() and "failed" not in result.lower()
        else:
            success = True

        # Record action
        self.display.record_action(success=success)

        # Update activity
        self.display.set_activity(f"called tool: {tool_name}")

    def on_round_end(self, round_number: int, **kwargs: Any) -> None:
        """
        Called at end of each round.

        Renders the status display showing progress and performance.

        Args:
            round_number: Current round number
            **kwargs: Additional context (context_stats, subtask_rounds, max_rounds, etc.)
        """
        if not self.display:
            return

        # Get context stats if available
        context_stats = kwargs.get("context_stats", None)
        subtask_rounds = kwargs.get("subtask_rounds", 0)
        max_rounds = kwargs.get("max_rounds", 6)

        # Render status display
        status_output = self.display.render(
            round_no=round_number,
            context_stats=context_stats,
            in_place=False,  # Don't use in-place updates (can cause issues in some terminals)
            subtask_rounds=subtask_rounds,
            max_rounds=max_rounds,
            show_hierarchical=self.show_hierarchical
        )

        # Print to console
        print(status_output)

    def on_timeout(self, elapsed_seconds: float, **kwargs: Any) -> None:
        """
        Called when goal times out.

        Updates runtime and shows final status.

        Args:
            elapsed_seconds: Time elapsed since goal start
            **kwargs: Additional context
        """
        if not self.display:
            return

        # Update runtime
        self.display.update_runtime()
        self.display.set_activity("timed out")

        print(f"\n[status_display] Goal timed out after {elapsed_seconds:.1f}s")

    def on_goal_complete(self, success: bool, **kwargs: Any) -> None:
        """
        Called when goal completes.

        Shows final performance summary.

        Args:
            success: True if goal succeeded, False if failed
            **kwargs: Additional context
        """
        if not self.display:
            return

        # Update runtime
        self.display.update_runtime()

        # Update activity
        status_str = "completed successfully" if success else "failed"
        self.display.set_activity(status_str)

        # Show final summary
        print("\n" + "="*70)
        print("FINAL PERFORMANCE SUMMARY")
        print("="*70)
        print(self.display._render_stats())
        print("="*70 + "\n")

    def get_instructions(self) -> str:
        """
        Return status display instructions.

        Returns:
            Instructions about progress tracking
        """
        return """
STATUS DISPLAY:
Your progress is tracked and displayed automatically at each round.

The display shows:
- Current position in task hierarchy
- Performance statistics (LLM timing, success rate)
- Context usage visualization
- Recent actions and errors

You don't need to manually report progress - the system handles it automatically.
"""
