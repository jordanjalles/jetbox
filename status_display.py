"""Clean, hierarchical status display for the local coding agent.

Provides real-time visibility into:
- Task hierarchy (Goal → Task → Subtask)
- Current progress and status
- Performance statistics (timing, tokens, success rates)
- Recent activity and errors
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from context_manager import ContextManager


# ----------------------------
# Helper Functions
# ----------------------------
def get_model_context_size() -> int:
    """Get the context window size for the current Ollama model.

    Returns the num_ctx parameter from the model, or a sensible default.
    """
    try:
        from ollama import Client
        client = Client()
        model_name = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")

        # Try to get model info
        info = client.show(model_name)
        if info and "modelinfo" in info:
            # Look for num_ctx in parameters
            params = info["modelinfo"]
            if isinstance(params, dict) and "num_ctx" in params:
                return int(params["num_ctx"])

        # Common defaults for known models
        if "gpt-oss" in model_name or "qwen" in model_name:
            return 32768  # Common for 7B-20B models
        elif "llama3" in model_name:
            return 8192
        else:
            return 32768  # Conservative default

    except Exception:
        # If we can't query, use conservative default
        return 32768


# ----------------------------
# Performance Tracking
# ----------------------------
@dataclass
class PerformanceStats:
    """Tracks agent performance metrics."""

    # Timing stats
    total_runtime: float = 0.0  # seconds
    llm_call_times: list[float] = field(default_factory=list)
    tool_call_times: list[float] = field(default_factory=list)

    # Token stats (estimated)
    total_tokens_estimated: int = 0
    messages_sent: int = 0

    # Task completion stats
    subtasks_completed: int = 0
    subtasks_failed: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0

    # Action stats
    actions_total: int = 0
    actions_successful: int = 0
    actions_failed: int = 0
    loops_detected: int = 0

    # Agent activity tracking
    current_activity: str = "idle"  # e.g., "thinking", "calling tool: write_file", "decomposing"
    last_activity_time: float = 0.0

    def avg_llm_time(self) -> float:
        """Average LLM call time in seconds."""
        return sum(self.llm_call_times) / len(self.llm_call_times) if self.llm_call_times else 0.0

    def avg_subtask_time(self) -> float:
        """Estimated average time per subtask in seconds."""
        total = self.subtasks_completed + self.subtasks_failed
        return self.total_runtime / total if total > 0 else 0.0

    def success_rate(self) -> float:
        """Overall action success rate (0-1)."""
        if self.actions_total == 0:
            return 0.0
        return self.actions_successful / self.actions_total

    def format_duration(self, seconds: float) -> str:
        """Format seconds as human-readable duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "total_runtime": self.total_runtime,
            "llm_call_times": self.llm_call_times,
            "tool_call_times": self.tool_call_times,
            "total_tokens_estimated": self.total_tokens_estimated,
            "messages_sent": self.messages_sent,
            "subtasks_completed": self.subtasks_completed,
            "subtasks_failed": self.subtasks_failed,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "actions_total": self.actions_total,
            "actions_successful": self.actions_successful,
            "actions_failed": self.actions_failed,
            "loops_detected": self.loops_detected,
            "current_activity": self.current_activity,
            "last_activity_time": self.last_activity_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerformanceStats:
        """Load from dict."""
        return cls(**data)


# ----------------------------
# Status Display
# ----------------------------
class StatusDisplay:
    """Manages clean, hierarchical status output."""

    def __init__(self, ctx: ContextManager, reset_stats: bool = False) -> None:
        self.ctx = ctx
        self.stats = PerformanceStats()
        self.session_start = time.time()
        self.stats_file = Path(".agent_context/stats.json")
        if not reset_stats:
            self._load_stats()

    def _load_stats(self) -> None:
        """Load performance stats from disk if available."""
        if self.stats_file.exists():
            try:
                data = json.loads(self.stats_file.read_text(encoding="utf-8"))
                self.stats = PerformanceStats.from_dict(data)
            except Exception:
                pass  # Use fresh stats on error

    def reset_stats(self) -> None:
        """Reset all performance statistics."""
        self.stats = PerformanceStats()
        self.session_start = time.time()
        self._save_stats()

    def _save_stats(self) -> None:
        """Save performance stats to disk."""
        try:
            self.stats_file.parent.mkdir(exist_ok=True)
            self.stats_file.write_text(
                json.dumps(self.stats.to_dict(), indent=2), encoding="utf-8"
            )
        except Exception:
            pass  # Non-critical

    def record_llm_call(self, duration: float, messages_count: int) -> None:
        """Record an LLM call for statistics."""
        self.stats.llm_call_times.append(duration)
        self.stats.messages_sent += messages_count
        # Rough token estimation: ~100 tokens per message
        self.stats.total_tokens_estimated += messages_count * 100
        self._save_stats()

    def record_action(self, success: bool) -> None:
        """Record a tool action result."""
        self.stats.actions_total += 1
        if success:
            self.stats.actions_successful += 1
        else:
            self.stats.actions_failed += 1
        self._save_stats()

    def record_subtask_complete(self, success: bool) -> None:
        """Record subtask completion."""
        if success:
            self.stats.subtasks_completed += 1
        else:
            self.stats.subtasks_failed += 1
        self._save_stats()

    def record_task_complete(self, success: bool) -> None:
        """Record task completion."""
        if success:
            self.stats.tasks_completed += 1
        else:
            self.stats.tasks_failed += 1
        self._save_stats()

    def record_loop(self) -> None:
        """Record a detected loop."""
        self.stats.loops_detected += 1
        self._save_stats()

    def update_runtime(self) -> None:
        """Update total runtime."""
        self.stats.total_runtime = time.time() - self.session_start
        self._save_stats()

    def set_activity(self, activity: str) -> None:
        """Set current agent activity for live status display."""
        self.stats.current_activity = activity
        self.stats.last_activity_time = time.time()
        self._save_stats()

    def render(self, round_no: int = 0, context_stats: dict[str, int] | None = None,
               in_place: bool = False, subtask_rounds: int = 0, max_rounds: int = 6) -> str:
        """
        Render complete status display.

        Returns a clean text summary showing:
        - Performance stats (top)
        - Context usage visualization
        - Current position in task hierarchy (with elbow connectors)
        - Turn counter (circles showing rounds used vs limit)
        - Agent status and current activity (bottom)

        Args:
            round_no: Current round number
            context_stats: Optional dict with keys: system_prompt, task_desc, agent_output, files_read
            in_place: If True, prepend ANSI codes to clear previous output for in-place update
            subtask_rounds: Number of rounds used on current subtask
            max_rounds: Max rounds allowed before forced decomposition
        """
        self.update_runtime()

        # If in_place update requested, output ANSI codes to clear previous display
        prefix = ""
        if in_place and hasattr(self, '_last_line_count') and self._last_line_count > 0:
            # Strategy: Move cursor up to start of previous output, then clear line by line
            # \033[F moves to beginning of previous line
            # \033[2K clears entire line
            # Do this for each line of previous output
            moves = []
            for _ in range(self._last_line_count):
                moves.append("\033[F\033[2K")  # Move up and clear line
            prefix = "".join(moves)

        lines = []
        lines.append("=" * 80)
        lines.append(self._render_header(round_no))
        lines.append("=" * 80)
        lines.append("")

        # PERFORMANCE at top
        lines.append(self._render_stats())
        lines.append("")

        # Context usage visualization (if provided)
        if context_stats:
            lines.append(self._render_context_usage(context_stats))
            lines.append("")

        # TURN COUNTER before task tree
        lines.append(self._render_turn_counter(subtask_rounds, max_rounds))
        lines.append("")

        # Enhanced hierarchy with elbow connectors
        lines.append(self._render_hierarchy_with_elbows())
        lines.append("")

        # AGENT STATUS at bottom
        lines.append(self._render_agent_status())
        lines.append("=" * 80)

        output = "\n".join(lines)
        # Track line count for next in-place update (including the final newline from print())
        self._last_line_count = len(lines) + 1  # +1 for the spacing line added by print()

        return prefix + output

    def _render_header(self, round_no: int) -> str:
        """Render header with round and runtime."""
        runtime = self.stats.format_duration(self.stats.total_runtime)
        return f"AGENT STATUS - Round {round_no} | Runtime: {runtime}"

    def _render_agent_status(self) -> str:
        """Render current agent activity status with latest action."""
        activity = self.stats.current_activity
        activity_emoji = {
            "idle": "💤",
            "thinking": "🤔",
            "reading": "📖",
            "writing": "✍️",
            "running": "⚙️",
            "calling": "🔄",
            "decomposing": "🔀",
            "reflecting": "💭",
            "planning": "📋",
            "marking": "✅",
        }

        # Extract emoji based on activity keywords
        emoji = "🔵"  # default
        for keyword, em in activity_emoji.items():
            if keyword in activity.lower():
                emoji = em
                break

        lines = []
        lines.append(f"AGENT STATUS: {emoji} {activity}")

        # Show latest action inline with status (not in separate section)
        task = self.ctx._get_current_task()
        if task:
            subtask = task.active_subtask()
            if subtask and subtask.actions:
                latest = subtask.actions[-1]  # Just the very last action
                icon = "✓" if latest.result == "success" else "✗"
                action_desc = latest.name
                # Truncate if too long
                if len(action_desc) > 40:
                    action_desc = action_desc[:37] + "..."
                lines.append(f"  Latest: {icon} {action_desc}")

        return "\n".join(lines)

    def _render_context_usage(self, context_stats: dict[str, int]) -> str:
        """
        Render discrete chunk visualization of context token usage.

        Args:
            context_stats: Dict with keys: system_prompt, task_desc, agent_output, system_feedback

        Returns discrete chunk bar showing token distribution
        """
        # Get actual model context size
        MAX_CONTEXT = get_model_context_size()

        sys_tokens = context_stats.get("system_prompt", 0)
        task_tokens = context_stats.get("task_desc", 0)
        agent_tokens = context_stats.get("agent_output", 0)
        feedback_tokens = context_stats.get("system_feedback", 0)  # All tool outputs

        total = sys_tokens + task_tokens + agent_tokens + feedback_tokens
        remaining = MAX_CONTEXT - total

        lines = []
        lines.append("CONTEXT USAGE:")

        # Use discrete chunks - each block represents a fixed amount of tokens
        bar_width = 40
        tokens_per_block = MAX_CONTEXT // bar_width

        # Calculate blocks for each category (round up if there are any tokens)
        # This ensures even small amounts show at least 1 block
        sys_blocks = (sys_tokens + tokens_per_block - 1) // tokens_per_block if sys_tokens > 0 else 0
        task_blocks = (task_tokens + tokens_per_block - 1) // tokens_per_block if task_tokens > 0 else 0
        agent_blocks = (agent_tokens + tokens_per_block - 1) // tokens_per_block if agent_tokens > 0 else 0
        feedback_blocks = (feedback_tokens + tokens_per_block - 1) // tokens_per_block if feedback_tokens > 0 else 0

        used_blocks = sys_blocks + task_blocks + agent_blocks + feedback_blocks
        # Ensure we don't exceed bar width due to rounding up
        if used_blocks > bar_width:
            used_blocks = bar_width
        remaining_blocks = bar_width - used_blocks

        # Build colored bar with discrete chunks (using ANSI color codes)
        # Red=system, Yellow=task, Green=agent, Blue=system feedback, Gray=remaining
        bar = (
            "\033[91m" + "█" * sys_blocks + "\033[0m" +      # Red for system prompt
            "\033[93m" + "█" * task_blocks + "\033[0m" +     # Yellow for task desc
            "\033[92m" + "█" * agent_blocks + "\033[0m" +    # Green for agent output
            "\033[94m" + "█" * feedback_blocks + "\033[0m" + # Blue for system feedback
            "\033[90m" + "░" * remaining_blocks + "\033[0m"  # Gray for unused
        )

        lines.append(f"  [{bar}]")
        lines.append(f"  {total:,}/{MAX_CONTEXT:,} tokens ({100*total/MAX_CONTEXT:.1f}% used)")
        lines.append(f"  Each █ = ~{tokens_per_block:,} tokens")
        lines.append("")
        lines.append("  Legend:")
        lines.append(f"    \033[91m■\033[0m System Prompt:   {sys_tokens:>7,} tokens ({sys_blocks} blocks)")
        lines.append(f"    \033[93m■\033[0m Task Desc:       {task_tokens:>7,} tokens ({task_blocks} blocks)")
        lines.append(f"    \033[92m■\033[0m Agent Output:    {agent_tokens:>7,} tokens ({agent_blocks} blocks)")
        lines.append(f"    \033[94m■\033[0m System Feedback: {feedback_tokens:>7,} tokens ({feedback_blocks} blocks)")
        lines.append(f"    \033[90m■\033[0m Remaining:       {remaining:>7,} tokens ({remaining_blocks} blocks)")

        return "\n".join(lines)

    def _render_hierarchy_with_elbows(self) -> str:
        """Render task hierarchy with ASCII elbow connectors."""
        if not self.ctx.state.goal:
            return "No active goal."

        lines = []
        goal = self.ctx.state.goal

        # Goal level
        lines.append(f"GOAL: {goal.description}")
        lines.append("")

        # Tasks level with elbow connectors
        if goal.tasks:
            total_tasks = len(goal.tasks)
            completed_tasks = sum(1 for t in goal.tasks if t.status == "completed")

            lines.append(f"TASK TREE ({completed_tasks}/{total_tasks} completed):")
            for i, task in enumerate(goal.tasks):
                is_current = i == self.ctx.state.current_task_idx
                is_last_task = i == len(goal.tasks) - 1

                # Render task with elbow connectors
                lines.extend(self._render_task_with_elbows(task, is_current, is_last_task))

        return "\n".join(lines)

    def _render_task_with_elbows(self, task: Any, is_current: bool, is_last: bool) -> list[str]:
        """Render a task and its subtasks with ASCII elbow connectors."""
        lines = []

        # Task line
        status_icon = self._get_status_icon(task.status, is_current)
        connector = "└─" if is_last else "├─"
        task_desc = task.description[:65] + "..." if len(task.description) > 65 else task.description

        # Don't highlight the task itself - only subtasks get highlighting
        # Just show a simple arrow for current task
        if is_current and task.status != "completed":
            prefix = "► "
        else:
            prefix = "  "

        # Show approach attempt count if > 0
        attempt_info = f" (attempt {task.approach_attempts}/{3})" if hasattr(task, 'approach_attempts') and task.approach_attempts > 0 else ""
        lines.append(f"{connector}{prefix}{status_icon} {task_desc}{attempt_info}")

        # Render subtasks with proper indentation
        if task.subtasks:
            for j, subtask in enumerate(task.subtasks):
                is_last_subtask = j == len(task.subtasks) - 1
                indent_char = "  " if is_last else "│ "
                lines.extend(self._render_subtask_with_elbows(subtask, is_last_subtask, indent_char))

        return lines

    def _render_subtask_with_elbows(self, subtask: Any, is_last: bool, parent_indent: str) -> list[str]:
        """Render a subtask and its children with ASCII elbow connectors."""
        lines = []

        # Determine if this subtask is active
        is_current = subtask.status == "in_progress"

        # Subtask line
        status_icon = self._get_status_icon(subtask.status, is_current)
        connector = "└─" if is_last else "├─"
        desc = subtask.description[:60] + "..." if len(subtask.description) > 60 else subtask.description

        # Make current subtask VERY visible with highlighting
        if is_current:
            # Use bold and color for current subtask
            prefix = "► \033[1m\033[96m"  # Bold cyan
            suffix = "\033[0m"  # Reset
        else:
            prefix = "  "
            suffix = ""

        # Show depth indicator
        depth_info = f" [L{subtask.depth}]" if hasattr(subtask, 'depth') and subtask.depth > 1 else ""

        lines.append(f"{parent_indent}{connector}{prefix}{status_icon} {desc}{depth_info}{suffix}")

        # Show failure reason if blocked/failed
        if subtask.status in ["blocked", "failed"] and hasattr(subtask, 'failure_reason') and subtask.failure_reason:
            failure_short = subtask.failure_reason[:45] + "..." if len(subtask.failure_reason) > 45 else subtask.failure_reason
            fail_indent = "  " if is_last else "│ "
            lines.append(f"{parent_indent}{fail_indent}  └─ ⚠ {failure_short}")

        # Render child subtasks recursively
        if hasattr(subtask, 'child_subtasks') and subtask.child_subtasks:
            for k, child in enumerate(subtask.child_subtasks):
                is_last_child = k == len(subtask.child_subtasks) - 1
                child_indent = parent_indent + ("  " if is_last else "│ ")
                lines.extend(self._render_subtask_with_elbows(child, is_last_child, child_indent))

        return lines

    def _render_hierarchy(self) -> str:
        """Render full task hierarchy tree structure."""
        if not self.ctx.state.goal:
            return "No active goal."

        lines = []
        goal = self.ctx.state.goal

        # Goal level
        lines.append(f"GOAL: {goal.description}")
        lines.append("")

        # Tasks level
        if goal.tasks:
            total_tasks = len(goal.tasks)
            completed_tasks = sum(1 for t in goal.tasks if t.status == "completed")

            lines.append(f"TASK TREE ({completed_tasks}/{total_tasks} completed):")
            for i, task in enumerate(goal.tasks):
                is_current = i == self.ctx.state.current_task_idx
                # Render task and its full subtask tree
                lines.extend(self._render_task_tree(task, is_current, indent=1))

        return "\n".join(lines)

    def _render_task_tree(self, task: Any, is_current: bool, indent: int) -> list[str]:
        """Render a task and all its subtasks as a tree."""
        lines = []
        indent_str = "  " * indent

        # Render task
        status_icon = self._get_status_icon(task.status, is_current)
        prefix = "► " if is_current else "  "
        task_desc = task.description[:70] + "..." if len(task.description) > 70 else task.description

        # Show approach attempt count if > 0
        attempt_info = f" (attempt {task.approach_attempts}/{3})" if hasattr(task, 'approach_attempts') and task.approach_attempts > 0 else ""
        lines.append(f"{indent_str}{prefix}{status_icon} {task_desc}{attempt_info}")

        # Render all subtasks recursively
        if task.subtasks:
            for subtask in task.subtasks:
                lines.extend(self._render_subtask_tree(subtask, indent + 1))

        return lines

    def _render_subtask_tree(self, subtask: Any, indent: int) -> list[str]:
        """Render a subtask and all its children recursively as a tree."""
        lines = []
        indent_str = "  " * indent

        # Determine if this subtask is active
        is_current = subtask.status == "in_progress"

        # Render subtask
        status_icon = self._get_status_icon(subtask.status, is_current)
        prefix = "► " if is_current else "  "
        desc = subtask.description[:65] + "..." if len(subtask.description) > 65 else subtask.description

        # Show depth indicator
        depth_info = f" [L{subtask.depth}]" if hasattr(subtask, 'depth') and subtask.depth > 1 else ""

        # Show failure reason if blocked/failed
        if subtask.status in ["blocked", "failed"] and hasattr(subtask, 'failure_reason') and subtask.failure_reason:
            failure_short = subtask.failure_reason[:40] + "..." if len(subtask.failure_reason) > 40 else subtask.failure_reason
            lines.append(f"{indent_str}{prefix}{status_icon} {desc}{depth_info}")
            lines.append(f"{indent_str}   └─ ⚠ {failure_short}")
        else:
            lines.append(f"{indent_str}{prefix}{status_icon} {desc}{depth_info}")

        # Render child subtasks recursively
        if hasattr(subtask, 'child_subtasks') and subtask.child_subtasks:
            for child in subtask.child_subtasks:
                lines.extend(self._render_subtask_tree(child, indent + 1))

        return lines

    def _render_turn_counter(self, current: int, max_turns: int) -> str:
        """Render turn counter with circles showing rounds used before forced decomposition.

        Args:
            current: Current round number for this subtask
            max_turns: Max rounds allowed before decomposition
        """
        lines = []
        lines.append("TURNS UNTIL FORCED DECOMPOSITION:")

        # Use filled circles (●) for used turns, empty circles (○) for remaining
        filled = "●" * current
        empty = "○" * (max_turns - current)
        circles = filled + empty

        # Add warning color if close to limit
        if current >= max_turns * 0.8:  # 80% or more used
            status_color = "\033[91m"  # Red
            status_text = " ⚠ Near limit"
        elif current >= max_turns * 0.5:  # 50% or more used
            status_color = "\033[93m"  # Yellow
            status_text = " ⚡ Half used"
        else:
            status_color = ""  # No color for early turns
            status_text = ""  # No status text

        lines.append(f"  {circles}  {status_color}{current}/{max_turns} turns{status_text}\033[0m")

        return "\n".join(lines)

    def _render_stats(self) -> str:
        """Render performance statistics."""
        lines = []
        lines.append("PERFORMANCE:")

        # Timing stats
        avg_llm = self.stats.avg_llm_time()
        avg_subtask = self.stats.avg_subtask_time()
        lines.append(f"  Avg LLM call:      {avg_llm:.2f}s")
        lines.append(f"  Avg subtask time:  {self.stats.format_duration(avg_subtask)}")

        # Throughput
        lines.append(f"  LLM calls:         {len(self.stats.llm_call_times)}")
        lines.append(f"  Actions executed:  {self.stats.actions_total}")
        lines.append(f"  Tokens (est):      {self.stats.total_tokens_estimated:,}")

        # Issues
        if self.stats.loops_detected > 0:
            lines.append(f"  ⚠ Loops detected:  {self.stats.loops_detected}")

        return "\n".join(lines)

    def _render_recent_activity(self) -> str:
        """Render recent actions (errors only shown if blocking progress)."""
        lines = []
        lines.append("RECENT ACTIVITY:")

        # Get recent actions from current subtask
        task = self.ctx._get_current_task()
        has_actions = False

        if task:
            subtask = task.active_subtask()
            if subtask and subtask.actions:
                recent = subtask.actions[-3:]  # Last 3 actions
                has_actions = True
                for action in recent:
                    icon = "✓" if action.result == "success" else "✗"
                    lines.append(f"  {icon} {action.name}")
                    # Only show error inline if action failed
                    if action.result != "success" and action.error_msg:
                        err_short = action.error_msg[:50] + "..." if len(action.error_msg) > 50 else action.error_msg
                        lines.append(f"    └─ {err_short}")

        # Only show recent errors section if there are unresolved errors AND low success rate
        # This makes errors "sticky" only when they're actually blocking progress
        if self.ctx.state.last_probe_state and self.ctx.state.last_probe_state.get("recent_errors"):
            errors = self.ctx.state.last_probe_state["recent_errors"]
            # Only show if success rate < 70% (indicates ongoing issues)
            if errors and self.stats.success_rate() < 0.7 and self.stats.actions_total > 3:
                lines.append("")
                lines.append("  ⚠ Blocking errors:")
                for err in errors[-2:]:  # Last 2 errors
                    err_short = err[:60] + "..." if len(err) > 60 else err
                    lines.append(f"    • {err_short}")
                has_actions = True

        if not has_actions:
            lines.append("  (none)")

        return "\n".join(lines)

    def _get_status_icon(self, status: str, is_current: bool) -> str:
        """Get status icon for task/subtask."""
        # Status takes priority over is_current - completed items always show checkmark
        if status == "completed":
            return "✓"
        elif status == "failed":
            return "✗"
        elif status == "blocked":
            return "⊗"
        elif is_current or status == "in_progress":
            return "⟳"  # In progress
        else:
            return "○"  # Pending

    def _render_progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """Render a text progress bar (continuous - legacy)."""
        if total == 0:
            return "[" + " " * width + "]"

        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def _render_discrete_bar(self, completed: int, total: int, max_width: int = 20) -> str:
        """Render a discrete chunk bar showing individual items.

        Each item is represented as a distinct block, making it easy to count.
        """
        if total == 0:
            return "[empty]"

        # If we have few items, show each one as a block
        if total <= max_width:
            # Show each item as a discrete block
            bar = "█" * completed + "░" * (total - completed)
            return f"[{bar}]"
        else:
            # Too many items - group them into chunks
            # Each visual block represents multiple items
            items_per_block = (total + max_width - 1) // max_width  # Round up
            filled_blocks = completed // items_per_block
            remaining = max_width - filled_blocks

            # Show grouping info
            bar = "█" * filled_blocks + "░" * remaining
            return f"[{bar}] (each █ = ~{items_per_block} items)"

    def render_compact(self) -> str:
        """
        Render a compact one-line status for frequent updates.

        Example: "Round 5 | Task 2/3 | Subtask 3/5 | ✓92% | 2m 15s"
        """
        if not self.ctx.state.goal:
            return "No active goal"

        total_tasks = len(self.ctx.state.goal.tasks)
        current_task = min(self.ctx.state.current_task_idx + 1, total_tasks)

        task = self.ctx._get_current_task()
        if task:
            total_subtasks = len(task.subtasks)
            # Count completed subtasks
            completed = sum(1 for st in task.subtasks if st.status == "completed")
            current_subtask = min(completed + 1, total_subtasks)

            success_pct = int(self.stats.success_rate() * 100)
            runtime = self.stats.format_duration(self.stats.total_runtime)

            return (
                f"Task {current_task}/{total_tasks} | "
                f"Subtask {current_subtask}/{total_subtasks} | "
                f"✓{success_pct}% | {runtime}"
            )
        else:
            return f"Task {total_tasks}/{total_tasks} | Complete | {self.stats.format_duration(self.stats.total_runtime)}"


# ----------------------------
# Helper functions
# ----------------------------
def clear_screen() -> None:
    """Clear terminal screen (cross-platform)."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def print_status_update(display: StatusDisplay, round_no: int, clear: bool = False) -> None:
    """Print status update to console."""
    if clear:
        clear_screen()

    print(display.render(round_no))
