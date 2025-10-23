"""Clean, hierarchical status display for the local coding agent.

Provides real-time visibility into:
- Task hierarchy (Goal → Task → Subtask)
- Current progress and status
- Performance statistics (timing, tokens, success rates)
- Recent activity and errors
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from context_manager import ContextManager


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

    def render(self, round_no: int = 0) -> str:
        """
        Render complete status display.

        Returns a clean text summary showing:
        - Current position in task hierarchy
        - Progress indicators
        - Performance stats
        - Recent activity
        """
        self.update_runtime()

        lines = []
        lines.append("=" * 70)
        lines.append(self._render_header(round_no))
        lines.append("=" * 70)
        lines.append("")
        lines.append(self._render_hierarchy())
        lines.append("")
        lines.append(self._render_progress())
        lines.append("")
        lines.append(self._render_stats())
        lines.append("")
        lines.append(self._render_recent_activity())
        lines.append("=" * 70)

        return "\n".join(lines)

    def _render_header(self, round_no: int) -> str:
        """Render header with round and runtime."""
        runtime = self.stats.format_duration(self.stats.total_runtime)
        return f"AGENT STATUS - Round {round_no} | Runtime: {runtime}"

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

    def _render_progress(self) -> str:
        """Render progress indicators."""
        if not self.ctx.state.goal or not self.ctx.state.goal.tasks:
            return "No progress yet."

        total_tasks = len(self.ctx.state.goal.tasks)
        current_task_idx = min(self.ctx.state.current_task_idx, total_tasks)  # Cap at total

        # Calculate overall progress
        total_subtasks = 0
        completed_subtasks = 0

        for task in self.ctx.state.goal.tasks:
            total_subtasks += len(task.subtasks)
            for subtask in task.subtasks:
                if subtask.status == "completed":
                    completed_subtasks += 1

        # Progress bars
        lines = []
        lines.append("PROGRESS:")

        # Task progress
        task_pct = (current_task_idx / total_tasks * 100) if total_tasks > 0 else 0
        task_bar = self._render_progress_bar(current_task_idx, total_tasks)
        lines.append(f"  Tasks:    {task_bar} {task_pct:.0f}%")

        # Subtask progress
        subtask_pct = (completed_subtasks / total_subtasks * 100) if total_subtasks > 0 else 0
        subtask_bar = self._render_progress_bar(completed_subtasks, total_subtasks)
        lines.append(f"  Subtasks: {subtask_bar} {subtask_pct:.0f}%")

        # Success rate
        success_pct = self.stats.success_rate() * 100
        lines.append(f"  Success:  {success_pct:.0f}%")

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
        """Render recent actions and errors."""
        lines = []
        lines.append("RECENT ACTIVITY:")

        # Get recent actions from current subtask
        task = self.ctx._get_current_task()
        if task:
            subtask = task.active_subtask()
            if subtask and subtask.actions:
                recent = subtask.actions[-3:]  # Last 3 actions
                for action in recent:
                    icon = "✓" if action.result == "success" else "✗"
                    lines.append(f"  {icon} {action.name}")
                    if action.error_msg:
                        err_short = action.error_msg[:50] + "..." if len(action.error_msg) > 50 else action.error_msg
                        lines.append(f"    └─ {err_short}")

        # Show recent errors from probe state
        if self.ctx.state.last_probe_state and self.ctx.state.last_probe_state.get("recent_errors"):
            errors = self.ctx.state.last_probe_state["recent_errors"]
            if errors:
                lines.append("")
                lines.append("  Recent errors:")
                for err in errors[-2:]:  # Last 2 errors
                    err_short = err[:60] + "..." if len(err) > 60 else err
                    lines.append(f"    • {err_short}")

        if len(lines) == 1:
            lines.append("  (none)")

        return "\n".join(lines)

    def _get_status_icon(self, status: str, is_current: bool) -> str:
        """Get status icon for task/subtask."""
        if is_current:
            return "⟳"  # In progress
        elif status == "completed":
            return "✓"
        elif status == "failed":
            return "✗"
        elif status == "blocked":
            return "⊗"
        else:
            return "○"  # Pending

    def _render_progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """Render a text progress bar."""
        if total == 0:
            return "[" + " " * width + "]"

        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

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
