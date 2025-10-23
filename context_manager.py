"""Hierarchical context manager for crash-resilient agent.

Design principles:
1. Hierarchical structure: Goal → Task → Subtask → Action
2. Need-to-know only: Each level sees only relevant parent context
3. Loop detection: Automatically detect repeated attempts and escalate
4. Crash recovery: Rebuild full context from minimal state files
5. Automatic compaction: Prune stale/irrelevant context periodically
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Load configuration
from agent_config import config

# ----------------------------
# Configuration
# ----------------------------
CONTEXT_DIR = Path(".agent_context")
STATE_FILE = CONTEXT_DIR / "state.json"
HISTORY_FILE = CONTEXT_DIR / "history.jsonl"
LOOPS_FILE = CONTEXT_DIR / "loops.json"

# Loop detection thresholds (from config)
MAX_ACTION_REPEATS = config.loop_detection.max_action_repeats
MAX_SUBTASK_REPEATS = config.loop_detection.max_subtask_repeats
MAX_CONTEXT_AGE = config.loop_detection.max_context_age


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Action:
    """Single atomic action (tool call)."""

    name: str  # Tool name (write_file, run_cmd, etc.)
    args: dict[str, Any]  # Normalized arguments
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    result: str | None = None  # "success", "error", or None if pending
    error_msg: str = ""
    attempt_count: int = 1

    def signature(self) -> str:
        """Unique signature for deduplication."""
        args_str = json.dumps(self.args, sort_keys=True)
        return f"{self.name}::{args_str}"


@dataclass
class Subtask:
    """Small concrete task (e.g., 'write mathx/__init__.py')."""

    description: str
    actions: list[Action] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed, blocked
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    failure_reason: str = ""
    attempt_count: int = 1

    # Phase 2: Hierarchy tracking
    depth: int = 1  # How deep in hierarchy (1 = top-level subtask)
    parent_subtask: str = ""  # Description of parent (empty if top-level)
    child_subtasks: list[Subtask] = field(default_factory=list)  # Nested subtasks
    rounds_used: int = 0  # Rounds spent on this subtask

    # Accomplishment tracking (for partial success and learning from failures)
    accomplishments: list[str] = field(default_factory=list)  # What was successfully done
    tried_approaches: list[str] = field(default_factory=list)  # What was attempted but failed
    context_notes: str = ""  # Additional context about progress/blockers

    def signature(self) -> str:
        """Unique signature for loop detection."""
        return self.description.lower().strip()

    def can_add_child(self, max_depth: int | None = None, max_siblings: int | None = None) -> bool:
        """Check if can add child subtask (depth and sibling limits)."""
        # Use config defaults if not specified
        if max_depth is None:
            max_depth = config.hierarchy.max_depth
        if max_siblings is None:
            max_siblings = config.hierarchy.max_siblings

        if self.depth >= max_depth:
            return False  # Too deep
        if len(self.child_subtasks) >= max_siblings:
            return False  # Too many siblings
        return True


@dataclass
class Task:
    """Mid-level task (e.g., 'create mathx package')."""

    description: str
    subtasks: list[Subtask] = field(default_factory=list)
    status: str = "pending"
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    parent_goal: str = ""
    approach_attempts: int = 0  # Number of times we've retried this task from root
    failed_approaches: list[str] = field(default_factory=list)  # Track what didn't work

    def active_subtask(self) -> Subtask | None:
        """Get current in-progress subtask."""
        for st in self.subtasks:
            if st.status == "in_progress":
                return st
        return None

    def next_pending_subtask(self) -> Subtask | None:
        """Get next pending subtask."""
        for st in self.subtasks:
            if st.status == "pending":
                return st
        return None


@dataclass
class Goal:
    """Top-level goal (user request)."""

    description: str
    tasks: list[Task] = field(default_factory=list)
    status: str = "pending"
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def active_task(self) -> Task | None:
        """Get current in-progress task."""
        for task in self.tasks:
            if task.status == "in_progress":
                return task
        return None


@dataclass
class ContextState:
    """Full hierarchical context state."""

    goal: Goal | None = None
    current_task_idx: int = 0
    current_subtask_idx: int = 0
    loop_counts: dict[str, int] = field(default_factory=dict)
    blocked_actions: set[str] = field(default_factory=set)
    last_probe_state: dict[str, Any] = field(default_factory=dict)
    session_start: float = field(default_factory=lambda: datetime.now().timestamp())


# ----------------------------
# Context Manager
# ----------------------------
class ContextManager:
    """Manages hierarchical context with loop detection and crash recovery."""

    def __init__(self) -> None:
        self.state = ContextState()
        self.action_history: list[Action] = []
        self.loop_detector = LoopDetector()
        self.loop_callback = None  # Optional callback for loop detection
        self.is_new_goal = False  # Track if this is a fresh goal
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create context directory if needed."""
        CONTEXT_DIR.mkdir(exist_ok=True)

    def load_or_init(self, goal_text: str) -> None:
        """Load existing state or initialize new goal. Returns True if new goal."""
        if STATE_FILE.exists():
            self._load_state()
            if self.state.goal and self.state.goal.description == goal_text:
                # Check if goal was previously completed
                all_tasks_done = all(
                    task.status == "completed"
                    for task in self.state.goal.tasks
                ) if self.state.goal.tasks else False

                if all_tasks_done:
                    # Goal was completed - start fresh
                    print("[context] Previous run completed. Starting fresh run.")
                    self.state.goal = Goal(description=goal_text)
                    self.state.current_task_idx = 0
                    self.state.current_subtask_idx = 0
                    self.is_new_goal = True
                    self._save_state()
                else:
                    # Resume in-progress goal
                    self.is_new_goal = False
                    return
            else:
                # Different goal - reset indices
                print("[context] Different goal detected. Starting fresh.")
                self.is_new_goal = True
        else:
            self.is_new_goal = True
        # New goal - reset all state
        self.state.goal = Goal(description=goal_text)
        self.state.current_task_idx = 0
        self.state.current_subtask_idx = 0
        self._save_state()

    def _load_subtask(self, st_data: dict[str, Any]) -> Subtask:
        """Recursively load subtask with children."""
        return Subtask(
            description=st_data["description"],
            actions=[Action(**a) for a in st_data.get("actions", [])],
            status=st_data.get("status", "pending"),
            timestamp=st_data.get("timestamp", 0),
            failure_reason=st_data.get("failure_reason", ""),
            attempt_count=st_data.get("attempt_count", 1),
            depth=st_data.get("depth", 1),
            parent_subtask=st_data.get("parent_subtask", ""),
            child_subtasks=[
                self._load_subtask(child) for child in st_data.get("child_subtasks", [])
            ],
            rounds_used=st_data.get("rounds_used", 0),
            accomplishments=st_data.get("accomplishments", []),
            tried_approaches=st_data.get("tried_approaches", []),
            context_notes=st_data.get("context_notes", ""),
        )

    def _load_state(self) -> None:
        """Load state from disk."""
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            # Reconstruct from dict
            goal_data = data.get("goal")
            if goal_data:
                tasks = [
                    Task(
                        description=t["description"],
                        subtasks=[
                            self._load_subtask(st)
                            for st in t.get("subtasks", [])
                        ],
                        status=t.get("status", "pending"),
                        timestamp=t.get("timestamp", 0),
                        parent_goal=t.get("parent_goal", ""),
                        approach_attempts=t.get("approach_attempts", 0),
                        failed_approaches=t.get("failed_approaches", []),
                    )
                    for t in goal_data.get("tasks", [])
                ]
                self.state.goal = Goal(
                    description=goal_data["description"],
                    tasks=tasks,
                    status=goal_data.get("status", "pending"),
                    timestamp=goal_data.get("timestamp", 0),
                )
            self.state.current_task_idx = data.get("current_task_idx", 0)
            self.state.current_subtask_idx = data.get("current_subtask_idx", 0)
            self.state.loop_counts = data.get("loop_counts", {})
            self.state.blocked_actions = set(data.get("blocked_actions", []))
        except Exception as e:
            print(f"[context] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            data = {
                "goal": asdict(self.state.goal) if self.state.goal else None,
                "current_task_idx": self.state.current_task_idx,
                "current_subtask_idx": self.state.current_subtask_idx,
                "loop_counts": self.state.loop_counts,
                "blocked_actions": list(self.state.blocked_actions),
                "last_probe_state": self.state.last_probe_state,
                "session_start": self.state.session_start,
            }
            STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[context] Failed to save state: {e}")

    def update_probe_state(self, state: dict[str, Any]) -> None:
        """Update last known probe state (filesystem, tests, etc.)."""
        self.state.last_probe_state = state
        self._save_state()

    def record_action(
        self, name: str, args: dict[str, Any], result: str, error_msg: str = ""
    ) -> bool:
        """
        Record an action attempt and check for loops.

        Returns True if action should proceed, False if blocked (loop detected).
        """
        action = Action(name=name, args=args, result=result, error_msg=error_msg)
        sig = action.signature()

        # Check if action is blocked
        if sig in self.state.blocked_actions:
            return False

        # Check for loops
        if self.loop_detector.is_loop(action, self.action_history):
            self.state.blocked_actions.add(sig)
            self.state.loop_counts[sig] = self.state.loop_counts.get(sig, 0) + 1
            self._save_state()
            self._log_loop(action)
            if self.loop_callback:
                self.loop_callback()  # Notify status display
            return False

        # Record action
        self.action_history.append(action)
        self._append_history(action)

        # Update current subtask if exists
        if self.state.goal:
            task = self._get_current_task()
            if task:
                subtask = task.active_subtask()
                if subtask:
                    subtask.actions.append(action)

        self._save_state()
        return True

    def _get_current_task(self) -> Task | None:
        """Get current active task."""
        if not self.state.goal or not self.state.goal.tasks:
            return None
        if self.state.current_task_idx < len(self.state.goal.tasks):
            return self.state.goal.tasks[self.state.current_task_idx]
        return None

    def _append_history(self, action: Action) -> None:
        """Append action to history file."""
        try:
            line = json.dumps(asdict(action)) + "\n"
            with HISTORY_FILE.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            print(f"[context] Failed to append history: {e}")

    def _log_loop(self, action: Action) -> None:
        """Log detected loop."""
        sig = action.signature()
        loop_data = {
            "timestamp": datetime.now().isoformat(),
            "action_signature": sig,
            "attempt_count": self.state.loop_counts.get(sig, 0),
        }
        try:
            if LOOPS_FILE.exists():
                loops = json.loads(LOOPS_FILE.read_text(encoding="utf-8"))
            else:
                loops = []
            loops.append(loop_data)
            LOOPS_FILE.write_text(json.dumps(loops, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[context] Failed to log loop: {e}")

    def get_compact_context(self, max_chars: int = 2000) -> str:
        """
        Generate compact, need-to-know context for current level.

        Returns a hierarchical summary focused on the active path:
        Goal → Current Task → Current Subtask → Recent Actions
        """
        if not self.state.goal:
            return "No active goal."

        lines = []

        # Level 1: Goal
        lines.append(f"GOAL: {self.state.goal.description}")
        lines.append(f"Status: {self.state.goal.status}")

        # Level 2: Current Task
        task = self._get_current_task()
        if task:
            lines.append(f"\nCURRENT TASK: {task.description}")
            lines.append(f"Status: {task.status}")

            # Level 3: Current Subtask
            subtask = task.active_subtask()
            if subtask:
                lines.append(f"\nACTIVE SUBTASK: {subtask.description}")
                lines.append(f"Status: {subtask.status}")
                if subtask.failure_reason:
                    lines.append(f"Last failure: {subtask.failure_reason}")

                # Level 4: Recent Actions (last 3)
                if subtask.actions:
                    lines.append("\nRecent actions:")
                    for act in subtask.actions[-3:]:
                        result_str = f"→ {act.result}" if act.result else "→ pending"
                        lines.append(f"  - {act.name} {result_str}")
                        if act.error_msg:
                            lines.append(f"    Error: {act.error_msg[:100]}")

            # Show next pending subtask
            next_st = task.next_pending_subtask()
            if next_st:
                lines.append(f"\nNEXT: {next_st.description}")

        # Probe state summary
        if self.state.last_probe_state:
            lines.append("\nCURRENT STATE:")
            for key, val in self.state.last_probe_state.items():
                if isinstance(val, bool):
                    lines.append(f"  {key}: {'✓' if val else '✗'}")

        # Loop warnings
        if self.state.blocked_actions:
            lines.append(f"\n⚠ {len(self.state.blocked_actions)} blocked (loops)")

        context = "\n".join(lines)
        if len(context) > max_chars:
            context = context[:max_chars] + "..."
        return context

    def mark_subtask_complete(self, success: bool, reason: str = "") -> None:
        """Mark current subtask as complete or failed."""
        task = self._get_current_task()
        if not task:
            return

        subtask = task.active_subtask()
        if not subtask:
            return

        # Analyze actions to extract accomplishments and failed attempts
        self._extract_subtask_context(subtask)

        if success:
            subtask.status = "completed"
        else:
            subtask.status = "failed"
            subtask.failure_reason = reason
            subtask.attempt_count += 1

            # Check if subtask has been tried too many times
            if subtask.attempt_count >= MAX_SUBTASK_REPEATS:
                subtask.status = "blocked"
                # Escalate to task level
                task.status = "blocked"

        self._save_state()

    def _extract_subtask_context(self, subtask: Subtask) -> None:
        """
        Extract accomplishments and tried approaches from subtask actions.

        This analyzes what was done successfully vs what failed to provide
        context for future retry attempts.
        """
        # Extract successful actions as accomplishments
        for action in subtask.actions:
            if action.result == "success":
                if action.name == "write_file":
                    path = action.args.get("path", "unknown")
                    if path not in [a for a in subtask.accomplishments if path in a]:
                        subtask.accomplishments.append(f"Created {path}")
                elif action.name == "run_cmd":
                    cmd = " ".join(action.args.get("cmd", []))
                    if "pytest" in cmd or "ruff" in cmd:
                        subtask.accomplishments.append(f"Ran {cmd}")

        # Extract failed actions as tried approaches
        for action in subtask.actions:
            if action.result == "error":
                approach = f"{action.name}"
                if action.args.get("path"):
                    approach += f" {action.args['path']}"
                elif action.args.get("cmd"):
                    approach += f" {' '.join(action.args['cmd'][:2])}"

                error_short = action.error_msg[:50] if action.error_msg else "unknown error"
                tried = f"{approach}: {error_short}"

                # Avoid duplicates
                if not any(tried[:30] in t for t in subtask.tried_approaches):
                    subtask.tried_approaches.append(tried)

        # Generate context notes summarizing the situation
        if subtask.accomplishments or subtask.tried_approaches:
            notes = []
            if subtask.accomplishments:
                notes.append(f"Completed: {len(subtask.accomplishments)} actions")
            if subtask.tried_approaches:
                notes.append(f"Failed: {len(subtask.tried_approaches)} attempts")
            if subtask.rounds_used > 0:
                notes.append(f"Rounds: {subtask.rounds_used}")
            subtask.context_notes = ", ".join(notes)

    def advance_to_next_subtask(self) -> bool:
        """
        Move to next pending subtask.

        Returns True if there's a next subtask, False if task is complete.
        """
        task = self._get_current_task()
        if not task:
            return False

        next_st = task.next_pending_subtask()
        if next_st:
            next_st.status = "in_progress"
            self._save_state()
            return True

        # No more subtasks - task is complete
        task.status = "completed"
        self._save_state()
        return False

    def get_loop_summary(self) -> str:
        """Get summary of detected loops."""
        if not self.state.loop_counts:
            return "No loops detected."

        lines = ["Detected loops:"]
        for sig, count in sorted(
            self.state.loop_counts.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  - {sig[:60]}: {count} attempts")
        return "\n".join(lines)


# ----------------------------
# Loop Detector
# ----------------------------
class LoopDetector:
    """Detects repeated action patterns that indicate loops."""

    def __init__(self) -> None:
        self.signature_counts: defaultdict[str, int] = defaultdict(int)

    def is_loop(self, action: Action, history: list[Action]) -> bool:
        """
        Check if this action represents a loop.

        A loop is detected if:
        1. Same action signature appears MAX_ACTION_REPEATS+ times
        2. Recent history shows alternating pattern (A→B→A→B)
        3. Action failed with same error multiple times
        """
        sig = action.signature()
        self.signature_counts[sig] += 1

        # Simple repeat check
        if self.signature_counts[sig] >= MAX_ACTION_REPEATS:
            return True

        # Check recent history for same signature
        recent = history[-10:]  # Last 10 actions
        same_sig_count = sum(1 for a in recent if a.signature() == sig)
        if same_sig_count >= MAX_ACTION_REPEATS - 1:
            return True

        # Check for alternating pattern (A-B-A-B)
        if len(recent) >= 4:
            sigs = [a.signature() for a in recent[-4:]]
            if sigs[0] == sigs[2] and sigs[1] == sigs[3] and sigs[0] != sigs[1]:
                return True

        return False


# ----------------------------
# Helper functions
# ----------------------------
def compact_action_list(actions: list[Action], max_actions: int = 5) -> str:
    """Compact action list to string summary."""
    if not actions:
        return "No actions yet."

    recent = actions[-max_actions:]
    lines = []
    for a in recent:
        status = "✓" if a.result == "success" else "✗" if a.result == "error" else "⋯"
        lines.append(f"{status} {a.name}")
    return " → ".join(lines)
