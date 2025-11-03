"""
WorkspaceTaskNotesBehavior - Persistent context summaries across task boundaries.

This behavior provides auto-summarization and context persistence functionality
by managing workspace_task_notes.md files within agent workspaces.

Features:
- Event: on_goal_complete(success, **kwargs)
- Event: on_timeout(elapsed_seconds, **kwargs)
- Context enhancement: loads existing notes
- No tools (utility behavior)

All implementation is self-contained in this module.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Any
from behaviors.base import AgentBehavior

# Module-level state for workspace notes system
_workspace = None  # Global reference to workspace manager (set by behavior at runtime)


# ============================================================================
# UTILITY FUNCTIONS (workspace task notes implementation)
# ============================================================================

def set_workspace(workspace_manager) -> None:
    """Set the workspace manager for notes file access."""
    global _workspace
    _workspace = workspace_manager


def _get_notes_file() -> Path | None:
    """Get the workspace task notes file path."""
    if not _workspace:
        return None
    return _workspace.workspace_dir / "workspace_task_notes.md"


def append_to_notes(content: str, section: str = "task") -> bool:
    """
    Append content to workspace_task_notes.md in workspace.

    Args:
        content: Text to append (markdown formatted)
        section: Type of entry ("task", "goal_success", "goal_failure")

    Returns:
        True if successful, False otherwise
    """
    notes_file = _get_notes_file()
    if not notes_file:
        return False

    try:
        # Create file with header if doesn't exist
        if not notes_file.exists():
            notes_file.write_text("# Workspace Task Notes\n\n", encoding="utf-8")

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format based on section
        if section == "task":
            entry = f"## Task Complete - {timestamp}\n\n{content}\n\n---\n\n"
        elif section == "goal_success":
            entry = f"## ✓ GOAL COMPLETE - {timestamp}\n\n{content}\n\n---\n\n"
        elif section == "goal_failure":
            entry = f"## ✗ GOAL FAILED - {timestamp}\n\n{content}\n\n---\n\n"
        elif section == "timeout":
            entry = f"## ⏱ TIMEOUT - {timestamp}\n\n{content}\n\n---\n\n"
        else:
            entry = f"## Note - {timestamp}\n\n{content}\n\n---\n\n"

        # Append
        with notes_file.open("a", encoding="utf-8") as f:
            f.write(entry)

        print(f"[workspace_task_notes] Appended {section} summary to workspace_task_notes.md")
        return True

    except Exception as e:
        print(f"[workspace_task_notes] Error appending to notes: {e}")
        return False


def load_notes(max_chars: int = 2000) -> str | None:
    """
    Load workspace task notes from workspace file.

    Args:
        max_chars: Maximum characters to return (tail of file if larger)

    Returns:
        Notes content or None if file doesn't exist
    """
    notes_file = _get_notes_file()
    if not notes_file or not notes_file.exists():
        return None

    try:
        content = notes_file.read_text(encoding="utf-8")

        # Truncate if too large (keep tail - most recent)
        if len(content) > max_chars:
            content = "[... earlier notes truncated ...]\n\n" + content[-max_chars:]

        return content

    except Exception as e:
        print(f"[workspace_task_notes] Error loading notes: {e}")
        return None


def prompt_for_task_summary(task_description: str) -> str:
    """
    Prompt agent to summarize completed task.

    STRATEGY-AGNOSTIC: Takes only a task description string, works with any
    context strategy.

    Args:
        task_description: Description of the completed task

    Returns:
        Summary text from agent
    """
    if not _llm_call_func:
        return f"Task completed: {task_description}\n(Summary generation not available)"

    prompt = f"""You just completed this task: "{task_description}"

Briefly summarize what was accomplished in 2-4 bullet points. Be specific and factual:
- What was built/created/modified
- Key decisions made or approaches used
- Important files, functions, or resources created

Keep it concise - focus on facts that future tasks might need to know.

Format: Use bullet points starting with "-"."""

    try:
        response = _llm_call_func(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Low temperature for factual summary
            timeout=30,
        )

        content = response.get("message", {}).get("content", "")
        if not content:
            return f"- Completed: {task_description}"

        return content.strip()

    except Exception as e:
        print(f"[workspace_task_notes] Error generating task summary: {e}")
        return f"- Completed: {task_description}\n- (Summary generation timed out)"


def prompt_for_goal_summary(
    goal_description: str,
    success: bool,
    reason: str = "",
    task_summaries: list[str] = None,
) -> str:
    """
    Prompt agent to summarize goal completion or failure.

    STRATEGY-AGNOSTIC: Takes only string descriptions, works with any
    context strategy. task_summaries is optional generic context.

    Args:
        goal_description: Description of the goal
        success: True if goal succeeded, False if failed
        reason: Reason for failure (if applicable)
        task_summaries: Optional list of task summaries for context

    Returns:
        Summary text from agent
    """
    if success:
        prompt = f"""Goal completed successfully: "{goal_description}"

Provide a concise final summary (3-6 bullet points):
- What was accomplished overall
- Key features/components created
- Important files or entry points
- Any critical decisions or approaches
- Suggestions for next steps or improvements (if any)

Be specific and factual. Focus on what matters for someone continuing this work.

Format: Use bullet points starting with "-"."""

        # Add context from task summaries if available
        if task_summaries:
            task_context = "\n".join(f"  • {summary}" for summary in task_summaries)
            prompt += f"\n\nTask summaries for context:\n{task_context}"

    else:
        prompt = f"""Goal failed: "{goal_description}"
Reason: {reason}

Provide a concise failure summary (3-5 bullet points):
- What was attempted
- How far did progress get
- What blocked or prevented completion
- What was learned or discovered
- Suggestions for retry or alternative approach

Be specific and factual. Help someone understand what happened.

Format: Use bullet points starting with "-"."""

    try:
        # Call LLM directly with ollama.chat (one-shot, no context needed)
        from ollama import chat

        response = chat(
            model="gpt-oss:20b",  # Default model
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )

        content = response.get("message", {}).get("content", "")
        if not content:
            status = "succeeded" if success else "failed"
            return f"- Goal {status}: {goal_description}"

        return content.strip()

    except Exception as e:
        print(f"[workspace_task_notes] Error generating goal summary: {e}")
        status = "completed" if success else "failed"
        return f"- Goal {status}: {goal_description}\n- (Summary generation timed out)"


def get_notes_summary_for_display() -> str | None:
    """
    Get notes content formatted for console display.

    Returns:
        Formatted notes or None if no notes exist
    """
    content = load_notes(max_chars=1000)  # Shorter for display
    if not content:
        return None

    return f"\n{'='*70}\nWORKSPACE TASK NOTES\n{'='*70}\n{content}\n{'='*70}\n"


def create_timeout_summary(goal=None, elapsed_seconds: float = 0, action_history: list = None) -> None:
    """
    Create a workspace task notes summary when goal times out.

    Generic implementation that works with any context strategy by using
    action_history instead of hierarchical task trees.

    Args:
        goal: Goal object (optional, uses goal.description if provided)
        elapsed_seconds: Total elapsed time
        action_history: List of Action objects from context manager (optional)
    """
    if not _workspace:
        return

    # Extract goal description
    goal_description = goal.description if goal else _workspace.goal

    # Build action summary from action_history (strategy-agnostic)
    if action_history:
        # Count successful vs failed actions
        total_actions = len(action_history)
        successful_actions = sum(1 for a in action_history if a.result == "success")
        failed_actions = sum(1 for a in action_history if a.result == "error")

        # Get recent actions (last 10)
        recent_actions = action_history[-10:] if len(action_history) > 10 else action_history

        # Group actions by tool type
        actions_by_tool = {}
        for action in action_history:
            tool = action.name
            if tool not in actions_by_tool:
                actions_by_tool[tool] = {"success": 0, "error": 0, "total": 0}
            actions_by_tool[tool]["total"] += 1
            if action.result == "success":
                actions_by_tool[tool]["success"] += 1
            elif action.result == "error":
                actions_by_tool[tool]["error"] += 1

        # Find last action
        last_action = action_history[-1] if action_history else None

        # Build progress context
        progress_lines = [
            f"- Total actions: {total_actions} (success: {successful_actions}, failed: {failed_actions})",
            f"- Actions by tool:",
        ]
        for tool, counts in actions_by_tool.items():
            progress_lines.append(f"  • {tool}: {counts['total']} total ({counts['success']} success, {counts['error']} failed)")

        progress_context = "\n".join(progress_lines)

        # Build recent actions context
        recent_context = "RECENT ACTIONS (last 10):\n"
        for i, action in enumerate(recent_actions, 1):
            status = "✓" if action.result == "success" else "✗" if action.result == "error" else "?"
            args_preview = str(action.args)[:50]
            recent_context += f"{i}. {status} {action.name}({args_preview}...)\n"
            if action.error_msg:
                recent_context += f"   Error: {action.error_msg[:100]}\n"

        last_action_context = f"- Last action: {last_action.name} ({last_action.result})" if last_action else "- No actions recorded"
    else:
        # No action history - basic timeout summary
        progress_context = "- No action history available"
        recent_context = ""
        last_action_context = "- No actions recorded"

    # Build prompt for LLM to create summary (strategy-agnostic)
    prompt = f"""The agent timed out after {elapsed_seconds:.1f} seconds working on this goal.

GOAL: {goal_description}

PROGRESS:
{progress_context}

LAST ACTION:
{last_action_context}

{recent_context}

Please write a concise summary (3-5 bullet points) covering:
1. What was successfully accomplished before timeout (based on actions)
2. What was being worked on when timeout occurred (last actions)
3. What blocking issue or complexity caused the timeout (based on errors)
4. Suggested next steps if retrying

Format: Dense bullets focused on facts."""

    # Get LLM summary
    try:
        # Call LLM directly with ollama.chat (one-shot, no context needed)
        from ollama import chat

        response = chat(
            model="gpt-oss:20b",  # Default model
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        summary = response.get("message", {}).get("content", "")

        # Append to notes
        timeout_header = f"## TIMEOUT ({elapsed_seconds:.0f}s)"
        append_to_notes(f"{timeout_header}\n{summary}", "timeout")

        print(f"[workspace_task_notes] Created timeout summary ({len(summary)} chars)")

    except Exception as e:
        # Fallback to basic summary (strategy-agnostic)
        fallback_lines = [
            f"## TIMEOUT ({elapsed_seconds:.0f}s)",
            f"- Goal: {goal_description}",
        ]

        if action_history:
            total = len(action_history)
            success = sum(1 for a in action_history if a.result == "success")
            fallback_lines.append(f"- Actions: {total} total ({success} successful)")
            if action_history:
                last = action_history[-1]
                fallback_lines.append(f"- Last action: {last.name} ({last.result})")
        else:
            fallback_lines.append("- No action history available")

        fallback_lines.append("- Summary generation failed - see action history for details")

        fallback = "\n".join(fallback_lines)
        append_to_notes(fallback, "timeout")
        print(f"[workspace_task_notes] Created fallback timeout summary (LLM failed: {e})")


# ============================================================================
# BEHAVIOR CLASS
# ============================================================================

class WorkspaceTaskNotesBehavior(AgentBehavior):
    """
    Behavior that provides persistent workspace task notes (context summaries).

    Automatically:
    - Loads existing notes on context enhancement
    - Creates summaries on goal completion/failure
    - Creates timeout summaries when agent times out
    - Persists summaries to workspace_task_notes.md in workspace

    This is a utility behavior (no tools) that integrates the
    workspace task notes system with the behavior framework.
    """

    def __init__(self, **kwargs):
        """
        Initialize workspace task notes behavior.

        Accepts any config parameters for forward compatibility.
        Common params: enabled (bool)
        """
        self.workspace_manager = None
        self.notes_content = None

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "workspace_task_notes"

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Load existing notes and inject into context.

        Warns if notes exceed 10% of max context.

        Args:
            context: Current context
            **kwargs: Additional context (agent, workspace, etc.)

        Returns:
            Modified context with notes injected (if notes exist)
        """
        # Set workspace if provided
        if "workspace_manager" in kwargs and not self.workspace_manager:
            self.workspace_manager = kwargs["workspace_manager"]
            set_workspace(self.workspace_manager)

        # Load notes (cached)
        if self.notes_content is None:
            self.notes_content = load_notes(max_chars=2000)

        # Inject notes into context if they exist
        if self.notes_content and len(context) > 0:
            # Check if notes are too large (warning threshold: 10% of max context)
            agent = kwargs.get("agent")
            if agent:
                max_tokens = self._get_max_tokens(agent)
                if max_tokens:
                    # Estimate tokens (chars / 4 is rough heuristic)
                    notes_tokens = len(self.notes_content) // 4
                    threshold_tokens = max_tokens * 0.10

                    if notes_tokens > threshold_tokens:
                        pct = (notes_tokens / max_tokens) * 100
                        print(f"⚠️  Workspace task notes file is {pct:.1f}% of max context ({notes_tokens}/{max_tokens} tokens)")

            # Insert after system prompt (index 1)
            notes_message = {
                "role": "user",
                "content": f"## Previous Context (from workspace task notes)\n\n{self.notes_content}"
            }
            context.insert(1, notes_message)

        return context

    def _get_max_tokens(self, agent: Any) -> int | None:
        """
        Get max_tokens from agent.

        Tries to extract max_tokens from various agent attributes:
        - CompactWhenNearFullBehavior in agent.behaviors
        - SubAgentContextBehavior in agent.behaviors
        - agent.token_threshold (orchestrator)
        - agent.context_window (orchestrator)

        Args:
            agent: Agent instance

        Returns:
            Max tokens or None if not found
        """
        # Try to find context behavior with max_tokens
        for behavior in getattr(agent, "behaviors", []):
            behavior_name = behavior.get_name()
            if behavior_name in ["compact_when_near_full", "subagent_context", "hierarchical_context"]:
                max_tokens = getattr(behavior, "max_tokens", None)
                if max_tokens:
                    return max_tokens

        # Try orchestrator attributes
        if hasattr(agent, "context_window"):
            return agent.context_window

        # Try token_threshold (orchestrator fallback)
        if hasattr(agent, "token_threshold"):
            # token_threshold is 75% of context_window, so estimate full context
            return int(agent.token_threshold / 0.75)

        return None

    def on_goal_start(self, goal: str, **kwargs: Any) -> None:
        """
        Called when goal starts.

        Sets up workspace and LLM caller for notes system.

        Args:
            goal: The goal string
            **kwargs: Additional context (workspace_manager, llm_call_func, etc.)
        """
        # Set workspace manager
        if "workspace_manager" in kwargs:
            self.workspace_manager = kwargs["workspace_manager"]
            set_workspace(self.workspace_manager)

        # Clear cached notes to force reload
        self.notes_content = None

    def on_goal_complete(self, success: bool, **kwargs: Any) -> None:
        """
        Called when goal completes.

        Generates and saves goal summary to workspace_task_notes.md.

        Args:
            success: True if goal succeeded, False if failed
            **kwargs: Additional context (goal, reason, workspace_manager, llm_call_func, etc.)
        """
        # Set workspace manager from kwargs
        if "workspace_manager" in kwargs:
            self.workspace_manager = kwargs["workspace_manager"]
            set_workspace(self.workspace_manager)

        goal_description = kwargs.get("goal", "Unknown goal")
        reason = kwargs.get("reason", "")
        task_summaries = kwargs.get("task_summaries", None)

        # Generate summary via module functions
        summary = prompt_for_goal_summary(
            goal_description=goal_description,
            success=success,
            reason=reason,
            task_summaries=task_summaries
        )

        # Append to notes
        section = "goal_success" if success else "goal_failure"
        append_to_notes(summary, section=section)

        # Print summary for console
        if success:
            print("\n" + "="*70)
            print("GOAL COMPLETE - Summary:")
            print("="*70)
            print(summary)
            print("="*70 + "\n")
        else:
            print("\n" + "="*70)
            print("GOAL FAILED - Summary:")
            print("="*70)
            print(summary)
            print("="*70 + "\n")

    def on_timeout(self, elapsed_seconds: float, **kwargs: Any) -> None:
        """
        Called when goal times out.

        Generates and saves timeout summary to workspace_task_notes.md.

        Args:
            elapsed_seconds: Time elapsed since goal start
            **kwargs: Additional context (goal, action_history, workspace_manager, llm_call_func, etc.)
        """
        # Set workspace manager from kwargs
        if "workspace_manager" in kwargs:
            self.workspace_manager = kwargs["workspace_manager"]
            set_workspace(self.workspace_manager)

        goal = kwargs.get("goal", None)
        action_history = kwargs.get("action_history", None)

        # Generate timeout summary via module functions
        create_timeout_summary(
            goal=goal,
            elapsed_seconds=elapsed_seconds,
            action_history=action_history
        )

    def get_instructions(self) -> str:
        """
        Return workspace task notes instructions.

        Returns:
            Instructions about persistent context summaries
        """
        return """
WORKSPACE TASK NOTES:
Your work is automatically summarized and persisted to workspace task notes (workspace_task_notes.md).

Context persistence:
- Task summaries are created when tasks complete
- Goal summaries are created on completion/failure
- Timeout summaries are created if time runs out
- Notes are loaded automatically on subsequent runs

The system handles this automatically - you don't need to manage it manually.
"""
