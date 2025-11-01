"""
Jetbox Notes System - Persistent context across task boundaries.

Automatically captures key context at task/goal completion and persists
it in jetboxnotes.md within the goal workspace.

STRATEGY-AGNOSTIC DESIGN:
This module is designed to work with ANY context strategy (Hierarchical,
AppendUntilFull, SubAgent, Architect, etc.) by using generic information:
- action_history: List of Action objects from context_manager (all strategies)
- workspace_manager: For accessing workspace and goal description
- Goal description: Simple string, not hierarchical structure

Summary functions (create_timeout_summary, prompt_for_task_summary, etc.)
do NOT assume hierarchical task trees exist. They extract information from
action_history and workspace context instead.
"""
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
from typing import Any

# Global reference to workspace manager (set by agent at runtime)
_workspace = None
_llm_call_func = None  # Function to call LLM (injected by agent)


def set_workspace(workspace_manager) -> None:
    """Set the workspace manager for notes file access."""
    global _workspace
    _workspace = workspace_manager


def set_llm_caller(llm_func) -> None:
    """Set the LLM calling function."""
    global _llm_call_func
    _llm_call_func = llm_func


def _get_notes_file() -> Path | None:
    """Get the jetbox notes file path."""
    if not _workspace:
        return None
    return _workspace.workspace_dir / "jetboxnotes.md"


def append_to_jetbox_notes(content: str, section: str = "task") -> bool:
    """
    Append content to jetboxnotes.md in workspace.

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
            notes_file.write_text("# Jetbox Notes\n\n", encoding="utf-8")

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format based on section
        if section == "task":
            entry = f"## Task Complete - {timestamp}\n\n{content}\n\n---\n\n"
        elif section == "goal_success":
            entry = f"## ✓ GOAL COMPLETE - {timestamp}\n\n{content}\n\n---\n\n"
        elif section == "goal_failure":
            entry = f"## ✗ GOAL FAILED - {timestamp}\n\n{content}\n\n---\n\n"
        else:
            entry = f"## Note - {timestamp}\n\n{content}\n\n---\n\n"

        # Append
        with notes_file.open("a", encoding="utf-8") as f:
            f.write(entry)

        print(f"[jetbox_notes] Appended {section} summary to jetboxnotes.md")
        return True

    except Exception as e:
        print(f"[jetbox_notes] Error appending to notes: {e}")
        return False


def load_jetbox_notes(max_chars: int = 2000) -> str | None:
    """
    Load jetbox notes from workspace file.

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
        print(f"[jetbox_notes] Error loading notes: {e}")
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
        print(f"[jetbox_notes] Error generating task summary: {e}")
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
    if not _llm_call_func:
        status = "completed" if success else "failed"
        return f"Goal {status}: {goal_description}"

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
        response = _llm_call_func(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=30,
        )

        content = response.get("message", {}).get("content", "")
        if not content:
            status = "succeeded" if success else "failed"
            return f"- Goal {status}: {goal_description}"

        return content.strip()

    except Exception as e:
        print(f"[jetbox_notes] Error generating goal summary: {e}")
        status = "completed" if success else "failed"
        return f"- Goal {status}: {goal_description}\n- (Summary generation timed out)"


def get_notes_summary_for_display() -> str | None:
    """
    Get notes content formatted for console display.

    Returns:
        Formatted notes or None if no notes exist
    """
    content = load_jetbox_notes(max_chars=1000)  # Shorter for display
    if not content:
        return None

    return f"\n{'='*70}\nJETBOX NOTES\n{'='*70}\n{content}\n{'='*70}\n"


def create_timeout_summary(goal=None, elapsed_seconds: float = 0, action_history: list = None) -> None:
    """
    Create a jetbox notes summary when goal times out.

    Generic implementation that works with any context strategy by using
    action_history instead of hierarchical task trees.

    Args:
        goal: Goal object (optional, uses goal.description if provided)
        elapsed_seconds: Total elapsed time
        action_history: List of Action objects from context manager (optional)
    """
    if not _workspace or not _llm_call_func:
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
        response = _llm_call_func(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=30,
        )
        summary = response.get("message", {}).get("content", "")

        # Append to notes
        timeout_header = f"## TIMEOUT ({elapsed_seconds:.0f}s)"
        append_to_jetbox_notes(f"{timeout_header}\n{summary}", "timeout")

        print(f"[jetbox] Created timeout summary ({len(summary)} chars)")

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
        append_to_jetbox_notes(fallback, "timeout")
        print(f"[jetbox] Created fallback timeout summary (LLM failed: {e})")


# For testing/debugging
def _test_notes_system():
    """Test the notes system with mock data."""
    print("Testing Jetbox Notes System...")

    # Test append
    append_to_jetbox_notes("- Created project structure\n- Added main.py", "task")
    append_to_jetbox_notes("- Implemented feature X\n- Tests passing", "task")
    append_to_jetbox_notes(
        "- Built complete system\n- All tests pass\n- Ready for use",
        "goal_success"
    )

    # Test load
    notes = load_jetbox_notes()
    if notes:
        print(f"\nLoaded notes:\n{notes}")
    else:
        print("No notes loaded")

    print("\nTest complete!")


if __name__ == "__main__":
    # For standalone testing
    import tempfile
    from pathlib import Path

    class MockWorkspace:
        def __init__(self):
            self.workspace_dir = Path(tempfile.mkdtemp())

    set_workspace(MockWorkspace())
    _test_notes_system()
