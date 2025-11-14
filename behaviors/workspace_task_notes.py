"""
WorkspaceTaskNotesBehavior - Persistent context summaries across task boundaries.

This behavior provides auto-summarization and context persistence functionality
by managing workspace_task_notes.md files within agent workspaces.

Features:
- Event: on_goal_start(agent, goal) - Initialize workspace and file snapshots
- Event: on_initial_context(agent, context) - Load existing notes once at start
- Event: on_goal_complete(agent, success, summary) - Generate changelog-style summaries
- Event: on_timeout(agent, elapsed_seconds) - Generate timeout summaries
- No tools (utility behavior)

All implementation is self-contained in this module.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Any
from behaviors.base import AgentBehavior
from behaviors.rule_of_two_types import RuleOfTwoProperty

# ============================================================================
# UTILITY FUNCTIONS (workspace task notes implementation)
# ============================================================================

def _get_notes_file(workspace_manager) -> Path | None:
    """Get the workspace task notes file path."""
    if not workspace_manager:
        return None
    # Store in .agent_context for workspace-specific state
    context_dir = workspace_manager.workspace_dir / ".agent_context"
    context_dir.mkdir(exist_ok=True)
    return context_dir / "workspace_task_notes.md"


def append_to_notes(content: str, section: str = "task", workspace_manager=None) -> bool:
    """
    Append content to workspace_task_notes.md in workspace.

    Args:
        content: Text to append (markdown formatted)
        section: Type of entry ("task", "goal_success", "goal_failure")
        workspace_manager: WorkspaceManager instance for file access

    Returns:
        True if successful, False otherwise
    """
    notes_file = _get_notes_file(workspace_manager)
    if not notes_file:
        return False

    try:
        # Create file with header if doesn't exist
        if not notes_file.exists():
            notes_file.write_text("# Workspace Task Notes\n\n", encoding="utf-8")

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format based on section (matter-of-fact, not celebratory)
        if section == "task":
            entry = f"## Task marked done - {timestamp}\n\n{content}\n\n---\n\n"
        elif section == "goal_success":
            entry = f"## Goal marked done - {timestamp}\n\n{content}\n\n---\n\n"
        elif section == "goal_failure":
            entry = f"## Goal marked failed - {timestamp}\n\n{content}\n\n---\n\n"
        elif section == "timeout":
            entry = f"## Agent timeout - {timestamp}\n\n{content}\n\n---\n\n"
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


def load_notes(max_chars: int = 2000, workspace_manager=None) -> str | None:
    """
    Load workspace task notes from workspace file.

    Args:
        max_chars: Maximum characters to return (tail of file if larger)

    Returns:
        Notes content or None if file doesn't exist
    """
    notes_file = _get_notes_file(workspace_manager)
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

    Note: This function is currently unused. It was part of the original
    workspace task notes implementation but has been superseded by the
    changelog-style summaries in onGoalComplete.
    """
    prompt = f"""You just completed this task: "{task_description}"

Briefly summarize what was accomplished in 2-4 bullet points. Be specific and factual:
- What was built/created/modified
- Key decisions made or approaches used
- Important files, functions, or resources created

Keep it concise - focus on facts that future tasks might need to know.

Format: Use bullet points starting with "-"."""

    try:
        from utils.llm_utils import summarize_with_llm

        content = summarize_with_llm(
            prompt=prompt,
            model="gpt-oss:20b",
            temperature=0.2,
        )

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
        from utils.llm_utils import summarize_with_llm

        content = summarize_with_llm(
            prompt=prompt,
            model="gpt-oss:20b",
            temperature=0.2,
        )

        if not content:
            status = "succeeded" if success else "failed"
            return f"- Goal {status}: {goal_description}"

        return content.strip()

    except Exception as e:
        print(f"[workspace_task_notes] Error generating goal summary: {e}")
        status = "completed" if success else "failed"
        return f"- Goal {status}: {goal_description}\n- (Summary generation timed out)"


def get_notes_summary_for_display(workspace_manager=None) -> str | None:
    """
    Get notes content formatted for console display.

    Returns:
        Formatted notes or None if no notes exist
    """
    content = load_notes(max_chars=1000, workspace_manager=workspace_manager)  # Shorter for display
    if not content:
        return None

    return f"\n{'='*70}\nWORKSPACE TASK NOTES\n{'='*70}\n{content}\n{'='*70}\n"


def create_timeout_summary(goal=None, elapsed_seconds: float = 0, action_history: list = None, workspace_manager=None) -> None:
    """
    Create a workspace task notes summary when goal times out.

    Generic implementation that works with any context strategy by using
    action_history instead of hierarchical task trees.

    Args:
        goal: Goal description string or goal object (optional, uses goal.description if object)
        elapsed_seconds: Total elapsed time
        action_history: List of Action objects from context manager (optional)
        workspace_manager: WorkspaceManager instance for file access
    """
    if not workspace_manager:
        return

    # Extract goal description (handle both string and object)
    if isinstance(goal, str):
        goal_description = goal
    elif goal and hasattr(goal, 'description'):
        goal_description = goal.description
    else:
        goal_description = workspace_manager.goal if workspace_manager else "Unknown goal"

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
            "- Actions by tool:",
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
        from utils.llm_utils import summarize_with_llm

        summary = summarize_with_llm(
            prompt=prompt,
            model="gpt-oss:20b",
            temperature=0.2,
        )

        # Append to notes
        timeout_header = f"## TIMEOUT ({elapsed_seconds:.0f}s)"
        append_to_notes(f"{timeout_header}\n{summary}", "timeout", workspace_manager=workspace_manager)

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
        append_to_notes(fallback, "timeout", workspace_manager=workspace_manager)
        print(f"[workspace_task_notes] Created fallback timeout summary (LLM failed: {e})")


# ============================================================================
# BEHAVIOR CLASS
# ============================================================================

class WorkspaceTaskNotesBehavior(AgentBehavior):
    """
    Behavior that provides persistent workspace task notes (context summaries).

    Automatically:
    - Loads existing notes on initial context (once at start)
    - Captures file snapshots on goal start for change tracking
    - Creates changelog-style summaries on goal completion/failure
    - Creates timeout summaries when agent times out
    - Persists summaries to workspace_task_notes.md in workspace

    This is a utility behavior (no tools) that integrates the
    workspace task notes system with the behavior framework.

    Lifecycle:
    1. on_goal_start(agent, goal) - Initialize workspace and snapshots
    2. on_initial_context(agent, context) - Load existing notes once
    3. on_goal_complete(agent, success, summary) - Generate summaries
    4. on_timeout(agent, elapsed_seconds) - Generate timeout summaries

    Security: [] None (reads/writes agent-generated notes, not user data)
    """

    # Rule of Two: Empty (utility behavior for context management)
    rule_of_two_properties = set()

    def __init__(self, **kwargs):
        """
        Initialize workspace task notes behavior.

        Accepts any config parameters for forward compatibility.
        Common params: enabled (bool)
        """
        self.workspace_manager = None
        self.notes_content = None
        self.initial_files = None  # Snapshot of files at goal start for change tracking

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "workspace_task_notes"

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Load existing notes and inject into context ONCE at start.

        Warns if notes exceed 10% of max context.

        Args:
            agent: Agent instance
            context: Initial context (system prompt only)

        Returns:
            Modified context with notes injected (if notes exist)
        """
        # Get workspace_manager from agent
        if hasattr(agent, "workspace_manager"):
            workspace_manager = agent.workspace_manager
            if not self.workspace_manager:
                self.workspace_manager = workspace_manager
        else:
            workspace_manager = self.workspace_manager

        # Load notes (cached)
        if self.notes_content is None:
            self.notes_content = load_notes(max_chars=2000, workspace_manager=workspace_manager)

        # Inject notes into context if they exist
        if self.notes_content and len(context) > 0:
            # Check if notes are too large (warning threshold: 10% of max context)
            max_tokens = self._get_max_tokens(agent)
            if max_tokens:
                # Estimate tokens (chars / 4 is rough heuristic)
                notes_tokens = len(self.notes_content) // 4
                threshold_tokens = max_tokens * 0.10

                if notes_tokens > threshold_tokens:
                    pct = (notes_tokens / max_tokens) * 100
                    print(f"⚠️  Workspace task notes file is {pct:.1f}% of max context ({notes_tokens}/{max_tokens} tokens)")

            # Insert after system prompt with clear delimiters
            notes_with_delimiters = (
                "<Begin Reading Workspace Task Notes File>\n\n"
                f"{self.notes_content}\n\n"
                "<End Reading Workspace Task Notes File>"
            )
            context = self.inject_user_message_after_system(context, notes_with_delimiters)

        return context

    def _get_max_tokens(self, agent: Any) -> int | None:
        """
        Get max_tokens from agent.

        Tries to extract max_tokens from various agent attributes:
        - CompactWhenNearFullBehavior in agent.behaviors
        - Core goal tracking in BaseAgent
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

    def _get_snapshot_file(self, workspace_manager=None) -> Path | None:
        """Get path to the file snapshot JSON file."""
        if not workspace_manager:
            return None
        # Store in .agent_context for workspace-specific state
        context_dir = workspace_manager.workspace_dir / ".agent_context"
        context_dir.mkdir(exist_ok=True)
        return context_dir / "wtn_file_snapshot.json"

    def _save_snapshot(self, snapshot: dict[str, dict[str, Any]], workspace_manager=None) -> None:
        """Save file snapshot to workspace for future reuse."""
        snapshot_file = self._get_snapshot_file(workspace_manager)
        if not snapshot_file:
            return

        try:
            import json
            snapshot_file.write_text(json.dumps(snapshot, indent=2))
            print(f"[workspace_task_notes] Saved initial snapshot: {len(snapshot)} files")
        except Exception as e:
            print(f"[workspace_task_notes] Failed to save snapshot: {e}")

    def _get_workspace_files(self, workspace_manager=None) -> dict[str, dict[str, Any]]:
        """
        Get snapshot of workspace files with metadata.

        Args:
            workspace_manager: WorkspaceManager instance

        Returns:
            Dict mapping file path -> {size, mtime} for all workspace files
        """
        import os

        files = {}
        wm = workspace_manager or self.workspace_manager
        if not wm:
            return files

        workspace_path = Path(wm.workspace_dir)
        if not workspace_path.exists():
            return files

        for item in workspace_path.rglob("*"):
            if item.is_file():
                # Skip hidden files, workspace_task_notes, and snapshot file
                if (item.name.startswith('.') or
                    'workspace_task_notes' in item.name or
                    item.name == '.wtn_file_snapshot.json'):
                    continue

                rel_path = str(item.relative_to(workspace_path))
                try:
                    stat = os.stat(item)
                    files[rel_path] = {
                        'size': stat.st_size,
                        'mtime': stat.st_mtime
                    }
                except Exception:
                    # File disappeared or inaccessible
                    pass

        return files

    def _compute_file_changes(self, initial_files: dict, final_files: dict) -> dict[str, list[str]]:
        """
        Compute file changes between initial and final snapshots.

        Args:
            initial_files: Snapshot from goal start
            final_files: Snapshot from goal end

        Returns:
            Dict with keys: created, edited, deleted (each a list of file paths)
        """
        changes = {
            'created': [],
            'edited': [],
            'deleted': []
        }

        if not initial_files:
            # No initial snapshot - all files are created
            changes['created'] = sorted(final_files.keys())
            return changes

        # Find created and edited files
        for path, final_meta in final_files.items():
            if path not in initial_files:
                changes['created'].append(path)
            else:
                initial_meta = initial_files[path]
                # Check if modified (size or mtime changed)
                if (final_meta['size'] != initial_meta['size'] or
                    final_meta['mtime'] != initial_meta['mtime']):
                    changes['edited'].append(path)

        # Find deleted files
        for path in initial_files:
            if path not in final_files:
                changes['deleted'].append(path)

        # Sort for consistent output
        changes['created'] = sorted(changes['created'])
        changes['edited'] = sorted(changes['edited'])
        changes['deleted'] = sorted(changes['deleted'])

        return changes

    def on_goal_start(self, agent: Any, goal: str) -> None:
        """
        Called when goal starts.

        Sets up workspace and LLM caller for notes system.
        Captures initial file snapshot for change tracking.

        Args:
            agent: Agent instance
            goal: The goal string
        """
        # Get workspace manager from agent
        if hasattr(agent, "workspace_manager"):
            workspace_manager = agent.workspace_manager
            self.workspace_manager = workspace_manager
        else:
            workspace_manager = None

        # Clear cached notes to force reload
        self.notes_content = None

        # Load or capture initial file snapshot for change tracking
        snapshot_file = self._get_snapshot_file(workspace_manager)
        if snapshot_file and snapshot_file.exists():
            # Workspace reuse - load existing snapshot
            import json
            try:
                self.initial_files = json.loads(snapshot_file.read_text())
                print(f"[workspace_task_notes] Loaded initial snapshot: {len(self.initial_files)} files")
            except Exception as e:
                print(f"[workspace_task_notes] Failed to load snapshot: {e}")
                self.initial_files = self._get_workspace_files(workspace_manager)
        else:
            # New workspace - capture initial snapshot
            self.initial_files = self._get_workspace_files(workspace_manager)
            # Save snapshot for future reuse
            self._save_snapshot(self.initial_files, workspace_manager)

    def on_goal_complete(self, agent: Any, success: bool, summary: str) -> None:
        """
        Called when goal completes.

        Generates and saves changelog-style summary to workspace_task_notes.md.

        Args:
            agent: Agent instance
            success: True if goal succeeded, False if failed
            summary: Summary message (from mark_complete or mark_failed)
        """
        # Get workspace manager from agent
        if hasattr(agent, "workspace_manager"):
            workspace_manager = agent.workspace_manager
            self.workspace_manager = workspace_manager
        else:
            workspace_manager = self.workspace_manager

        # Get goal description from agent
        goal_description = agent.goal if hasattr(agent, "goal") else "Unknown goal"

        # Use summary as reason for failures
        reason = summary if not success else ""

        # Get final file snapshot
        final_files = self._get_workspace_files(workspace_manager)

        # Compute file changes
        changes = self._compute_file_changes(self.initial_files or {}, final_files)

        # Find documentation files
        doc_files = []
        for path in final_files.keys():
            lower_path = path.lower()
            if any(doc_marker in lower_path for doc_marker in ['readme', 'doc', 'guide', '.md']):
                doc_files.append(path)

        # Build changelog-style summary
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "marked done" if success else "marked failed"

        summary_lines = [
            f"Timestamp: {timestamp}",
            f"Goal: {goal_description}",
            f"Status: {status}",
            ""
        ]

        # File changes section
        has_changes = any(changes.values())
        if has_changes:
            summary_lines.append("File Changes:")

            if changes['created']:
                summary_lines.append("  Created:")
                for path in changes['created']:
                    summary_lines.append(f"    - {path}")

            if changes['edited']:
                summary_lines.append("  Edited:")
                for path in changes['edited']:
                    summary_lines.append(f"    - {path}")

            if changes['deleted']:
                summary_lines.append("  Deleted:")
                for path in changes['deleted']:
                    summary_lines.append(f"    - {path}")

            summary_lines.append("")
        else:
            summary_lines.append("File Changes: none")
            summary_lines.append("")

        # Documentation pointers
        if doc_files:
            summary_lines.append("Documentation:")
            for doc in sorted(doc_files):
                summary_lines.append(f"  - {doc}")
            summary_lines.append("")

        # Failure reason (if applicable)
        if not success and reason:
            summary_lines.append(f"Failure Reason:")
            summary_lines.append(f"  {reason}")
            summary_lines.append("")

        # Total file count
        total_files = len(final_files)
        summary_lines.append(f"Workspace Files: {total_files} total")

        summary = "\n".join(summary_lines)

        # Append to notes with matter-of-fact section header
        section = "goal_success" if success else "goal_failure"
        append_to_notes(summary, section=section, workspace_manager=workspace_manager)

        # Update snapshot for next round (save final state as new baseline)
        # This ensures subsequent rounds can track Created vs Edited correctly
        if success:
            self._save_snapshot(final_files, workspace_manager)

        # Print summary for console (matter-of-fact)
        print("\n" + "="*70)
        if success:
            print("Goal marked done")
        else:
            print("Goal marked failed")
        print("="*70)
        print(summary)
        print("="*70 + "\n")

    def on_timeout(self, agent: Any, elapsed_seconds: float) -> None:
        """
        Called when goal times out.

        Generates and saves timeout summary to workspace_task_notes.md.

        Args:
            agent: Agent instance
            elapsed_seconds: Time elapsed since goal start
        """
        # Get workspace manager from agent
        if hasattr(agent, "workspace_manager"):
            workspace_manager = agent.workspace_manager
            self.workspace_manager = workspace_manager
        else:
            workspace_manager = self.workspace_manager

        # Get goal and action history from agent
        goal = agent.goal if hasattr(agent, "goal") else None
        action_history = None
        if hasattr(agent, "state") and hasattr(agent.state, "action_history"):
            action_history = agent.state.action_history

        # Generate timeout summary via module functions
        create_timeout_summary(
            goal=goal,
            elapsed_seconds=elapsed_seconds,
            action_history=action_history,
            workspace_manager=workspace_manager
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
