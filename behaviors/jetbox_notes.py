"""
JetboxNotesBehavior - Persistent context summaries across task boundaries.

This behavior wraps the jetbox_notes module to provide auto-summarization
and context persistence functionality through the AgentBehavior interface.

Features:
- Event: on_goal_complete(success, **kwargs)
- Event: on_timeout(elapsed_seconds, **kwargs)
- Context enhancement: loads existing notes
- No tools (utility behavior)

The behavior delegates to the jetbox_notes module for actual implementation.
"""

from typing import Any
from behaviors.base import AgentBehavior
import jetbox_notes


class JetboxNotesBehavior(AgentBehavior):
    """
    Behavior that provides persistent context summaries.

    Automatically:
    - Loads existing notes on context enhancement
    - Creates summaries on goal completion/failure
    - Creates timeout summaries when agent times out
    - Persists summaries to jetboxnotes.md in workspace

    This is a utility behavior (no tools) that integrates the
    jetbox_notes system with the behavior framework.
    """

    def __init__(self):
        """Initialize jetbox notes behavior."""
        self.workspace_manager = None
        self.llm_call_func = None
        self.notes_content = None

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "jetbox_notes"

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Load existing notes and inject into context.

        Args:
            context: Current context
            **kwargs: Additional context (agent, workspace, etc.)

        Returns:
            Modified context with notes injected (if notes exist)
        """
        # Set workspace if provided
        if "workspace_manager" in kwargs and not self.workspace_manager:
            self.workspace_manager = kwargs["workspace_manager"]
            jetbox_notes.set_workspace(self.workspace_manager)

        # Set LLM caller if provided
        if "llm_call_func" in kwargs and not self.llm_call_func:
            self.llm_call_func = kwargs["llm_call_func"]
            jetbox_notes.set_llm_caller(self.llm_call_func)

        # Load notes (cached)
        if self.notes_content is None:
            self.notes_content = jetbox_notes.load_jetbox_notes(max_chars=2000)

        # Inject notes into context if they exist
        if self.notes_content and len(context) > 0:
            # Insert after system prompt (index 1)
            notes_message = {
                "role": "user",
                "content": f"## Previous Context (from jetboxnotes.md)\n\n{self.notes_content}"
            }
            context.insert(1, notes_message)

        return context

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
            jetbox_notes.set_workspace(self.workspace_manager)

        # Set LLM caller
        if "llm_call_func" in kwargs:
            self.llm_call_func = kwargs["llm_call_func"]
            jetbox_notes.set_llm_caller(self.llm_call_func)

        # Clear cached notes to force reload
        self.notes_content = None

    def on_goal_complete(self, success: bool, **kwargs: Any) -> None:
        """
        Called when goal completes.

        Generates and saves goal summary to jetboxnotes.md.

        Args:
            success: True if goal succeeded, False if failed
            **kwargs: Additional context (goal, reason, etc.)
        """
        goal_description = kwargs.get("goal", "Unknown goal")
        reason = kwargs.get("reason", "")
        task_summaries = kwargs.get("task_summaries", None)

        # Generate summary via jetbox_notes module
        summary = jetbox_notes.prompt_for_goal_summary(
            goal_description=goal_description,
            success=success,
            reason=reason,
            task_summaries=task_summaries
        )

        # Append to notes
        section = "goal_success" if success else "goal_failure"
        jetbox_notes.append_to_jetbox_notes(summary, section=section)

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

        Generates and saves timeout summary to jetboxnotes.md.

        Args:
            elapsed_seconds: Time elapsed since goal start
            **kwargs: Additional context (goal, action_history, etc.)
        """
        goal = kwargs.get("goal", None)
        action_history = kwargs.get("action_history", None)

        # Generate timeout summary via jetbox_notes module
        jetbox_notes.create_timeout_summary(
            goal=goal,
            elapsed_seconds=elapsed_seconds,
            action_history=action_history
        )

    def get_instructions(self) -> str:
        """
        Return jetbox notes instructions.

        Returns:
            Instructions about persistent context summaries
        """
        return """
JETBOX NOTES:
Your work is automatically summarized and persisted to jetboxnotes.md in the workspace.

Context persistence:
- Task summaries are created when tasks complete
- Goal summaries are created on completion/failure
- Timeout summaries are created if time runs out
- Notes are loaded automatically on subsequent runs

The system handles this automatically - you don't need to manage it manually.
"""
