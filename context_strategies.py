"""
Context building strategies for Jetbox agents.

Provides different context management strategies:
- Hierarchical: For task-focused agents (TaskExecutor, standalone agent)
- Append-until-full: For conversational agents (Orchestrator)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from context_manager import ContextManager
    from agent_config import AgentConfig


class ContextEnhancement(ABC):
    """
    Base class for context enhancements.

    Enhancements inject additional context sections, tools, and instructions
    without replacing the core context strategy. Multiple enhancements can
    be composed together.

    Examples: TaskManagement, JetboxNotes, ArchitectDocs
    """

    @abstractmethod
    def get_context_injection(
        self,
        context_manager: Any,
        workspace: Any = None,
        **kwargs
    ) -> dict[str, str] | None:
        """
        Get context to inject into the agent's context.

        Returns:
            Message dict to inject (role: "user", content: "...") or None
        """
        pass

    @abstractmethod
    def get_enhancement_tools(self) -> list[dict[str, Any]]:
        """
        Get tools provided by this enhancement.

        Returns:
            List of tool definitions
        """
        pass

    @abstractmethod
    def get_enhancement_instructions(self) -> str:
        """
        Get instructions for using this enhancement.

        Returns:
            Additional instructions to append to system prompt
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get enhancement name for reporting.

        Returns:
            Enhancement name (e.g., "task_management", "jetbox_notes")
        """
        pass


class ContextStrategy(ABC):
    """Abstract base class for context management strategies."""

    def __init__(self):
        """Initialize strategy with loop detection."""
        # Loop detection is a core context management feature
        self.action_history: list[dict[str, Any]] = []
        self.loop_warnings: list[str] = []
        self.max_action_repeats = 5  # Default, can be overridden

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert an object to a JSON-serializable format.

        Handles common non-serializable types:
        - Objects: convert to string representation
        - Dicts: recursively process values
        - Lists: recursively process items
        - Primitives: pass through

        Args:
            obj: Object to make serializable

        Returns:
            JSON-serializable version of the object
        """
        import json

        # Try direct serialization first (fast path for primitives)
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            pass

        # Handle different types
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        elif isinstance(obj, dict):
            # Recursively process dict values
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # Recursively process list/tuple items
            return [self._make_serializable(item) for item in obj]
        else:
            # Non-serializable object - use string representation
            # Include type name for debugging
            return f"<{type(obj).__name__}>"

    def record_action(self, tool_name: str, args: dict[str, Any], result: Any, success: bool) -> dict[str, Any] | None:
        """
        Record an action and check for loops.

        This is a core context management responsibility - all strategies should detect
        when agents are repeating the same actions and warn them.

        Args:
            tool_name: Name of tool executed
            args: Tool arguments (may contain non-serializable objects)
            result: Tool result (for detecting same-result loops)
            success: Whether tool succeeded

        Returns:
            Loop warning dict if loop detected, None otherwise
            Format: {"warning": "...", "suggestion": "..."}
        """
        import hashlib
        import json

        # Create action signature (tool + args)
        # Filter out non-serializable objects from args before JSON encoding
        serializable_args = self._make_serializable(args)
        args_str = json.dumps(serializable_args, sort_keys=True)
        action_sig = f"{tool_name}::{args_str}"

        # Create result signature (action + result hash for detecting repeated failures)
        result_str = str(result)[:500]  # First 500 chars of result
        result_hash = hashlib.sha256(result_str.encode('utf-8', errors='ignore')).hexdigest()[:16]
        result_sig = f"{action_sig}::{result_hash}"

        # Record action
        self.action_history.append({
            "action_sig": action_sig,
            "result_sig": result_sig,
            "success": success,
            "tool_name": tool_name,
        })

        # Check for loops in recent history (last 20 actions)
        recent = self.action_history[-20:]

        # Count identical action+result pairs
        same_result_count = sum(1 for a in recent if a["result_sig"] == result_sig)

        # Count identical actions (regardless of result)
        same_action_count = sum(1 for a in recent if a["action_sig"] == action_sig)

        # Detect loop
        if same_result_count >= self.max_action_repeats:
            warning = {
                "warning": f"Action repeated {same_result_count} times with identical results",
                "action": tool_name,
                "suggestion": (
                    "This approach isn't working. Consider:\n"
                    "  1. Try a COMPLETELY DIFFERENT approach\n"
                    "  2. Read error messages more carefully\n"
                    "  3. If core task is complete, call mark_complete() even if tests fail\n"
                    "  4. If truly blocked, call mark_failed() with detailed reason"
                )
            }
            self.loop_warnings.append(f"{tool_name} repeated {same_result_count}x")
            return warning

        # Warn about repeated attempts even if results differ slightly
        if same_action_count >= self.max_action_repeats + 2:
            warning = {
                "warning": f"Action attempted {same_action_count} times (results vary)",
                "action": tool_name,
                "suggestion": (
                    "You've tried this many times. Consider:\n"
                    "  1. Is there a pattern to the different results?\n"
                    "  2. Try a fundamentally different approach\n"
                    "  3. Accept 'good enough' and call mark_complete()"
                )
            }
            self.loop_warnings.append(f"{tool_name} attempted {same_action_count}x")
            return warning

        return None

    def get_loop_warnings_context(self) -> str | None:
        """
        Get loop warnings to inject into context.

        Returns:
            Warning text to add to context, or None if no warnings
        """
        if not self.loop_warnings:
            return None

        warnings_text = "⚠️  LOOP DETECTION WARNING:\n"
        warnings_text += "You appear to be repeating actions:\n"
        for warning in self.loop_warnings[-3:]:  # Last 3 warnings
            warnings_text += f"  • {warning}\n"
        warnings_text += "\nConsider trying a different approach or marking task complete if done."

        return warnings_text

    @abstractmethod
    def build_context(
        self,
        context_manager: ContextManager,
        messages: list[dict[str, Any]],
        system_prompt: str,
        config: Any,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Build context for LLM based on strategy.

        Args:
            context_manager: ContextManager with goal/task/subtask hierarchy
            messages: Full message history
            system_prompt: System prompt for the agent
            config: Configuration object
            **kwargs: Strategy-specific parameters

        Returns:
            Context list ready for LLM
        """
        pass

    @abstractmethod
    def should_clear_on_transition(self) -> bool:
        """
        Whether messages should be cleared on subtask transitions.

        Returns:
            True if messages should be cleared, False otherwise
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get strategy name for reporting.

        Returns:
            Strategy name
        """
        pass

    @abstractmethod
    def estimate_context_size(self, context: list[dict[str, Any]]) -> int:
        """
        Estimate context size in tokens.

        Args:
            context: Built context

        Returns:
            Estimated token count
        """
        pass

    def get_strategy_instructions(self) -> str:
        """
        Get strategy-specific instructions to inject into system prompt.

        Override this method to add workflow instructions specific to your strategy.
        Examples:
        - Hierarchical: instructions about decompose_task and mark_subtask_complete
        - Simple: instructions about when task is considered complete

        Returns:
            Additional instructions to append to base system prompt (empty string = no additions)
        """
        return ""

    def get_strategy_tools(self) -> list[dict[str, Any]]:
        """
        Get strategy-specific tools to add to agent's tool list.

        Override this method to provide tools needed by your strategy.
        Examples:
        - Hierarchical: decompose_task, mark_subtask_complete
        - Other strategies: custom completion markers, progress trackers, etc.

        Returns:
            List of tool definitions in Ollama format (empty list = no additional tools)
        """
        return []

    def should_use_jetbox_notes(self) -> bool:
        """
        Whether this strategy should use jetbox notes for context persistence.

        Jetbox notes provide automatic summarization of completed tasks/goals
        and load those summaries in subsequent runs for context continuity.

        Override this to enable/disable jetbox notes per strategy:
        - Hierarchical: typically ON (task-focused, benefits from summaries)
        - Conversational: typically OFF (maintains full conversation history)
        - Orchestrator: typically OFF (delegates all work, no need for notes)

        Returns:
            True to enable jetbox notes, False to disable
        """
        return False  # Default: OFF (strategies opt-in)


class HierarchicalStrategy(ContextStrategy):
    """
    Hierarchical context strategy.

    Keeps only recent N message exchanges and clears on subtask transitions.
    Optimized for focused task execution.

    Features:
    - System prompt + current goal/task/subtask
    - Loop detection warnings
    - Filesystem probe state
    - Jetbox notes enabled by default
    - Last N message exchanges only
    - Clears messages on subtask transitions
    - Provides decompose_task and mark_subtask_complete tools
    - Injects workflow instructions into system prompt
    """

    def __init__(self, history_keep: int = 12, use_jetbox_notes: bool = True):
        """
        Initialize hierarchical strategy.

        Args:
            history_keep: Number of message exchanges to keep in history
            use_jetbox_notes: Whether to use jetbox notes for context persistence (default: True)
        """
        super().__init__()  # Initialize loop detection
        self.history_keep = history_keep
        self.use_jetbox_notes = use_jetbox_notes

    def get_strategy_instructions(self) -> str:
        """
        Hierarchical strategy workflow instructions.

        Returns:
            Instructions for using decompose_task and mark_subtask_complete
        """
        return """
HIERARCHICAL WORKFLOW:
1. If NO TASKS YET: Use decompose_task to break down the goal into a task structure
2. Work on your current subtask using the available tools
3. When you complete a subtask, call mark_subtask_complete(success=True)
4. If you cannot complete it, call mark_subtask_complete(success=False, reason="...")
5. The system will automatically advance you to the next subtask

Hierarchical-specific guidelines:
- For simple goals, create ONE task with ONE subtask, then immediately work on it
- When you finish your current subtask, ALWAYS call mark_subtask_complete
- Focus on the current subtask shown in the context

Additional tools for hierarchical workflow:
- mark_subtask_complete(success, reason): Mark current subtask done
- decompose_task(subtasks): Break down goal into tasks (only if NO TASKS YET)
"""

    def get_strategy_tools(self) -> list[dict[str, Any]]:
        """
        Provide hierarchical-specific tools.

        Returns:
            Tool definitions for decompose_task and mark_subtask_complete
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "mark_subtask_complete",
                    "description": "Mark current subtask as complete and advance to next subtask. Call this when you finish a subtask.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "description": "Whether the subtask was completed successfully"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief explanation of what was done or why it failed"
                            }
                        },
                        "required": ["success"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "decompose_task",
                    "description": "Break down the goal into tasks and subtasks. Use this when NO TASKS YET to create initial structure.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subtasks": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of subtask descriptions (2-6 subtasks recommended)"
                            }
                        },
                        "required": ["subtasks"]
                    }
                }
            }
        ]

    def should_use_jetbox_notes(self) -> bool:
        """Hierarchical strategy enables jetbox notes by default."""
        return self.use_jetbox_notes

    def build_context(
        self,
        context_manager: ContextManager,
        messages: list[dict[str, Any]],
        system_prompt: str,
        config: Any,
        **kwargs
    ) -> list[dict[str, Any]]:
        """Build context using hierarchical strategy."""
        probe_state_func = kwargs.get('probe_state_func')
        workspace = kwargs.get('workspace')

        # Start with system prompt
        context = [{"role": "system", "content": system_prompt}]

        # Add current goal/task/subtask context
        if context_manager.state.goal:
            task = context_manager._get_current_task()
            subtask = task.active_subtask() if task else None

            context_info = [
                f"GOAL: {context_manager.state.goal.description}",
            ]

            if task:
                context_info.append(f"CURRENT TASK: {task.description}")

                if subtask:
                    context_info.append(f"ACTIVE SUBTASK: {subtask.description}")

                    # Add depth/rounds info if config provided
                    if hasattr(config, 'hierarchy') and hasattr(config.hierarchy, 'max_depth'):
                        context_info.append(f"Subtask Depth: {subtask.depth}/{config.hierarchy.max_depth}")

                    if hasattr(config, 'rounds') and hasattr(config.rounds, 'max_per_subtask'):
                        context_info.append(f"Rounds Used: {subtask.rounds_used}/{config.rounds.max_per_subtask}")
                else:
                    context_info.append("ACTIVE SUBTASK: (none - call mark_subtask_complete to advance)")
            else:
                # No tasks yet - need to decompose goal
                context_info.append("")
                context_info.append("⚠️  NO TASKS YET")
                context_info.append("The goal has not been broken down into tasks.")
                context_info.append("Use decompose_task to create an initial task/subtask structure.")

            # Add loop detection warnings
            if context_manager.state.loop_counts:
                loop_warnings = []
                for sig, count in context_manager.state.loop_counts.items():
                    if count > 0:
                        loop_warnings.append(f"  • Action repeated {count}x: {sig[:80]}")

                if loop_warnings:
                    context_info.append("")
                    context_info.append("⚠️  LOOP DETECTION WARNING:")
                    context_info.append("You appear to be repeating actions. Consider:")
                    context_info.append("- Trying a COMPLETELY DIFFERENT approach")
                    context_info.append("- Reading error messages carefully")
                    context_info.append("- Checking if assumptions are wrong")
                    context_info.append("- Asking yourself: 'Why didn't the last attempt work?'")
                    context_info.append("")
                    context_info.append("Detected loops:")
                    context_info.extend(loop_warnings)
                    context_info.append("")
                    context_info.append("Try something NEW this round.")

            # Add filesystem probe state if function provided
            if probe_state_func:
                probe = probe_state_func()

                # Add workspace warning if present
                if probe.get("warning"):
                    context_info.append("")
                    context_info.append(probe["warning"])
                    context_info.append("")

                if probe.get("files_exist"):
                    context_info.append(f"FILES CREATED: {', '.join(probe['files_exist'])}")
                if probe.get("recent_errors"):
                    context_info.append(f"RECENT ERRORS: {probe['recent_errors'][-1][:100]}")

            # Add jetbox notes if enabled by strategy and workspace provided
            if self.should_use_jetbox_notes() and workspace:
                import jetbox_notes
                notes_content = jetbox_notes.load_jetbox_notes(max_chars=2000)
                if notes_content:
                    context_info.append("")
                    context_info.append("="*70)
                    context_info.append("JETBOX NOTES (Previous Work Summary)")
                    context_info.append("="*70)
                    context_info.append(notes_content)
                    context_info.append("="*70)
                    context_info.append("")

            context.append({"role": "user", "content": "\n".join(context_info)})

        # Add last N message exchanges with compaction fallback
        history_keep = self.history_keep
        if hasattr(config, 'context') and hasattr(config.context, 'history_keep'):
            history_keep = config.context.history_keep

        recent = messages[-history_keep * 2:] if len(messages) > history_keep * 2 else messages
        context.extend(recent)

        # Fallback: If context too large, use LLM-based summarization
        estimated_tokens = self.estimate_context_size(context)
        max_context = 131072  # 128K for gpt-oss:20b

        if estimated_tokens > max_context * 0.75:  # >75% of context window
            print(f"[context_compaction] Context at {estimated_tokens:,} tokens ({estimated_tokens/max_context*100:.1f}% of {max_context:,}) - triggering LLM summarization")

            # Summarize old messages, keep recent ones
            # Keep only 8 recent messages (4 exchanges) to ensure we have room
            keep_recent = 8
            to_summarize = messages[:-keep_recent] if len(messages) > keep_recent else []

            if to_summarize:
                # Use LLM to summarize the old messages
                summary = self._summarize_messages(to_summarize)

                # Rebuild context: system + goal/task + summary + recent messages
                context_base_len = len(context) - len(recent)
                context = context[:context_base_len]

                # Add summary as a user message
                context.append({
                    "role": "user",
                    "content": f"Previous work summary (compacted from {len(to_summarize)} messages):\n{summary}"
                })

                # Add recent messages
                context.extend(messages[-keep_recent:])

                new_tokens = self.estimate_context_size(context)
                print(f"[context_compaction] Reduced from {estimated_tokens:,} to {new_tokens:,} tokens ({new_tokens/max_context*100:.1f}%)")
            else:
                # Not enough messages to summarize, just keep recent
                print(f"[context_compaction] Not enough messages to summarize, keeping last {keep_recent}")
                context_base_len = len(context) - len(recent)
                context = context[:context_base_len]
                context.extend(messages[-keep_recent:])

        return context

    def _summarize_messages(self, messages: list[dict[str, Any]]) -> str:
        """
        Use LLM to summarize a sequence of messages.

        Args:
            messages: List of message dicts to summarize

        Returns:
            Concise summary of the messages
        """
        from llm_utils import chat_with_inactivity_timeout

        # Build a prompt asking for summary - aggressively truncate for compactness
        messages_text = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                # Just show tool names, not full arguments
                tool_names = str(tool_calls)[:200]
                messages_text.append(f"[{role}] {tool_names}")
            elif content:
                # Aggressively truncate - we only need gist for summarization
                if role == "tool":
                    # Tool results: just first 100 chars (usually enough to see success/error)
                    content = content[:100] + ("..." if len(content) > 100 else "")
                else:
                    # Other messages: first 300 chars
                    content = content[:300] + ("..." if len(content) > 300 else "")
                messages_text.append(f"[{role}] {content}")

        prompt = f"""Provide an extremely concise summary (max 200 words) of this conversation, focusing ONLY on:
1. Files created/modified (just names, not content)
2. Commands run (results only if they failed)
3. Current state/progress

Omit: successful tool outputs, file contents, verbose explanations.
Format: Dense bullet points.

Conversation:
{chr(10).join(messages_text)}

Concise summary (max 200 words):"""

        try:
            response = chat_with_inactivity_timeout(
                model="gpt-oss:20b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
                inactivity_timeout=30,
                max_total_time=60,  # 1 minute max for summarization
            )
            return response.get("message", {}).get("content", "Unable to generate summary.")
        except Exception as e:
            # If summarization fails, return a basic summary
            return f"[Summarization failed: {e}] Previous work included {len(messages)} message exchanges with tool calls and results."

    def should_clear_on_transition(self) -> bool:
        """Hierarchical strategy clears messages on subtask transitions."""
        return True

    def get_name(self) -> str:
        """Get strategy name."""
        return "hierarchical"

    def estimate_context_size(self, context: list[dict[str, Any]]) -> int:
        """Estimate context size using 4 chars per token heuristic."""
        total_chars = 0
        for msg in context:
            total_chars += len(str(msg.get("content", "")))
        return total_chars // 4


class AppendUntilFullStrategy(ContextStrategy):
    """
    Append-until-full context strategy.

    Appends all messages until approaching token limit, then compacts.
    Optimized for conversational agents.

    Features:
    - System prompt + all messages
    - Compacts when near token limit (80%)
    - Keeps recent N messages intact during compaction
    - Summarizes older messages
    - Does NOT clear on subtask transitions
    - Jetbox notes disabled by default (conversational agents maintain full history)
    """

    def __init__(self, max_tokens: int = 8000, recent_keep: int = 20, use_jetbox_notes: bool = False):
        """
        Initialize append-until-full strategy.

        Args:
            max_tokens: Maximum context tokens before compaction
            recent_keep: Number of recent messages to keep intact during compaction
            use_jetbox_notes: Whether to use jetbox notes (default: False for conversational agents)
        """
        super().__init__()  # Initialize loop detection
        self.max_tokens = max_tokens
        self.recent_keep = recent_keep
        self.use_jetbox_notes = use_jetbox_notes

    def build_context(
        self,
        context_manager: ContextManager,
        messages: list[dict[str, Any]],
        system_prompt: str,
        config: Any,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Build context using append-until-full strategy.

        Includes:
        - System prompt (with tools)
        - Goal
        - Jetbox notes (if available)
        - All messages (or compacted if near limit)

        Does NOT include hierarchical task/subtask structure.
        """
        workspace = kwargs.get('workspace')

        context = [{"role": "system", "content": system_prompt}]

        # Build goal context (simple, not hierarchical)
        context_parts = []

        if context_manager.state.goal:
            context_parts.append(f"GOAL: {context_manager.state.goal.description}")

        # Add jetbox notes if enabled by strategy
        if self.should_use_jetbox_notes() and workspace:
            import jetbox_notes
            notes_content = jetbox_notes.load_jetbox_notes(max_chars=2000)
            if notes_content:
                context_parts.append("")
                context_parts.append("="*70)
                context_parts.append("JETBOX NOTES (Previous Work Summary)")
                context_parts.append("="*70)
                context_parts.append(notes_content)
                context_parts.append("="*70)

        if context_parts:
            context.append({"role": "user", "content": "\n".join(context_parts)})

        # Add all messages first, then check if compaction needed
        context.extend(messages)

        # Check if context exceeds 75% of configured capacity
        estimated_tokens = self.estimate_context_size(context)
        max_context = self.max_tokens  # Use configured max_tokens from __init__

        if estimated_tokens > max_context * 0.75:  # >75% of context window
            print(f"[context_compaction] Append strategy: Context at {estimated_tokens:,} tokens ({estimated_tokens/max_context*100:.1f}% of {max_context:,}) - triggering LLM summarization")

            # Keep recent N messages, summarize the rest
            keep_recent = self.recent_keep
            to_summarize = messages[:-keep_recent] if len(messages) > keep_recent else []

            if to_summarize:
                # Use LLM to summarize old messages (same method as hierarchical)
                summary = self._summarize_messages(to_summarize)

                # Rebuild context: system + goal + summary + recent messages
                # Find where messages start in context (after system and goal)
                context_base = []
                for msg in context:
                    if msg in messages:
                        break
                    context_base.append(msg)

                # Start fresh with base + summary + recent
                context = context_base
                context.append({
                    "role": "user",
                    "content": f"Previous work summary (compacted from {len(to_summarize)} messages):\n{summary}"
                })
                context.extend(messages[-keep_recent:])

                new_tokens = self.estimate_context_size(context)
                print(f"[context_compaction] Reduced from {estimated_tokens:,} to {new_tokens:,} tokens ({new_tokens/max_context*100:.1f}%)")
            else:
                # Not enough messages to summarize
                print(f"[context_compaction] Not enough messages to summarize, keeping last {keep_recent}")

        return context

    def _summarize_messages(self, messages: list[dict[str, Any]]) -> str:
        """
        Use LLM to summarize old messages (same implementation as HierarchicalStrategy).

        Args:
            messages: Messages to summarize

        Returns:
            Concise summary string
        """
        from llm_utils import chat_with_inactivity_timeout

        # Build prompt with truncated messages to avoid recursive explosion
        messages_text = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                # Show tool names only
                tool_names = str(tool_calls)[:200]
                messages_text.append(f"[{role}] {tool_names}")
            elif content:
                # Truncate tool results heavily, keep other messages longer
                if role == "tool":
                    content = content[:100] + ("..." if len(content) > 100 else "")
                else:
                    content = content[:300] + ("..." if len(content) > 300 else "")
                messages_text.append(f"[{role}] {content}")

        prompt = f"""Provide an extremely concise summary (max 200 words) of this conversation, focusing ONLY on:
1. Files created/modified (just names, not content)
2. Commands run (results only if they failed)
3. Current state/progress

Omit: successful tool outputs, file contents, verbose explanations.
Format: Dense bullet points.

Conversation:
{chr(10).join(messages_text)}

Concise summary (max 200 words):"""

        try:
            response = chat_with_inactivity_timeout(
                model="gpt-oss:20b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
                inactivity_timeout=30,
                max_total_time=60,  # 1 minute max for summarization
            )
            return response.get("message", {}).get("content", "Unable to generate summary.")
        except Exception as e:
            # If summarization fails, return a basic summary
            return f"[Summarization failed: {e}] Previous work included {len(messages)} message exchanges with tool calls and results."

    def should_clear_on_transition(self) -> bool:
        """Append strategy does NOT clear messages on transitions."""
        return False

    def should_use_jetbox_notes(self) -> bool:
        """Append strategy disables jetbox notes by default (conversational agents maintain full history)."""
        return self.use_jetbox_notes

    def get_strategy_instructions(self) -> str:
        """
        Append-until-full strategy workflow instructions.

        Returns:
            Simple workflow for direct execution without hierarchical decomposition
        """
        return """
WORKFLOW:
Your goal is shown at the start of the conversation. Simply complete the work using the available tools.

- Use write_file to create/modify files
- Use run_bash to run commands (tests, linters, etc.)
- Use read_file to check existing files
- Use list_dir to explore directories

Work directly on the goal - no need to decompose into subtasks.
When all work is complete and tests pass, call mark_goal_complete().
"""

    def get_strategy_tools(self) -> list[dict[str, Any]]:
        """
        Provide simple completion tool for append-until-full strategy.

        Returns:
            Tool definition for mark_goal_complete
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "mark_goal_complete",
                    "description": "Mark the goal as complete. Call this when you have finished all work and tests pass.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Brief summary of what was accomplished"
                            }
                        },
                        "required": ["summary"]
                    }
                }
            }
        ]

    def get_name(self) -> str:
        """Get strategy name."""
        return "append_until_full"

    def estimate_context_size(self, context: list[dict[str, Any]]) -> int:
        """Estimate context size using 4 chars per token heuristic."""
        total_chars = 0
        for msg in context:
            total_chars += len(str(msg.get("content", "")))
        return total_chars // 4


class SubAgentStrategy(ContextStrategy):
    """
    Sub-agent context strategy for delegated work.

    Designed for agents that are invoked by a controlling agent (orchestrator)
    to complete a specific delegated task. This strategy:

    - Inserts the delegated goal at the top of context
    - Provides mark_complete/mark_failed tools for reporting results
    - Nudges completion on timeouts
    - Works with enhancements (jetbox notes, task management)
    - Appends all messages until token limit

    Features:
    - System prompt + delegated goal
    - Jetbox notes from workspace (previous work context)
    - All messages (append-until-full style)
    - Compaction when near token limit
    - Completion tools to communicate with controlling agent
    """

    def __init__(self, max_tokens: int = 128000, recent_keep: int = 20, use_jetbox_notes: bool = True):
        """
        Initialize sub-agent strategy.

        Args:
            max_tokens: Maximum context tokens before compaction (default: 128K for gpt-oss:20b)
            recent_keep: Number of recent messages to keep intact during compaction
            use_jetbox_notes: Whether to use jetbox notes (default: True for context continuity)
        """
        super().__init__()  # Initialize loop detection
        self.max_tokens = max_tokens
        self.recent_keep = recent_keep
        self.use_jetbox_notes = use_jetbox_notes

    def build_context(
        self,
        context_manager: ContextManager,
        messages: list[dict[str, Any]],
        system_prompt: str,
        config: Any,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Build context for sub-agent.

        Includes:
        - System prompt with tools
        - Delegated goal
        - Jetbox notes (if available)
        - All messages (or compacted if near limit)

        Args:
            context_manager: ContextManager instance
            messages: Full message history
            system_prompt: System prompt
            config: Agent configuration
            **kwargs: Additional parameters (workspace)

        Returns:
            Context list ready for LLM
        """
        workspace = kwargs.get('workspace')

        context = [{"role": "system", "content": system_prompt}]

        # Build goal context
        context_parts = []

        if context_manager.state.goal:
            context_parts.append(f"DELEGATED GOAL: {context_manager.state.goal.description}")
            context_parts.append("")
            context_parts.append("You are working on a task delegated by the orchestrator.")
            context_parts.append("When complete, call mark_complete(summary) with what you accomplished.")
            context_parts.append("If you cannot complete it, call mark_failed(reason) explaining why.")

        # Add jetbox notes if enabled
        if self.should_use_jetbox_notes() and workspace:
            import jetbox_notes
            notes_content = jetbox_notes.load_jetbox_notes(max_chars=2000)
            if notes_content:
                context_parts.append("")
                context_parts.append("="*70)
                context_parts.append("PREVIOUS WORK (from jetboxnotes.md)")
                context_parts.append("="*70)
                context_parts.append(notes_content)
                context_parts.append("="*70)

        if context_parts:
            context.append({"role": "user", "content": "\n".join(context_parts)})

        # Add all messages
        context.extend(messages)

        # Check if context exceeds threshold
        estimated_tokens = self.estimate_context_size(context)

        if estimated_tokens > self.max_tokens * 0.75:  # >75% of context window
            print(f"[context_compaction] SubAgent strategy: Context at {estimated_tokens:,} tokens ({estimated_tokens/self.max_tokens*100:.1f}% of {self.max_tokens:,}) - triggering LLM summarization")

            # Keep recent N messages, summarize the rest
            keep_recent = self.recent_keep
            to_summarize = messages[:-keep_recent] if len(messages) > keep_recent else []

            if to_summarize:
                # Use LLM to summarize old messages
                summary = self._summarize_messages(to_summarize)

                # Rebuild context: system + goal + summary + recent messages
                context_base = []
                for msg in context:
                    if msg in messages:
                        break
                    context_base.append(msg)

                # Start fresh with base + summary + recent
                context = context_base
                context.append({
                    "role": "user",
                    "content": f"Previous work summary (compacted from {len(to_summarize)} messages):\n{summary}"
                })
                context.extend(messages[-keep_recent:])

                new_tokens = self.estimate_context_size(context)
                print(f"[context_compaction] Reduced from {estimated_tokens:,} to {new_tokens:,} tokens ({new_tokens/self.max_tokens*100:.1f}%)")

        return context

    def _summarize_messages(self, messages: list[dict[str, Any]]) -> str:
        """
        Use LLM to summarize old messages.

        Args:
            messages: Messages to summarize

        Returns:
            Concise summary string
        """
        from llm_utils import chat_with_inactivity_timeout

        # Build prompt with truncated messages
        messages_text = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                # Show tool names only
                tool_names = str(tool_calls)[:200]
                messages_text.append(f"[{role}] {tool_names}")
            elif content:
                # Truncate tool results heavily
                if role == "tool":
                    content = content[:100] + ("..." if len(content) > 100 else "")
                else:
                    content = content[:300] + ("..." if len(content) > 300 else "")
                messages_text.append(f"[{role}] {content}")

        prompt = f"""Provide an extremely concise summary (max 200 words) of this conversation, focusing ONLY on:
1. Files created/modified (just names, not content)
2. Commands run (results only if they failed)
3. Current state/progress

Omit: successful tool outputs, file contents, verbose explanations.
Format: Dense bullet points.

Conversation:
{chr(10).join(messages_text)}

Concise summary (max 200 words):"""

        try:
            response = chat_with_inactivity_timeout(
                model="gpt-oss:20b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
                inactivity_timeout=30,
                max_total_time=60,
            )
            return response.get("message", {}).get("content", "Unable to generate summary.")
        except Exception as e:
            return f"[Summarization failed: {e}] Previous work included {len(messages)} message exchanges."

    def should_clear_on_transition(self) -> bool:
        """SubAgent strategy does NOT clear messages on transitions."""
        return False

    def should_use_jetbox_notes(self) -> bool:
        """SubAgent strategy enables jetbox notes for context continuity."""
        return self.use_jetbox_notes

    def get_strategy_instructions(self) -> str:
        """
        Sub-agent strategy workflow instructions.

        Returns:
            Instructions for completing delegated work and reporting results
        """
        return """
SUB-AGENT WORKFLOW:
You are working on a task delegated by a controlling agent (orchestrator).

Your job is to:
1. Complete the delegated goal using available tools
2. Report results back to the controlling agent when done

Available tools:
- write_file: Create/modify files
- run_bash: Run commands (tests, linters, build tools, etc.)
- read_file: Check existing files
- list_dir: Explore directories
- mark_complete(summary): Signal successful completion with summary
- mark_failed(reason): Signal failure with explanation

IMPORTANT - YOU MUST SIGNAL COMPLETION:
- When work is done and tests pass: call mark_complete(summary="what you accomplished")
- If you cannot complete the task: call mark_failed(reason="why it failed")
- DO NOT just stop - always call one of these tools to report results

The controlling agent is waiting for your completion signal.
"""

    def get_strategy_tools(self) -> list[dict[str, Any]]:
        """
        Provide completion tools for sub-agent.

        Returns:
            Tool definitions for mark_complete and mark_failed
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "mark_complete",
                    "description": "Mark the delegated task as complete and report success to controlling agent. REQUIRED when work is finished.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Brief summary of what was accomplished (2-4 sentences)"
                            }
                        },
                        "required": ["summary"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_failed",
                    "description": "Mark the delegated task as failed and report reason to controlling agent. Use when you cannot complete the task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Explanation of why the task could not be completed"
                            }
                        },
                        "required": ["reason"]
                    }
                }
            }
        ]

    def get_name(self) -> str:
        """Get strategy name."""
        return "sub_agent"

    def estimate_context_size(self, context: list[dict[str, Any]]) -> int:
        """Estimate context size using 4 chars per token heuristic."""
        total_chars = 0
        for msg in context:
            total_chars += len(str(msg.get("content", "")))
        return total_chars // 4


class ArchitectStrategy(ContextStrategy):
    """
    Architect context strategy.

    Optimized for architecture design conversations where artifacts
    are the primary output (not summaries).

    Features:
    - System prompt + all messages
    - Compacts when near token limit
    - Keeps recent messages intact during compaction
    - Does NOT use jetbox notes (artifacts are persistent output)
    - Higher token limit for verbose architecture discussions
    """

    def __init__(self, max_tokens: int = 32000, recent_keep: int = 20, use_jetbox_notes: bool = False):
        """
        Initialize architect strategy.

        Args:
            max_tokens: Maximum context tokens before compaction (default: 32K for architecture docs)
            recent_keep: Number of recent messages to keep intact during compaction
            use_jetbox_notes: Whether to use jetbox notes (default: False, artifacts are output)
        """
        super().__init__()  # Initialize loop detection
        self.max_tokens = max_tokens
        self.recent_keep = recent_keep
        self.use_jetbox_notes = use_jetbox_notes

    def build_context(
        self,
        context_manager: ContextManager,
        messages: list[dict[str, Any]],
        system_prompt: str,
        config: Any,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Build context using architect strategy.

        Similar to append-until-full but with higher token budget
        for architecture discussions.
        """
        workspace = kwargs.get('workspace')

        context = [{"role": "system", "content": system_prompt}]

        # Build context parts (goal info if present)
        context_parts = []

        if context_manager.state.goal:
            context_parts.append(f"PROJECT: {context_manager.state.goal.description}")

        # Note: Architect doesn't use jetbox notes - artifacts are the output
        # But we respect the interface
        if self.should_use_jetbox_notes() and workspace:
            import jetbox_notes
            notes_content = jetbox_notes.load_jetbox_notes(max_chars=2000)
            if notes_content:
                context_parts.append("")
                context_parts.append("="*70)
                context_parts.append("PREVIOUS WORK SUMMARY")
                context_parts.append("="*70)
                context_parts.append(notes_content)
                context_parts.append("="*70)

        if context_parts:
            context.append({"role": "user", "content": "\n".join(context_parts)})

        # Add all messages first
        context.extend(messages)

        # Check if context exceeds 75% of limit
        estimated_tokens = self.estimate_context_size(context)
        max_context = 131072  # 128K for large models

        if estimated_tokens > max_context * 0.75:
            print(f"[context_compaction] Architect: Context at {estimated_tokens:,} tokens ({estimated_tokens/max_context*100:.1f}%) - triggering summarization")

            # Keep recent messages, summarize the rest
            keep_recent = self.recent_keep
            to_summarize = messages[:-keep_recent] if len(messages) > keep_recent else []

            if to_summarize:
                summary = self._summarize_messages(to_summarize)

                # Rebuild: system + goal + summary + recent
                context_base = []
                for msg in context:
                    if msg in messages:
                        break
                    context_base.append(msg)

                context = context_base
                context.append({
                    "role": "user",
                    "content": f"Previous discussion summary ({len(to_summarize)} messages):\n{summary}"
                })
                context.extend(messages[-keep_recent:])

                new_tokens = self.estimate_context_size(context)
                print(f"[context_compaction] Reduced from {estimated_tokens:,} to {new_tokens:,} tokens")

        return context

    def _summarize_messages(self, messages: list[dict[str, Any]]) -> str:
        """Summarize messages for architect (focus on decisions and requirements)."""
        from llm_utils import chat_with_inactivity_timeout

        # Build concise summary prompt focusing on architecture decisions
        messages_text = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                tool_names = [tc["function"]["name"] for tc in tool_calls]
                messages_text.append(f"[{role}] {', '.join(tool_names)}")
            elif content:
                # Keep architecture-relevant content longer
                if role == "tool":
                    content = content[:100] + ("..." if len(content) > 100 else "")
                else:
                    content = content[:400] + ("..." if len(content) > 400 else "")
                messages_text.append(f"[{role}] {content}")

        prompt = f"""Summarize this architecture discussion (max 300 words). Focus on:
1. Project requirements and constraints
2. Architecture decisions made
3. Technology choices and rationale
4. Outstanding questions or concerns

Omit: Tool outputs, verbose explanations.
Format: Bullet points.

Discussion:
{chr(10).join(messages_text)}

Summary (max 300 words):"""

        try:
            response = chat_with_inactivity_timeout(
                model="qwen3:14b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
                inactivity_timeout=30,
                max_total_time=60,
            )
            return response.get("message", {}).get("content", "Unable to generate summary.")
        except Exception as e:
            return f"[Summarization failed: {e}] Previous discussion included {len(messages)} exchanges about architecture design."

    def should_clear_on_transition(self) -> bool:
        """Architect strategy does NOT clear messages."""
        return False

    def should_use_jetbox_notes(self) -> bool:
        """Architect strategy disables jetbox notes (artifacts are persistent output)."""
        return self.use_jetbox_notes

    def get_name(self) -> str:
        """Get strategy name."""
        return "architect"

    def estimate_context_size(self, context: list[dict[str, Any]]) -> int:
        """Estimate context size using 4 chars per token heuristic."""
        total_chars = 0
        for msg in context:
            total_chars += len(str(msg.get("content", "")))
        return total_chars // 4


# ============================================================================
# Backward Compatibility Functions
# These delegate to the new strategy classes
# ============================================================================

def build_hierarchical_context(
    context_manager: ContextManager,
    messages: list[dict[str, Any]],
    system_prompt: str,
    config: Any,
    probe_state_func: callable | None = None,
    workspace=None,
) -> list[dict[str, Any]]:
    """
    Build context using hierarchical strategy (backward compatibility wrapper).

    Delegates to HierarchicalStrategy class.

    Args:
        context_manager: ContextManager with goal/task/subtask hierarchy
        messages: Full message history
        system_prompt: System prompt for the agent
        config: Configuration object
        probe_state_func: Optional function to probe filesystem state
        workspace: Optional workspace path for jetbox notes

    Returns:
        Context list ready for LLM
    """
    strategy = HierarchicalStrategy()
    return strategy.build_context(
        context_manager=context_manager,
        messages=messages,
        system_prompt=system_prompt,
        config=config,
        probe_state_func=probe_state_func,
        workspace=workspace,
    )


def build_simple_hierarchical_context(
    context_manager: ContextManager,
    messages: list[dict[str, Any]],
    system_prompt: str,
    config: Any,
) -> list[dict[str, Any]]:
    """
    Simplified hierarchical context builder (backward compatibility wrapper).

    Delegates to HierarchicalStrategy without probe/loop features.

    Args:
        context_manager: ContextManager with goal/task/subtask hierarchy
        messages: Full message history
        system_prompt: System prompt for the agent
        config: Configuration object

    Returns:
        Context list ready for LLM
    """
    strategy = HierarchicalStrategy()
    return strategy.build_context(
        context_manager=context_manager,
        messages=messages,
        system_prompt=system_prompt,
        config=config,
    )


class JetboxNotesEnhancement(ContextEnhancement):
    """
    Jetbox Notes enhancement.

    Injects jetbox notes summary into context for task continuity.
    Used by TaskExecutor agents when jetbox notes are enabled.

    Features:
    - Loads jetbox notes from workspace (if they exist)
    - Injects notes summary into context for continuity
    - No additional tools (notes are auto-generated)
    - Provides instructions about automatic note-taking
    """

    def __init__(self, workspace_manager=None):
        """
        Initialize jetbox notes enhancement.

        Args:
            workspace_manager: Workspace manager for accessing notes file
        """
        self.workspace_manager = workspace_manager

    def get_context_injection(
        self,
        context_manager: Any,
        workspace: Any = None,
        **kwargs
    ) -> dict[str, str] | None:
        """
        Inject jetbox notes into context if they exist.

        Returns:
            Message dict with notes content or None if no notes
        """
        if not self.workspace_manager:
            return None

        # Load notes using jetbox_notes module
        import jetbox_notes
        notes_content = jetbox_notes.load_jetbox_notes(max_chars=2000)

        if not notes_content:
            return None

        # Format notes for context injection
        context_parts = []
        context_parts.append("="*70)
        context_parts.append("JETBOX NOTES (Previous Work Summary)")
        context_parts.append("="*70)
        context_parts.append(notes_content)
        context_parts.append("="*70)

        return {
            "role": "user",
            "content": "\n".join(context_parts)
        }

    def get_enhancement_instructions(self) -> str:
        """
        Jetbox notes workflow instructions.

        Returns:
            Instructions about automatic note-taking
        """
        return """
JETBOX NOTES - AUTOMATIC CONTEXT PERSISTENCE:

The system automatically captures summaries of your work:
- When you complete a subtask: Brief 2-4 bullet summary is saved
- When you complete the goal: Comprehensive 3-6 bullet summary is saved
- When goal times out: Analysis of what was completed and what remains

These notes are loaded automatically when you resume work in the same workspace.
You don't need to do anything - the system handles it for you.

Notes are saved to jetboxnotes.md in your workspace directory.
"""

    def get_enhancement_tools(self) -> list[dict[str, Any]]:
        """
        Jetbox notes provides no tools (notes are auto-generated).

        Returns:
            Empty list
        """
        return []

    def get_name(self) -> str:
        """Get enhancement name."""
        return "jetbox_notes"


class TaskManagementEnhancement(ContextEnhancement):
    """
    Task Management enhancement.

    Injects task status and provides CRUD operations on task breakdown.
    Used by Orchestrator and Architect agents when working with structured tasks.

    Features:
    - Injects task status into context (pending/in_progress/completed/failed counts)
    - Shows next pending task with dependency checking
    - Provides task management tools (read, get_next, mark_status, update)
    - Displays first 5 tasks with visual status icons
    """

    def __init__(self, workspace_manager=None):
        """
        Initialize task management enhancement.

        Args:
            workspace_manager: Workspace manager for accessing task breakdown
        """
        self.workspace_manager = workspace_manager

        # Configure task management tools with workspace
        if workspace_manager:
            import task_management_tools
            task_management_tools.set_workspace(workspace_manager)

    def get_context_injection(
        self,
        context_manager: Any,
        workspace: Any = None,
        **kwargs
    ) -> dict[str, str] | None:
        """
        Inject task breakdown status into context.

        Returns:
            Message dict with task status or None if no tasks
        """
        if not self.workspace_manager:
            return None

        import task_management_tools
        breakdown = task_management_tools.read_task_breakdown()

        if breakdown.get("status") != "success" or breakdown.get("total_tasks", 0) == 0:
            return None

        # Build task status context
        context_parts = []
        context_parts.append("="*70)
        context_parts.append("TASK BREAKDOWN STATUS")
        context_parts.append("="*70)
        context_parts.append(f"Total Tasks: {breakdown['total_tasks']}")
        context_parts.append(f"  Pending:     {breakdown.get('pending_count', 0)}")
        context_parts.append(f"  In Progress: {breakdown.get('in_progress_count', 0)}")
        context_parts.append(f"  Completed:   {breakdown.get('completed_count', 0)}")
        context_parts.append(f"  Failed:      {breakdown.get('failed_count', 0)}")

        # Show next pending task
        next_task_result = task_management_tools.get_next_task(skip_dependencies=False)
        if next_task_result.get("status") == "success":
            next_task = next_task_result.get("task")
            if next_task:
                context_parts.append("")
                context_parts.append("NEXT PENDING TASK:")
                context_parts.append(f"  [{next_task['id']}] {next_task['description']}")
                context_parts.append(f"  Module: {next_task.get('module', 'N/A')}")
                context_parts.append(f"  Complexity: {next_task.get('estimated_complexity', 'unknown')}")
                if next_task.get('dependencies'):
                    context_parts.append(f"  Dependencies: {', '.join(next_task['dependencies'])} (completed)")
            else:
                context_parts.append("")
                context_parts.append("NEXT PENDING TASK: None")
                context_parts.append(next_task_result.get("message", "All tasks completed or blocked"))

        # Show task list (first 5 tasks)
        tasks = breakdown.get("tasks", [])
        if tasks:
            context_parts.append("")
            context_parts.append("TASK LIST (first 5):")
            for task in tasks[:5]:
                status_icon = {
                    "pending": "○",
                    "in_progress": "⟳",
                    "completed": "✓",
                    "failed": "✗"
                }.get(task.get("status", "pending"), "○")
                deps = f" (deps: {', '.join(task['dependencies'])})" if task.get('dependencies') else ""
                context_parts.append(f"  {status_icon} [{task['id']}] {task['description']}{deps}")

            if len(tasks) > 5:
                context_parts.append(f"  ... and {len(tasks) - 5} more tasks")

        context_parts.append("="*70)

        return {
            "role": "user",
            "content": "\n".join(context_parts)
        }

    def get_enhancement_instructions(self) -> str:
        """
        Task management workflow instructions.

        Returns:
            Instructions for using task management tools
        """
        return """
TASK MANAGEMENT WORKFLOW:
1. Use read_task_breakdown() to see all tasks and their status
2. Use get_next_task() to find the next pending task (respects dependencies)
3. Work on the task using your available tools (delegate_to_executor, consult_architect, etc.)
4. Use mark_task_status(task_id, status, notes) to update task status:
   - "in_progress" when starting a task
   - "completed" when task is done
   - "failed" if task cannot be completed
5. Use update_task(task_id, updates) to modify task properties if needed

Task management guidelines:
- Always check task breakdown before delegating work
- Mark tasks in_progress when starting, completed when done
- Respect task dependencies - don't start tasks whose dependencies aren't completed
- Include notes when marking status to track progress

Available task management tools:
- read_task_breakdown(): Get all tasks with status counts
- get_next_task(skip_dependencies): Find next task to work on
- mark_task_status(task_id, status, notes): Update task status
- update_task(task_id, updates): Modify task properties
"""

    def get_enhancement_tools(self) -> list[dict[str, Any]]:
        """
        Provide task management tools.

        Returns:
            Tool definitions for task CRUD operations
        """
        import task_management_tools
        return task_management_tools.get_task_management_tool_definitions()

    def get_name(self) -> str:
        """Get enhancement name."""
        return "task_management"


# Legacy TaskManagementStrategy for backward compatibility
# (wraps TaskManagementEnhancement + AppendUntilFullStrategy)
class TaskManagementStrategy(AppendUntilFullStrategy):
    """
    DEPRECATED: Use AppendUntilFullStrategy + TaskManagementEnhancement instead.

    This class exists for backward compatibility only.
    """

    def __init__(self, workspace_manager=None, recent_keep: int = 20, use_jetbox_notes: bool = False):
        super().__init__(max_tokens=131072, recent_keep=recent_keep, use_jetbox_notes=use_jetbox_notes)
        self.workspace_manager = workspace_manager
        self.task_enhancement = TaskManagementEnhancement(workspace_manager)

    def build_context(
        self,
        context_manager: ContextManager,
        messages: list[dict[str, Any]],
        system_prompt: str,
        config: Any,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Build context with task status injected.

        Delegates to AppendUntilFullStrategy and injects task management enhancement.
        """
        # Build base context using AppendUntilFullStrategy
        context = super().build_context(context_manager, messages, system_prompt, config, **kwargs)

        # Inject task management context after system prompt (index 1)
        task_injection = self.task_enhancement.get_context_injection(
            context_manager=context_manager,
            workspace=kwargs.get('workspace'),
            **kwargs
        )

        if task_injection:
            # Insert after system prompt
            context.insert(1, task_injection)

        return context

    def get_strategy_tools(self) -> list[dict[str, Any]]:
        """Get task management tools."""
        return self.task_enhancement.get_enhancement_tools()

    def get_strategy_instructions(self) -> str:
        """Get task management instructions."""
        return self.task_enhancement.get_enhancement_instructions()

    def get_name(self) -> str:
        """Get strategy name."""
        return "task_management"


# ============================================================================
# Backward Compatibility Functions
# These delegate to the new strategy classes
# ============================================================================

def build_append_context(
    messages: list[dict[str, Any]],
    system_prompt: str,
    max_tokens: int = 8000,
    recent_keep: int = 20,
) -> list[dict[str, Any]]:
    """
    Build context using append-until-full strategy (backward compatibility wrapper).

    Delegates to AppendUntilFullStrategy class.

    Args:
        messages: Full message history
        system_prompt: System prompt for the agent
        max_tokens: Maximum context tokens
        recent_keep: Number of recent messages to keep intact

    Returns:
        Context list ready for LLM
    """
    # Create dummy context_manager and config for compatibility
    from context_manager import ContextManager
    from agent_config import AgentConfig

    dummy_cm = ContextManager()
    dummy_config = type('Config', (), {})()

    strategy = AppendUntilFullStrategy(max_tokens=max_tokens, recent_keep=recent_keep)
    return strategy.build_context(
        context_manager=dummy_cm,
        messages=messages,
        system_prompt=system_prompt,
        config=dummy_config,
    )
