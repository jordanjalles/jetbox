"""
Context building strategies for Jetbox agents.

Provides different context management strategies:
- Hierarchical: For task-focused agents (TaskExecutor, standalone agent)
- Append-until-full: For conversational agents (Orchestrator)
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from context_manager import ContextManager
    from agent_config import AgentConfig


def build_hierarchical_context(
    context_manager: ContextManager,
    messages: list[dict[str, Any]],
    system_prompt: str,
    config: Any,
    probe_state_func: callable | None = None,
    workspace=None,  # Add workspace parameter for jetbox notes
) -> list[dict[str, Any]]:
    """
    Build context using hierarchical strategy.

    Strategy:
    - System prompt
    - Current goal/task/subtask info
    - Loop detection warnings (if applicable)
    - Filesystem probe state (if provided)
    - Last N message exchanges

    This keeps the agent focused on the current subtask and prevents
    context bloat by limiting message history.

    Args:
        context_manager: ContextManager with goal/task/subtask hierarchy
        messages: Full message history
        system_prompt: System prompt for the agent
        config: Configuration object with rounds.max_per_subtask, etc.
        probe_state_func: Optional function to probe filesystem state

    Returns:
        Context list ready for LLM
    """
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

        # Add jetbox notes if workspace provided and notes exist
        if workspace:
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

    # Add last N message exchanges
    # Use HISTORY_KEEP from config if available, otherwise default to 12
    history_keep = 12
    if hasattr(config, 'context') and hasattr(config.context, 'history_keep'):
        history_keep = config.context.history_keep

    recent = messages[-history_keep * 2:] if len(messages) > history_keep * 2 else messages
    context.extend(recent)

    return context


def build_simple_hierarchical_context(
    context_manager: ContextManager,
    messages: list[dict[str, Any]],
    system_prompt: str,
    config: Any,
) -> list[dict[str, Any]]:
    """
    Simplified hierarchical context builder without loop detection or probing.

    This is used by task_executor_agent.py which doesn't have loop detection
    or filesystem probing built in.

    Args:
        context_manager: ContextManager with goal/task/subtask hierarchy
        messages: Full message history
        system_prompt: System prompt for the agent
        config: Configuration object

    Returns:
        Context list ready for LLM
    """
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

            # Add depth/rounds info
            if hasattr(config, 'hierarchy') and hasattr(config.hierarchy, 'max_depth'):
                context_info.append(f"Subtask Depth: {subtask.depth}/{config.hierarchy.max_depth}")

            if hasattr(config, 'rounds') and hasattr(config.rounds, 'max_per_subtask'):
                context_info.append(f"Rounds Used: {subtask.rounds_used}/{config.rounds.max_per_subtask}")
        else:
            context_info.append("ACTIVE SUBTASK: (none - call mark_subtask_complete to advance)")

        context.append({"role": "user", "content": "\n".join(context_info)})

    # Add last N message exchanges
    history_keep = config.context.history_keep if hasattr(config, 'context') else 12
    recent = messages[-history_keep * 2:] if len(messages) > history_keep * 2 else messages
    context.extend(recent)

    return context


def build_append_context(
    messages: list[dict[str, Any]],
    system_prompt: str,
    max_tokens: int = 8000,
    recent_keep: int = 20,
) -> list[dict[str, Any]]:
    """
    Build context using append-until-full strategy.

    Strategy:
    - System prompt
    - Append all messages until approaching token limit
    - When near limit (80%), compact:
      * Keep recent N messages intact
      * Summarize older messages

    This preserves conversation history for conversational agents
    like the Orchestrator.

    Args:
        messages: Full message history
        system_prompt: System prompt for the agent
        max_tokens: Maximum context tokens (default 8000)
        recent_keep: Number of recent messages to keep intact (default 20)

    Returns:
        Context list ready for LLM
    """
    context = [{"role": "system", "content": system_prompt}]

    # Simple heuristic: 4 chars per token
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    # Calculate current token usage
    total_tokens = estimate_tokens(system_prompt)
    for msg in messages:
        total_tokens += estimate_tokens(str(msg.get("content", "")))

    # If under 80% of limit, just append all
    if total_tokens < max_tokens * 0.8:
        context.extend(messages)
    else:
        # Compact: keep recent messages, summarize older ones
        if len(messages) <= recent_keep:
            context.extend(messages)
        else:
            # Summarize older messages
            older_messages = messages[:-recent_keep]
            summary_parts = []
            for msg in older_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                summary_parts.append(f"[{role}]: {content[:100]}...")

            summary = "\n".join(summary_parts[:50])  # Limit to 50 older messages
            context.append({
                "role": "system",
                "content": f"[Earlier conversation summary]:\n{summary}"
            })

            # Add recent messages intact
            context.extend(messages[-recent_keep:])

    return context
