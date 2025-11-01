"""
SubAgentContextBehavior - Context management for delegated work.

This behavior is for agents invoked by a controlling agent (orchestrator)
to complete a specific delegated task.

Features:
- Context: "DELEGATED GOAL" injection at top
- Tools: mark_complete, mark_failed
- Max tokens: 128000 (large limit for delegated work)
- Timeout nudging logic
- Append-until-full with compaction
"""

from typing import Any
from behaviors.base import AgentBehavior


class SubAgentContextBehavior(AgentBehavior):
    """
    Context behavior for sub-agents working on delegated tasks.

    Designed for agents invoked by a controlling agent (orchestrator)
    to complete a specific delegated task. Provides completion tools
    to report results back to the controlling agent.

    Features:
    - Inserts "DELEGATED GOAL" at top of context
    - Provides mark_complete/mark_failed tools
    - Appends all messages until near token limit
    - Compacts when needed (at 75% threshold)
    - Higher token limit (128K) for complex delegated work
    """

    def __init__(
        self,
        max_tokens: int = 128000,
        compact_threshold: float = 0.75,
        keep_recent: int = 20,
    ):
        """
        Initialize sub-agent context behavior.

        Args:
            max_tokens: Maximum context tokens before compaction (default: 128K)
            compact_threshold: Trigger compaction at this fraction (default: 0.75)
            keep_recent: Number of recent messages to keep during compaction (default: 20)
        """
        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self.keep_recent = keep_recent

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "subagent_context"

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Enhance context with delegated goal information.

        This method:
        1. Injects "DELEGATED GOAL" header after system prompt
        2. Adds completion instructions
        3. Includes jetbox notes if available
        4. Compacts if context exceeds threshold

        Args:
            context: Current context (system + messages)
            **kwargs: Additional context (context_manager, workspace)

        Returns:
            Modified context with delegated goal info
        """
        context_manager = kwargs.get('context_manager')
        workspace = kwargs.get('workspace')

        if not context_manager:
            return context

        # Build delegated goal context
        context_parts = []

        if context_manager.state.goal:
            context_parts.append(f"DELEGATED GOAL: {context_manager.state.goal.description}")
            context_parts.append("")
            context_parts.append("You are working on a task delegated by the orchestrator.")
            context_parts.append("When complete, call mark_complete(summary) with what you accomplished.")
            context_parts.append("If you cannot complete it, call mark_failed(reason) explaining why.")

        # Add jetbox notes if available
        if workspace:
            notes_content = self._get_jetbox_notes()
            if notes_content:
                context_parts.append("")
                context_parts.append("="*70)
                context_parts.append("PREVIOUS WORK (from jetboxnotes.md)")
                context_parts.append("="*70)
                context_parts.append(notes_content)
                context_parts.append("="*70)

        # Insert after system prompt (index 1)
        if context_parts and len(context) > 0:
            context.insert(1, {
                "role": "user",
                "content": "\n".join(context_parts)
            })

        # Check if compaction needed
        # Find where messages start (after system + delegated goal)
        messages_start_idx = 2  # system + delegated goal
        if len(context) > messages_start_idx:
            messages = context[messages_start_idx:]

            # Estimate token count
            estimated_tokens = self._estimate_context_size(context)

            if estimated_tokens > self.max_tokens * self.compact_threshold:
                print(f"[subagent_context] Context at {estimated_tokens:,} tokens "
                      f"({estimated_tokens/self.max_tokens*100:.1f}% of {self.max_tokens:,}) "
                      f"- triggering LLM summarization")

                # Keep recent N messages, summarize the rest
                keep_recent = self.keep_recent
                to_summarize = messages[:-keep_recent] if len(messages) > keep_recent else []

                if to_summarize:
                    # Use LLM to summarize old messages
                    summary = self._summarize_messages(to_summarize)

                    # Rebuild context: base + summary + recent
                    context_base = context[:messages_start_idx]
                    context_base.append({
                        "role": "user",
                        "content": f"Previous work summary (compacted from {len(to_summarize)} messages):\n{summary}"
                    })
                    context_base.extend(messages[-keep_recent:])

                    new_tokens = self._estimate_context_size(context_base)
                    print(f"[subagent_context] Reduced from {estimated_tokens:,} to {new_tokens:,} tokens "
                          f"({new_tokens/self.max_tokens*100:.1f}%)")

                    return context_base

        return context

    def _get_jetbox_notes(self) -> str | None:
        """
        Get jetbox notes content if available.

        Returns:
            Notes content string or None
        """
        try:
            import jetbox_notes
            return jetbox_notes.load_jetbox_notes(max_chars=2000)
        except Exception:
            return None

    def _estimate_context_size(self, context: list[dict[str, Any]]) -> int:
        """
        Estimate context size using 4 chars per token heuristic.

        Args:
            context: Context to estimate

        Returns:
            Estimated token count
        """
        total_chars = 0
        for msg in context:
            total_chars += len(str(msg.get("content", "")))
        return total_chars // 4

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

    def get_tools(self) -> list[dict[str, Any]]:
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

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Handle tool calls for this behavior.

        Args:
            tool_name: Tool being called
            args: Tool arguments
            **kwargs: Additional context

        Returns:
            Tool result dict
        """
        context_manager = kwargs.get('context_manager')

        if tool_name == "mark_complete":
            summary = args.get('summary', 'Task completed')

            # Mark goal as complete in context manager if available
            if context_manager and hasattr(context_manager, 'state') and context_manager.state.goal:
                context_manager.state.goal.mark_complete(success=True)

            return {
                "success": True,
                "result": f"Task marked complete: {summary}",
                "summary": summary
            }

        elif tool_name == "mark_failed":
            reason = args.get('reason', 'Task failed')

            # Mark goal as failed in context manager if available
            if context_manager and hasattr(context_manager, 'state') and context_manager.state.goal:
                context_manager.state.goal.mark_complete(success=False)

            return {
                "success": False,
                "result": f"Task marked failed: {reason}",
                "reason": reason
            }

        return super().dispatch_tool(tool_name, args, **kwargs)

    def get_instructions(self) -> str:
        """
        Return sub-agent workflow instructions.

        Returns:
            Instructions for completing delegated work
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
