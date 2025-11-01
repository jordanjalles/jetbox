"""
CompactWhenNearFullBehavior - Append messages until near limit, then compact via LLM.

This behavior merges the functionality of:
- AppendUntilFullStrategy: Append all messages until token limit
- ContextCompaction: Compact via LLM summarization at 75% threshold

Features:
- Appends all messages to context (no truncation)
- Monitors token usage (estimate via character count)
- Compacts at 75% threshold via LLM summarization
- Preserves recent messages (last 3-5 exchanges)
- Tools: mark_goal_complete
"""

from typing import Any
from behaviors.base import AgentBehavior


class CompactWhenNearFullBehavior(AgentBehavior):
    """
    Context behavior that appends all messages until near token limit.

    When context exceeds 75% of max_tokens, this behavior:
    1. Keeps recent N messages intact
    2. Summarizes older messages via LLM
    3. Rebuilds context: system + goal + summary + recent messages

    This provides the best of both worlds:
    - Full message history when possible
    - Automatic compaction when needed
    - Recent context always preserved
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        compact_threshold: float = 0.75,
        keep_recent: int = 20,
    ):
        """
        Initialize compact-when-near-full behavior.

        Args:
            max_tokens: Maximum context tokens before compaction (default: 8000)
            compact_threshold: Trigger compaction at this fraction of max_tokens (default: 0.75)
            keep_recent: Number of recent messages to keep intact during compaction (default: 20)
        """
        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self.keep_recent = keep_recent

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "compact_when_near_full"

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Enhance context by appending all messages and compacting if needed.

        This method:
        1. Assumes context already has system prompt + goal + messages
        2. Estimates token count
        3. If > threshold: summarizes old messages, keeps recent ones
        4. Returns modified context

        Args:
            context: Current context (system + messages)
            **kwargs: Additional context (context_manager, workspace, etc.)

        Returns:
            Modified context (possibly compacted)
        """
        # Early exit if no messages to compact
        if len(context) <= 2:  # Just system + goal
            return context

        # Find where messages start (after system prompt and goal injection)
        messages_start_idx = 1
        for i, msg in enumerate(context):
            if i == 0:  # Skip system prompt
                continue
            if msg.get('role') == 'user' and ('GOAL:' in msg.get('content', '') or 'DELEGATED GOAL:' in msg.get('content', '')):
                messages_start_idx = i + 1
                break

        # Extract messages (everything after system + goal)
        messages = context[messages_start_idx:]

        if not messages:
            return context

        # Check if context exceeds threshold
        estimated_tokens = self._estimate_context_size(context)

        if estimated_tokens > self.max_tokens * self.compact_threshold:
            print(f"[compact_when_near_full] Context at {estimated_tokens:,} tokens "
                  f"({estimated_tokens/self.max_tokens*100:.1f}% of {self.max_tokens:,}) "
                  f"- triggering LLM summarization")

            # Keep recent N messages, summarize the rest
            keep_recent = self.keep_recent
            to_summarize = messages[:-keep_recent] if len(messages) > keep_recent else []

            if to_summarize:
                # Use LLM to summarize old messages
                summary = self._summarize_messages(to_summarize)

                # Rebuild context: base (system + goal) + summary + recent messages
                context_base = context[:messages_start_idx]

                # Add summary as user message
                context_base.append({
                    "role": "user",
                    "content": f"Previous work summary (compacted from {len(to_summarize)} messages):\n{summary}"
                })

                # Add recent messages
                context_base.extend(messages[-keep_recent:])

                new_tokens = self._estimate_context_size(context_base)
                print(f"[compact_when_near_full] Reduced from {estimated_tokens:,} to {new_tokens:,} tokens "
                      f"({new_tokens/self.max_tokens*100:.1f}%)")

                return context_base
            else:
                # Not enough messages to summarize, just keep recent
                print(f"[compact_when_near_full] Not enough messages to summarize, keeping last {keep_recent}")
                context_base = context[:messages_start_idx]
                context_base.extend(messages[-keep_recent:])
                return context_base

        return context

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

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Provide mark_goal_complete tool.

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
        if tool_name == "mark_goal_complete":
            # Signal goal completion
            context_manager = kwargs.get('context_manager')
            summary = args.get('summary', 'Goal completed')

            # Mark goal as complete in context manager if available
            if context_manager and hasattr(context_manager, 'state') and context_manager.state.goal:
                context_manager.state.goal.mark_complete(success=True)

            return {
                "success": True,
                "result": f"Goal marked complete: {summary}",
                "summary": summary
            }

        return super().dispatch_tool(tool_name, args, **kwargs)

    def get_instructions(self) -> str:
        """
        Return workflow instructions for this behavior.

        Returns:
            Instructions for using mark_goal_complete
        """
        return """
WORKFLOW:
Your goal is shown at the start of the conversation. Simply complete the work using the available tools.

- Use write_file to create/modify files
- Use run_bash to run commands (tests, linters, etc.)
- Use read_file to check existing files
- Use list_dir to explore directories

Work directly on the goal - no need to decompose into subtasks.
When all work is complete and tests pass, call mark_goal_complete(summary="what you accomplished").
"""
