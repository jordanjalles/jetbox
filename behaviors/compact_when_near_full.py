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

    def on_round_start(
        self,
        agent: Any,
        round_number: int,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Called at start of every round - compact context if needed.

        This method:
        1. Assumes context already has system prompt + goal + messages
        2. Estimates token count
        3. If > threshold: summarizes old messages, keeps recent ones
        4. Returns modified context

        Args:
            agent: Agent instance (not currently used by this behavior)
            round_number: Current round number
            context: Current context (system + messages)

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
            percent_used = estimated_tokens / self.max_tokens * 100
            print(f"[compact_when_near_full] Context at {estimated_tokens:,} tokens "
                  f"({percent_used:.1f}% of {self.max_tokens:,}) "
                  f"- triggering compaction")

            # AGGRESSIVE: If already way over limit (>100%), keep only 5 recent messages
            # Otherwise use configured keep_recent
            if estimated_tokens > self.max_tokens:
                keep_recent = min(5, self.keep_recent)
                print(f"[compact_when_near_full] ⚠️  OVER LIMIT ({percent_used:.0f}%) - emergency compaction, keeping only {keep_recent} messages")
            else:
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

                # HARD LIMIT: If still over max_tokens, aggressively drop messages
                if new_tokens > self.max_tokens:
                    print("[compact_when_near_full] ⚠️  STILL OVER LIMIT after compaction - dropping oldest messages")
                    # Keep system + goal + last 5 messages only
                    system_and_goal = context_base[:messages_start_idx]
                    recent_msgs = context_base[-5:] if len(context_base) > 5 else context_base[messages_start_idx:]
                    context_base = system_and_goal + recent_msgs
                    final_tokens = self._estimate_context_size(context_base)
                    print(f"[compact_when_near_full] Emergency truncation: {new_tokens:,} → {final_tokens:,} tokens")

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

        FIXED: Now includes tool_calls, role overhead, and message structure.

        Args:
            context: Context to estimate

        Returns:
            Estimated token count
        """
        total_chars = 0
        for msg in context:
            # Count content
            content = msg.get("content", "")
            if content:
                total_chars += len(str(content))

            # Count tool_calls (can be huge!)
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # Convert to string to get approximate size
                # (ToolCall objects may not be JSON serializable)
                total_chars += len(str(tool_calls))

            # Add overhead for role, structure (approximately 20 chars per message)
            total_chars += 20

        # 4 chars per token is standard heuristic
        return total_chars // 4

    def _summarize_messages(self, messages: list[dict[str, Any]]) -> str:
        """
        Use LLM to summarize a sequence of messages.

        Args:
            messages: List of message dicts to summarize

        Returns:
            Concise summary of the messages
        """
        from utils.llm_utils import summarize_messages

        try:
            summary = summarize_messages(
                messages=messages,
                model="gpt-oss:20b",
                temperature=0.2,
            )
            return summary if summary else "Unable to generate summary."
        except Exception as e:
            # If summarization fails, return a basic summary
            return f"[Summarization failed: {e}] Previous work included {len(messages)} message exchanges with tool calls and results."

    def get_instructions(self) -> str:
        """
        Return workflow instructions for this behavior.

        Returns:
            Instructions for completing work
        """
        return """
WORKFLOW:
Your goal is shown at the start of the conversation. Simply complete the work using the available tools.

- Use write_file to create/modify files
- Use run_bash to run commands (tests, linters, etc.)
- Use read_file to check existing files
- Use list_dir to explore directories

Work directly on the goal - no need to decompose into subtasks.
"""
