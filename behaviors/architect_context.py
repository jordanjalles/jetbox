"""
ArchitectContextBehavior - Context management for architecture design.

This behavior is optimized for architecture design conversations where
artifacts (documents, specs) are the primary output.

Features:
- Context: Architecture design focus
- Max tokens: 32000 (higher for verbose discussions)
- No jetbox notes by default (artifacts are the output)
- Similar to CompactWhenNearFull but architecture-optimized
"""

from typing import Any
from behaviors.base import AgentBehavior


class ArchitectContextBehavior(AgentBehavior):
    """
    Context behavior for architecture design agents.

    Optimized for architecture design conversations where artifacts
    (architecture docs, module specs, task lists) are the primary output.

    Features:
    - Higher token limit (32K) for verbose architecture discussions
    - Architecture-focused message summarization
    - Does NOT use jetbox notes by default (artifacts persist)
    - Compacts when near limit (75% threshold)
    """

    def __init__(
        self,
        max_tokens: int = 32000,
        compact_threshold: float = 0.75,
        keep_recent: int = 20,
    ):
        """
        Initialize architect context behavior.

        Args:
            max_tokens: Maximum context tokens before compaction (default: 32K)
            compact_threshold: Trigger compaction at this fraction (default: 0.75)
            keep_recent: Number of recent messages to keep during compaction (default: 20)
        """
        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self.keep_recent = keep_recent

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "architect_context"

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Enhance context with project information and compact if needed.

        This method:
        1. Injects "PROJECT" header after system prompt
        2. Compacts when context exceeds threshold
        3. Uses architecture-focused summarization

        Args:
            context: Current context (system + messages)
            **kwargs: Additional context (context_manager, workspace)

        Returns:
            Modified context with project info
        """
        context_manager = kwargs.get('context_manager')

        if not context_manager:
            return context

        # Build project context (simple, not hierarchical)
        context_parts = []

        if context_manager.state.goal:
            context_parts.append(f"PROJECT: {context_manager.state.goal.description}")

        # Insert after system prompt (index 1)
        if context_parts and len(context) > 0:
            context.insert(1, {
                "role": "user",
                "content": "\n".join(context_parts)
            })

        # Check if compaction needed
        # Find where messages start (after system + project)
        messages_start_idx = 2  # system + project
        if len(context) > messages_start_idx:
            messages = context[messages_start_idx:]

            # Estimate token count
            estimated_tokens = self._estimate_context_size(context)

            # Use 131072 (128K) as upper bound for safety
            max_context = 131072

            if estimated_tokens > max_context * 0.75:
                print(f"[architect_context] Context at {estimated_tokens:,} tokens "
                      f"({estimated_tokens/max_context*100:.1f}%) - triggering summarization")

                # Keep recent N messages, summarize the rest
                keep_recent = self.keep_recent
                to_summarize = messages[:-keep_recent] if len(messages) > keep_recent else []

                if to_summarize:
                    # Use architecture-focused summarization
                    summary = self._summarize_architecture_messages(to_summarize)

                    # Rebuild context: base + summary + recent
                    context_base = context[:messages_start_idx]
                    context_base.append({
                        "role": "user",
                        "content": f"Previous discussion summary ({len(to_summarize)} messages):\n{summary}"
                    })
                    context_base.extend(messages[-keep_recent:])

                    new_tokens = self._estimate_context_size(context_base)
                    print(f"[architect_context] Reduced from {estimated_tokens:,} to {new_tokens:,} tokens")

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

    def _summarize_architecture_messages(self, messages: list[dict[str, Any]]) -> str:
        """
        Summarize messages with focus on architecture decisions.

        Args:
            messages: Messages to summarize

        Returns:
            Architecture-focused summary
        """
        from llm_utils import chat_with_inactivity_timeout

        # Build concise summary prompt focusing on architecture decisions
        messages_text = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                tool_names = [tc.get("function", {}).get("name", "unknown") for tc in tool_calls]
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

    def get_instructions(self) -> str:
        """
        Return architect workflow instructions.

        Returns:
            Instructions for architecture design work
        """
        return """
ARCHITECTURE DESIGN WORKFLOW:
You are designing the architecture for a software project.

Your job is to:
1. Understand project requirements and constraints
2. Design system architecture and component interactions
3. Make technology choices with clear rationale
4. Document decisions in architecture artifacts

Available tools:
- write_architecture_doc: Create architecture overview document
- write_module_spec: Create detailed module specifications
- write_task_list: Create implementation task breakdown
- clarify_with_user: Ask questions about requirements

IMPORTANT:
- Focus on high-level design, not implementation details
- Consider scalability, maintainability, and technology tradeoffs
- Document your reasoning for major decisions
- Ask clarifying questions when requirements are unclear
- Create comprehensive artifacts that guide implementation

Your artifacts (docs, specs) are the primary output of this conversation.
"""
