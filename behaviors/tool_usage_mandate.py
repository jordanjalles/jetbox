"""Tool Usage Mandate Behavior

BEHAVIOR: Injects critical tool usage instruction into initial context
PURPOSE: Ensures LLM understands it MUST use tools, not respond with text only

Lifecycle:
- Event: on_initial_context(agent, context) - Inject tool usage mandate once
- No tools (context enhancement only)

Why This Exists:
- After KERNEL prompt shrinkage, tool calling instructions were removed
- qwen3-coder and other models need explicit instruction to use tools
- Without this, models may respond with text instead of calling tools

Design:
- Injects instruction BEFORE tool descriptions
- Should be loaded early in behavior list
- Applies to all agents that use tools
"""

from typing import Any
from behaviors.base import AgentBehavior


class ToolUsageMandateBehavior(AgentBehavior):
    """
    Injects critical tool usage instruction into agent context.

    This behavior ensures the LLM knows it must use tools, not respond with text only.
    Should be loaded by all agents that provide tools (task_executor, orchestrator, etc.)
    """

    def __init__(self, params: dict[str, Any] | None = None):
        """Initialize behavior with optional parameters.

        Args:
            params: Configuration parameters (currently unused)
        """
        super().__init__()
        self.params = params or {}

    def get_name(self) -> str:
        """Return behavior name for debugging/logging."""
        return "tool_usage_mandate"

    def get_instructions(self) -> str:
        """
        Return instructions to add to system prompt.

        Note: We inject via on_initial_context instead to ensure it appears
        AFTER system prompt but BEFORE tool descriptions.
        """
        return ""

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject tool usage mandate into initial context (called ONCE).

        Injects a user message right after the system prompt instructing
        the LLM to always use tools instead of responding with text only.

        Args:
            agent: The agent instance
            context: Current context (system prompt + any initial messages)

        Returns:
            Enhanced context with tool usage instruction
        """
        instruction = (
            "IMPORTANT: You MUST use the available tools to complete tasks. "
            "Never respond with explanatory text only - always call appropriate tools. "
            "If you need to explain something, call a tool AND provide explanation in the tool's output."
        )

        # Inject after system prompt (position 1)
        return self.inject_user_message_after_system(context, instruction)

    def get_tools(self) -> list[dict[str, Any]]:
        """This behavior doesn't provide tools."""
        return []

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any
    ) -> dict[str, Any] | None:
        """This behavior doesn't handle tool calls."""
        return None
