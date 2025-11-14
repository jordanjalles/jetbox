"""
{BEHAVIOR_NAME}Behavior - {ONE_SENTENCE_DESCRIPTION}

Enhances context by: {WHAT_IT_INJECTS}
"""

from typing import Any
from behaviors.base import AgentBehavior


class {BEHAVIOR_CLASS_NAME}Behavior(AgentBehavior):
    """
    Behavior that enhances context with {WHAT_INFORMATION}.
    """

    def __init__(self, **kwargs):
        self.state = {}  # Track any state needed

    def get_name(self) -> str:
        return "{BEHAVIOR_NAME}"

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject {WHAT_INFORMATION} into initial context (called ONCE).

        Use this for static content that doesn't change between rounds.
        This is also where behaviors with tools should inject tool documentation.

        Args:
            agent: Agent instance (access agent.goal, agent.workspace, etc.)
            context: Initial context (system prompt only)

        Returns:
            Modified context with injected information
        """
        # Extract info from agent
        goal = agent.goal if hasattr(agent, 'goal') else ''

        # Build message to inject
        message = f"{CONTEXT_HEADER}: {goal}"

        # Use helper to inject after system prompt
        # Adjust role as needed: "system" for framework, "user" for context
        return self.inject_message_after_system(context, message, role="user")

        # NOTE: If this behavior provides tools, inject tool documentation here.
        # See behavior_with_tools_template.py for the pattern.

    def on_round_start(
        self,
        agent: Any,
        round_number: int,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject dynamic {WHAT_INFORMATION} into context (called EVERY round).

        Use this for dynamic content that changes between rounds (warnings, prompts).

        Args:
            agent: Agent instance
            round_number: Current round number
            context: Current context

        Returns:
            Modified context
        """
        # Example: Inject dynamic warning based on state
        if self.state.get('should_warn'):
            warning = "⚠️ {DYNAMIC_WARNING}"
            # Adjust role as needed
                context = self.inject_message_after_system(context, warning, role="user")

        return context
