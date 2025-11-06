"""
{BEHAVIOR_NAME}Behavior - {ONE_SENTENCE_DESCRIPTION}

Enhances context by: {WHAT_IT_INJECTS}
"""

from typing import Any, TYPE_CHECKING
from behaviors.base import AgentBehavior

if TYPE_CHECKING:
    from behaviors.base_agent import BaseAgent


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
        agent: "BaseAgent",
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject {WHAT_INFORMATION} into initial context (called ONCE).

        Use this for static content that doesn't change between rounds.

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
        return self.inject_user_message_after_system(context, message)

    def on_round_start(
        self,
        agent: "BaseAgent",
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
            context = self.inject_user_message_after_system(context, warning)

        return context
