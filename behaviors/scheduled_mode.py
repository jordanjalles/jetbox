"""Scheduled mode behavior for clean task execution in scheduled agents."""

import logging

from behaviors.base import AgentBehavior

logger = logging.getLogger(__name__)


class ScheduledModeBehavior(AgentBehavior):
    """Behavior that helps agents run cleanly in scheduled mode.

    In scheduled mode, agents should:
    - Complete their assigned task
    - Exit cleanly (not run indefinitely)
    - Report completion status

    This behavior injects context to guide the agent's behavior.

    Security: No Rule of Two properties - just context injection.
    """

    # No security properties - this behavior doesn't access files,
    # sensitive data, or perform external actions
    rule_of_two_properties = set()

    def __init__(self, max_rounds: int = 10):
        """Initialize scheduled mode behavior.

        Args:
            max_rounds: Maximum rounds before forcing completion (default: 10)
        """
        self.max_rounds = max_rounds
        self.round_count = 0

    def get_name(self) -> str:
        """Get behavior name."""
        return "scheduled_mode"

    def on_initial_context(self, agent, context: list[dict]) -> list[dict]:
        """Inject scheduled mode instructions into initial context.

        Args:
            agent: The agent instance
            context: Current context messages

        Returns:
            Modified context
        """
        # Add scheduled mode instructions to system prompt
        scheduled_mode_instructions = {
            "role": "system",
            "content": """
# SCHEDULED MODE

You are running in SCHEDULED MODE. This means:

1. **Complete your task efficiently** - You have one clear goal to accomplish
2. **Exit cleanly when done** - Don't loop indefinitely or wait for more input
3. **Report results** - Clearly indicate what you accomplished
4. **Handle errors gracefully** - If you can't complete the task, explain why

**When you're done with your task, use the `mark_subtask_complete()` tool to signal completion.**

Focus on executing the assigned task and exiting successfully.
""",
        }

        # Insert after system prompt
        context.insert(1, scheduled_mode_instructions)

        return context

    def on_round_start(
        self, agent, round_number: int, context: list[dict]
    ) -> list[dict]:
        """Track rounds and nudge toward completion if needed.

        Args:
            agent: The agent instance
            round_number: Current round number
            context: Current context messages

        Returns:
            Modified context
        """
        self.round_count = round_number

        # If approaching max rounds, add a nudge
        if round_number >= self.max_rounds - 2:
            nudge = {
                "role": "system",
                "content": f"""
**REMINDER**: You are in scheduled mode. You've used {round_number}/{self.max_rounds} rounds.
Please complete your task and call `mark_subtask_complete()` to exit cleanly.
""",
            }

            context.append(nudge)

        return context

    def on_llm_response(self, agent, response: dict) -> dict:
        """Monitor LLM response for completion signals.

        Args:
            agent: The agent instance
            response: LLM response

        Returns:
            The response unchanged
        """
        # Check if agent is trying to loop or wait
        content = response.get("content", "").lower()

        if any(
            phrase in content
            for phrase in [
                "waiting for",
                "will check again",
                "monitoring",
                "continuously",
            ]
        ):
            logger.warning(
                "Agent may be trying to run continuously in scheduled mode"
            )

        return response

    def on_goal_complete(self, agent, success: bool, summary: str) -> None:
        """Handle goal completion.

        Args:
            agent: The agent instance
            success: Whether goal was completed successfully
            summary: Completion summary
        """
        if success:
            logger.info(f"Scheduled task completed successfully: {summary}")
        else:
            logger.warning(f"Scheduled task failed: {summary}")
