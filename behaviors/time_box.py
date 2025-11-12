"""
TimeBoxBehavior - Automatic temporal awareness for agents.

Provides:
- Automatic factual time nudges at configurable percentages
- Tool for agents to schedule custom reminders to themselves
- Wall-clock or round-based time tracking
"""
from __future__ import annotations
import time
from typing import Any
from behaviors.base import AgentBehavior


class TimeBoxBehavior(AgentBehavior):
    """
    Automatic temporal awareness with agent-scheduled reminders.

    Injects factual time nudges (e.g., "25% of time elapsed") at configurable
    intervals. Agents can schedule custom reminders using schedule_reminder().

    Time tracking uses wall-clock if budget set, otherwise falls back to rounds.

    Security: [] None (utility behavior, no security properties)
    """

    # Rule of Two: Empty (utility behavior for time tracking)
    rule_of_two_properties = set()

    def __init__(
        self,
        total_budget_minutes: int | None = None,
        default_nudges: list[int] | None = None
    ):
        """
        Initialize TimeBoxBehavior.

        Args:
            total_budget_minutes: Wall-clock budget (None = use rounds only)
            default_nudges: Percentages for factual nudges (default: [25, 50, 75])
        """
        super().__init__()
        self.budget_minutes = total_budget_minutes
        self.default_nudges = default_nudges or [25, 50, 75]
        self.custom_reminders: list[tuple[float, str]] = []
        self.triggered: set[float] = set()

    def get_name(self) -> str:
        """Return behavior name."""
        return "timebox"

    def on_goal_start(self, agent, **kwargs) -> None:
        """Track goal start time."""
        agent.goal_start_time = time.time()
        self.triggered.clear()
        self.custom_reminders.clear()

    def get_tools(self) -> list[dict[str, Any]]:
        """Provide schedule_reminder tool."""
        return [{
            "type": "function",
            "function": {
                "name": "schedule_reminder",
                "description": """Schedule a message to yourself at a future time percentage. Use this to plan phase transitions, strategic checkpoints, scope adjustments, or remind yourself of tradeoffs as time progresses. You receive automatic time nudges at 25%, 50%, 75%.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "at_percent": {
                            "type": "number",
                            "description": "Percentage (0-100) when this reminder should trigger",
                            "minimum": 0,
                            "maximum": 100
                        },
                        "message": {
                            "type": "string",
                            "description": "The reminder message to show yourself at that time"
                        }
                    },
                    "required": ["at_percent", "message"]
                }
            }
        }]

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        agent=None,
        **kwargs
    ) -> dict[str, Any]:
        """Handle schedule_reminder tool calls."""
        if tool_name == "schedule_reminder":
            percent = args["at_percent"]
            message = args["message"]

            # Store reminder
            self.custom_reminders.append((percent, message))

            return {
                "success": True,
                "scheduled_at": f"{percent}%",
                "message_preview": message[:60] + ("..." if len(message) > 60 else "")
            }

        return super().dispatch_tool(tool_name, args, agent=agent, **kwargs)

    def on_round_start(
        self,
        agent,
        round_number: int,
        context: list[dict[str, Any]],
        **kwargs
    ) -> list[dict[str, Any]]:
        """Auto-inject nudges and reminders at thresholds."""
        # Calculate current percentage
        if self.budget_minutes:
            # Wall-clock based
            elapsed = time.time() - agent.goal_start_time
            percent = (elapsed / 60) / self.budget_minutes * 100
        else:
            # Round-based fallback
            percent = (round_number / agent.max_rounds) * 100

        # Inject default factual nudges
        for nudge_percent in self.default_nudges:
            if percent >= nudge_percent and nudge_percent not in self.triggered:
                self._inject_factual_nudge(agent, nudge_percent, context)
                self.triggered.add(nudge_percent)

        # Inject custom agent-scheduled reminders
        for reminder_percent, message in self.custom_reminders:
            if percent >= reminder_percent and reminder_percent not in self.triggered:
                self._inject_custom_reminder(agent, reminder_percent, message, context)
                self.triggered.add(reminder_percent)

        return context

    def _inject_factual_nudge(
        self,
        agent,
        percent: float,
        context: list[dict[str, Any]]
    ) -> None:
        """Inject neutral factual time nudge into persistent message history."""
        goal = getattr(agent, 'goal', 'current task')

        # Neutral, factual tone
        if self.budget_minutes:
            elapsed_min = (time.time() - agent.goal_start_time) / 60
            msg = f"{percent}% of your {self.budget_minutes}-minute time budget for '{goal}' has elapsed ({elapsed_min:.1f} minutes used)."
        else:
            msg = f"{percent}% of your time for '{goal}' has elapsed."

        # Add to persistent message history (not transient context)
        nudge_message = {"role": "user", "content": msg}
        agent.state.messages.append(nudge_message)

    def _inject_custom_reminder(
        self,
        agent,
        percent: float,
        message: str,
        context: list[dict[str, Any]]
    ) -> None:
        """Inject agent-scheduled custom reminder into persistent message history."""
        msg = f"[Scheduled reminder at {percent}%]\n{message}"

        # Add to persistent message history (not transient context)
        reminder_message = {"role": "user", "content": msg}
        agent.state.messages.append(reminder_message)
