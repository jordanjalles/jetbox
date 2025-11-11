"""
Event system for triggering lifecycle events on agent behaviors.

This module handles all event propagation from the agent to registered behaviors.

Lifecycle Events:
- on_goal_start: Called once when a goal is set
- on_initial_context: Called once at start of run() to inject initial context
- on_round_start: Called at start of each round (can modify context)
- on_llm_response: Called after LLM responds but before tool dispatch
- on_tool_call: Called after each tool execution
- on_round_end: Called at end of each round
- on_goal_complete: Called when goal completes (success or failure)
- on_timeout: Called when goal times out

Custom Events:
- on_custom_event: Called when behaviors broadcast custom events
- fire_custom_event: Method to broadcast custom events to all behaviors

Custom events enable inter-behavior communication without tight coupling.
Behaviors can use this for mode coordination, state synchronization, and
other cross-behavior communication patterns.
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
import inspect

if TYPE_CHECKING:
    from base_agent import BaseAgent


class EventSystem:
    """
    Manages event propagation from agent to behaviors.

    This class centralizes all behavior event triggering logic, making it
    easier to understand and maintain the event flow.
    """

    def __init__(self, agent: BaseAgent):
        """
        Initialize event system.

        Args:
            agent: Reference to the agent that owns this event system
        """
        self.agent = agent

    def trigger_goal_start(self, goal: str) -> None:
        """
        Trigger on_goal_start event on all behaviors.

        Called ONCE when a goal is set, before any rounds.

        Args:
            goal: Goal description string
        """
        for behavior in self.agent._behaviors:
            try:
                # Try new API first
                if hasattr(behavior, 'on_goal_start') and callable(behavior.on_goal_start):
                    behavior.on_goal_start(agent=self.agent, goal=goal)
                # Fall back to old API if new one doesn't exist
                elif hasattr(behavior, 'onGoalStart') and callable(behavior.onGoalStart):
                    behavior.onGoalStart(goal=goal, agent=self.agent)
            except Exception as e:
                print(f"[{self.agent.name}] Behavior {behavior.get_name()} on_goal_start error: {e}")

    def inject_initial_context(self) -> None:
        """
        Trigger on_initial_context event on all behaviors ONCE.

        Called at start of run(), before first LLM call.
        This allows behaviors to inject static context that doesn't change per round.

        NOTE: This modifies self.agent.state.messages to inject initial context messages.
        """
        # Build initial context (system prompt only, no history yet)
        initial_context = [
            {"role": "system", "content": self.agent.get_system_prompt()}
        ]

        # Inject goal context if goal is set (before behaviors)
        if self.agent.goal:
            initial_context = self.agent._inject_goal_context(initial_context)

        # Let behaviors inject initial context
        for behavior in self.agent._behaviors:
            try:
                if hasattr(behavior, 'on_initial_context') and callable(behavior.on_initial_context):
                    initial_context = behavior.on_initial_context(agent=self.agent, context=initial_context)
            except Exception as e:
                print(f"[{self.agent.name}] Behavior {behavior.get_name()} on_initial_context error: {e}")

        # Extract any user messages that were injected (skip system prompt)
        # and prepend them to message history
        if len(initial_context) > 1:
            injected_messages = initial_context[1:]  # Everything after system prompt
            # Prepend to existing messages (if any)
            self.agent.state.messages = injected_messages + self.agent.state.messages

    def trigger_round_start(self, round_number: int, context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Trigger on_round_start event on all behaviors.

        Called at the start of EVERY round, before LLM call.

        Args:
            round_number: Current round number (1-indexed)
            context: Current context to be sent to LLM

        Returns:
            Modified context after all behaviors have processed it
        """
        for behavior in self.agent._behaviors:
            try:
                # Try new API first
                if hasattr(behavior, 'on_round_start') and callable(behavior.on_round_start):
                    context = behavior.on_round_start(
                        agent=self.agent,
                        round_number=round_number,
                        context=context
                    )
                # Fall back to enhance_context if on_round_start doesn't exist
                elif hasattr(behavior, 'enhance_context') and callable(behavior.enhance_context):
                    context = behavior.enhance_context(
                        context,
                        agent=self.agent,
                        workspace=self.agent.workspace,
                        round_number=round_number,
                        workspace_manager=self.agent.workspace_manager,
                    )
            except Exception as e:
                print(f"[{self.agent.name}] Behavior {behavior.get_name()} on_round_start error: {e}")

        return context

    def trigger_llm_response(self, response: dict[str, Any]) -> None:
        """
        Trigger on_llm_response event on all behaviors.

        Called after LLM responds but before tool dispatch.

        Args:
            response: LLM response dict (contains 'message' with role/content/tool_calls)
        """
        for behavior in self.agent._behaviors:
            try:
                if hasattr(behavior, 'on_llm_response') and callable(behavior.on_llm_response):
                    behavior.on_llm_response(agent=self.agent, response=response)
            except Exception as e:
                print(f"[{self.agent.name}] Behavior {behavior.get_name()} on_llm_response error: {e}")

    def trigger_tool_call(self, tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        """
        Trigger on_tool_call event on all behaviors.

        Called after each tool execution. Behaviors can use this for tracking,
        loop detection, or other side effects.

        Args:
            tool_name: Name of the tool that was called
            args: Arguments passed to the tool
            result: Result returned by the tool
        """
        for behavior in self.agent._behaviors:
            try:
                # Try new API signature first (agent as first param)
                if hasattr(behavior, 'on_tool_call'):
                    # Check signature to determine which API to use
                    sig = inspect.signature(behavior.on_tool_call)
                    params = list(sig.parameters.keys())

                    # New API: on_tool_call(agent, tool_name, args, result)
                    if len(params) >= 4 and params[0] == 'agent':
                        behavior.on_tool_call(
                            agent=self.agent,
                            tool_name=tool_name,
                            args=args,
                            result=result
                        )
                    # Old API: on_tool_call(tool_name, args, result, **kwargs)
                    else:
                        behavior.on_tool_call(
                            tool_name=tool_name,
                            args=args,
                            result=result,
                            agent=self.agent
                        )
            except Exception as e:
                print(f"[{self.agent.name}] Behavior {behavior.get_name()} on_tool_call error: {e}")

    def trigger_round_end(self, round_number: int) -> None:
        """
        Trigger on_round_end event on all behaviors.

        Called at end of each round, after all tool calls executed.

        Args:
            round_number: Current round number (1-indexed)
        """
        for behavior in self.agent._behaviors:
            try:
                if hasattr(behavior, 'on_round_end') and callable(behavior.on_round_end):
                    behavior.on_round_end(agent=self.agent, round_number=round_number)
            except Exception as e:
                print(f"[{self.agent.name}] Behavior {behavior.get_name()} on_round_end error: {e}")

    def trigger_goal_complete(self, success: bool, summary: str) -> None:
        """
        Trigger on_goal_complete event on all behaviors.

        Called when goal completes (via mark_complete or mark_failed).

        Args:
            success: True if goal succeeded, False if failed
            summary: Summary message (from mark_complete or mark_failed)
        """
        for behavior in self.agent._behaviors:
            try:
                if hasattr(behavior, 'on_goal_complete') and callable(behavior.on_goal_complete):
                    behavior.on_goal_complete(agent=self.agent, success=success, summary=summary)
            except Exception as e:
                print(f"[{self.agent.name}] Behavior {behavior.get_name()} on_goal_complete error: {e}")

    def trigger_timeout(self, elapsed_seconds: float) -> None:
        """
        Trigger on_timeout event on all behaviors.

        Called when goal times out (circuit breaker triggered).

        Args:
            elapsed_seconds: Time elapsed since goal start
        """
        for behavior in self.agent._behaviors:
            try:
                if hasattr(behavior, 'on_timeout') and callable(behavior.on_timeout):
                    behavior.on_timeout(agent=self.agent, elapsed_seconds=elapsed_seconds)
            except Exception as e:
                print(f"[{self.agent.name}] Behavior {behavior.get_name()} on_timeout error: {e}")

    def fire_custom_event(self, event_name: str, **event_data) -> None:
        """
        Fire a custom event to all behaviors for inter-behavior communication.

        This method allows behaviors to broadcast custom events to other behaviors,
        enabling mode coordination and other cross-behavior communication patterns.

        Custom events are behavior-specific and not part of the standard lifecycle.
        Each behavior can choose which custom events to listen for and respond to.

        Args:
            event_name: Name of the custom event (e.g., "mode_changed", "context_full")
            **event_data: Event-specific data to pass to listeners

        Example:
            # In one behavior, fire a custom event
            agent.event_system.fire_custom_event(
                "mode_changed",
                mode="architect",
                reason="User requested design phase"
            )

            # In another behavior, listen for the event
            def on_custom_event(self, agent, event_name, **event_data):
                if event_name == "mode_changed":
                    self.current_mode = event_data.get("mode")
        """
        for behavior in self.agent._behaviors:
            try:
                if hasattr(behavior, 'on_custom_event') and callable(behavior.on_custom_event):
                    behavior.on_custom_event(
                        agent=self.agent,
                        event_name=event_name,
                        **event_data
                    )
            except Exception as e:
                print(f"[{self.agent.name}] Behavior {behavior.get_name()} on_custom_event({event_name}) error: {e}")

    def trigger_legacy_event(self, event_name: str, **kwargs) -> None:
        """
        Trigger a legacy event on all behaviors.

        DEPRECATED: This method is kept for backwards compatibility with old event names.
        New code should use the specific event trigger methods.

        Args:
            event_name: Event method name (e.g., "onGoalSet")
            **kwargs: Event-specific arguments
        """
        for behavior in self.agent._behaviors:
            try:
                event_method = getattr(behavior, event_name, None)
                if event_method and callable(event_method):
                    event_method(agent=self.agent, **kwargs)
            except Exception as e:
                print(f"[{self.agent.name}] Behavior {behavior.get_name()} {event_name} error: {e}")
