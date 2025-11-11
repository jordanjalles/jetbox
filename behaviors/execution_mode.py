"""
ExecutionModeBehavior - Manages execution mode for task completion.

This behavior handles execution mode state and tool usage enforcement:
- Activates when: set_goal() called or activate_execution_mode() called
- Deactivates when: mark_complete(), mark_failed(), or conflicting mode activates
- Enforces tool usage (no text-only responses allowed)
- Detects empty rounds and nudges agent
- Detects completion signals and prompts for mark_complete()

Key features:
- self.is_active state tracking
- activate() / deactivate() lifecycle methods
- Event-based coordination via on_custom_event()
- Tool usage enforcement (consecutive empty round detection)
- Completion nudging (detects "task complete" signals)
- Broadcasts 'mode_activated' event when activating

Configuration:
- enable_completion_nudging: Enable completion signal detection (default: True)
- min_rounds_before_nudge: Minimum rounds before completion nudging (default: 3)
- max_empty_rounds: Max consecutive empty rounds before violation warning (default: 3)
"""

from typing import Any
from behaviors.base import AgentBehavior


class ExecutionModeBehavior(AgentBehavior):
    """
    Manages execution mode for task completion.

    Activates when: set_goal() called or activate_execution_mode() called
    Deactivates when: mark_complete(), mark_failed(), or conflicting mode activates

    Features:
    - Tool usage enforcement (no text-only responses)
    - Empty round detection and warnings
    - Completion signal detection and nudging
    - Event-based mode coordination
    """

    def __init__(self, params: dict[str, Any] | None = None):
        """
        Initialize execution mode behavior.

        Args:
            params: Configuration parameters:
                - enable_completion_nudging: Enable completion detection (default: True)
                - min_rounds_before_nudge: Minimum rounds before nudging (default: 3)
                - max_empty_rounds: Max consecutive empty rounds (default: 3)
        """
        super().__init__()
        self.params = params or {}

        # Mode state (owned by behavior)
        self.is_active = False

        # Execution mode config
        self.enable_completion_nudging = self.params.get(
            'enable_completion_nudging', True
        )
        self.min_rounds_before_nudge = self.params.get(
            'min_rounds_before_nudge', 3
        )
        self.max_empty_rounds = self.params.get('max_empty_rounds', 3)

        # State tracking
        self.consecutive_empty_rounds = 0
        self.pending_nudge: str | None = None
        self.last_round_tool_count = 0

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "execution_mode"

    # ============================================
    # Mode Lifecycle
    # ============================================

    def activate(self, agent: Any, **context: Any) -> dict[str, Any]:
        """
        Activate execution mode.

        Called when set_goal() or activate_execution_mode() tool is used.
        Broadcasts mode activation to other behaviors.

        Args:
            agent: Agent instance
            **context: Optional context data

        Returns:
            Activation result dict
        """
        if self.is_active:
            return {"already_active": True}

        self.is_active = True

        # Append mode activation message to history
        goal = agent.goal if hasattr(agent, 'goal') else "Unknown"
        activation_message = f"""
╔════════════════════════════════════════════════════════╗
║  🔧 EXECUTION MODE ACTIVATED                           ║
╚════════════════════════════════════════════════════════╝

Goal: {goal}

EXECUTION MODE REQUIREMENTS:
- You MUST use tools to make progress
- No text-only responses allowed
- Make concrete progress each round
- Call mark_complete() when done or mark_failed() if blocked

Tools available: write_file, read_file, run_bash, list_dir, etc.

Let's get to work!
"""

        agent.add_message({"role": "user", "content": activation_message})

        # Broadcast mode activation event to other behaviors
        # This allows conflicting modes (like chat) to deactivate
        if hasattr(agent, 'event_system'):
            agent.event_system.fire_custom_event(
                'mode_activated',
                mode_name='execution',
                behavior=self
            )

        return {
            "success": True,
            "mode": "execution",
            "active": True
        }

    def deactivate(self, agent: Any, reason: str = "complete") -> dict[str, Any]:
        """
        Deactivate execution mode.

        Called when:
        - mark_complete()/mark_failed() is called
        - Conflicting mode (chat) activates
        - User explicitly deactivates

        Args:
            agent: Agent instance
            reason: Reason for deactivation ("complete", "failed", "conflict", etc.)

        Returns:
            Deactivation result dict
        """
        if not self.is_active:
            return {"already_inactive": True}

        self.is_active = False

        # Append deactivation message
        if reason == "complete":
            message = """
╔════════════════════════════════════════════════════════╗
║  ✅ EXECUTION MODE COMPLETE                            ║
╚════════════════════════════════════════════════════════╝

Task completed successfully!
"""
        elif reason == "failed":
            message = """
╔════════════════════════════════════════════════════════╗
║  ⚠️  EXECUTION MODE FAILED                             ║
╚════════════════════════════════════════════════════════╝

Task could not be completed.
"""
        elif reason == "conflict":
            message = """
╔════════════════════════════════════════════════════════╗
║  🔧 EXECUTION MODE DEACTIVATED                         ║
╚════════════════════════════════════════════════════════╝

Switching to another mode...
"""
        else:
            message = f"🔧 Execution mode deactivated ({reason})"

        agent.add_message({"role": "user", "content": message})

        # Reset state
        self.consecutive_empty_rounds = 0
        self.pending_nudge = None

        return {
            "success": True,
            "mode": "execution",
            "active": False,
            "reason": reason
        }

    # ============================================
    # Initial Context (Static, KV Cached)
    # ============================================

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Explain execution mode in initial context (STATIC, KV cached).

        This is part of the system prompt that never changes.
        Explains what execution mode IS and requirements.

        Args:
            agent: Agent instance
            context: Initial context (system prompt only)

        Returns:
            Modified context with execution mode explanation
        """
        mode_explanation = f"""
═══════════════════════════════════════════════════════════
EXECUTION MODE (activated when working on tasks)
═══════════════════════════════════════════════════════════

Purpose: Complete coding tasks using available tools

Requirements when execution mode is active:
- You MUST use tools to make progress (no text-only responses)
- Write files, run commands, read outputs
- Verify work (run tests, check linting)
- Call mark_complete(summary) when task is done
- Call mark_failed(reason) if blocked

Tools available: write_file, read_file, list_dir, run_bash, etc.

Activation: Automatically activated when set_goal() is called
Status: {"🔧 ACTIVE" if self.is_active else "⏸️  INACTIVE"}
"""

        return self.inject_user_message_after_system(context, mode_explanation)

    # ============================================
    # Event Handlers
    # ============================================

    def on_custom_event(
        self,
        agent: Any,
        event_name: str,
        **event_data: Any
    ) -> None:
        """
        Listen for mode activation events from other behaviors.

        If a conflicting mode activates (e.g., chat), deactivate execution mode.

        Args:
            agent: Agent instance
            event_name: Name of the event
            **event_data: Event data
        """
        if event_name == "mode_activated":
            activated_mode = event_data.get('mode_name')

            # If chat mode activates and we're active, deactivate
            if activated_mode == "chat" and self.is_active:
                self.deactivate(agent, reason="conflict")

    # ============================================
    # Runtime Hooks (Only when active)
    # ============================================

    def on_round_start(
        self,
        agent: Any,
        round_number: int,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject pending nudges (only when execution mode is active).

        Args:
            agent: Agent instance
            round_number: Current round number
            context: Current context

        Returns:
            Modified context with nudges (if any)
        """
        if not self.is_active:
            return context

        if self.pending_nudge:
            context = self.inject_user_message_after_system(context, self.pending_nudge)
            self.pending_nudge = None

        return context

    def on_round_end(self, agent: Any, round_number: int) -> None:
        """
        Detect empty rounds and completion signals.

        Only active when execution mode is active.
        This is where tool usage enforcement happens.

        Args:
            agent: Agent instance
            round_number: Current round number
        """
        if not self.is_active:
            return

        # Track empty rounds (NO TOOLS CALLED = VIOLATION)
        # We need to detect if any tools were called this round
        # We do this by checking if tool count changed since last round

        # Get current tool count from agent's action history or tool dispatcher
        current_tool_count = 0
        if (
            hasattr(agent, 'tool_dispatcher')
            and hasattr(agent.tool_dispatcher, 'tool_call_count')
        ):
            current_tool_count = agent.tool_dispatcher.tool_call_count
        elif hasattr(agent, 'state') and hasattr(
            agent.state, 'tool_call_count'
        ):
            current_tool_count = agent.state.tool_call_count

        # Calculate tools called this round
        tools_called_this_round = current_tool_count - self.last_round_tool_count
        self.last_round_tool_count = current_tool_count

        if tools_called_this_round == 0:
            self.consecutive_empty_rounds += 1

            if self.consecutive_empty_rounds >= self.max_empty_rounds:
                self.pending_nudge = (
                    f"⚠️ EXECUTION MODE VIOLATION: Empty round detected "
                    f"({self.consecutive_empty_rounds} consecutive).\n\n"
                    "You are in EXECUTION MODE. You MUST call tools to make progress.\n"
                    "- If task is complete: call mark_complete(summary)\n"
                    "- If blocked: call mark_failed(reason)\n"
                    "- Otherwise: call write_file(), run_bash(), or other tools"
                )
        else:
            self.consecutive_empty_rounds = 0

        # Check for completion signals
        if (
            self.enable_completion_nudging
            and round_number >= self.min_rounds_before_nudge
        ):
            # Get last assistant message to check for completion signals
            if hasattr(agent, 'state') and agent.state.messages:
                last_assistant_msg = None
                for msg in reversed(agent.state.messages):
                    if msg.get('role') == 'assistant':
                        last_assistant_msg = msg
                        break

                if last_assistant_msg:
                    last_response = last_assistant_msg.get('content', '')
                    if self._detect_completion_signal(last_response):
                        self.pending_nudge = (
                            "🎯 It looks like you've completed the task. "
                            "Please call mark_complete(summary) with a brief summary."
                        )

    def _detect_completion_signal(self, text: str) -> bool:
        """
        Detect completion indicators in text.

        Args:
            text: Text to analyze

        Returns:
            True if completion signal detected, False otherwise
        """
        completion_phrases = [
            "task completed", "task complete", "successfully completed",
            "implementation complete", "all done", "finished implementing",
            "tests pass", "linting clean", "all requirements met"
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in completion_phrases)

    # ============================================
    # Tools (Mode Activation + Completion)
    # ============================================

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Provide tools for execution mode.

        Note: mark_complete/mark_failed are provided by ToolDispatcher.
        We just need to handle the mode deactivation in dispatch_tool.

        Returns:
            Tool definitions for activate_execution_mode
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "activate_execution_mode",
                    "description": (
                        "Explicitly activate execution mode for working on tasks. "
                        "Usually activated automatically by set_goal(), but can be "
                        "called explicitly if needed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        agent: Any,
        tool_name: str,
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle execution mode tools.

        Args:
            agent: Agent instance
            tool_name: Tool name
            args: Tool arguments

        Returns:
            Tool result dict

        Raises:
            NotImplementedError: If tool not handled by this behavior
        """
        if tool_name == "activate_execution_mode":
            return self.activate(agent)

        return super().dispatch_tool(agent, tool_name, args)

    def get_instructions(self) -> str:
        """
        Return execution mode instructions.

        Returns:
            Instructions about execution mode
        """
        return """
EXECUTION MODE:
When a goal is set, execution mode activates automatically.

In execution mode:
- You MUST use tools to make progress (no text-only responses)
- Make concrete changes to files, run commands, verify results
- Call mark_complete(summary) when the task is done
- Call mark_failed(reason) if you're blocked

Empty rounds (no tool calls) are not allowed in execution mode.
"""
