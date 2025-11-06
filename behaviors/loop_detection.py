"""
LoopDetectionBehavior - Detects repeated actions and warns agent.

This behavior tracks all tool calls and detects when the agent is
repeating the same actions (infinite loops). When loops are detected,
it injects warnings into context to nudge the agent toward different approaches.

Features:
- Event: on_tool_call(tool_name, args, result, **kwargs)
- Track action signatures (tool_name + args)
- Track result signatures
- Detect repeated failures (same action, same error)
- Inject warnings into context when loops detected
- Max repeats: 5 (configurable)
"""

from typing import Any
import hashlib
import json
from behaviors.base import AgentBehavior


class LoopDetectionBehavior(AgentBehavior):
    """
    Behavior that detects and warns about repeated actions (loops).

    Tracks all tool calls and their results to detect when the agent
    is stuck in a loop (repeating the same actions with same results).

    Features:
    - Tracks action signatures (tool_name + args hash)
    - Tracks result signatures (action + result hash)
    - Detects repeated failures (same action → same error)
    - Injects warnings into context when loops detected
    - Configurable max_repeats threshold
    """

    def __init__(self, max_repeats: int = 5, max_empty_rounds: int = 3, auto_fail_empty_rounds: int = 6, delegation_tool_names: list[str] | None = None):
        """
        Initialize loop detection behavior.

        Args:
            max_repeats: Maximum times an action can repeat before warning (default: 5)
            max_empty_rounds: Maximum consecutive rounds without tool calls before intervention (default: 3)
            auto_fail_empty_rounds: Maximum consecutive empty rounds before auto-failing (default: 6)
            delegation_tool_names: Optional list of delegation tool names to nudge when idle (e.g., ['delegate_to_executor', 'consult_architect'])
                                   If None, generic idle detection is used. Used for orchestrator-type agents.
        """
        self.max_repeats = max_repeats
        self.max_empty_rounds = max_empty_rounds
        self.auto_fail_empty_rounds = auto_fail_empty_rounds
        self.delegation_tool_names = set(delegation_tool_names) if delegation_tool_names else None
        self.action_history: list[dict[str, Any]] = []
        self.loop_warnings: list[str] = []
        self.consecutive_empty_rounds = 0
        self.recovery_prompt_injected = False
        self.last_round_action_count = 0

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "loop_detection"

    def on_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Called after each tool execution to track actions.

        Args:
            tool_name: Name of tool that was called
            args: Arguments passed to the tool
            result: Result returned by the tool
            **kwargs: Additional context
        """
        # Reset empty rounds counter - a tool was called
        self.consecutive_empty_rounds = 0
        self.recovery_prompt_injected = False

        # Create action signature (tool + args)
        serializable_args = self._make_serializable(args)
        args_str = json.dumps(serializable_args, sort_keys=True)
        action_sig = f"{tool_name}::{args_str}"

        # Create result signature (action + result hash for detecting repeated failures)
        result_str = str(result)[:500]  # First 500 chars of result
        result_hash = hashlib.sha256(result_str.encode('utf-8', errors='ignore')).hexdigest()[:16]
        result_sig = f"{action_sig}::{result_hash}"

        # Determine success - handle both dict and string results
        if isinstance(result, dict):
            success = not ("error" in result or result.get("success") is False)
        elif isinstance(result, str):
            # String results - check for error indicators
            success = "error" not in result.lower() and "failed" not in result.lower()
        else:
            # Unknown type - assume success
            success = True

        # Record action
        self.action_history.append({
            "action_sig": action_sig,
            "result_sig": result_sig,
            "success": success,
            "tool_name": tool_name,
        })

        # Check for loops in recent history (last 20 actions)
        recent = self.action_history[-20:]

        # Count identical action+result pairs
        same_result_count = sum(1 for a in recent if a["result_sig"] == result_sig)

        # Count identical actions (regardless of result)
        same_action_count = sum(1 for a in recent if a["action_sig"] == action_sig)

        # Detect loop
        if same_result_count >= self.max_repeats:
            warning = f"{tool_name} repeated {same_result_count}x with identical results"
            if warning not in self.loop_warnings:
                self.loop_warnings.append(warning)

        # Warn about repeated attempts even if results differ slightly
        elif same_action_count >= self.max_repeats + 2:
            warning = f"{tool_name} attempted {same_action_count}x (results vary)"
            if warning not in self.loop_warnings:
                self.loop_warnings.append(warning)

    def onRoundEnd(self, round_number: int, **kwargs: Any) -> None:
        """
        Called at end of each round to detect empty rounds.

        Args:
            round_number: Current round number
            **kwargs: Additional context
        """
        # Check if action count changed this round
        current_action_count = len(self.action_history)

        if current_action_count == self.last_round_action_count:
            # No tool was called this round (empty round)
            self.consecutive_empty_rounds += 1
            print(f"[loop_detection] ⚠️  Empty round #{self.consecutive_empty_rounds} - LLM did not call any tools")

            # Log diagnostic info
            agent = kwargs.get('agent')
            if agent and hasattr(agent, 'state') and agent.state.messages:
                # Get last assistant message
                last_msg = None
                for msg in reversed(agent.state.messages):
                    if msg.get('role') == 'assistant':
                        last_msg = msg
                        break

                if last_msg:
                    content_preview = last_msg.get('content', '')[:200]
                    print(f"[loop_detection] LLM response: {content_preview}...")

            # AUTO-FAIL: If recovery prompt was injected but LLM continues having empty rounds,
            # agent is stuck and should fail fast instead of wasting time
            if self.consecutive_empty_rounds >= self.auto_fail_empty_rounds:
                print(f"[loop_detection] ❌ AUTO-FAIL: {self.consecutive_empty_rounds} consecutive "
                      f"empty rounds exceeded threshold ({self.auto_fail_empty_rounds})")
                print("[loop_detection] Recovery prompts were injected but LLM did not respond with tool calls.")
                print("[loop_detection] Agent is stuck - forcing failure to prevent infinite loop.")

                # Force agent to fail by raising an exception
                # This will be caught by the agent's run loop and trigger mark_failed
                raise RuntimeError(
                    f"Auto-fail: Agent stuck with {self.consecutive_empty_rounds} consecutive "
                    f"empty rounds (threshold: {self.auto_fail_empty_rounds}). Recovery prompts "
                    f"were injected but LLM did not call any tools."
                )
        else:
            # Tool was called - counter already reset by on_tool_call
            pass

        # Update last count for next round
        self.last_round_action_count = current_action_count

    def _build_delegation_nudge(self, agent: Any) -> list[str]:
        """
        Build delegation nudge message for stuck delegating agents.

        Args:
            agent: Agent instance

        Returns:
            List of message lines
        """
        delegation_nudge = [
            "🚨 NOTICE: You have not called any tools for 2 consecutive rounds.",
            "",
            "You are a delegating agent. Your job is to DELEGATE work to specialized agents.",
            "",
            "AVAILABLE DELEGATION OPTIONS:",
        ]

        # Get delegation tool specs from agent tools
        if agent:
            tools = agent.get_tools()
            for tool in tools:
                if isinstance(tool, dict) and 'function' in tool:
                    tool_name = tool['function'].get('name', '')
                    if tool_name in self.delegation_tool_names:
                        tool_desc = tool['function'].get('description', '')
                        delegation_nudge.append(f"  • {tool_name}: {tool_desc}")

        delegation_nudge.extend([
            "",
            "ACTION REQUIRED:",
            "Call the appropriate delegation tool to assign work to a specialized agent.",
            "",
            "DO NOT try to implement the task yourself - delegate to the appropriate agent NOW."
        ])

        return delegation_nudge

    def _detect_goal(self, agent: Any, context_manager: Any) -> tuple[str, bool]:
        """
        Try multiple sources to detect agent's goal and execution mode.

        Args:
            agent: Agent instance
            context_manager: Context manager instance

        Returns:
            Tuple of (goal_description, has_goal)
        """
        goal_desc = None
        has_goal = False

        # Source 1: context_manager
        if context_manager and hasattr(context_manager, 'state') and context_manager.state.goal:
            goal_desc = context_manager.state.goal.description
            has_goal = True

        # Source 2: Core goal tracking (all agents)
        if not goal_desc and agent and hasattr(agent, 'goal') and agent.goal:
            goal_desc = agent.goal
            has_goal = True

        # Fallback
        if not goal_desc:
            goal_desc = "your assigned goal (check earlier messages)"

        return goal_desc, has_goal

    def _get_agent_tools_info(self, agent: Any) -> tuple[list[str], list[str], str]:
        """
        Extract tool information from agent.

        Args:
            agent: Agent instance

        Returns:
            Tuple of (tool_names, completion_tools, tools_list_str)
        """
        tool_names = []
        completion_tools = []

        if agent:
            tools = agent.get_tools()
            for tool in tools:
                if isinstance(tool, dict) and 'function' in tool:
                    tool_name = tool['function'].get('name', 'unknown')
                    tool_names.append(tool_name)

                    # Track completion tools
                    if tool_name in ['mark_complete', 'mark_failed', 'mark_goal_complete']:
                        completion_tools.append(tool_name)

        tools_list = ", ".join(tool_names) if tool_names else "(none available)"
        return tool_names, completion_tools, tools_list

    def _build_empty_round_recovery(
        self,
        consecutive_empty_rounds: int,
        goal_desc: str,
        tools_list: str,
        completion_tools: list[str]
    ) -> list[str]:
        """
        Build recovery prompt for empty rounds.

        Args:
            consecutive_empty_rounds: Number of consecutive empty rounds
            goal_desc: Goal description
            tools_list: Comma-separated tool names
            completion_tools: List of completion tool names

        Returns:
            List of message lines
        """
        urgency_level = "CRITICAL" if consecutive_empty_rounds >= 10 else "WARNING"
        recovery = [
            f"🚨 {urgency_level}: {consecutive_empty_rounds} consecutive empty rounds - NO TOOLS CALLED!",
            "",
            "GOAL:",
            f"  {goal_desc}",
            "",
            "YOUR TOOLS:",
            f"  {tools_list}",
            "",
            "ACTION REQUIRED NOW:",
        ]

        # Emphasize completion tools if available
        if completion_tools:
            recovery.append(f"  • If work is DONE: call {' or '.join(completion_tools)}")
            recovery.append("  • If work is BLOCKED: call mark_failed(reason=\"...\")")
            recovery.append("  • If work CONTINUES: call the next tool needed")
        else:
            recovery.append("  • Call the appropriate tool to continue")

        recovery.extend([
            "",
            "YOU CANNOT PROCEED WITHOUT CALLING A TOOL.",
            "Look at the tool list above and call one NOW."
        ])

        return recovery

    def _build_loop_warnings(self) -> list[str] | None:
        """
        Build loop detection warning message.

        Returns:
            List of message lines, or None if no warnings
        """
        if not self.loop_warnings:
            return None

        warnings_text = ["⚠️  LOOP DETECTION WARNING:"]
        warnings_text.append("You appear to be repeating actions:")
        for warning in self.loop_warnings[-3:]:  # Last 3 warnings
            warnings_text.append(f"  • {warning}")
        warnings_text.append("")
        warnings_text.append("Consider trying a COMPLETELY DIFFERENT approach:")
        warnings_text.append("  1. Read error messages more carefully")
        warnings_text.append("  2. Check if assumptions are wrong")
        warnings_text.append("  3. Try a fundamentally different strategy")
        warnings_text.append("  4. If core task is complete, call mark_complete() even if tests fail")
        warnings_text.append("  5. If truly blocked, call mark_failed() with detailed reason")

        return warnings_text

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Inject loop warnings and recovery prompts into context.

        Args:
            context: Current context
            **kwargs: Additional context (agent, context_manager)

        Returns:
            Modified context with warnings/recovery prompts (if any)
        """
        warnings_to_inject = []
        agent = kwargs.get('agent')

        # DELEGATION NUDGE: For orchestrator-type agents that should delegate rather than implement
        if self.delegation_tool_names and self.consecutive_empty_rounds >= 2 and not self.recovery_prompt_injected:
            # Check if any delegation tool has been called
            delegation_called = any(
                action['tool_name'] in self.delegation_tool_names
                for action in self.action_history
            )

            if not delegation_called:
                # Agent is stuck before delegating - inject delegation nudge
                delegation_nudge = self._build_delegation_nudge(agent)
                warnings_to_inject.append("\n".join(delegation_nudge))
                self.recovery_prompt_injected = True

        # EMPTY ROUND RECOVERY: For general agents
        else:
            context_manager = kwargs.get('context_manager')

            # Detect goal and execution mode
            goal_desc, has_goal = self._detect_goal(agent, context_manager)

            # CRITICAL: In execute mode (has_goal=True), inject after 1 empty round
            # In chatbot mode (has_goal=False), inject after max_empty_rounds (default 3)
            min_empty_rounds_for_injection = 1 if has_goal else self.max_empty_rounds

            if self.consecutive_empty_rounds >= min_empty_rounds_for_injection:
                should_inject = (
                    not self.recovery_prompt_injected or
                    (self.consecutive_empty_rounds % 5 == 0)  # Repeat every 5 empty rounds
                )

                if should_inject:
                    # Get tool information
                    tool_names, completion_tools, tools_list = self._get_agent_tools_info(agent)

                    # Build recovery prompt
                    recovery = self._build_empty_round_recovery(
                        self.consecutive_empty_rounds,
                        goal_desc,
                        tools_list,
                        completion_tools
                    )

                    warnings_to_inject.append("\n".join(recovery))
                    self.recovery_prompt_injected = True
                    print(f"[loop_detection] Injecting empty round recovery (round {self.consecutive_empty_rounds})")

            # Check for action loops
            loop_warnings = self._build_loop_warnings()
            if loop_warnings:
                warnings_to_inject.append("\n".join(loop_warnings))

        # Inject warnings if any
        if warnings_to_inject:
            # Insert after system prompt (index 1)
            for warning in reversed(warnings_to_inject):  # Reverse so they appear in correct order
                self.inject_user_message_after_system(context, warning)

        return context

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert an object to a JSON-serializable format.

        Handles common non-serializable types:
        - Objects: convert to string representation
        - Dicts: recursively process values
        - Lists: recursively process items
        - Primitives: pass through

        Args:
            obj: Object to make serializable

        Returns:
            JSON-serializable version of the object
        """
        # Try direct serialization first (fast path for primitives)
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            pass

        # Handle different types
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        elif isinstance(obj, dict):
            # Recursively process dict values
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # Recursively process list/tuple items
            return [self._make_serializable(item) for item in obj]
        else:
            # Non-serializable object - use string representation
            # Include type name for debugging
            return f"<{type(obj).__name__}>"

    def get_instructions(self) -> str:
        """
        Return loop detection instructions.

        Returns:
            Instructions about loop detection
        """
        return """
LOOP DETECTION:
The system monitors your actions and warns you if you're repeating the same approach.

If you see a loop warning:
- STOP and reconsider your approach
- Read error messages carefully
- Check if your assumptions are wrong
- Try a COMPLETELY DIFFERENT strategy
- Don't just tweak parameters - change the fundamental approach

Loop detection helps you avoid wasting time on approaches that aren't working.
"""
