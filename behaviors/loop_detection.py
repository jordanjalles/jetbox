"""
LoopDetectionBehavior - Detects repeated actions and warns agent.

This behavior tracks all tool calls and detects when the agent is
repeating the same actions (infinite loops). When loops are detected,
it injects warnings into context to nudge the agent toward different approaches.

NOTE: Empty round detection has been moved to ExecutionModeBehavior.
This behavior only tracks repeated tool calls and failures.

Features:
- Event: on_tool_call(agent, tool_name, args, result)
- Event: on_round_start(agent, round_number, context)
- Track action signatures (tool_name + args)
- Track result signatures
- Detect repeated failures (same action, same error)
- Inject warnings into context when loops detected
- Mode-aware: Only tracks in execution mode (not chat mode)
- Max repeats: 5 (configurable)
"""

from typing import Any
import hashlib
import json
from behaviors.base import AgentBehavior
from behaviors.rule_of_two_types import RuleOfTwoProperty


class LoopDetectionBehavior(AgentBehavior):
    """
    Behavior that detects and warns about repeated actions (loops).

    Tracks all tool calls and their results to detect when the agent
    is stuck in a loop (repeating the same actions with same results).

    NOTE: Empty round detection has been moved to ExecutionModeBehavior.

    Security: No security properties (utility behavior)
    - Only observes and warns (no input/access/action)
    - Purely analytical behavior

    Features:
    - Tracks action signatures (tool_name + args hash)
    - Tracks result signatures (action + result hash)
    - Detects repeated failures (same action → same error)
    - Injects warnings into context when loops detected
    - Mode-aware: Only tracks in execution mode (not chat mode)
    - Configurable max_repeats threshold
    """

    # Rule of Two: No properties - utility behavior only
    rule_of_two_properties = set()

    def __init__(self, max_repeats: int = 5):
        """
        Initialize loop detection behavior.

        Args:
            max_repeats: Maximum times an action can repeat before warning (default: 5)
        """
        self.max_repeats = max_repeats
        self.action_history: list[dict[str, Any]] = []
        self.loop_warnings: list[str] = []

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "loop_detection"

    def on_tool_call(
        self,
        agent: Any,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any]
    ) -> None:
        """
        Called after each tool execution to track actions.

        Args:
            agent: Agent instance
            tool_name: Name of tool that was called
            args: Arguments passed to the tool
            result: Result returned by the tool
        """
        # Check if execution mode is active
        execution_active = False
        for behavior in agent.behaviors:
            if hasattr(behavior, 'get_name') and behavior.get_name() == 'execution_mode':
                execution_active = behavior.is_active
                break

        if not execution_active:
            return  # Don't track in chat mode

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


    def _detect_reading_loop(self) -> str | None:
        """
        Detect when agent is stuck reading files instead of implementing.

        Checks recent tool history for pattern of excessive reading without
        any writing/implementation actions.

        Returns:
            Warning message if reading loop detected, None otherwise
        """
        # Need at least 5 actions to detect pattern
        if len(self.action_history) < 5:
            return None

        # Check last 6 actions
        recent = self.action_history[-6:]

        # Categorize tools
        read_tools = {'read_file', 'list_dir'}
        write_tools = {'write_file', 'run_bash', 'mark_subtask_complete', 'mark_complete'}

        # Count read vs write actions
        read_count = sum(1 for a in recent if a["tool_name"] in read_tools)
        write_count = sum(1 for a in recent if a["tool_name"] in write_tools)

        # Detect reading loop: 4+ reads, 0 writes in last 6 actions
        if read_count >= 4 and write_count == 0:
            return (
                "⚠️  READING LOOP DETECTED\n"
                f"You've spent {read_count} recent actions reading files without writing any code.\n"
                "Architecture docs are for reference - you don't need to read them all.\n"
                "START IMPLEMENTING NOW. You can refer back to docs as needed."
            )

        return None

    def _build_loop_warnings(self) -> list[str] | None:
        """
        Build loop detection warning message.

        Returns:
            List of message lines, or None if no warnings
        """
        # Check for reading loop first
        reading_loop_warning = self._detect_reading_loop()
        if reading_loop_warning:
            return [reading_loop_warning]

        # Check for action loops
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

    def on_round_start(
        self,
        agent: Any,
        round_number: int,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject loop warnings into context.

        Args:
            agent: Agent instance
            round_number: Current round number
            context: Current context

        Returns:
            Modified context with warnings (if any)
        """
        # Check if execution mode is active
        execution_active = False
        for behavior in agent.behaviors:
            if hasattr(behavior, 'get_name') and behavior.get_name() == 'execution_mode':
                execution_active = behavior.is_active
                break

        if not execution_active:
            return context  # Don't inject warnings in chat mode

        # Check for action loops
        loop_warnings = self._build_loop_warnings()
        if loop_warnings:
            warning_text = "\n".join(loop_warnings)
            # Insert after system prompt (index 1)
            self.inject_user_message_after_system(context, warning_text)

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
