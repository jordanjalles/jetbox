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
        # Create action signature (tool + args)
        serializable_args = self._make_serializable(args)
        args_str = json.dumps(serializable_args, sort_keys=True)
        action_sig = f"{tool_name}::{args_str}"

        # Create result signature (action + result hash for detecting repeated failures)
        result_str = str(result)[:500]  # First 500 chars of result
        result_hash = hashlib.sha256(result_str.encode('utf-8', errors='ignore')).hexdigest()[:16]
        result_sig = f"{action_sig}::{result_hash}"

        # Determine success
        success = not ("error" in result or result.get("success") is False)

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

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Inject loop warnings into context if loops detected.

        Args:
            context: Current context
            **kwargs: Additional context

        Returns:
            Modified context with loop warnings (if any)
        """
        if not self.loop_warnings:
            return context

        # Build warning message
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

        # Insert warning after system prompt (index 1)
        if len(context) > 0:
            context.insert(1, {
                "role": "user",
                "content": "\n".join(warnings_text)
            })

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
