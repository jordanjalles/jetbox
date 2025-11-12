"""
ContextInspectorBehavior - Captures context snapshots for analysis.

This behavior captures the full context (messages, tools, behaviors) at each
LLM call without modifying agent behavior. Snapshots are saved as JSON files
for offline analysis.

Features:
- Event: on_initial_context() - Capture initial context (round 0)
- Event: on_round_start() - Capture pre-LLM context (every round)
- Event: on_round_end() - Capture post-LLM response with thinking tokens
- Zero overhead when disabled (config-driven)
- No context modification (pure observer)
- Structured JSON snapshots with metrics

Use case:
- Debug context growth issues
- Analyze duplication
- Attribute token usage to behaviors
- Optimize context management strategies
- Capture LLM reasoning (thinking tokens) for analysis
"""

from typing import Any
import json
import time
from pathlib import Path
from behaviors.base import AgentBehavior


class ContextInspectorBehavior(AgentBehavior):
    """
    Behavior that captures context snapshots at each LLM call.

    This is a pure observer behavior - it doesn't modify context or
    influence agent behavior. It only saves snapshots for later analysis.

    Features:
    - Captures full context (messages, tools, behaviors)
    - Calculates metrics (lengths, counts)
    - Saves as JSON with timestamps
    - Handles large contexts gracefully
    - Creates output directory automatically

    Security: [] None (writes local JSON snapshots, not network communication)
    """

    # Rule of Two: Empty (utility behavior for debugging/analysis)
    rule_of_two_properties = set()

    def __init__(
        self,
        output_dir: str = ".context_inspection",
        save_full_context: bool = True,
        compress_large_contexts: bool = False,
        capture_tools: bool = True,
        capture_metrics: bool = True
    ):
        """
        Initialize context inspector behavior.

        Args:
            output_dir: Directory to save snapshots (default: .context_inspection)
            save_full_context: Whether to save full message content (default: True)
            compress_large_contexts: Truncate large messages (default: False)
            capture_tools: Whether to capture tool definitions (default: True)
            capture_metrics: Whether to calculate metrics (default: True)
        """
        self.output_dir = Path(output_dir)
        self.save_full_context = save_full_context
        self.compress_large_contexts = compress_large_contexts
        self.capture_tools = capture_tools
        self.capture_metrics = capture_metrics
        self.round_count = 0
        self.agent_name = "unknown"

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "context_inspector"

    def get_sequence_number(self) -> int:
        """
        Return behavior sequence number for execution ordering.

        Behaviors are sorted by sequence number before execution.
        Higher numbers run later in the chain. Default is 0 (no method defined).
        Behaviors with the same sequence number maintain config file order (stable sort).

        ContextInspector should run LAST to capture the final context
        that the LLM actually sees (including nudges from other behaviors).

        Returns:
            999 (runs last, after all other behaviors modify context)
        """
        return 999

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Capture initial context (round 0).

        Args:
            agent: Agent instance
            context: Initial context (system prompt only)

        Returns:
            Context unchanged (pure observer)
        """
        # Store agent name for snapshot filenames
        self.agent_name = agent.name if hasattr(agent, 'name') else "unknown"

        # Capture snapshot
        self._capture_snapshot(agent, context, round_number=0, phase="initial")

        # Return context unchanged (pure observer)
        return context

    def on_round_start(
        self,
        agent: Any,
        round_number: int,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Capture pre-LLM context every round.

        Args:
            agent: Agent instance
            round_number: Current round number
            context: Current context to be sent to LLM

        Returns:
            Context unchanged (pure observer)
        """
        # Capture snapshot
        self._capture_snapshot(agent, context, round_number=round_number, phase="pre_llm")

        # Return context unchanged (pure observer)
        return context

    def on_round_end(self, agent: Any, round_number: int) -> None:
        """
        Capture post-LLM data (response, thinking tokens, tool calls).

        Args:
            agent: Agent instance
            round_number: Current round number
        """
        try:
            # Build post-LLM snapshot
            snapshot = {
                "agent_name": self.agent_name,
                "round": round_number,
                "phase": "post_llm",
                "timestamp": time.time(),
            }

            # Capture LLM response if available
            if hasattr(agent, 'last_response'):
                response = agent.last_response
                snapshot["response"] = {
                    "content": response.get("content", "") if isinstance(response, dict) else "",
                    "thinking": response.get("thinking", "") if isinstance(response, dict) else "",
                    "role": response.get("role", "assistant") if isinstance(response, dict) else "assistant",
                }

                # Capture tool calls if present
                if isinstance(response, dict) and "tool_calls" in response:
                    snapshot["response"]["tool_calls"] = self._serialize_tool_calls(response["tool_calls"])

            # Capture tools that were actually called
            if hasattr(agent, 'last_tools_executed'):
                snapshot["tools_executed"] = agent.last_tools_executed

            # Save to file
            filename = f"{self.agent_name}_round_{round_number:03d}_post_llm.json"
            filepath = self.output_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)

            print(f"[context_inspector] Saved post-LLM snapshot: {filename}")

        except Exception as e:
            # Don't fail the agent if snapshot capture fails
            print(f"[context_inspector] Error capturing post-LLM snapshot: {e}")

    def _serialize_tool_calls(self, tool_calls: Any) -> list[dict[str, Any]]:
        """
        Serialize tool calls to JSON-compatible format.

        Args:
            tool_calls: Tool calls (may be ToolCall objects or dicts)

        Returns:
            List of serialized tool call dicts
        """
        if not tool_calls:
            return []

        serialized = []
        for tc in tool_calls:
            if hasattr(tc, "model_dump"):
                # Pydantic model
                serialized.append(tc.model_dump())
            elif hasattr(tc, "to_dict"):
                # Has to_dict method
                serialized.append(tc.to_dict())
            elif isinstance(tc, dict):
                # Already a dict
                serialized.append(tc)
            else:
                # Manual extraction
                try:
                    serialized.append({
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(tc.function, "name", "") if hasattr(tc, "function") else "",
                            "arguments": getattr(tc.function, "arguments", {}) if hasattr(tc, "function") else {}
                        }
                    })
                except Exception:
                    # Fallback: convert to string representation
                    serialized.append({"_error": "Could not serialize ToolCall", "_repr": str(tc)})

        return serialized

    def _capture_snapshot(
        self,
        agent: Any,
        context: list[dict[str, Any]],
        round_number: int,
        phase: str
    ) -> None:
        """
        Capture and save a context snapshot.

        Args:
            agent: Agent instance
            context: Current context
            round_number: Round number (0 for initial)
            phase: Phase name ("initial" or "pre_llm")
        """
        try:
            # Build snapshot
            snapshot = {
                "agent_name": self.agent_name,
                "round": round_number,
                "phase": phase,
                "timestamp": time.time(),
                "_note": "Context captured by ContextInspectorBehavior (sequence=999, runs last). This is the final context sent to LLM, including all nudges and modifications from other behaviors."
            }

            # Add context (messages)
            if self.save_full_context:
                snapshot["context"] = self._process_context(context)
            else:
                # Just include counts
                snapshot["context_message_count"] = len(context)

            # Add tools
            if self.capture_tools:
                snapshot["tools"] = self._get_tools(agent)

            # Add behaviors loaded
            snapshot["behaviors_loaded"] = self._get_behaviors(agent)

            # Add metrics
            if self.capture_metrics:
                snapshot["metrics"] = self._calculate_metrics(context, snapshot.get("tools", []))

            # Save to file
            filename = f"{self.agent_name}_round_{round_number:03d}_{phase}.json"
            filepath = self.output_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)

            print(f"[context_inspector] Saved snapshot: {filename}")

        except Exception as e:
            # Don't fail the agent if snapshot capture fails
            print(f"[context_inspector] Error capturing snapshot: {e}")

    def _process_context(self, context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process context messages for snapshot.

        Args:
            context: List of message dicts

        Returns:
            Processed context (possibly truncated if compress_large_contexts=True)
        """
        # Always serialize to handle ToolCall objects and other non-JSON types
        processed = []
        for msg in context:
            processed_msg = self._serialize_message(msg)

            # Compress large messages if configured
            if self.compress_large_contexts:
                content = processed_msg.get('content', '')
                if isinstance(content, str) and len(content) > 5000:
                    # Truncate large messages
                    processed_msg['content'] = content[:5000] + f"\n... [truncated {len(content) - 5000} chars]"
                    processed_msg['_truncated'] = True

            processed.append(processed_msg)

        return processed

    def _serialize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """
        Serialize a message to JSON-compatible format.

        Handles ToolCall objects and other non-JSON-serializable types.

        Args:
            message: Message dict (may contain ToolCall objects)

        Returns:
            JSON-serializable message dict
        """
        serialized = {}
        for key, value in message.items():
            if key == "tool_calls" and value is not None:
                # Convert ToolCall objects to dicts
                serialized_calls = []
                for tc in value:
                    if hasattr(tc, "model_dump"):
                        # Pydantic model
                        serialized_calls.append(tc.model_dump())
                    elif hasattr(tc, "to_dict"):
                        # Has to_dict method
                        serialized_calls.append(tc.to_dict())
                    elif isinstance(tc, dict):
                        # Already a dict
                        serialized_calls.append(tc)
                    else:
                        # Manual extraction
                        try:
                            serialized_calls.append({
                                "id": getattr(tc, "id", None),
                                "type": getattr(tc, "type", "function"),
                                "function": {
                                    "name": getattr(tc.function, "name", "") if hasattr(tc, "function") else "",
                                    "arguments": getattr(tc.function, "arguments", {}) if hasattr(tc, "function") else {}
                                }
                            })
                        except Exception:
                            # Fallback: convert to string representation
                            serialized_calls.append({"_error": "Could not serialize ToolCall", "_repr": str(tc)})
                serialized[key] = serialized_calls
            else:
                serialized[key] = value
        return serialized

    def _get_tools(self, agent: Any) -> list[dict[str, Any]]:
        """
        Get tool definitions from agent.

        Args:
            agent: Agent instance

        Returns:
            List of tool definitions
        """
        try:
            if hasattr(agent, 'get_tools'):
                return agent.get_tools()
        except Exception as e:
            print(f"[context_inspector] Error getting tools: {e}")

        return []

    def _get_behaviors(self, agent: Any) -> list[str]:
        """
        Get list of loaded behaviors from agent.

        Args:
            agent: Agent instance

        Returns:
            List of behavior names
        """
        try:
            if hasattr(agent, '_behaviors'):
                return [b.get_name() for b in agent._behaviors]
        except Exception as e:
            print(f"[context_inspector] Error getting behaviors: {e}")

        return []

    def _calculate_metrics(
        self,
        context: list[dict[str, Any]],
        tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Calculate context metrics.

        Args:
            context: List of message dicts
            tools: List of tool definitions

        Returns:
            Dict of metrics
        """
        metrics = {
            "total_messages": len(context),
            "tool_count": len(tools),
        }

        # Calculate lengths
        system_prompt_length = 0
        total_context_length = 0

        for msg in context:
            content = msg.get('content', '')
            if isinstance(content, str):
                msg_length = len(content)
                total_context_length += msg_length

                if msg.get('role') == 'system':
                    system_prompt_length += msg_length

        metrics["system_prompt_length"] = system_prompt_length
        metrics["total_context_length"] = total_context_length

        # Calculate tool definition length
        tool_definition_length = 0
        try:
            tool_json = json.dumps(tools)
            tool_definition_length = len(tool_json)
        except Exception:
            pass

        metrics["tool_definition_length"] = tool_definition_length

        # Rough token estimate (chars / 4)
        metrics["estimated_tokens"] = total_context_length // 4 + tool_definition_length // 4

        return metrics

    def get_instructions(self) -> str:
        """
        Return context inspector instructions.

        Returns:
            Instructions about context inspection (empty - this is an observer)
        """
        # This behavior doesn't add instructions to the LLM
        # It's a pure observer for debugging/analysis
        return ""
