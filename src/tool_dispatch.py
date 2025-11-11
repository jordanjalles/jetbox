"""
Tool dispatch system for BaseAgent.

This module handles tool registration and dispatch to behaviors.
All tool-related logic is centralized here.

Key responsibilities:
- Maintain tool_registry mapping (tool_name → behavior)
- Validate tool parameters against schemas
- Dispatch tool calls to appropriate behaviors
- Collect tools from all behaviors
- Log parameter hallucinations for analysis
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import json

if TYPE_CHECKING:
    from base_agent import BaseAgent


class ToolDispatcher:
    """
    Manages tool registration and dispatch for an agent.

    This class is responsible for:
    1. Registering tools with behaviors when behaviors are loaded
    2. Dispatching tool calls to the appropriate behavior
    3. Validating tool parameters against schemas
    4. Collecting tools from all behaviors
    5. Logging parameter hallucinations to wishlist

    The ToolDispatcher is created by BaseAgent and maintains a reference
    back to the agent for accessing behaviors and triggering events.
    """

    def __init__(self, agent: BaseAgent):
        """
        Initialize tool dispatcher.

        Args:
            agent: Reference to BaseAgent instance
        """
        self.agent = agent
        self.tool_registry: dict[str, Any] = {}  # Map tool_name → behavior

    def register_tool(self, tool_name: str, behavior: Any) -> None:
        """
        Register a tool with a behavior.

        Called when behaviors are loaded to populate the tool registry.

        Args:
            tool_name: Name of the tool (e.g., "write_file")
            behavior: Behavior instance that provides this tool

        Raises:
            ValueError: If tool name already registered by another behavior
        """
        if tool_name in self.tool_registry:
            existing_behavior = self.tool_registry[tool_name]
            raise ValueError(
                f"Tool '{tool_name}' already registered by "
                f"{existing_behavior.get_name()}"
            )
        self.tool_registry[tool_name] = behavior

    def dispatch(
        self, tool_call: dict[str, Any], **extra_context
    ) -> dict[str, Any]:
        """
        Dispatch a tool call to the appropriate behavior.

        This is the main entry point for tool dispatch. It:
        1. Validates parameters
        2. Handles core completion tools (mark_complete/mark_failed)
        3. Dispatches to behavior system
        4. Triggers tool call events

        Args:
            tool_call: Tool call dict with function name and arguments
                Format: {"function": {"name": "tool_name", "arguments": {...}}}
            **extra_context: Additional context to pass to behaviors

        Returns:
            Tool result dict
        """
        # Validate parameters before dispatch
        validation_result = self._validate_parameters(tool_call)
        if validation_result:
            # Invalid parameters detected - return feedback to LLM
            return validation_result

        # Extract tool name
        tool_name = tool_call.get("function", {}).get("name")

        # Handle core completion tools (mark_complete, mark_failed)
        if tool_name in ["mark_complete", "mark_failed"]:
            return self._dispatch_completion_tool(tool_call)

        # Dispatch to behavior system for other tools
        result = self._dispatch_to_behavior(tool_call, **extra_context)

        # Trigger on_tool_call event on behaviors
        args = tool_call.get("function", {}).get("arguments", {})
        self.agent.event_system.trigger_tool_call(tool_name, args, result)

        return result

    def _dispatch_completion_tool(
        self, tool_call: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Dispatch core completion tools (mark_complete, mark_failed).

        These are core agent tools available to all agents with goals.

        Args:
            tool_call: Tool call dict

        Returns:
            Tool result dict with completion status
        """
        tool_name = tool_call["function"]["name"]
        args = tool_call["function"].get("arguments", {})

        if tool_name == "mark_complete":
            summary = args.get("summary", "Task completed")

            return {
                "success": True,
                "result": f"Task marked complete: {summary}",
                "summary": summary,
                "status": "goal_complete",
            }

        elif tool_name == "mark_failed":
            reason = args.get("reason", "Task failed")

            return {
                "success": False,
                "result": f"Task marked failed: {reason}",
                "reason": reason,
                "status": "goal_failed",
            }

        return {"error": f"Unknown completion tool: {tool_name}"}

    def _dispatch_to_behavior(
        self, tool_call: dict[str, Any], **extra_context
    ) -> dict[str, Any]:
        """
        Dispatch tool call to appropriate behavior.

        Args:
            tool_call: Tool call dict with function name and arguments
            **extra_context: Additional context to pass to behaviors

        Returns:
            Tool result dict
        """
        tool_name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]

        # Find behavior that owns this tool
        behavior = self.tool_registry.get(tool_name)
        if not behavior:
            return {"error": f"Unknown tool: {tool_name}"}

        # Dispatch to behavior with lifecycle API (agent as first parameter)
        try:
            result = behavior.dispatch_tool(
                self.agent, tool_name, args  # agent (positional)  # tool_name
                # (positional)  # args (positional)
            )
        except Exception as e:
            return {"error": f"Tool {tool_name} failed: {e}"}

        # Normalize result: behaviors may return string or dict
        # Convert strings to dict format for consistent handling
        if isinstance(result, str):
            # Determine success based on content (error messages start with "Error:")
            is_error = result.startswith("Error:")
            return {
                "success": not is_error,
                "result": result,
                **({"error": result} if is_error else {})
            }
        elif isinstance(result, dict):
            # Already in dict format
            return result
        elif isinstance(result, list):
            # List results (e.g., from list_dir) - wrap in dict
            return {
                "success": True,
                "result": result
            }
        else:
            # Unknown type - convert to string
            return {
                "success": True,
                "result": str(result)
            }

    def _validate_parameters(
        self, tool_call: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Validate tool call parameters against tool schema.

        If invalid parameters found:
        1. Returns feedback dict with error message and correct spec
        2. Logs hallucinated parameter to wishlist file

        Args:
            tool_call: Tool call dict with function name and arguments

        Returns:
            None if valid, dict with error message if invalid
        """
        tool_name = tool_call.get("function", {}).get("name")
        args = tool_call.get("function", {}).get("arguments", {})

        if not tool_name or not isinstance(args, dict):
            return None

        # Get tool spec from agent's tools
        tool_spec = None
        for tool in self.get_all_tools():
            if tool.get("function", {}).get("name") == tool_name:
                tool_spec = tool.get("function")
                break

        if not tool_spec:
            # Tool not found in spec - let dispatch handle it
            return None

        # Get valid parameters from schema
        schema = tool_spec.get("parameters", {})
        valid_params = set(schema.get("properties", {}).keys())

        # Check for invalid parameters
        provided_params = set(args.keys())
        invalid_params = provided_params - valid_params

        if not invalid_params:
            # All parameters valid
            return None

        # Log hallucinated parameters to wishlist
        self._log_parameter_wishlist(tool_name, invalid_params)

        # Build feedback message with correct spec
        param_specs = []
        for param_name, param_info in schema.get("properties", {}).items():
            param_type = param_info.get("type", "unknown")
            param_desc = param_info.get("description", "")
            required = param_name in schema.get("required", [])
            req_marker = " (required)" if required else " (optional)"
            param_specs.append(
                f"  - {param_name}: {param_type}{req_marker} - {param_desc}"
            )

        feedback = f"""⚠️  Tool call used invalid parameters

Tool: {tool_name}
Invalid parameters: {', '.join(invalid_params)}

These parameters were IGNORED because they don't exist in the tool spec.

CORRECT TOOL SPEC FOR {tool_name}:
{tool_spec.get('description', '')}

Valid parameters:
{chr(10).join(param_specs) if param_specs else '  (no parameters)'}

Please retry the tool call using only the valid parameters listed above.
"""

        return {
            "status": "parameter_error",
            "message": feedback,
            "tool_name": tool_name,
            "invalid_params": list(invalid_params),
        }

    def _log_parameter_wishlist(
        self, tool_name: str, invalid_params: set[str]
    ) -> None:
        """
        Log hallucinated parameters to wishlist file for future consideration.

        Args:
            tool_name: Tool that was called
            invalid_params: Set of invalid parameter names
        """
        wishlist_file = Path(".agent_context") / "parameter_wishlist.jsonl"
        wishlist_file.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "hallucinated_params": list(invalid_params),
            "agent": self.agent.name,
        }

        # Append to JSONL file
        with open(wishlist_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_all_tools(self) -> list[dict[str, Any]]:
        """
        Collect tools from all registered behaviors.

        This method is called by BaseAgent.get_tools() to assemble
        the complete tool list for LLM calls.

        Returns:
            List of tool definitions in Ollama format
        """
        tools = []

        # Core completion tools (only available when agent has a goal)
        if self.agent.goal:
            core_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "mark_complete",
                        "description": (
                            "Mark the task/goal as complete and report "
                            "success. REQUIRED when work is finished."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "summary": {
                                    "type": "string",
                                    "description": (
                                        "Brief summary of what was "
                                        "accomplished (2-4 sentences)"
                                    ),
                                }
                            },
                            "required": ["summary"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "mark_failed",
                        "description": (
                            "Mark the task/goal as failed and report reason. "
                            "Use when you cannot complete the task."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reason": {
                                    "type": "string",
                                    "description": (
                                        "Explanation of why the task could "
                                        "not be completed"
                                    ),
                                }
                            },
                            "required": ["reason"],
                        },
                    },
                },
            ]
            tools.extend(core_tools)

        # Behavior tools
        for behavior in self.agent.behaviors:
            tools.extend(behavior.get_tools())

        return tools

    def generate_tool_documentation(self) -> str:
        """
        Generate tool documentation from loaded behaviors.

        DEPRECATED: Tool documentation should now be injected by behaviors via
        on_initial_context() method. This method is kept for backward
        compatibility.

        Returns a formatted string listing all available tools with their
        signatures and descriptions.

        Returns:
            Tool documentation string (empty if no behaviors loaded)
        """
        if not self.agent.behaviors:
            return ""

        tool_docs = []
        for behavior in self.agent.behaviors:
            tools = behavior.get_tools()
            for tool in tools:
                func = tool.get("function", {})
                name = func.get("name", "unknown")
                desc = func.get("description", "")
                params = func.get("parameters", {}).get("properties", {})
                required = func.get("parameters", {}).get("required", [])

                # Build parameter signature
                param_strs = []
                for param_name, param_spec in params.items():
                    param_type = param_spec.get("type", "any")
                    # Mark required params
                    if param_name in required:
                        param_strs.append(f"{param_name}: {param_type}")
                    else:
                        # Optional params with default if specified
                        default = param_spec.get("default")
                        if default is not None:
                            param_strs.append(
                                f"{param_name}: {param_type} = {default}"
                            )
                        else:
                            param_strs.append(f"{param_name}?: {param_type}")

                param_sig = ", ".join(param_strs) if param_strs else ""
                tool_docs.append(f"  - {name}({param_sig}): {desc}")

        if tool_docs:
            return "\n\nAvailable tools:\n" + "\n".join(tool_docs)
        return ""
