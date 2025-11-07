"""
Base agent class providing common functionality for all agent types.

All agents inherit from BaseAgent and can override:
- get_tools(): Returns list of tools (default: returns behavior tools)
- get_system_prompt(): Returns system prompt (default: config + behaviors + tool docs)
- get_context_strategy(): Returns context strategy (DEPRECATED - use behaviors)

The behavior system provides the primary extensibility mechanism.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from pathlib import Path
import json
import time
import os
from datetime import datetime
import re
import importlib


@dataclass
class AgentState:
    """Base state that all agents maintain."""
    name: str
    role: str
    messages: list[dict[str, Any]]
    start_time: float
    total_rounds: int

    def _serialize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert message to JSON-serializable format."""
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
                        serialized_calls.append(tc.to_dict())
                    elif isinstance(tc, dict):
                        serialized_calls.append(tc)
                    else:
                        # Try to extract attributes manually
                        serialized_calls.append({
                            "id": getattr(tc, "id", None),
                            "type": getattr(tc, "type", "function"),
                            "function": {
                                "name": getattr(tc.function, "name", ""),
                                "arguments": getattr(tc.function, "arguments", {})
                            } if hasattr(tc, "function") else {}
                        })
                serialized[key] = serialized_calls
            else:
                serialized[key] = value
        return serialized

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "messages": [self._serialize_message(msg) for msg in self.messages],
            "start_time": self.start_time,
            "total_rounds": self.total_rounds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        return cls(**data)


class BaseAgent:
    """
    Base class for all agents.

    Provides common functionality:
    - LLM calling with tool support
    - Message history management
    - State persistence
    - Tool dispatch
    - Behavior system integration

    Subclasses can override methods like get_tools(), get_system_prompt(),
    but most customization should be done via behaviors in YAML config.
    """

    def __init__(
        self,
        name: str,
        workspace: Path,
        config_file: str,
        exclude_behaviors: list[str] | None = None,
        timeout_seconds: int = 600,
    ):
        """
        Initialize base agent.

        Args:
            name: Agent identifier (e.g., "orchestrator", "task_executor")
            workspace: Working directory for this agent
            config_file: Path to agent-specific config YAML (e.g., "config/agents/task_executor.yaml")
            exclude_behaviors: List of behavior names to exclude (e.g., ["ChatbotBehavior"])
            timeout_seconds: Subprocess timeout in seconds (default: 600 = 10 minutes)
        """
        import yaml
        from agent_config import config as global_config

        self.name = name
        self.workspace = Path(workspace)
        self.config = global_config  # Global config for behavior defaults
        self.timeout_seconds = timeout_seconds  # Subprocess timeout (for delegation)

        # Load agent-specific config file
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Agent config file not found: {config_file}")

        # Optional: Validate config with pydantic (enable with VALIDATE_CONFIGS=1)
        if os.environ.get("VALIDATE_CONFIGS") == "1":
            try:
                from utils.config_validator import validate_agent_config
                is_valid, error_msg, validated_config = validate_agent_config(config_path)
                if not is_valid:
                    raise ValueError(f"Invalid config file {config_file}:\n{error_msg}")
                agent_config = validated_config
            except ImportError:
                # Validation not available, fall back to normal loading
                with open(config_path) as f:
                    agent_config = yaml.safe_load(f) or {}
        else:
            with open(config_path) as f:
                agent_config = yaml.safe_load(f) or {}

        # Extract role from agent config
        self.role = agent_config.get("role", f"{name} agent")

        # Create workspace if needed
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self.state = AgentState(
            name=name,
            role=self.role,
            messages=[],
            start_time=time.time(),
            total_rounds=0,
        )

        # State file location
        self.state_file = self.workspace / ".agent_context" / f"{name}_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Try to load existing state
        self.load_state()

        # Phase 1 additions: Optional subsystems (can be initialized by subclasses)
        self.context_manager = None  # For hierarchical task tracking (TaskExecutor)
        self.workspace_manager = None  # For workspace isolation
        self.perf_stats = None  # For performance tracking
        self.server_manager = None  # For server management (Orchestrator)
        self.registry = None  # For agent registry (Orchestrator)

        # Phase 4 additions: Behavior system
        self._behaviors: list[Any] = []  # List of registered behaviors (AgentBehavior instances)
        self.behaviors: list[Any] = self._behaviors  # Public alias
        self.tool_registry: dict[str, Any] = {}  # Map tool_name -> behavior that provides it
        self.config_system_prompt: str | None = None  # System prompt loaded from config (if any)
        self.config_blurb: str | None = None  # Agent blurb loaded from config (if any)
        self.exclude_behaviors: list[str] = exclude_behaviors or []  # Behaviors to exclude from loading

        # Core goal tracking (moved from SubAgentModeBehavior)
        self.goal: str | None = None  # Current goal description
        self.is_subagent: bool = False  # True if agent was invoked by parent agent
        self.enable_completion_nudging: bool = True  # Enable completion signal detection
        self.min_rounds_before_nudge: int = 3  # Min rounds before checking completion signals
        self.pending_nudge: str | None = None  # Completion nudge to inject next round
        self.goal_start_time: float | None = None  # Wall-clock time when goal started

        # Timeout handling
        self.consecutive_timeouts = 0
        self.total_timeouts = 0

        # Load timeout settings from config
        if self.config and hasattr(self.config, 'llm') and hasattr(self.config.llm, 'timeout') and self.config.llm.timeout:
            self.inactivity_timeout = self.config.llm.timeout.inactivity_timeout
            self.max_call_time = self.config.llm.timeout.max_call_time
            self.max_consecutive_timeouts = self.config.llm.timeout.max_consecutive_timeouts
            self.auto_restart_ollama = getattr(self.config.llm.timeout, 'auto_restart_ollama', False)
        else:
            # Fallback defaults
            self.inactivity_timeout = 30
            self.max_call_time = 180
            self.max_consecutive_timeouts = 3
            self.auto_restart_ollama = False

        # Load behaviors from agent config
        # This must happen at the end of __init__ after all attributes are set
        self._load_behaviors_from_config_dict(agent_config)

    # ===========================
    # Abstract methods (must implement)
    # ===========================

    def get_context_strategy(self) -> str:
        """
        Return context management strategy name.

        DEPRECATED: Context strategies should be defined via behaviors in YAML config.
        This method is kept for backward compatibility only.

        Options:
            "hierarchical" - Keep last N exchanges (TaskExecutor)
            "append_until_full" - Append until token limit, then compact (Orchestrator)

        Returns:
            Strategy name (default: "append_until_full")
        """
        # Default to append_until_full for backward compatibility
        # New agents should define context strategy via behaviors in config YAML
        return "append_until_full"

    # ===========================
    # Methods with default implementations
    # (Agents can override for custom behavior)
    # ===========================

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return tool definitions for this agent.

        Default implementation: Returns behavior tools + core completion tools.
        Core completion tools (mark_complete/mark_failed) are only included
        if the agent has a goal set. This allows pure conversational agents
        to operate without goal completion semantics.

        Override this method if you need custom tool handling.

        Returns:
            List of tool definitions in Ollama format
        """
        tools = []

        # Core completion tools (only available when agent has a goal)
        if self.goal:
            core_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "mark_complete",
                        "description": "Mark the task/goal as complete and report success. REQUIRED when work is finished.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "summary": {
                                    "type": "string",
                                    "description": "Brief summary of what was accomplished (2-4 sentences)"
                                }
                            },
                            "required": ["summary"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "mark_failed",
                        "description": "Mark the task/goal as failed and report reason. Use when you cannot complete the task.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reason": {
                                    "type": "string",
                                    "description": "Explanation of why the task could not be completed"
                                }
                            },
                            "required": ["reason"]
                        }
                    }
                }
            ]
            tools.extend(core_tools)

        # Behavior tools
        behavior_tools = self.get_behavior_tools()
        tools.extend(behavior_tools)

        return tools

    def get_system_prompt(self) -> str:
        """
        Return system prompt for this agent.

        Default implementation: Returns config prompt + behavior instructions.
        Tool documentation is now injected by behaviors via on_initial_context().

        Override this method to provide custom system prompt.

        Returns:
            System prompt string
        """
        # New architecture: always use behavior system
        # Use config prompt if available, otherwise empty base
        base_prompt = self.config_system_prompt if self.config_system_prompt else ""

        parts = [base_prompt] if base_prompt else []

        # Add behavior instructions
        behavior_instructions = self.get_behavior_instructions()
        if behavior_instructions:
            parts.append(behavior_instructions)

        # Tool documentation is now handled by behaviors in on_initial_context()
        # (removed from here)

        return "\n\n".join(parts)

    def build_context(self) -> list[dict[str, Any]]:
        """
        Build context for LLM call using behavior system.

        Builds basic context with system prompt + messages, then lets behaviors enhance it.

        Returns:
            List of messages to send to LLM
        """
        # Build basic context
        context = [
            {"role": "system", "content": self.get_system_prompt()},
            *self.state.messages
        ]

        # Inject goal context if goal is set (core agent functionality)
        if self.goal:
            context = self._inject_goal_context(context)

        # NOTE: Context enhancement by behaviors is now handled by lifecycle events:
        # - on_initial_context() for first-time setup
        # - on_round_start() for per-round modifications
        # These are called from run() method, not here.

        return context

    def _inject_goal_context(self, context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Inject goal information into context after system prompt.

        Args:
            context: Current context (system prompt + messages)

        Returns:
            Modified context with goal info injected
        """
        # Build goal context parts
        context_parts = []

        # Use different header based on delegation status
        if self.is_subagent:
            context_parts.append(f"DELEGATED GOAL: {self.goal}")
            context_parts.append("")
            context_parts.append("You are working on a task delegated by a parent agent.")
        else:
            context_parts.append(f"GOAL: {self.goal}")
            context_parts.append("")
            context_parts.append("You are working on a standalone task.")

        context_parts.append("When complete, call mark_complete(summary) with what you accomplished.")
        context_parts.append("If you cannot complete it, call mark_failed(reason) explaining why.")

        # Insert after system prompt (index 1)
        goal_message = {"role": "user", "content": "\n".join(context_parts)}
        context.insert(1, goal_message)

        # Inject completion nudge if pending
        if self.pending_nudge:
            context.append({"role": "user", "content": self.pending_nudge})
            self.pending_nudge = None  # Clear after injection

        return context

    def _check_completion_signals(self, response: dict[str, Any], tool_calls: list[dict[str, Any]]) -> None:
        """
        Check for completion signals in LLM response and set pending nudge.

        This detects when the LLM says things like "task complete" without
        calling mark_complete, and sets up a nudge for the next round.

        Args:
            response: LLM response dict
            tool_calls: List of tool calls from this round
        """
        from completion_detector import analyze_llm_response

        # Extract LLM response text
        llm_response = ""
        if "message" in response:
            msg = response["message"]
            if isinstance(msg, dict):
                llm_response = msg.get("content", "")
            elif hasattr(msg, "content"):
                llm_response = msg.content

        # Skip if no response to analyze
        if not llm_response:
            return

        # Analyze for completion signals
        analysis = analyze_llm_response(
            llm_response,
            tool_calls,
            current_subtask=self.goal
        )

        # Set pending nudge if completion signal detected
        if analysis["should_nudge"]:
            matched = analysis["matched_phrases"][0] if analysis["matched_phrases"] else "completion signal"

            # Compose goal-specific nudge
            self.pending_nudge = (
                f"💡 REMINDER: You mentioned '{matched}'. "
                f"If '{self.goal}' is complete, please call mark_complete(summary='...') with what you accomplished."
            )

            print(f"[{self.name}] 💡 Completion signal detected: '{matched[:50]}' - will nudge next round")

    # ===========================
    # Shared functionality
    # ===========================


    def _handle_timeout(self, error: TimeoutError) -> dict[str, Any]:
        """
        Handle LLM timeout with circuit breaker logic.

        Implements circuit breaker pattern:
        - Tracks consecutive timeouts
        - Attempts Ollama restart after threshold
        - Returns special markers for different states

        Args:
            error: The TimeoutError that occurred

        Returns:
            Response dict with timeout/circuit breaker information
        """
        print(f"\n⚠️  LLM TIMEOUT: {error}")
        print("[timeout] Incrementing timeout counter...")

        # Increment timeout counter
        self.consecutive_timeouts = getattr(self, 'consecutive_timeouts', 0) + 1
        self.total_timeouts = getattr(self, 'total_timeouts', 0) + 1

        # Check circuit breaker threshold
        if self.consecutive_timeouts >= self.max_consecutive_timeouts:
            print(f"[timeout] {self.consecutive_timeouts} consecutive timeouts (max: {self.max_consecutive_timeouts})")
            print("[timeout] Circuit breaker triggered - LLM service appears unavailable")

            # Attempt Ollama restart if configured
            if self.auto_restart_ollama:
                print("[timeout] auto_restart_ollama is enabled - attempting restart...")
                from llm_utils import restart_ollama
                restart_success = restart_ollama()
                if restart_success:
                    print("[timeout] Ollama restarted successfully - resetting timeout counter")
                    self.consecutive_timeouts = 0
                    # Return special marker to indicate restart occurred
                    return {
                        "message": {
                            "role": "assistant",
                            "content": "__OLLAMA_RESTARTED__ - will retry immediately",
                        },
                        "_ollama_restarted": True,
                    }
                else:
                    print("[timeout] Ollama restart failed - circuit breaker will trigger")
            else:
                print("[timeout] auto_restart_ollama is disabled (set to true in agent_config.yaml to enable)")

            # Return a special response indicating circuit breaker triggered
            # The calling agent's run() method should detect this and save partial progress
            return {
                "message": {
                    "role": "assistant",
                    "content": "__CIRCUIT_BREAKER_TRIGGERED__",
                },
                "_circuit_breaker": True,
                "_consecutive_timeouts": self.consecutive_timeouts,
            }

        # Otherwise, return timeout message but allow retry
        print(f"[timeout] Timeout {self.consecutive_timeouts}/{self.max_consecutive_timeouts} - will retry")
        return {
            "message": {
                "role": "assistant",
                "content": f"__LLM_TIMEOUT__ (attempt {self.consecutive_timeouts}/{self.max_consecutive_timeouts})",
            },
            "_timeout": True,
            "_consecutive_timeouts": self.consecutive_timeouts,
        }

    def call_llm(
        self,
        model: str,
        temperature: float,
        timeout: int = 120,
    ) -> dict[str, Any]:
        """
        Call LLM with current context and tools.

        DEPRECATED: This method builds context internally using build_context().
        New code should use _call_llm_with_context() which accepts context as a parameter.

        Args:
            model: Model name (e.g., "gpt-oss:20b")
            temperature: Sampling temperature
            timeout: Timeout in seconds

        Returns:
            LLM response dict with 'message' key
        """
        context = self.build_context()
        return self._call_llm_with_context(context, model, temperature, timeout)

    def _call_llm_with_context(
        self,
        context: list[dict[str, Any]],
        model: str,
        temperature: float,
        timeout: int = 120,
    ) -> dict[str, Any]:
        """
        Call LLM with provided context and tools.

        This is the preferred internal method that accepts context as a parameter,
        allowing lifecycle events to modify context before the LLM call.

        Args:
            context: Context messages to send to LLM
            model: Model name (e.g., "gpt-oss:20b")
            temperature: Sampling temperature
            timeout: Timeout in seconds

        Returns:
            LLM response dict with 'message' key
        """
        from llm_utils import chat_with_inactivity_timeout

        tools = self.get_tools()

        # Build options dict with temperature
        options = {"temperature": temperature}

        # Add num_ctx if specified in llm_config (otherwise llm_utils uses model-specific default)
        if hasattr(self, 'config') and self.config:
            if hasattr(self.config, 'llm') and self.config.llm:
                if self.config.llm.max_tokens is not None:
                    options["num_ctx"] = self.config.llm.max_tokens

        try:
            response = chat_with_inactivity_timeout(
                model=model,
                messages=context,
                options=options,
                tools=tools,
                inactivity_timeout=timeout,
            )

            # Reset timeout counter on successful LLM call
            self.consecutive_timeouts = 0

            return response

        except TimeoutError as e:
            return self._handle_timeout(e)

        except Exception as e:
            # Parse error to provide actionable feedback to LLM
            error_str = str(e)

            # Check if it's a tool call parsing error
            if "error parsing tool call" in error_str.lower():
                # Extract the malformed output from the error
                import re
                match = re.search(r"raw='(.*?)'", error_str, re.DOTALL)
                if match:
                    malformed_output = match.group(1)[:200]  # First 200 chars
                else:
                    malformed_output = "unknown"

                # Provide clear, actionable feedback
                feedback = (
                    "ERROR: Your last response had a malformed tool call.\n\n"
                    f"What you generated: {malformed_output}...\n\n"
                    "PROBLEM: Tool calls must be pure JSON with NO text before or after.\n\n"
                    "CORRECT FORMAT:\n"
                    "  {\n"
                    "    \"name\": \"tool_name\",\n"
                    "    \"arguments\": {\"arg1\": \"value1\"}\n"
                    "  }\n\n"
                    "INCORRECT (what you did):\n"
                    "  Let me do this: {\"name\": \"tool_name\", ...}  ← NO TEXT BEFORE JSON\n\n"
                    "Try again with ONLY the JSON tool call, no explanatory text."
                )

                return {
                    "message": {
                        "role": "user",  # Send as user message so LLM treats it as feedback
                        "content": feedback,
                    }
                }

            # For other errors, provide generic feedback
            return {
                "message": {
                    "role": "user",
                    "content": f"ERROR: LLM call failed with: {error_str}\n\nPlease try again.",
                }
            }

    def _validate_tool_parameters(self, tool_call: dict[str, Any]) -> dict[str, Any] | None:
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
        for tool in self.get_tools():
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
            param_specs.append(f"  - {param_name}: {param_type}{req_marker} - {param_desc}")

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

    def _log_parameter_wishlist(self, tool_name: str, invalid_params: set[str]) -> None:
        """
        Log hallucinated parameters to wishlist file for future consideration.

        Args:
            tool_name: Tool that was called
            invalid_params: Set of invalid parameter names
        """
        import json

        wishlist_file = Path(".agent_context") / "parameter_wishlist.jsonl"
        wishlist_file.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "hallucinated_params": list(invalid_params),
            "agent": self.name,
        }

        # Append to JSONL file
        with open(wishlist_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def dispatch_tool(self, tool_call: dict[str, Any], **extra_context) -> dict[str, Any]:
        """
        Dispatch a tool call to the appropriate handler.

        Default implementation: Handles core tools (mark_complete/mark_failed) then
        dispatches to behavior system.

        Args:
            tool_call: Tool call dict with function name and arguments
            **extra_context: Additional context to pass to behaviors

        Returns:
            Tool result dict
        """
        # Validate parameters before dispatch
        validation_result = self._validate_tool_parameters(tool_call)
        if validation_result:
            # Invalid parameters detected - return feedback to LLM
            return validation_result

        # Handle core completion tools (mark_complete, mark_failed)
        tool_name = tool_call.get("function", {}).get("name")
        if tool_name in ["mark_complete", "mark_failed"]:
            return self._dispatch_completion_tool(tool_call)

        # Dispatch to behavior system for other tools
        return self.dispatch_tool_to_behavior(tool_call, **extra_context)

    def _dispatch_completion_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch core completion tools (mark_complete, mark_failed).

        Args:
            tool_call: Tool call dict

        Returns:
            Tool result dict
        """
        tool_name = tool_call["function"]["name"]
        args = tool_call["function"].get("arguments", {})

        if tool_name == "mark_complete":
            summary = args.get('summary', 'Task completed')

            # Mark goal as complete in context manager if available
            if self.context_manager and hasattr(self.context_manager, 'state') and self.context_manager.state.goal:
                self.context_manager.state.goal.status = "success"

            return {
                "success": True,
                "result": f"Task marked complete: {summary}",
                "summary": summary,
                "status": "goal_complete"
            }

        elif tool_name == "mark_failed":
            reason = args.get('reason', 'Task failed')

            # Mark goal as failed in context manager if available
            if self.context_manager and hasattr(self.context_manager, 'state') and self.context_manager.state.goal:
                self.context_manager.state.goal.status = "failed"

            return {
                "success": False,
                "result": f"Task marked failed: {reason}",
                "reason": reason,
                "status": "goal_failed"
            }

        return {"error": f"Unknown completion tool: {tool_name}"}

    def persist_state(self) -> None:
        """Save agent state to disk."""
        with open(self.state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def load_state(self) -> None:
        """Load agent state from disk if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    self.state = AgentState.from_dict(data)
            except Exception:
                # If load fails, keep fresh state
                pass

    def add_message(self, message: dict[str, Any]) -> None:
        """
        Add a message to history.

        Args:
            message: Message dict with role and content
        """
        self.state.messages.append(message)

    def get_message_history(self) -> list[dict[str, Any]]:
        """Get full message history."""
        return self.state.messages

    def clear_messages(self) -> None:
        """Clear message history (useful for fresh starts)."""
        self.state.messages = []

    def increment_round(self) -> None:
        """Increment round counter and active subtask rounds."""
        self.state.total_rounds += 1

        # Also increment the active subtask's rounds_used
        if self.context_manager and self.context_manager.state.goal:
            current_task = self.context_manager._get_current_task()
            if current_task:
                active_subtask = current_task.active_subtask()
                if active_subtask:
                    active_subtask.rounds_used += 1
                    self.context_manager._save_state()

    # ===========================
    # Phase 1 additions: Helper methods for subsystems
    # ===========================

    def init_context_manager(self) -> None:
        """Initialize context manager for hierarchical task tracking (DEPRECATED)."""
        # Context manager is deprecated with the behavior system
        # Behaviors now handle context management directly
        pass

    def init_workspace_manager(self, goal_slug: str, workspace_path: Path | str | None = None) -> None:
        """
        Initialize workspace manager for this goal.

        Args:
            goal_slug: Goal description slug for workspace directory name
            workspace_path: Optional existing workspace path to reuse (for iteration)
        """
        from workspace_manager import WorkspaceManager
        if self.workspace_manager is None:
            self.workspace_manager = WorkspaceManager(
                goal=goal_slug,
                base_dir=self.workspace,
                workspace_path=workspace_path
            )

    def init_perf_stats(self) -> None:
        """Initialize performance stats tracking (DEPRECATED)."""
        # Performance stats are deprecated with the behavior system
        # Stats are now tracked by StatusDisplayBehavior if enabled
        pass

    def init_server_manager(self) -> None:
        """
        Initialize server manager for managing development servers.

        Used by orchestrator for coordinating servers across delegated tasks.
        """
        from server_manager import ServerManager
        if self.server_manager is None:
            self.server_manager = ServerManager(self.workspace)
            self.server_manager.start_monitoring()

    def init_registry(self, team_name: str = "default") -> None:
        """
        Initialize agent registry for delegation.

        Args:
            team_name: Name of team config file (default: "default")
        """
        from agent_registry import AgentRegistry
        if self.registry is None:
            self.registry = AgentRegistry(team_name=team_name, workspace=self.workspace)

    # ===========================
    # Phase 4 additions: Behavior system methods
    # ===========================

    def load_behaviors_from_config(self, config_file: str) -> None:
        """
        Load and register behaviors from YAML config file.

        DEPRECATED: This method is kept for backward compatibility.
        New code should pass config_file to __init__ instead.

        Args:
            config_file: Path to YAML config file
        """
        import yaml

        config_path = Path(config_file)
        if not config_path.exists():
            print(f"[{self.name}] Warning: Config file not found: {config_file}")
            return

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if not config:
            print(f"[{self.name}] Empty config file")
            return

        self._load_behaviors_from_config_dict(config)

    def _load_behaviors_from_config_dict(self, config: dict[str, Any]) -> None:
        """
        Internal method to load behaviors from config dict.

        Also loads system_prompt and blurb if present.
        Auto-adds DelegationBehavior based on team config.

        Behavior parameters are merged from global defaults (config/behavior_defaults.yaml)
        and agent-specific overrides (this config dict).

        Args:
            config: Config dict loaded from YAML
        """
        if not config:
            return

        # Load system prompt if present
        if "system_prompt" in config:
            self.config_system_prompt = config["system_prompt"]
            print(f"[{self.name}] Loaded system prompt from config ({len(self.config_system_prompt)} chars)")

        # Load blurb if present
        if "blurb" in config:
            self.config_blurb = config["blurb"]
            print(f"[{self.name}] Loaded blurb from config ({len(self.config_blurb)} chars)")

        # Auto-add DelegationBehavior if this agent can delegate
        self._auto_add_delegation_behavior()

        # Load behaviors
        if "behaviors" not in config:
            print(f"[{self.name}] No behaviors defined in config")
            return

        print(f"[{self.name}] Loading behaviors from config")

        # Load global behavior defaults
        global_defaults = self._load_global_behavior_defaults()

        for behavior_spec in config.get("behaviors", []):
            behavior_type = behavior_spec["type"]

            # Skip behaviors in the exclude list
            if behavior_type in self.exclude_behaviors:
                print(f"[{self.name}] Skipping excluded behavior: {behavior_type}")
                continue

            # Get global defaults for this behavior type
            default_params = global_defaults.get(behavior_type, {})
            # If default_params is None (empty YAML entry), convert to empty dict
            if default_params is None:
                default_params = {}

            # Get agent-specific overrides
            agent_params = behavior_spec.get("params", {})
            # If agent_params is None, convert to empty dict
            if agent_params is None:
                agent_params = {}

            # Merge: agent params override global defaults
            behavior_params = {**default_params, **agent_params}

            # Dynamically import and instantiate behavior
            try:
                behavior_class = self._import_behavior_class(behavior_type)
                behavior = behavior_class(**behavior_params)
                self.add_behavior(behavior)

                # Log parameter source
                if agent_params:
                    print(f"[{self.name}] Loaded behavior: {behavior_type} (agent-specific params: {agent_params})")
                elif default_params:
                    print(f"[{self.name}] Loaded behavior: {behavior_type} (using global defaults)")
                else:
                    print(f"[{self.name}] Loaded behavior: {behavior_type} (no parameters)")
            except Exception as e:
                print(f"[{self.name}] Failed to load behavior {behavior_type}: {e}")

    def _load_global_behavior_defaults(self) -> dict[str, dict[str, Any]]:
        """
        Load global behavior parameter defaults from config/behavior_defaults.yaml.

        Returns:
            Dict mapping behavior type name to parameter dict
            Example: {"LoopDetectionBehavior": {"max_repeats": 5}, ...}
        """
        from agent_config import load_behavior_defaults

        try:
            return load_behavior_defaults()
        except Exception as e:
            print(f"[{self.name}] Warning: Failed to load global behavior defaults: {e}")
            return {}

    def _load_target_agent_config(self, target_agent: str, agents: dict) -> dict:
        """
        Load configuration for a target agent from its config file.

        Args:
            target_agent: Name of target agent
            agents: Agents dict from team config

        Returns:
            Agent info dict with delegation_tool and blurb (if available)
        """
        from agent_config import load_agent_config

        agent_info = agents.get(target_agent, {})

        try:
            target_config = load_agent_config(target_agent)

            # Add delegation_tool if present
            if target_config and "delegation_tool" in target_config:
                agent_info["delegation_tool"] = target_config["delegation_tool"]

            # Add blurb if present
            if target_config and "blurb" in target_config:
                agent_info["blurb"] = target_config["blurb"]

        except Exception as e:
            print(f"[{self.name}] Warning: Failed to load config for {target_agent}: {e}")

        return agent_info

    def _auto_add_delegation_behavior(self) -> None:
        """
        Auto-add DelegationBehavior if this agent can delegate to others.

        DelegationBehavior is delegator-only - provides tools for delegating TO other agents.
        For being delegated to, use core goal tracking (available in ALL agents).

        Reads team config to check if this agent has can_delegate_to list.
        If yes, adds DelegationBehavior with delegation tools (consult_X, delegate_to_X).
        """
        from agent_config import load_team_config

        # Early return: DelegationBehavior already added
        if any(b.get_name() == "delegation" for b in self._behaviors):
            return

        try:
            # Load team config (default team)
            team_config = load_team_config("default")

            # Early return: invalid config structure
            if not team_config or "agents" not in team_config:
                return

            agents = team_config["agents"]

            # Early return: agent not found in config
            agent_config = agents.get(self.name)
            if not agent_config:
                return

            # Early return: no delegation capability
            can_delegate_to = agent_config.get("can_delegate_to", [])
            if not can_delegate_to:
                return

            # Build agent relationships dict
            agent_relationships = {"can_delegate_to": can_delegate_to}

            # Load config for each target agent
            for target_agent in can_delegate_to:
                agent_info = self._load_target_agent_config(target_agent, agents)
                agent_relationships[target_agent] = agent_info

            # Create and add DelegationBehavior
            from behaviors.delegation import DelegationBehavior
            delegation_behavior = DelegationBehavior(agent_relationships=agent_relationships)
            self.add_behavior(delegation_behavior)
            print(f"[{self.name}] Auto-added DelegationBehavior (can delegate to: {', '.join(can_delegate_to)})")

        except Exception as e:
            print(f"[{self.name}] Failed to auto-add DelegationBehavior: {e}")


    def _import_behavior_class(self, behavior_type: str):
        """
        Dynamically import behavior class by name.

        Args:
            behavior_type: CamelCase behavior class name (e.g., "FileToolsBehavior")

        Returns:
            Behavior class

        Raises:
            ImportError: If behavior module/class not found
        """
        # Convert CamelCase to snake_case for module name
        module_name = self._to_snake_case(behavior_type)

        # Import from behaviors module
        module = importlib.import_module(f"behaviors.{module_name}")
        return getattr(module, behavior_type)

    def _to_snake_case(self, name: str) -> str:
        """
        Convert CamelCase to snake_case, removing "Behavior" suffix.

        Examples:
            FileToolsBehavior -> file_tools
            LoopDetectionBehavior -> loop_detection
            (SubAgentModeBehavior is now core functionality in BaseAgent)
            ChatbotBehavior -> chatbot
            ArchitectToolsBehavior -> architect_tools

        Args:
            name: CamelCase name

        Returns:
            snake_case name without "_behavior" suffix
        """
        # Remove "Behavior" suffix if present
        if name.endswith("Behavior"):
            name = name[:-8]  # Remove "Behavior" (8 chars)

        # Special cases for known compound words
        # SubAgent -> subagent (not sub_agent)
        name = name.replace("SubAgent", "Subagent")

        # Convert to snake_case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def add_behavior(self, behavior: Any) -> None:
        """
        Register a behavior with this agent.

        Args:
            behavior: AgentBehavior instance

        Raises:
            ValueError: If behavior tool names conflict with existing tools
        """
        # Attach behavior to agent (sets self.agent on behavior)
        behavior.agent = self

        # Check for tool name conflicts
        for tool in behavior.get_tools():
            tool_name = tool["function"]["name"]
            if tool_name in self.tool_registry:
                existing_behavior = self.tool_registry[tool_name]
                raise ValueError(
                    f"Tool '{tool_name}' already registered by "
                    f"{existing_behavior.get_name()}"
                )
            self.tool_registry[tool_name] = behavior

        self._behaviors.append(behavior)

    def get_behavior_tools(self) -> list[dict[str, Any]]:
        """
        Collect tools from all registered behaviors.

        Returns:
            List of tool definitions from all behaviors
        """
        tools = []
        for behavior in self._behaviors:
            tools.extend(behavior.get_tools())
        return tools

    def get_behavior_instructions(self) -> str:
        """
        Collect instructions from all registered behaviors.

        Returns:
            Combined instructions from all behaviors
        """
        instructions = []
        for behavior in self._behaviors:
            inst = behavior.get_instructions()
            if inst:
                instructions.append(inst)
        return "\n\n".join(instructions)

    def generate_tool_documentation(self) -> str:
        """
        Generate tool documentation from loaded behaviors.

        DEPRECATED: Tool documentation should now be injected by behaviors via
        on_initial_context() method. This method is kept for backward compatibility
        but is no longer called by get_system_prompt().

        Returns a formatted string listing all available tools with their
        signatures and descriptions. This is dynamically generated based on
        which behaviors are loaded.

        Returns:
            Tool documentation string (empty if no behaviors loaded)
        """
        if not self._behaviors:
            return ""

        tool_docs = []
        for behavior in self._behaviors:
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
                            param_strs.append(f"{param_name}: {param_type} = {default}")
                        else:
                            param_strs.append(f"{param_name}?: {param_type}")

                param_sig = ", ".join(param_strs) if param_strs else ""
                tool_docs.append(f"  - {name}({param_sig}): {desc}")

        if tool_docs:
            return "\n\nAvailable tools:\n" + "\n".join(tool_docs)
        return ""

    def get_blurb(self) -> str:
        """
        Get agent blurb (description for parent agents).

        Tries multiple sources in order:
        1. config_blurb (from agent config file)
        2. blurb from team config
        3. Fallback: agent name + first 100 words of system prompt

        Returns:
            Agent blurb string
        """
        # First try config blurb
        if self.config_blurb:
            return self.config_blurb.strip()

        # Try team config
        from agent_config import load_team_config
        try:
            team_config = load_team_config("default")
            if team_config and "agents" in team_config:
                agent_config = team_config["agents"].get(self.name)
                if agent_config and "blurb" in agent_config:
                    return agent_config["blurb"].strip()
        except Exception:
            pass

        # Fallback: agent name + truncated system prompt
        system_prompt = self.get_system_prompt()
        words = system_prompt.split()[:100]  # First 100 words
        truncated = " ".join(words)
        if len(words) == 100:
            truncated += "..."
        return f"{self.name}: {truncated}"

    def dispatch_tool_to_behavior(self, tool_call: dict[str, Any], **extra_context) -> dict[str, Any]:
        """
        Dispatch tool call to appropriate behavior.

        Args:
            tool_call: Tool call dict with function name and arguments
            **extra_context: Additional context to pass to behaviors (e.g., registry, server_manager)

        Returns:
            Tool result dict
        """
        tool_name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]

        # Find behavior that owns this tool
        behavior = self.tool_registry.get(tool_name)
        if not behavior:
            return {"error": f"Unknown tool: {tool_name}"}

        # Dispatch to behavior with new lifecycle API (agent as first parameter)
        try:
            result = behavior.dispatch_tool(
                self,       # agent (positional)
                tool_name,  # tool_name (positional)
                args        # args (positional)
            )
        except Exception as e:
            return {"error": f"Tool {tool_name} failed: {e}"}

        # Notify all behaviors of tool call (for loop detection, etc.)
        for beh in self._behaviors:
            try:
                # Try new API signature first (agent as first param)
                if hasattr(beh, 'on_tool_call'):
                    # Check signature to determine which API to use
                    import inspect
                    sig = inspect.signature(beh.on_tool_call)
                    params = list(sig.parameters.keys())

                    # New API: on_tool_call(agent, tool_name, args, result)
                    if len(params) >= 4 and params[0] == 'agent':
                        beh.on_tool_call(
                            agent=self,
                            tool_name=tool_name,
                            args=args,
                            result=result
                        )
                    # Old API: on_tool_call(tool_name, args, result, **kwargs)
                    else:
                        beh.on_tool_call(
                            tool_name=tool_name,
                            args=args,
                            result=result,
                            agent=self
                        )
            except Exception as e:
                print(f"[{self.name}] Behavior {beh.get_name()} on_tool_call error: {e}")

        return result

    def trigger_behavior_event(self, event_name: str, **kwargs) -> None:
        """
        Trigger an event on all behaviors.

        DEPRECATED: This method is kept for backwards compatibility with old event names.
        New code should use the specific event trigger methods:
        - _trigger_on_goal_start()
        - _trigger_initial_context_setup()
        - _trigger_on_round_start()
        - _trigger_on_llm_response()
        - etc.

        Args:
            event_name: Event method name (e.g., "on_goal_start")
            **kwargs: Event-specific arguments
        """
        # Core agent initialization for onGoalSet (runs BEFORE behaviors)
        if event_name == "onGoalSet":
            self._handle_goal_set(**kwargs)

        # Trigger event on all behaviors
        for behavior in self._behaviors:
            try:
                event_method = getattr(behavior, event_name, None)
                if event_method and callable(event_method):
                    event_method(agent=self, **kwargs)
            except Exception as e:
                print(f"[{self.name}] Behavior {behavior.get_name()} {event_name} error: {e}")

    def _trigger_on_goal_start(self, goal: str) -> None:
        """
        Trigger on_goal_start event on all behaviors.

        Called ONCE when a goal is set, before any rounds.

        Args:
            goal: Goal description string
        """
        for behavior in self._behaviors:
            try:
                # Try new API first
                if hasattr(behavior, 'on_goal_start') and callable(behavior.on_goal_start):
                    behavior.on_goal_start(agent=self, goal=goal)
                # Fall back to old API if new one doesn't exist
                elif hasattr(behavior, 'onGoalStart') and callable(behavior.onGoalStart):
                    behavior.onGoalStart(goal=goal, agent=self)
            except Exception as e:
                print(f"[{self.name}] Behavior {behavior.get_name()} on_goal_start error: {e}")

    def _trigger_initial_context_setup(self) -> None:
        """
        Trigger on_initial_context event on all behaviors ONCE.

        Called at start of run(), before first LLM call.
        This allows behaviors to inject static context that doesn't change per round.

        NOTE: This modifies self.state.messages to inject initial context messages.
        """
        # Build initial context (system prompt only, no history yet)
        initial_context = [
            {"role": "system", "content": self.get_system_prompt()}
        ]

        # Inject goal context if goal is set (before behaviors)
        if self.goal:
            initial_context = self._inject_goal_context(initial_context)

        # Let behaviors inject initial context
        for behavior in self._behaviors:
            try:
                if hasattr(behavior, 'on_initial_context') and callable(behavior.on_initial_context):
                    initial_context = behavior.on_initial_context(agent=self, context=initial_context)
            except Exception as e:
                print(f"[{self.name}] Behavior {behavior.get_name()} on_initial_context error: {e}")

        # Extract any user messages that were injected (skip system prompt)
        # and prepend them to message history
        if len(initial_context) > 1:
            injected_messages = initial_context[1:]  # Everything after system prompt
            # Prepend to existing messages (if any)
            self.state.messages = injected_messages + self.state.messages

    def _trigger_on_round_start(self, round_number: int, context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Trigger on_round_start event on all behaviors.

        Called at the start of EVERY round, before LLM call.

        Args:
            round_number: Current round number (1-indexed)
            context: Current context to be sent to LLM

        Returns:
            Modified context after all behaviors have processed it
        """
        for behavior in self._behaviors:
            try:
                # Try new API first
                if hasattr(behavior, 'on_round_start') and callable(behavior.on_round_start):
                    context = behavior.on_round_start(agent=self, round_number=round_number, context=context)
                # Fall back to enhance_context if on_round_start doesn't exist
                elif hasattr(behavior, 'enhance_context') and callable(behavior.enhance_context):
                    context = behavior.enhance_context(
                        context,
                        agent=self,
                        workspace=self.workspace,
                        round_number=round_number,
                        context_manager=self.context_manager,
                        workspace_manager=self.workspace_manager,
                    )
            except Exception as e:
                print(f"[{self.name}] Behavior {behavior.get_name()} on_round_start error: {e}")

        return context

    def _trigger_on_llm_response(self, response: dict[str, Any]) -> None:
        """
        Trigger on_llm_response event on all behaviors.

        Called after LLM responds but before tool dispatch.

        Args:
            response: LLM response dict (contains 'message' with role/content/tool_calls)
        """
        for behavior in self._behaviors:
            try:
                if hasattr(behavior, 'on_llm_response') and callable(behavior.on_llm_response):
                    behavior.on_llm_response(agent=self, response=response)
            except Exception as e:
                print(f"[{self.name}] Behavior {behavior.get_name()} on_llm_response error: {e}")

    def _trigger_on_goal_complete(self, success: bool, summary: str) -> None:
        """
        Trigger on_goal_complete event on all behaviors.

        Called when goal completes (via mark_complete or mark_failed).

        Args:
            success: True if goal succeeded, False if failed
            summary: Summary message (from mark_complete or mark_failed)
        """
        for behavior in self._behaviors:
            try:
                if hasattr(behavior, 'on_goal_complete') and callable(behavior.on_goal_complete):
                    behavior.on_goal_complete(agent=self, success=success, summary=summary)
            except Exception as e:
                print(f"[{self.name}] Behavior {behavior.get_name()} on_goal_complete error: {e}")

    def _trigger_on_round_end(self, round_number: int) -> None:
        """
        Trigger on_round_end event on all behaviors.

        Called at end of each round, after all tool calls executed.

        Args:
            round_number: Current round number (1-indexed)
        """
        for behavior in self._behaviors:
            try:
                if hasattr(behavior, 'on_round_end') and callable(behavior.on_round_end):
                    behavior.on_round_end(agent=self, round_number=round_number)
            except Exception as e:
                print(f"[{self.name}] Behavior {behavior.get_name()} on_round_end error: {e}")

    def _trigger_on_timeout(self, elapsed_seconds: float) -> None:
        """
        Trigger on_timeout event on all behaviors.

        Called when goal times out (circuit breaker triggered).

        Args:
            elapsed_seconds: Time elapsed since goal start
        """
        for behavior in self._behaviors:
            try:
                if hasattr(behavior, 'on_timeout') and callable(behavior.on_timeout):
                    behavior.on_timeout(agent=self, elapsed_seconds=elapsed_seconds)
            except Exception as e:
                print(f"[{self.name}] Behavior {behavior.get_name()} on_timeout error: {e}")

    def _handle_goal_set(self, goal: str, **kwargs) -> None:
        """
        Core agent initialization when goal is set.

        This initializes workspace manager, performance tracking, and goal timing.
        Runs BEFORE behaviors receive the onGoalSet event.

        Args:
            goal: Goal description
            **kwargs: Additional context (workspace, workspace_manager, etc.)
        """
        import time

        # Store goal for context injection
        self.goal = goal

        # Initialize workspace manager
        workspace_param = kwargs.get('workspace')
        goal_slug = goal.lower()[:50].replace(" ", "-").replace("/", "-")

        if workspace_param:
            # Reuse mode: use existing workspace directory
            print(f"[{self.name}] Reusing workspace: {workspace_param}")
            self.init_workspace_manager(goal_slug, workspace_path=workspace_param)
        else:
            # Create new mode: create isolated workspace
            print(f"[{self.name}] Creating new workspace for goal")
            self.init_workspace_manager(goal_slug, workspace_path=None)

        # Update agent.workspace to point to the actual workspace directory
        if self.workspace_manager:
            self.workspace = self.workspace_manager.workspace_dir
            print(f"[{self.name}] Updated agent.workspace to: {self.workspace}")

        # Initialize performance tracking
        self.init_perf_stats()

        # Start wall-clock timer for goal
        self.goal_start_time = time.time()

        print(f"[{self.name}] Goal set: {goal}")

    def _save_partial_progress(self) -> dict:
        """
        Save progress when LLM becomes unavailable (circuit breaker triggered).

        This method scans the workspace for files created and checks for completed
        tasks in the context manager (if available). Returns a structured summary
        of work completed so far.

        Returns:
            dict: Summary with status, files_created, completed_tasks, etc.
        """
        # Count files created in workspace
        files_created = []
        if hasattr(self, 'workspace') and self.workspace and self.workspace.exists():
            try:
                all_files = list(self.workspace.rglob("*"))
                files_created = [f for f in all_files if f.is_file()]
            except Exception as e:
                print(f"[partial_progress] Error scanning workspace: {e}")

        # Get completed tasks from context manager if available
        completed_tasks = []
        if hasattr(self, 'context_manager') and self.context_manager:
            try:
                if hasattr(self.context_manager, 'state') and self.context_manager.state:
                    if hasattr(self.context_manager.state, 'goal') and self.context_manager.state.goal:
                        for task in self.context_manager.state.goal.tasks:
                            if hasattr(task, 'status') and task.status == "completed":
                                completed_tasks.append(task.description)
            except Exception as e:
                print(f"[partial_progress] Error extracting completed tasks: {e}")

        # Generate summary
        summary = {
            "status": "partial_success",
            "reason": f"LLM timeout - circuit breaker triggered after {self.consecutive_timeouts} consecutive failures",
            "files_created": len(files_created),
            "file_list": [str(f.relative_to(self.workspace)) for f in files_created] if files_created and self.workspace else [],
            "completed_tasks": len(completed_tasks),
            "task_list": completed_tasks,
            "workspace": str(self.workspace) if hasattr(self, 'workspace') and self.workspace else None,
            "total_timeouts": getattr(self, 'total_timeouts', 0),
            "agent": self.name,
        }

        # Print user-friendly summary
        print("\n" + "="*70)
        print(f"PARTIAL SUCCESS - Work Saved Despite Timeout ({self.name.upper()})")
        print("="*70)
        print(f"Files created: {len(files_created)}")
        for f in summary["file_list"][:10]:  # Show first 10
            print(f"  - {f}")
        if len(files_created) > 10:
            print(f"  ... and {len(files_created) - 10} more")

        if completed_tasks:
            print(f"\nCompleted tasks: {len(completed_tasks)}")
            for t in completed_tasks[:5]:  # Show first 5
                print(f"  - {t}")
            if len(completed_tasks) > 5:
                print(f"  ... and {len(completed_tasks) - 5} more")

        print(f"\nWorkspace: {summary['workspace']}")
        print(f"Total timeouts: {summary['total_timeouts']}")
        print("="*70)

        return summary

    # ===========================
    # CLI entry points (classmethod)
    # ===========================

    @classmethod
    def parse_cli_args(cls) -> dict[str, Any]:
        """
        Parse command line arguments.

        Returns:
            Dict with parsed arguments:
            - exit_after_initial: bool (--once flag)
            - force_chat_mode: bool (--chat flag)
            - custom_workspace: Path | None (--workspace path)
            - initial_message: str | None (remaining args as goal)
        """
        import sys

        exit_after_initial = False
        force_chat_mode = False
        custom_workspace = None
        args = sys.argv[1:]

        # Check for --once flag
        if "--once" in args:
            exit_after_initial = True
            args.remove("--once")

        # Check for --chat flag (enables ChatbotBehavior even with goal)
        if "--chat" in args:
            force_chat_mode = True
            args.remove("--chat")

        # Check for --workspace flag (custom workspace directory)
        if "--workspace" in args:
            idx = args.index("--workspace")
            if idx + 1 < len(args):
                custom_workspace = Path(args[idx + 1])
                args.pop(idx)  # Remove --workspace
                args.pop(idx)  # Remove path
            else:
                print("Error: --workspace requires a path argument")
                sys.exit(1)

        # Check for --timeout flag (subprocess timeout in seconds)
        timeout_seconds = 600  # Default: 10 minutes
        if "--timeout" in args:
            idx = args.index("--timeout")
            if idx + 1 < len(args):
                try:
                    timeout_seconds = int(args[idx + 1])
                    args.pop(idx)  # Remove --timeout
                    args.pop(idx)  # Remove value
                except ValueError:
                    print("Error: --timeout requires an integer value (seconds)")
                    sys.exit(1)
            else:
                print("Error: --timeout requires a timeout value in seconds")
                sys.exit(1)

        # Check for --clean-workspaces flag (workspace cleanup utility)
        clean_workspaces = False
        clean_older_than = None
        clean_dry_run = False

        if "--clean-workspaces" in args:
            clean_workspaces = True
            args.remove("--clean-workspaces")

            # Check for --older-than
            for arg in args[:]:
                if arg.startswith("--older-than="):
                    clean_older_than = arg.split("=", 1)[1]
                    args.remove(arg)

            # Check for --dry-run
            if "--dry-run" in args:
                clean_dry_run = True
                args.remove("--dry-run")

        if args:
            initial_message = " ".join(args).strip()
            # Treat empty string as None (no goal provided)
            if not initial_message:
                initial_message = None
        else:
            initial_message = None

        return {
            "exit_after_initial": exit_after_initial,
            "force_chat_mode": force_chat_mode,
            "custom_workspace": custom_workspace,
            "initial_message": initial_message,
            "timeout_seconds": timeout_seconds,
            "clean_workspaces": clean_workspaces,
            "clean_older_than": clean_older_than,
            "clean_dry_run": clean_dry_run,
        }

    @classmethod
    def setup_workspace(cls, args: dict[str, Any]) -> Path:
        """
        Determine workspace directory based on CLI args.

        Args:
            args: Parsed CLI arguments from parse_cli_args()

        Returns:
            Workspace Path
        """
        import re

        custom_workspace = args["custom_workspace"]
        initial_message = args["initial_message"]
        force_chat_mode = args["force_chat_mode"]

        if custom_workspace:
            workspace = custom_workspace
            if not workspace.exists():
                workspace.mkdir(parents=True, exist_ok=True)
            print(f"[{cls.__name__}] Using custom workspace: {workspace}")
        elif initial_message and not force_chat_mode:
            # Autonomous mode with goal: create isolated workspace for this specific goal
            slug = re.sub(r'[^a-z0-9]+', '-', initial_message.lower())
            slug = slug.strip('-')[:60]
            workspace = Path.cwd() / ".agent_workspaces" / slug
            workspace.mkdir(parents=True, exist_ok=True)
            print(f"[{cls.__name__}] Created isolated workspace: {workspace}")
        else:
            # Interactive/chat mode: use .agent_workspaces/{agent_name} for state
            agent_name = cls.__name__.replace("Agent", "").lower()
            workspace = Path.cwd() / ".agent_workspaces" / agent_name
            workspace.mkdir(parents=True, exist_ok=True)
            print(f"[{cls.__name__}] Using {agent_name} workspace for state: {workspace}")

        return workspace

    @classmethod
    def create_agent_instance(cls, workspace: Path, args: dict[str, Any]):
        """
        Create agent instance with appropriate configuration.

        Automatically handles ChatbotBehavior exclusion for agents:
        - Exclude ChatbotBehavior when goal string provided (unless --chat flag)
        - Include ChatbotBehavior when no goal (interactive) or --chat flag

        Subclasses can override to customize agent creation.

        Args:
            workspace: Workspace directory path
            args: Parsed CLI arguments

        Returns:
            Agent instance
        """
        initial_message = args.get("initial_message")
        force_chat_mode = args.get("force_chat_mode", False)
        timeout_seconds = args.get("timeout_seconds", 600)

        # Determine if ChatbotBehavior should be excluded
        exclude_behaviors = []
        if initial_message and not force_chat_mode:
            # Autonomous mode: exclude chatbot to run goal directly
            exclude_behaviors = ["ChatbotBehavior"]

        # Create agent with appropriate exclusions
        return cls(
            workspace=workspace,
            exclude_behaviors=exclude_behaviors,
            timeout_seconds=timeout_seconds
        )

    @classmethod
    def run_agent(cls, agent: BaseAgent, args: dict[str, Any]) -> None:
        """
        Execute the agent (interactive or autonomous mode).

        This method automatically detects ChatbotBehavior and enables multi-task
        chat mode if present. Otherwise, uses single-goal execution.

        Subclasses can override to customize execution logic.

        Args:
            agent: Agent instance
            args: Parsed CLI arguments
        """
        initial_message = args["initial_message"]
        exit_after_initial = args["exit_after_initial"]

        # Detect ChatbotBehavior for multi-task chat mode
        chatbot_behavior = None
        for behavior in agent.behaviors:
            if behavior.get_name() == "chatbot":
                chatbot_behavior = behavior
                break

        # Multi-task chat mode if ChatbotBehavior present
        if chatbot_behavior:
            cls._run_multi_task_chat_mode(agent, chatbot_behavior, initial_message, exit_after_initial)
        elif initial_message:
            # Single task mode (no ChatbotBehavior)
            print(f"User: {initial_message}\n")
            agent.trigger_behavior_event("onGoalSet", goal=initial_message, workspace=agent.workspace)
            result = agent.run()

            # Print result
            if result.get("status") == "success":
                print("\nTask completed successfully!")
                if "summary" in result:
                    print(f"Summary: {result['summary']}")
            else:
                print(f"\nTask failed: {result.get('reason', 'unknown')}")

            if exit_after_initial:
                print("\nExiting...")
                return
        else:
            # No ChatbotBehavior and no initial message - can't do anything
            print("Interactive mode not supported without ChatbotBehavior.")
            print("Usage: python {script} 'goal description'")

    @classmethod
    def _run_multi_task_chat_mode(
        cls,
        agent: BaseAgent,
        chatbot_behavior: Any,
        initial_message: str | None,
        exit_after_initial: bool
    ) -> None:
        """
        Run agent in multi-task chat mode using ChatbotBehavior.

        This enables any agent with ChatbotBehavior to handle multiple tasks
        in a conversational interface.

        Supports two modes:
        - Execute mode: Full task execution with workspace/tools (TaskExecutorAgent)
        - Chat mode: Pure conversation without task execution (simple chatbot)

        Args:
            agent: Agent instance
            chatbot_behavior: ChatbotBehavior instance
            initial_message: Optional initial message to execute
            exit_after_initial: Whether to exit after first task
        """
        # Determine mode: chat-only or task-execution
        # Chat-only if agent has no file/command tools
        has_file_tools = any(
            behavior.get_name() in ['read_file_tools', 'write_file_tools', 'directory_tools', 'command_tools']
            for behavior in agent.behaviors
        )
        chat_only_mode = not has_file_tools

        # Define task execution callback
        def execute_task(user_message: str) -> None:
            """Execute a single task in multi-task mode."""
            if chat_only_mode:
                # Pure chat mode - just run a single LLM round
                agent.run_single_llm_round(user_message)
            else:
                # Task execution mode - full workspace/tool setup
                # Call pre-task hooks (agent and behaviors)
                if hasattr(agent, 'pre_task_hook'):
                    agent.pre_task_hook()
                for behavior in agent.behaviors:
                    if hasattr(behavior, 'pre_task_hook'):
                        behavior.pre_task_hook(agent)

                # Reset ChatbotBehavior flags for new task
                chatbot_behavior.task_complete_flag = False
                chatbot_behavior.consecutive_empty_rounds = 0

                # Trigger onGoalSet to initialize workspace and subsystems
                # This is needed for agents like task_executor that need workspace setup
                agent.trigger_behavior_event(
                    "onGoalSet",
                    goal=user_message,
                    workspace=agent.workspace
                )

                # Execute task using base_agent's run_task_round_loop
                agent.run_task_round_loop(
                    user_message=user_message,
                    max_rounds=100,
                    check_completion_callback=lambda: chatbot_behavior.task_complete_flag
                )

        try:
            # Use ChatbotBehavior's chat loop
            if not exit_after_initial:
                # Multi-task chat mode
                chatbot_behavior.run_chat_loop(
                    agent=agent,
                    execute_task_callback=execute_task,
                    initial_message=initial_message
                )
            elif initial_message:
                # Single task mode (--once flag)
                print(f"User: {initial_message}\n")
                execute_task(initial_message)
                print("\nTask completed. Exiting...")
            else:
                # Interactive mode without initial message - use chat loop
                chatbot_behavior.run_chat_loop(
                    agent=agent,
                    execute_task_callback=execute_task,
                    initial_message=None
                )
        finally:
            # Call cleanup hooks (agent and behaviors)
            if hasattr(agent, 'cleanup_hook'):
                agent.cleanup_hook()
            for behavior in agent.behaviors:
                if hasattr(behavior, 'cleanup_hook'):
                    behavior.cleanup_hook(agent)

    @classmethod
    def main(cls) -> None:
        """
        Main entry point for CLI execution.

        This method orchestrates:
        1. UTF-8 encoding setup (Windows)
        2. CLI argument parsing
        3. Workspace setup
        4. Agent creation
        5. Agent execution

        Subclasses can override create_agent_instance() or run_agent()
        to customize behavior while reusing the CLI framework.
        """
        import sys

        # Ensure UTF-8 encoding on Windows
        if sys.platform == "win32":
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

        # Parse CLI arguments
        args = cls.parse_cli_args()

        # Handle workspace cleanup mode (exits after cleanup)
        if args.get("clean_workspaces"):
            from utils.workspace_cleanup import clean_workspaces, print_cleanup_report

            try:
                results = clean_workspaces(
                    older_than=args.get("clean_older_than"),
                    dry_run=args.get("clean_dry_run", False),
                )
                print_cleanup_report(results)
                sys.exit(0)
            except ValueError as e:
                print(f"Error: {e}")
                print("\nUsage: python agent.py --clean-workspaces --older-than=7d [--dry-run]")
                sys.exit(1)

        # Setup workspace
        workspace = cls.setup_workspace(args)

        # Create agent instance (can be overridden by subclasses)
        agent = cls.create_agent_instance(workspace, args)

        # Run agent (can be overridden by subclasses)
        try:
            cls.run_agent(agent, args)
        except KeyboardInterrupt:
            print("\n\nInterrupted. Shutting down...")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    # ===========================
    # Generic run() method (works for ALL agents)
    # ===========================

    def _setup_run(self, max_rounds: int | None = None) -> tuple[int, str, float]:
        """
        Setup run configuration and trigger start events.

        Args:
            max_rounds: Maximum rounds (None = use config default)

        Returns:
            Tuple of (max_rounds, model, temperature)
        """
        # Get max rounds from config or parameter
        if max_rounds is None:
            max_rounds = getattr(self.config.rounds, 'max_per_subtask', 128) if self.config else 128

        # Get model and temperature from config or defaults
        model = getattr(self, 'model', None) or getattr(self.config.llm, 'model', 'qwen3:8b') if self.config else 'qwen3:8b'
        temperature = getattr(self, 'temperature', None) or getattr(self.config.llm, 'temperature', 0.2) if self.config else 0.2

        # Trigger onGoalStart event (DEPRECATED - kept for backwards compatibility)
        # This event should fire whenever there's a goal, regardless of context_manager
        goal_desc = self._get_goal_description()
        if goal_desc:
            self.trigger_behavior_event("onGoalStart", goal=goal_desc)

        # Trigger new on_goal_start event (preferred API)
        if goal_desc:
            self._trigger_on_goal_start(goal_desc)

        # Trigger on_initial_context ONCE for first-round setup
        self._trigger_initial_context_setup()

        print(f"[{self.name}] Starting run loop (max_rounds={max_rounds}, model={model})")
        return max_rounds, model, temperature

    def _get_goal_description(self) -> str | None:
        """
        Get goal description from context manager or core goal tracking.

        Returns:
            Goal description string or None
        """
        # Try context_manager first
        if self.context_manager and self.context_manager.state.goal:
            return self.context_manager.state.goal.description

        # Try core goal tracking
        if self.goal:
            return self.goal

        return None

    def _check_completion_signal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        """
        Check if tool result contains a completion signal.

        Args:
            result: Tool execution result

        Returns:
            Completion dict if goal completed/failed, None otherwise
        """
        if not isinstance(result, dict):
            return None

        # IMPORTANT: Exclude delegation results (they have "target_agent" field)
        # Delegation success should NOT auto-complete the calling agent's goal
        is_delegation_result = "target_agent" in result
        if is_delegation_result:
            return None

        # Get goal description
        goal_desc = self._get_goal_description()

        # Check for mark_complete (success=True + summary)
        if result.get("success") is True and "summary" in result:
            print(f"[{self.name}] Goal marked complete")
            # Trigger new on_goal_complete event
            self._trigger_on_goal_complete(success=True, summary=result.get("summary", ""))
            # Keep old event for backwards compatibility
            self.trigger_behavior_event(
                "onGoalComplete",
                success=True,
                result=result,
                goal=goal_desc,
                llm_call_func=self.call_llm,
                workspace_manager=self.workspace_manager
            )
            return {
                "status": "success",
                "summary": result.get("summary"),
                "workspace": str(self.workspace) if self.workspace else None,
            }

        # Check for mark_failed (success=False + reason)
        if result.get("success") is False and "reason" in result:
            print(f"[{self.name}] Goal marked failed")
            # Trigger new on_goal_complete event
            self._trigger_on_goal_complete(success=False, summary=result.get("reason", ""))
            # Keep old event for backwards compatibility
            self.trigger_behavior_event(
                "onGoalComplete",
                success=False,
                result=result,
                goal=goal_desc,
                llm_call_func=self.call_llm,
                workspace_manager=self.workspace_manager
            )
            return {
                "status": "failure",
                "reason": result.get("reason"),
                "workspace": str(self.workspace) if self.workspace else None,
            }

        # Check for legacy goal_complete status
        actual_result = result.get("result", result)
        if isinstance(actual_result, dict) and actual_result.get("status") == "goal_complete":
            print(f"[{self.name}] Goal completed (legacy signal)")
            # Trigger new on_goal_complete event
            self._trigger_on_goal_complete(success=True, summary=actual_result.get("message", "Goal completed"))
            # Keep old event for backwards compatibility
            self.trigger_behavior_event(
                "onGoalComplete",
                success=True,
                result=actual_result,
                goal=goal_desc,
                llm_call_func=self.call_llm,
                workspace_manager=self.workspace_manager
            )
            return {
                "status": "success",
                "message": actual_result.get("message", "Goal completed"),
                "workspace": str(self.workspace) if self.workspace else None,
            }

        return None

    def _format_tool_call_preview(self, tool_name: str, args: dict[str, Any], max_length: int = 30) -> str:
        """
        Format a tool call with truncated argument preview.

        Args:
            tool_name: Name of the tool
            args: Tool arguments dict
            max_length: Maximum characters for each argument value preview

        Returns:
            Formatted string like "write_file(path=calculator.py, conte...)"
        """
        if not args:
            return f"{tool_name}()"

        # Format each argument with truncated value
        arg_previews = []
        for key, value in args.items():
            # Convert value to string and truncate
            value_str = str(value)
            if len(value_str) > max_length:
                value_preview = value_str[:max_length] + "..."
            else:
                value_preview = value_str

            # Clean up newlines and multiple spaces for display
            value_preview = value_preview.replace("\n", "\\n").replace("\r", "")
            value_preview = " ".join(value_preview.split())  # Collapse whitespace

            arg_previews.append(f"{key}={value_preview}")

        # Join arguments with comma
        args_str = ", ".join(arg_previews)
        return f"{tool_name}({args_str})"

    def _execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
        """
        Execute tool calls and check for completion after each.

        Args:
            tool_calls: List of tool call dicts

        Returns:
            Completion dict if goal completed, None otherwise
        """
        import json

        print(f"[{self.name}] Executing {len(tool_calls)} tool call(s)")

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            args = tool_call["function"].get("arguments", {})

            # Create preview of arguments
            preview = self._format_tool_call_preview(tool_name, args)
            print(f"[{self.name}] -> {preview}")

            # Dispatch tool
            result = self.dispatch_tool(tool_call)

            # Add tool result to messages
            tool_result_str = json.dumps(result)
            tool_message = {
                "role": "tool",
                "content": tool_result_str,
            }
            self.add_message(tool_message)

            # Check for completion signal
            completion = self._check_completion_signal(result)
            if completion:
                return completion

        return None

    def _execute_round(
        self,
        round_no: int,
        max_rounds: int,
        model: str,
        temperature: float
    ) -> dict[str, Any] | None:
        """
        Execute a single round of the agent loop.

        Args:
            round_no: Current round number
            max_rounds: Maximum rounds
            model: LLM model name
            temperature: Sampling temperature

        Returns:
            Completion dict if goal completed/failed, None to continue
        """
        # Trigger onRoundStart (DEPRECATED - kept for backwards compatibility)
        self.trigger_behavior_event("onRoundStart", round_number=round_no)

        # Build context for this round
        context = self.build_context()

        # Trigger on_round_start (NEW API - called every round before LLM)
        context = self._trigger_on_round_start(round_no, context)

        # Call LLM with modified context
        print(f"\n[{self.name}] Round {round_no}/{max_rounds}")
        response = self._call_llm_with_context(context, model=model, temperature=temperature)

        # Check for circuit breaker (consecutive timeouts)
        if response.get("_circuit_breaker"):
            print(f"[{self.name}] Circuit breaker triggered - saving partial progress")

            # Trigger timeout event (NEW)
            if self.goal_start_time:
                elapsed = time.time() - self.goal_start_time
                self._trigger_on_timeout(elapsed)

            return self._save_partial_progress()

        # Check for timeout (will retry)
        if response.get("_timeout"):
            print(f"[{self.name}] LLM timeout - retrying next round")
            return None

        # Trigger on_llm_response (NEW API - after LLM responds, before tool dispatch)
        if "message" in response:
            self._trigger_on_llm_response(response["message"])

        # Add assistant message to history
        had_tool_calls = False
        if "message" in response:
            msg = response["message"]
            self.add_message(msg)

            # Execute tool calls if present
            if "tool_calls" in msg and msg["tool_calls"]:
                had_tool_calls = True
                completion = self._execute_tool_calls(msg["tool_calls"])
                if completion:
                    return completion

        # Check for completion signals and set nudge (core agent functionality)
        if self.enable_completion_nudging and round_no >= self.min_rounds_before_nudge:
            self._check_completion_signals(response, msg.get("tool_calls", []) if "message" in response else [])

        # Trigger onRoundEnd (DEPRECATED - kept for backwards compatibility)
        self.trigger_behavior_event("onRoundEnd", round_number=round_no, had_tool_calls=had_tool_calls)

        # Trigger on_round_end (NEW API - preferred)
        self._trigger_on_round_end(round_no)

        # Increment round counter
        self.increment_round()

        return None

    def _handle_max_rounds(self, max_rounds: int) -> dict[str, Any]:
        """
        Handle case when max rounds reached without completion.

        Args:
            max_rounds: Maximum rounds that was reached

        Returns:
            Failure result dict
        """
        print(f"[{self.name}] Max rounds ({max_rounds}) reached without completion")
        goal_desc = self._get_goal_description() or "unknown"
        return {
            "status": "failure",
            "reason": f"Max rounds ({max_rounds}) exceeded",
            "goal": goal_desc,
            "workspace": str(self.workspace) if self.workspace else None,
        }

    def run(self, max_rounds: int | None = None) -> dict[str, Any]:
        """
        Generic agent run loop that works for all agent types.

        This method provides a standard execution loop that:
        1. Triggers behavior events (on_goal_start, on_round_start, etc.)
        2. Calls LLM in a loop
        3. Dispatches tool calls via dispatch_tool()
        4. Checks for completion (mark_complete, goal_complete, etc.)
        5. Handles timeouts and circuit breakers
        6. Returns structured results

        Override this method in subclasses if you need custom run logic.
        Most agents should be able to use this default implementation.

        Args:
            max_rounds: Maximum rounds before giving up (defaults to config value)

        Returns:
            Result dict with status, message, etc.
        """
        # Setup: get config and trigger start events
        max_rounds, model, temperature = self._setup_run(max_rounds)

        try:
            # Main loop: execute rounds until completion or max rounds
            for round_no in range(1, max_rounds + 1):
                result = self._execute_round(round_no, max_rounds, model, temperature)
                if result:
                    return result

            # Max rounds reached without completion
            return self._handle_max_rounds(max_rounds)

        except RuntimeError as e:
            # Handle auto-fail from loop detection
            error_msg = str(e)
            if "Auto-fail" in error_msg or "consecutive empty rounds" in error_msg:
                print(f"[{self.name}] Auto-fail triggered by loop detection: {e}")
                # Return as failure (not error) - this is expected behavior
                return {
                    "status": "failure",
                    "reason": f"Auto-failed due to stuck state: {error_msg}",
                    "workspace": str(self.workspace) if self.workspace else None,
                }
            else:
                # Other RuntimeErrors - treat as errors
                print(f"[{self.name}] RuntimeError during run: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "status": "error",
                    "reason": str(e),
                    "workspace": str(self.workspace) if self.workspace else None,
                }
        except Exception as e:
            print(f"[{self.name}] Exception during run: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "reason": str(e),
                "workspace": str(self.workspace) if self.workspace else None,
            }

    def run_single_llm_round(self, user_message: str) -> None:
        """
        Run a single LLM round for chat-only mode (no task execution).

        Used by simple chatbots that just need conversation without
        workspace setup, tool loops, or completion detection.

        Args:
            user_message: User's message/question
        """
        # Add user message to context
        self.add_message({"role": "user", "content": user_message})

        # Build context
        context = self.build_context()

        # Get model and temperature from config (same logic as _setup_run)
        model = getattr(self, 'model', None) or getattr(self.config.llm, 'model', 'qwen3:8b') if self.config else 'qwen3:8b'
        temperature = getattr(self, 'temperature', None) or getattr(self.config.llm, 'temperature', 0.2) if self.config else 0.2

        # Call LLM
        response = self._call_llm_with_context(
            context,
            model=model,
            temperature=temperature
        )

        # Add assistant response to history
        if "message" in response:
            self.add_message(response["message"])
            # Print assistant response
            if "content" in response["message"]:
                print(f"\n{self.name}: {response['message']['content']}\n")

    def run_task_round_loop(
        self,
        user_message: str,
        max_rounds: int = 100,
        check_completion_callback: Callable[[], bool] | None = None
    ) -> None:
        """
        Execute round loop for a single task in multi-task mode.

        This is used by orchestrator for multi-task chat mode where each user
        message is treated as a separate task that runs its own round loop.

        Unlike run(), this method:
        - Does NOT trigger on_goal_set events
        - Does NOT return completion dict
        - Adds user message to history
        - Runs rounds until completion callback returns True or max rounds

        Args:
            user_message: User message to add to history and execute
            max_rounds: Maximum rounds for this task (default: 100)
            check_completion_callback: Optional callback to check if task complete
                                      (e.g., ChatbotBehavior.task_complete_flag)
        """
        # Add user message to history
        self.add_message({"role": "user", "content": user_message})

        # Get model and temperature from config
        model = getattr(self.config.llm, 'model', 'qwen3:8b') if self.config else 'qwen3:8b'
        temperature = getattr(self.config.llm, 'temperature', 0.2) if self.config else 0.2

        # Execute rounds until completion
        for round_no in range(1, max_rounds + 1):
            # Check external completion signal (e.g., ChatbotBehavior detected idle)
            if check_completion_callback and check_completion_callback():
                print(f"[{self.name}] Task complete (detected by callback)")
                break

            # Execute one round
            result = self._execute_round(round_no, max_rounds, model, temperature)

            # Check for goal completion (mark_complete, etc.)
            if result:
                # Task completed via tool call (mark_complete, mark_failed, etc.)
                break

            # Check max rounds
            if round_no >= max_rounds:
                print(f"[{self.name}] Max rounds reached for task")
                break
