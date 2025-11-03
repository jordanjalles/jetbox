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
from typing import Any
from pathlib import Path
import json
import time
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
    ):
        """
        Initialize base agent.

        Args:
            name: Agent identifier (e.g., "orchestrator", "task_executor")
            workspace: Working directory for this agent
            config_file: Path to agent-specific config YAML (e.g., "task_executor_config.yaml")
            exclude_behaviors: List of behavior names to exclude (e.g., ["ChatbotBehavior"])
        """
        import yaml
        from agent_config import config as global_config

        self.name = name
        self.workspace = Path(workspace)
        self.config = global_config  # Global config for behavior defaults

        # Load agent-specific config file
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Agent config file not found: {config_file}")

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

        # Phase 4 additions: Behavior system
        self._behaviors: list[Any] = []  # List of registered behaviors (AgentBehavior instances)
        self.behaviors: list[Any] = self._behaviors  # Public alias
        self.tool_registry: dict[str, Any] = {}  # Map tool_name -> behavior that provides it
        self.config_system_prompt: str | None = None  # System prompt loaded from config (if any)
        self.config_blurb: str | None = None  # Agent blurb loaded from config (if any)
        self.exclude_behaviors: list[str] = exclude_behaviors or []  # Behaviors to exclude from loading

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

        Default implementation: Returns behavior tools (always enabled in new architecture).
        Override this method if you need custom tool handling.

        Returns:
            List of tool definitions in Ollama format
        """
        # New architecture: always use behavior tools
        return self.get_behavior_tools()

    def get_system_prompt(self) -> str:
        """
        Return system prompt for this agent.

        Default implementation: Returns config prompt + behavior instructions + tool docs.
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

        # Add dynamic tool documentation
        tool_docs = self.generate_tool_documentation()
        if tool_docs:
            parts.append(tool_docs)

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

        # Let behaviors enhance context
        context = self.enhance_context_with_behaviors(context)

        return context

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
        print(f"[timeout] Incrementing timeout counter...")

        # Increment timeout counter
        self.consecutive_timeouts = getattr(self, 'consecutive_timeouts', 0) + 1
        self.total_timeouts = getattr(self, 'total_timeouts', 0) + 1

        # Check circuit breaker threshold
        if self.consecutive_timeouts >= self.max_consecutive_timeouts:
            print(f"[timeout] {self.consecutive_timeouts} consecutive timeouts (max: {self.max_consecutive_timeouts})")
            print(f"[timeout] Circuit breaker triggered - LLM service appears unavailable")

            # Attempt Ollama restart if configured
            if self.auto_restart_ollama:
                print(f"[timeout] auto_restart_ollama is enabled - attempting restart...")
                from llm_utils import restart_ollama
                restart_success = restart_ollama()
                if restart_success:
                    print(f"[timeout] Ollama restarted successfully - resetting timeout counter")
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
                    print(f"[timeout] Ollama restart failed - circuit breaker will trigger")
            else:
                print(f"[timeout] auto_restart_ollama is disabled (set to true in agent_config.yaml to enable)")

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

        Args:
            model: Model name (e.g., "gpt-oss:20b")
            temperature: Sampling temperature
            timeout: Timeout in seconds

        Returns:
            LLM response dict with 'message' key
        """
        from ollama import chat
        from llm_utils import chat_with_inactivity_timeout

        context = self.build_context()
        tools = self.get_tools()

        try:
            response = chat_with_inactivity_timeout(
                model=model,
                messages=context,
                options={"temperature": temperature},
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
        from datetime import datetime

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

    def dispatch_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch a tool call to the appropriate handler.

        Default implementation: Always dispatches to behavior system (new architecture).
        Override this method if you need custom tool dispatch.

        Args:
            tool_call: Tool call dict with function name and arguments

        Returns:
            Tool result dict
        """
        # Validate parameters before dispatch
        validation_result = self._validate_tool_parameters(tool_call)
        if validation_result:
            # Invalid parameters detected - return feedback to LLM
            return validation_result

        # New architecture: always use behavior dispatch
        return self.dispatch_tool_to_behavior(tool_call)

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
        Auto-adds DelegationBehavior and SubAgentModeBehavior based on agents.yaml.

        Behavior parameters are merged from global defaults (agent_config.yaml)
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

        # Auto-add SubAgentContextBehavior if this agent is a subagent
        self._auto_add_subagent_context_behavior()

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
        Load global behavior parameter defaults from agent_config.yaml.

        Returns:
            Dict mapping behavior type name to parameter dict
            Example: {"LoopDetectionBehavior": {"max_repeats": 5}, ...}
        """
        import yaml

        config_path = Path(__file__).parent / "agent_config.yaml"

        if not config_path.exists():
            return {}

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            if not config:
                return {}

            return config.get("behavior_defaults", {})
        except Exception as e:
            print(f"[{self.name}] Warning: Failed to load global behavior defaults: {e}")
            return {}

    def _auto_add_delegation_behavior(self) -> None:
        """
        Auto-add DelegationBehavior if this agent can delegate to others.

        DelegationBehavior is delegator-only - provides tools for delegating TO other agents.
        For being delegated to, use SubAgentModeBehavior (added to ALL agents).

        Reads agents.yaml to check if this agent has can_delegate_to list.
        If yes, adds DelegationBehavior with delegation tools (consult_X, delegate_to_X).
        """
        import yaml

        agents_yaml = Path("agents.yaml")
        if not agents_yaml.exists():
            return

        try:
            with open(agents_yaml) as f:
                agents_config = yaml.safe_load(f)

            if not agents_config or "agents" not in agents_config:
                return

            agents = agents_config["agents"]

            # Find this agent's config
            agent_config = agents.get(self.name)
            if not agent_config:
                return

            # Check if agent can delegate to others
            can_delegate_to = agent_config.get("can_delegate_to", [])
            if not can_delegate_to:
                return  # No delegation capability

            # Check if DelegationBehavior already added
            has_delegation = any(b.get_name() == "delegation" for b in self._behaviors)
            if has_delegation:
                return  # Already added

            # Build agent relationships dict for DelegationBehavior
            agent_relationships = {
                "can_delegate_to": can_delegate_to
            }

            # Load delegation_tool and blurb from individual agent config files
            for target_agent in can_delegate_to:
                agent_info = agents.get(target_agent, {})

                # Try to load target agent's config file for delegation_tool and blurb
                agent_config_file = Path(f"{target_agent}_config.yaml")
                if agent_config_file.exists():
                    try:
                        with open(agent_config_file) as f:
                            target_config = yaml.safe_load(f)

                        # Add delegation_tool if present in config
                        if target_config and "delegation_tool" in target_config:
                            agent_info["delegation_tool"] = target_config["delegation_tool"]

                        # Add blurb if present in config
                        if target_config and "blurb" in target_config:
                            agent_info["blurb"] = target_config["blurb"]

                    except Exception as e:
                        print(f"[{self.name}] Warning: Failed to load {agent_config_file}: {e}")

                agent_relationships[target_agent] = agent_info

            # Create and add DelegationBehavior (delegator-only)
            from behaviors.delegation import DelegationBehavior
            delegation_behavior = DelegationBehavior(agent_relationships=agent_relationships)
            self.add_behavior(delegation_behavior)
            print(f"[{self.name}] Auto-added DelegationBehavior (can delegate to: {', '.join(can_delegate_to)})")

        except Exception as e:
            print(f"[{self.name}] Failed to auto-add DelegationBehavior: {e}")

    def _auto_add_subagent_context_behavior(self) -> None:
        """
        Auto-add SubAgentModeBehavior to ALL agents.

        SubAgentModeBehavior makes agents delegatable - they can receive work via:
        1. CLI: python agent.py "goal"
        2. Tool call: parent_agent.delegate_task(agent, goal)

        ALL agents get this behavior because ALL agents can be delegated to.
        This is different from DelegationBehavior which is only for agents that can delegate TO others.
        """
        # Check if SubAgentModeBehavior or SubAgentContextBehavior already added
        has_subagent_mode = any(
            b.get_name() in ["subagent_mode", "subagent_context"] for b in self._behaviors
        )

        if has_subagent_mode:
            return  # Already added

        # Add SubAgentModeBehavior to ALL agents
        try:
            from behaviors.subagent_mode import SubAgentModeBehavior
            behavior = SubAgentModeBehavior(is_subagent=True)
            self.add_behavior(behavior)
            print(f"[{self.name}] Auto-added SubAgentModeBehavior (agent is delegatable)")
        except ImportError:
            # Fall back to old name for backward compatibility
            try:
                from behaviors.subagent_context import SubAgentContextBehavior
                behavior = SubAgentContextBehavior()
                self.add_behavior(behavior)
                print(f"[{self.name}] Auto-added SubAgentContextBehavior (agent is delegatable)")
            except ImportError as e:
                print(f"[{self.name}] Failed to add SubAgentModeBehavior: {e}")

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
            SubAgentContextBehavior -> subagent_context (backward compat)
            SubAgentModeBehavior -> subagent_mode (new name)
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
        2. blurb from agents.yaml
        3. Fallback: agent name + first 100 words of system prompt

        Returns:
            Agent blurb string
        """
        # First try config blurb
        if self.config_blurb:
            return self.config_blurb.strip()

        # Try agents.yaml
        import yaml
        agents_yaml = Path("agents.yaml")
        if agents_yaml.exists():
            try:
                with open(agents_yaml) as f:
                    agents_config = yaml.safe_load(f)
                if agents_config and "agents" in agents_config:
                    agent_config = agents_config["agents"].get(self.name)
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

    def enhance_context_with_behaviors(
        self,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Let all behaviors enhance the context.

        Args:
            context: Current context (system prompt + messages)

        Returns:
            Enhanced context after all behavior modifications
        """
        # Let each behavior modify context in registration order
        for behavior in self._behaviors:
            context = behavior.enhance_context(
                context,
                agent=self,
                workspace=self.workspace,
                round_number=self.state.total_rounds,
                context_manager=self.context_manager,
                workspace_manager=self.workspace_manager,
            )

        return context

    def dispatch_tool_to_behavior(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch tool call to appropriate behavior.

        Args:
            tool_call: Tool call dict with function name and arguments

        Returns:
            Tool result dict
        """
        tool_name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]

        # Find behavior that owns this tool
        behavior = self.tool_registry.get(tool_name)
        if not behavior:
            return {"error": f"Unknown tool: {tool_name}"}

        # Dispatch to behavior
        try:
            result = behavior.dispatch_tool(
                tool_name=tool_name,
                args=args,
                agent=self,
                workspace=self.workspace,
                context_manager=self.context_manager,
                workspace_manager=self.workspace_manager,
                ledger_file=getattr(self, 'ledger_file', None)
            )
        except Exception as e:
            return {"error": f"Tool {tool_name} failed: {e}"}

        # Notify all behaviors of tool call (for loop detection, etc.)
        for beh in self._behaviors:
            try:
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

        Args:
            event_name: Event method name (e.g., "on_goal_start")
            **kwargs: Event-specific arguments
        """
        for behavior in self._behaviors:
            try:
                event_method = getattr(behavior, event_name, None)
                if event_method and callable(event_method):
                    event_method(agent=self, **kwargs)
            except Exception as e:
                print(f"[{self.name}] Behavior {behavior.get_name()} {event_name} error: {e}")

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
        model = getattr(self, 'model', None) or getattr(self.config.llm, 'model', 'gpt-oss:20b') if self.config else 'gpt-oss:20b'
        temperature = getattr(self, 'temperature', None) or getattr(self.config.llm, 'temperature', 0.2) if self.config else 0.2

        # Trigger on_goal_start event
        if self.context_manager and self.context_manager.state.goal:
            goal = self.context_manager.state.goal.description
            self.trigger_behavior_event("on_goal_start", goal=goal)

        print(f"[{self.name}] Starting run loop (max_rounds={max_rounds}, model={model})")
        return max_rounds, model, temperature

    def _get_goal_description(self) -> str | None:
        """
        Get goal description from context manager or SubAgentModeBehavior.

        Returns:
            Goal description string or None
        """
        # Try context_manager first
        if self.context_manager and self.context_manager.state.goal:
            return self.context_manager.state.goal.description

        # Try SubAgentModeBehavior
        for behavior in self._behaviors:
            if hasattr(behavior, 'goal') and behavior.goal:
                return behavior.goal

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
            self.trigger_behavior_event(
                "on_goal_complete",
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
            self.trigger_behavior_event(
                "on_goal_complete",
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
            self.trigger_behavior_event(
                "on_goal_complete",
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
            print(f"[{self.name}] -> {tool_name}")

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
        # Trigger on_round_start
        self.trigger_behavior_event("on_round_start", round_number=round_no)

        # Call LLM
        print(f"\n[{self.name}] Round {round_no}/{max_rounds}")
        response = self.call_llm(model=model, temperature=temperature)

        # Check for circuit breaker (consecutive timeouts)
        if response.get("_circuit_breaker"):
            print(f"[{self.name}] Circuit breaker triggered - saving partial progress")
            return self._save_partial_progress()

        # Check for timeout (will retry)
        if response.get("_timeout"):
            print(f"[{self.name}] LLM timeout - retrying next round")
            return None

        # Add assistant message to history
        if "message" in response:
            msg = response["message"]
            self.add_message(msg)

            # Execute tool calls if present
            if "tool_calls" in msg and msg["tool_calls"]:
                completion = self._execute_tool_calls(msg["tool_calls"])
                if completion:
                    return completion

        # Trigger on_round_end
        self.trigger_behavior_event("on_round_end", round_number=round_no)

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

        except Exception as e:
            print(f"[{self.name}] Exception during run: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "reason": str(e),
                "workspace": str(self.workspace) if self.workspace else None,
            }
