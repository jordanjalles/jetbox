"""
Base agent class providing common functionality for all agent types.

All agents inherit from BaseAgent and implement:
- get_tools(): Returns list of tools available to this agent
- get_system_prompt(): Returns the system prompt for this agent
- get_context_strategy(): Returns context management strategy name
"""
from __future__ import annotations
from abc import ABC, abstractmethod
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


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Provides common functionality:
    - LLM calling with tool support
    - Message history management
    - State persistence
    - Tool dispatch

    Subclasses must implement:
    - get_tools(): Tool definitions for this agent
    - get_system_prompt(): System prompt for this agent
    - get_context_strategy(): Context management strategy
    """

    def __init__(
        self,
        name: str,
        role: str,
        workspace: Path,
        config: Any = None,
    ):
        """
        Initialize base agent.

        Args:
            name: Agent identifier (e.g., "orchestrator", "task_executor")
            role: Human-readable role description
            workspace: Working directory for this agent
            config: Agent configuration (from agent_config.py)
        """
        self.name = name
        self.role = role
        self.workspace = Path(workspace)
        self.config = config

        # Create workspace if needed
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self.state = AgentState(
            name=name,
            role=role,
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
        self.behaviors: list[Any] = []  # List of registered behaviors (AgentBehavior instances)
        self.tool_registry: dict[str, Any] = {}  # Map tool_name -> behavior that provides it
        self.config_system_prompt: str | None = None  # System prompt loaded from config (if any)
        self.config_blurb: str | None = None  # Agent blurb loaded from config (if any)

    # ===========================
    # Abstract methods (must implement)
    # ===========================

    @abstractmethod
    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return tool definitions for this agent.

        Returns:
            List of tool definitions in Ollama format
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return system prompt for this agent.

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def get_context_strategy(self) -> str:
        """
        Return context management strategy name.

        Options:
            "hierarchical" - Keep last N exchanges (TaskExecutor)
            "append_until_full" - Append until token limit, then compact (Orchestrator)

        Returns:
            Strategy name
        """
        pass

    @abstractmethod
    def build_context(self) -> list[dict[str, Any]]:
        """
        Build context for LLM call using this agent's strategy.

        Returns:
            List of messages to send to LLM
        """
        pass

    # ===========================
    # Shared functionality
    # ===========================

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
            return response
        except Exception as e:
            return {
                "message": {
                    "role": "assistant",
                    "content": f"LLM call failed: {e}",
                }
            }

    def dispatch_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch a tool call to the appropriate handler.

        Args:
            tool_call: Tool call dict with function name and arguments

        Returns:
            Tool result dict
        """
        # This will be implemented by importing dispatch from agent.py
        # or by having each agent implement their own dispatch
        from agent import dispatch
        return dispatch(tool_call)

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
        """Initialize context manager for hierarchical task tracking."""
        from context_manager import ContextManager
        if self.context_manager is None:
            self.context_manager = ContextManager()

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
        """Initialize performance stats tracking."""
        from status_display import PerformanceStats
        if self.perf_stats is None:
            self.perf_stats = PerformanceStats()

    # ===========================
    # Phase 4 additions: Behavior system methods
    # ===========================

    def load_behaviors_from_config(self, config_file: str) -> None:
        """
        Load and register behaviors from YAML config file.

        Also loads system_prompt if present in config.
        Auto-adds DelegationBehavior if agent has can_delegate_to relationships.

        Behavior parameters are merged from global defaults (agent_config.yaml)
        and agent-specific overrides (this config file).

        Example config:
            system_prompt: |
              You are an agent that does X, Y, Z.

            behaviors:
              - type: FileToolsBehavior
                # No params - uses global defaults
              - type: LoopDetectionBehavior
                params:
                  max_repeats: 10  # Override global default

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

        print(f"[{self.name}] Loading behaviors from {config_file}")

        # Load global behavior defaults
        global_defaults = self._load_global_behavior_defaults()

        for behavior_spec in config.get("behaviors", []):
            behavior_type = behavior_spec["type"]

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
        Auto-add DelegationBehavior if this agent has delegation relationships.

        Reads agents.yaml to determine can_delegate_to list.
        Reads individual agent config files for delegation_tool schemas.
        If this agent can delegate, creates and adds DelegationBehavior.
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

            # Check if agent can delegate
            can_delegate_to = agent_config.get("can_delegate_to", [])
            if not can_delegate_to:
                return

            # Build agent relationships dict for DelegationBehavior
            # Include all agent info + this agent's delegation list
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

            # Create and add DelegationBehavior
            from behaviors.delegation import DelegationBehavior
            delegation_behavior = DelegationBehavior(agent_relationships)
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
            SubAgentContextBehavior -> subagent_context
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

        self.behaviors.append(behavior)

    def get_behavior_tools(self) -> list[dict[str, Any]]:
        """
        Collect tools from all registered behaviors.

        Returns:
            List of tool definitions from all behaviors
        """
        tools = []
        for behavior in self.behaviors:
            tools.extend(behavior.get_tools())
        return tools

    def get_behavior_instructions(self) -> str:
        """
        Collect instructions from all registered behaviors.

        Returns:
            Combined instructions from all behaviors
        """
        instructions = []
        for behavior in self.behaviors:
            inst = behavior.get_instructions()
            if inst:
                instructions.append(inst)
        return "\n\n".join(instructions)

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
        for behavior in self.behaviors:
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
        for beh in self.behaviors:
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
        for behavior in self.behaviors:
            try:
                event_method = getattr(behavior, event_name, None)
                if event_method and callable(event_method):
                    event_method(agent=self, **kwargs)
            except Exception as e:
                print(f"[{self.name}] Behavior {behavior.get_name()} {event_name} error: {e}")
