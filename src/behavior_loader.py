"""
Behavior loading system for BaseAgent.

This module handles:
- Loading behavior configurations from YAML files
- Dynamic behavior class importing
- Behavior parameter merging (global defaults + agent overrides)
- Behavior initialization and registration
- Auto-adding DelegationBehavior based on team config
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from pathlib import Path
import importlib
import re

if TYPE_CHECKING:
    from base_agent import BaseAgent


class BehaviorLoader:
    """
    Loads and initializes behaviors for an agent.

    This class handles all aspects of behavior loading:
    - YAML config parsing
    - Dynamic class importing
    - Parameter merging (global defaults + agent params)
    - Behavior instantiation
    - Tool registration
    - Auto-configuration (e.g., DelegationBehavior)

    Example:
        ```python
        # From BaseAgent.__init__
        self.behavior_loader = BehaviorLoader(self)
        self.behavior_loader.load_from_config_dict(agent_config)

        # Behavior loader populates:
        # - self._behaviors (list)
        # - self.tool_dispatcher.tool_registry (via registration)
        ```
    """

    def __init__(self, agent: BaseAgent):
        """
        Initialize behavior loader.

        Args:
            agent: Agent instance to load behaviors for
        """
        self.agent = agent

    def load_from_config_dict(self, config: dict[str, Any]) -> None:
        """
        Load behaviors from agent config dictionary.

        This is the main entry point for behavior loading. It:
        1. Loads system_prompt and blurb from config
        2. Auto-adds DelegationBehavior if agent can delegate
        3. Loads behaviors from config['behaviors']
        4. Merges parameters from global defaults
        5. Initializes and registers each behavior

        Args:
            config: Config dict loaded from YAML
                (e.g., config/agents/task_executor.yaml)
        """
        if not config:
            return

        # Load system prompt if present
        if "system_prompt" in config:
            self.agent.config_system_prompt = config["system_prompt"]
            print(
                f"[{self.agent.name}] Loaded system prompt from config "
                f"({len(self.agent.config_system_prompt)} chars)"
            )

            # Validate system prompt for common errors
            validation_errors = self._validate_system_prompt(
                self.agent.config_system_prompt
            )
            if validation_errors:
                print(
                    f"[{self.agent.name}] ⚠️  WARNING: "
                    "System prompt validation issues:"
                )
                for error in validation_errors:
                    print(f"[{self.agent.name}]    - {error}")

        # Load blurb if present
        if "blurb" in config:
            self.agent.config_blurb = config["blurb"]
            print(
                f"[{self.agent.name}] Loaded blurb from config "
                f"({len(self.agent.config_blurb)} chars)"
            )

        # Auto-add DelegationBehavior if this agent can delegate
        self._auto_add_delegation_behavior(config)

        # Load behaviors
        if "behaviors" not in config:
            print(f"[{self.agent.name}] No behaviors defined in config")
            return

        print(f"[{self.agent.name}] Loading behaviors from config")

        # Load global behavior defaults
        global_defaults = self._load_global_behavior_defaults()

        for behavior_spec in config.get("behaviors", []):
            behavior_type = behavior_spec["type"]

            # Skip behaviors in the exclude list
            if behavior_type in self.agent.exclude_behaviors:
                print(
                    f"[{self.agent.name}] Skipping excluded behavior: "
                    f"{behavior_type}"
                )
                continue

            # Merge parameters
            behavior_params = self._merge_behavior_params(
                behavior_type, behavior_spec.get("params"), global_defaults
            )

            # Create and register behavior
            try:
                behavior = self._create_behavior(behavior_type, behavior_params)
                self._register_behavior(
                    behavior, behavior_type, behavior_spec.get("params")
                )
            except Exception as e:
                print(
                    f"[{self.agent.name}] Failed to load behavior "
                    f"{behavior_type}: {e}"
                )

    def load_extra_behaviors(
        self, extra_behaviors: list[str] | None = None
    ) -> None:
        """
        Load additional behaviors from CLI flags or environment variable.

        Checks two sources (in priority order):
        1. extra_behaviors parameter (from direct instantiation)
        2. JETBOX_EXTRA_BEHAVIORS env var (for session-wide propagation)

        Behaviors loaded via this method use global defaults from
        config/behavior_defaults.yaml. If a behavior is already loaded
        via config or is in the exclude list, it will be skipped.

        Args:
            extra_behaviors: List of behavior class names to load
                (e.g., ["ContextInspectorBehavior"])
        """
        import os

        behaviors_to_load = []

        # From parameter (direct instantiation)
        if extra_behaviors:
            behaviors_to_load.extend(extra_behaviors)

        # From environment (for session-wide propagation to sub-agents)
        env_behaviors = os.environ.get("JETBOX_EXTRA_BEHAVIORS", "")
        if env_behaviors:
            behaviors_to_load.extend(
                [b.strip() for b in env_behaviors.split(",") if b.strip()]
            )

        if not behaviors_to_load:
            return

        print(
            f"[{self.agent.name}] Loading extra behaviors: "
            f"{behaviors_to_load}"
        )

        # Load global behavior defaults
        global_defaults = self._load_global_behavior_defaults()

        for behavior_type in behaviors_to_load:
            # Skip if already loaded or excluded
            if behavior_type in self.agent.exclude_behaviors:
                print(
                    f"[{self.agent.name}] Skipping excluded extra behavior: "
                    f"{behavior_type}"
                )
                continue

            # Check if already loaded (avoid duplicates)
            behavior_name = self._behavior_name_from_type(behavior_type)
            if any(b.get_name() == behavior_name for b in self.agent._behaviors):
                print(
                    f"[{self.agent.name}] Extra behavior {behavior_type} "
                    "already loaded, skipping"
                )
                continue

            # Get global defaults for this behavior
            default_params = global_defaults.get(behavior_type, {})
            if default_params is None:
                default_params = {}

            # Special handling for ChatbotBehavior: auto-detect tool_mode
            if behavior_type == "ChatbotBehavior":
                # Check if agent has tool-oriented behaviors (requires tool calls)
                # Map behavior class names to their get_name() values
                tool_oriented_behavior_names = [
                    'home_assistant',  # HomeAssistantBehavior
                    # Add other tool-required behavior names here as needed
                ]
                has_tool_behavior = any(
                    b.get_name() in tool_oriented_behavior_names
                    for b in self.agent._behaviors
                )
                if has_tool_behavior:
                    # Set tool_mode to 'required' for tool-oriented agents
                    default_params = default_params.copy()  # Don't modify global defaults
                    default_params['tool_mode'] = 'required'
                    print(
                        f"[{self.agent.name}] Auto-configured ChatbotBehavior: "
                        f"tool_mode='required' (tool-oriented agent detected)"
                    )

            # Dynamically import and instantiate
            try:
                behavior = self._create_behavior(behavior_type, default_params)
                self._register_behavior(behavior, behavior_type, None)
                print(
                    f"[{self.agent.name}] Loaded extra behavior: "
                    f"{behavior_type}"
                )
            except Exception as e:
                print(
                    f"[{self.agent.name}] Failed to load extra behavior "
                    f"{behavior_type}: {e}"
                )

    # ===========================
    # Private helper methods
    # ===========================

    def _create_behavior(
        self, behavior_type: str, params: dict[str, Any]
    ) -> Any:
        """
        Create and initialize a single behavior instance.

        Args:
            behavior_type: CamelCase behavior class name
                (e.g., "FileToolsBehavior")
            params: Merged parameters for behavior

        Returns:
            Initialized behavior instance

        Raises:
            ImportError: If behavior class not found
        """
        behavior_class = self._import_behavior_class(behavior_type)
        behavior = behavior_class(**params)
        behavior.agent = self.agent  # Give behavior reference to agent
        return behavior

    def _register_behavior(
        self,
        behavior: Any,
        behavior_type: str,
        agent_params: dict[str, Any] | None,
    ) -> None:
        """
        Register a behavior with the agent.

        This adds the behavior to the agent's behavior list and registers
        its tools with the tool dispatcher.

        Args:
            behavior: Behavior instance
            behavior_type: Behavior class name (for logging)
            agent_params: Agent-specific params (for logging),
                or None for extra behaviors
        """
        # Register tools with dispatcher
        for tool in behavior.get_tools():
            tool_name = tool["function"]["name"]
            self.agent.tool_dispatcher.register_tool(tool_name, behavior)

        # Add to behavior list
        self.agent._behaviors.append(behavior)

        # Log parameter source
        if agent_params:
            print(
                f"[{self.agent.name}] Loaded behavior: {behavior_type} "
                f"(agent-specific params: {agent_params})"
            )
        else:
            print(f"[{self.agent.name}] Loaded behavior: {behavior_type}")

    def _import_behavior_class(self, behavior_type: str):
        """
        Dynamically import behavior class by name.

        Args:
            behavior_type: CamelCase behavior class name
                (e.g., "FileToolsBehavior")

        Returns:
            Behavior class

        Raises:
            ImportError: If behavior module/class not found
                (with fuzzy match suggestions)
        """
        # Convert CamelCase to snake_case for module name
        module_name = self._to_snake_case(behavior_type)

        try:
            # Import from behaviors module
            module = importlib.import_module(f"behaviors.{module_name}")
            return getattr(module, behavior_type)
        except (ImportError, AttributeError) as e:
            # Provide helpful error message with fuzzy matching
            suggestions = self._get_similar_behaviors(behavior_type)

            error_msg = f"Behavior '{behavior_type}' not found"
            if suggestions:
                error_msg += f". Did you mean: {', '.join(suggestions[:3])}?"
            else:
                error_msg += ". No similar behaviors found."

            # Add hint about available behaviors
            available = self._get_available_behaviors()
            if available:
                error_msg += (
                    f"\n  Available behaviors: "
                    f"{', '.join(sorted(available[:10]))}"
                )
                if len(available) > 10:
                    error_msg += f"... ({len(available)} total)"

            raise ImportError(error_msg) from e

    def _merge_behavior_params(
        self,
        behavior_type: str,
        agent_params: dict[str, Any] | None,
        global_defaults: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Merge behavior parameters from global defaults and agent config.

        Agent-specific parameters override global defaults.

        Args:
            behavior_type: Behavior class name
            agent_params: Agent-specific params from config (may be None)
            global_defaults: Global defaults dict from behavior_defaults.yaml

        Returns:
            Merged parameter dict
        """
        # Get global defaults for this behavior type
        default_params = global_defaults.get(behavior_type, {})
        # If default_params is None (empty YAML entry), convert to empty dict
        if default_params is None:
            default_params = {}

        # Get agent-specific overrides
        if agent_params is None:
            agent_params = {}

        # Merge: agent params override global defaults
        return {**default_params, **agent_params}

    def _load_global_behavior_defaults(self) -> dict[str, dict[str, Any]]:
        """
        Load global behavior parameter defaults from
        config/behavior_defaults.yaml.

        Returns:
            Dict mapping behavior type name to parameter dict
            Example: {"LoopDetectionBehavior": {"max_repeats": 5}, ...}
        """
        from agent_config import load_behavior_defaults

        try:
            return load_behavior_defaults()
        except Exception as e:
            print(
                f"[{self.agent.name}] Warning: "
                f"Failed to load global behavior defaults: {e}"
            )
            return {}

    def _validate_system_prompt(self, prompt: str) -> list[str]:
        """
        Validate system prompt for common configuration errors.

        Args:
            prompt: System prompt string

        Returns:
            List of validation error messages (empty if no issues)
        """
        errors = []

        # Check for unresolved template placeholders
        if "{goal}" in prompt:
            errors.append(
                "System prompt contains '{goal}' placeholder but base_agent "
                "doesn't perform template substitution. "
                "Goal is automatically injected as a user message. "
                "Remove the placeholder."
            )

        # Check for other template-like syntax
        template_matches = re.findall(r"\{([^}]+)\}", prompt)
        if template_matches:
            # Filter out {goal} since we already reported it
            other_placeholders = [m for m in template_matches if m != "goal"]
            if other_placeholders:
                errors.append(
                    f"System prompt contains template-like syntax: "
                    f"{{{', '.join(other_placeholders)}}}. "
                    "If these are intentional placeholders, "
                    "ensure they're being replaced before use."
                )

        return errors

    def _auto_add_delegation_behavior(self, config: dict[str, Any]) -> None:
        """
        Auto-add DelegationBehavior if this agent can delegate to others.

        DelegationBehavior is delegator-only - provides tools for delegating
        TO other agents.
        For being delegated to, use core goal tracking (available in ALL
        agents).

        Reads team config to check if this agent has can_delegate_to list.
        Reads delegation configuration (workspace_strategy, etc.) from agent config.

        Args:
            config: Agent config dict containing delegation settings
        If yes, adds DelegationBehavior with delegation tools
        (consult_X, delegate_to_X).
        """
        from agent_config import load_team_config

        # Early return: DelegationBehavior already added
        if any(b.get_name() == "delegation" for b in self.agent._behaviors):
            return

        try:
            # Load team config (default team)
            team_config = load_team_config("default")

            # Early return: invalid config structure
            if not team_config or "agents" not in team_config:
                return

            agents = team_config["agents"]

            # Early return: agent not found in config
            agent_config = agents.get(self.agent.name)
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
                agent_info = self._load_target_agent_config(
                    target_agent, agents
                )
                agent_relationships[target_agent] = agent_info

            # Read delegation configuration (workspace strategy, etc.)
            delegation_config = config.get("delegation", {})
            workspace_strategy = delegation_config.get(
                "workspace_strategy", "enforce_inherit"
            )

            # Create and add DelegationBehavior
            from behaviors.delegation import DelegationBehavior

            delegation_behavior = DelegationBehavior(
                agent_relationships=agent_relationships,
                workspace_strategy=workspace_strategy
            )

            # Register with agent
            delegation_behavior.agent = self.agent
            for tool in delegation_behavior.get_tools():
                tool_name = tool["function"]["name"]
                self.agent.tool_dispatcher.register_tool(
                    tool_name, delegation_behavior
                )
            self.agent._behaviors.append(delegation_behavior)

            # Create human-readable strategy description
            strategy_descriptions = {
                "enforce_inherit": "sub-agents inherit workspace (prevents fragmentation)",
                "enforce_new": "sub-agents create isolated workspaces (testing isolation)",
                "llm_chooses": "LLM decides workspace mode (less predictable)"
            }
            strategy_desc = strategy_descriptions.get(
                workspace_strategy,
                workspace_strategy
            )

            print(
                f"[{self.agent.name}] Auto-added DelegationBehavior "
                f"(can delegate to: {', '.join(can_delegate_to)})\n"
                f"[{self.agent.name}]   Workspace strategy: {workspace_strategy} "
                f"({strategy_desc})"
            )

        except Exception as e:
            print(
                f"[{self.agent.name}] Failed to auto-add "
                f"DelegationBehavior: {e}"
            )

    def _load_target_agent_config(
        self, target_agent: str, agents: dict
    ) -> dict:
        """
        Load configuration for a target agent from its config file.

        Args:
            target_agent: Name of target agent
            agents: Agents dict from team config

        Returns:
            Agent info dict with delegation_tool, blurb, and behaviors (if available)
        """
        from agent_config import load_agent_config

        agent_info = agents.get(target_agent, {})

        try:
            target_config = load_agent_config(target_agent)

            # Add delegation_tool if present
            if target_config and "delegation_tool" in target_config:
                agent_info["delegation_tool"] = target_config[
                    "delegation_tool"
                ]

            # Add blurb if present
            if target_config and "blurb" in target_config:
                agent_info["blurb"] = target_config["blurb"]

            # Add behaviors list if present (for security property computation)
            if target_config and "behaviors" in target_config:
                agent_info["behaviors"] = target_config["behaviors"]

        except Exception as e:
            print(
                f"[{self.agent.name}] Warning: "
                f"Failed to load config for {target_agent}: {e}"
            )

        return agent_info

    def _behavior_name_from_type(self, behavior_type: str) -> str:
        """
        Get behavior instance name from class type.

        Example: LoopDetectionBehavior -> loop_detection

        Args:
            behavior_type: CamelCase behavior class name

        Returns:
            snake_case behavior name
        """
        return self._to_snake_case(behavior_type)

    def _to_snake_case(self, name: str) -> str:
        """
        Convert CamelCase to snake_case, removing "Behavior" suffix.

        Examples:
            FileToolsBehavior -> file_tools
            LoopDetectionBehavior -> loop_detection
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
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def _get_available_behaviors(self) -> list[str]:
        """
        Scan behaviors/ directory and return list of available
        behavior class names.

        Returns:
            List of behavior class names
            (e.g., ["FileToolsBehavior", "LoopDetectionBehavior"])
        """
        behaviors_dir = Path(__file__).parent.parent / "behaviors"
        if not behaviors_dir.exists():
            return []

        available = []
        for file_path in behaviors_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue  # Skip __init__.py, etc.

            # Read file and look for class definitions ending in "Behavior"
            try:
                content = file_path.read_text()
                # Match: class SomethingBehavior(...)
                matches = re.findall(r"class (\w+Behavior)\s*\(", content)
                available.extend(matches)
            except Exception:
                continue  # Skip files we can't read

        return available

    def _get_similar_behaviors(self, behavior_type: str) -> list[str]:
        """
        Find behavior names similar to the given name using fuzzy matching.

        Args:
            behavior_type: The behavior name to match against

        Returns:
            List of similar behavior names, sorted by similarity (best first)
        """
        import difflib

        available = self._get_available_behaviors()
        if not available:
            return []

        # Use difflib's SequenceMatcher for fuzzy matching
        # Get close matches with cutoff of 0.6 (60% similarity)
        matches = difflib.get_close_matches(
            behavior_type,
            available,
            n=5,  # Return up to 5 matches
            cutoff=0.6,  # 60% similarity threshold
        )

        return matches
