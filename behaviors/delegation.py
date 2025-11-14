"""
DelegationBehavior - Enables delegating work TO other agents.

This behavior is for agents that can delegate work to other agents (e.g., Orchestrator).
It provides delegation tools and injects agent descriptions into context.

Features:
- Auto-generates delegation tools based on can_delegate_to relationships
- Injects delegatable agent descriptions into context
- Handles delegation tool dispatch
- Tracks delegations for reporting
- No hardcoded agent relationships

Example:
    If agents.yaml defines:
        orchestrator:
            can_delegate_to: [architect, task_executor]

    Then DelegationBehavior will create:
        - consult_architect(project_description, requirements, constraints)
        - delegate_to_executor(task_description, workspace_mode, workspace_path)

This behavior does NOT handle being delegated to - that's core BaseAgent functionality.
"""

from typing import Any
from behaviors.base import AgentBehavior
from behaviors.rule_of_two_types import RuleOfTwoProperty


class DelegationBehavior(AgentBehavior):
    """
    Behavior that provides delegation tools for agents that delegate work to others.

    Automatically creates delegation tools for each agent in can_delegate_to list.
    This is a delegator-only behavior - it does NOT provide mark_complete/mark_failed.
    Being delegated to is core BaseAgent functionality (all agents support it).

    Security: DYNAMIC - inherits union of delegatable agents' properties
    - Computes properties from all agents in can_delegate_to list
    - Takes union of their aggregate security properties
    - Example: If delegates to [C] + [ABC] agents → inherits [ABC]

    This is capability propagation through delegation. The delegating agent
    effectively has all capabilities of its sub-agents.
    """

    # Rule of Two: Empty static fallback (dynamically computed at runtime)
    rule_of_two_properties = set()

    def __init__(
        self,
        agent_relationships: dict[str, Any],
        workspace_strategy: str = "enforce_inherit"
    ):
        """
        Initialize delegation behavior.

        Args:
            agent_relationships: Dict mapping agent name → {class, description, can_delegate_to}
            workspace_strategy: How to handle workspace delegation (default: "enforce_inherit")
                - "enforce_inherit": Always reuse calling agent's workspace (recommended, safe default)
                - "enforce_new": Always create new isolated workspaces (for testing isolation)
                - "llm_chooses": Let LLM decide via workspace_mode parameter (advanced, less predictable)
        """
        self.agent_relationships = agent_relationships
        self.workspace_strategy = workspace_strategy
        self.delegation_tools = []
        self.delegated_tasks: list[dict[str, Any]] = []  # Track delegated tasks
        self.tool_to_agent_map: dict[str, str] = {}  # Map tool names to agent names
        self._build_delegation_tools()

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "delegation"

    def get_rule_of_two_properties(self, agent, security_context):
        """
        Compute Rule of Two properties dynamically from delegatable agents.

        This method computes the union of security properties from all agents
        this behavior can delegate to. The delegating agent inherits all
        capabilities of its sub-agents (capability propagation).

        Args:
            agent: Agent instance (unused)
            security_context: SecurityContext to pass through to sub-behaviors

        Returns:
            Set of Rule of Two properties (union of all sub-agents)
        """
        aggregated_props = set()

        # For each agent we can delegate to
        for target_agent_name in self.agent_relationships.get("can_delegate_to", []):
            agent_info = self.agent_relationships.get(target_agent_name, {})
            behaviors_config = agent_info.get("behaviors", [])

            # Compute aggregate properties for this target agent
            target_props = self._compute_agent_aggregate_properties(
                target_agent_name, behaviors_config, security_context
            )
            aggregated_props.update(target_props)

        return aggregated_props

    def _compute_agent_aggregate_properties(
        self, agent_name: str, behaviors_config: list[dict], security_context
    ) -> set:
        """
        Compute aggregate security properties for a target agent.

        Creates behavior instances and calls their dynamic get_rule_of_two_properties()
        method to compute context-aware properties.

        Args:
            agent_name: Name of target agent
            behaviors_config: List of behavior specs from agent config
            security_context: SecurityContext to pass to sub-behaviors

        Returns:
            Set of aggregated Rule of Two properties
        """
        import importlib
        from behaviors.security_context import SecurityContext

        props = set()

        # Use provided context, or create default for Jetbox environment if None
        # (no untrusted files, no sensitive files, no network access)
        if security_context is None:
            security_context = SecurityContext(
                workspace_has_untrusted_files=False,
                workspace_has_sensitive_files=False,
                workspace_has_network_access=False
            )

        for behavior_spec in behaviors_config:
            behavior_type = behavior_spec.get("type")
            if not behavior_type:
                continue

            try:
                # Import behavior class
                module_name = self._behavior_type_to_module(behavior_type)
                module = importlib.import_module(module_name)
                behavior_class = getattr(module, behavior_type)

                # Instantiate behavior with params if provided
                params = behavior_spec.get("params", {})
                try:
                    behavior_instance = behavior_class(**params)
                except TypeError:
                    # Some behaviors don't accept params, try without
                    behavior_instance = behavior_class()

                # Call dynamic method to get properties
                if hasattr(behavior_instance, "get_rule_of_two_properties"):
                    behavior_props = behavior_instance.get_rule_of_two_properties(None, security_context)
                    if isinstance(behavior_props, set):
                        props.update(behavior_props)

            except (ImportError, AttributeError, TypeError) as e:
                # Behavior class not found or instantiation failed
                # Silently skip (this is expected for some behaviors)
                pass

        return props

    def _behavior_type_to_module(self, behavior_type: str) -> str:
        """
        Convert behavior type to module path.

        Examples:
            ReadFileToolsBehavior → behaviors.read_file_tools
            DelegationBehavior → behaviors.delegation
        """
        # Remove "Behavior" suffix
        name = behavior_type.replace("Behavior", "")

        # Convert CamelCase to snake_case
        import re
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

        return f"behaviors.{snake}"

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject tool documentation for delegation tools.

        Called once during agent initialization to document available tools.

        Args:
            agent: Agent instance
            context: Initial context (system prompt only)

        Returns:
            Context with tool documentation injected
        """
        tools = self.get_tools()
        if not tools:
            return context

        # Build tool documentation
        tool_docs = []
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
                is_required = param_name in required
                if is_required:
                    param_strs.append(f"{param_name}: {param_type}")
                else:
                    param_strs.append(f"{param_name}?: {param_type}")

            param_sig = ", ".join(param_strs) if param_strs else ""
            tool_docs.append(f"  - {name}({param_sig}): {desc}")

        if tool_docs:
            tool_message = f"\n{self.get_name()} tools:\n" + "\n".join(tool_docs)
            # Use default role="system" for framework tool documentation
            return self.inject_message_after_system(context, tool_message)

        return context

    def _build_delegation_tools(self) -> None:
        """
        Build delegation tools based on can_delegate_to relationships.

        For each delegatable agent, creates a tool definition from config.
        Also builds tool_to_agent_map for tool name → agent name resolution.
        """
        can_delegate_to = self.agent_relationships.get("can_delegate_to", [])

        for target_agent in can_delegate_to:
            # Get agent info from relationships
            agent_info = self.agent_relationships.get(target_agent, {})

            # Check if agent has delegation_tool defined in config
            if "delegation_tool" in agent_info:
                tool_config = agent_info["delegation_tool"]

                # Build tool parameters from config
                properties = {}
                required = []

                for param_name, param_config in tool_config.get("parameters", {}).items():
                    # Build property definition
                    prop = {
                        "type": param_config.get("type", "string"),
                        "description": param_config.get("description", "")
                    }

                    # Add enum if present
                    if "enum" in param_config:
                        prop["enum"] = param_config["enum"]

                    properties[param_name] = prop

                    # Add to required list if marked as required
                    if param_config.get("required", False):
                        required.append(param_name)

                # Build tool from config
                tool = {
                    "type": "function",
                    "function": {
                        "name": tool_config["name"],
                        "description": tool_config["description"],
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    }
                }

                # Map tool name to agent name
                tool_name = tool_config["name"]
                self.tool_to_agent_map[tool_name] = target_agent
            else:
                # Fallback: generic delegation tool for agents without delegation_tool config
                description = agent_info.get("description", f"Delegate to {target_agent}")
                tool_name = f"delegate_to_{target_agent}"
                tool = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Delegate work to {target_agent}: {description}",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_description": {
                                    "type": "string",
                                    "description": "Description of the task to delegate"
                                }
                            },
                            "required": ["task_description"]
                        }
                    }
                }

                # Map tool name to agent name
                self.tool_to_agent_map[tool_name] = target_agent

            self.delegation_tools.append(tool)

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return delegation tool definitions.

        Returns:
            List of dynamically generated delegation tools
        """
        return self.delegation_tools

    def dispatch_tool(
        self,
        agent: Any,
        tool_name: str,
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Dispatch delegation tool calls.

        Args:
            agent: Agent instance (calling agent)
            tool_name: Tool name (consult_architect, delegate_to_executor, etc.)
            args: Tool arguments

        Returns:
            Delegation result
        """
        # Get registry and server_manager from agent if available (orchestrator has these)
        # Initialize them lazily if agent supports it but hasn't initialized yet
        registry = getattr(agent, 'registry', None)
        if not registry and hasattr(agent, 'init_registry'):
            agent.init_registry()
            registry = agent.registry

        server_manager = getattr(agent, 'server_manager', None)
        if not server_manager and hasattr(agent, 'init_server_manager'):
            agent.init_server_manager()
            server_manager = agent.server_manager

        # Parse tool name to get target agent name
        target_agent_name = self._get_target_agent_for_tool(tool_name)

        if target_agent_name:
            # Use in-process delegation (instantiate BaseAgent with config)
            # This eliminates dependency on wrapper files and subprocess overhead
            return self._delegate_to_agent(target_agent_name, args, agent)
        else:
            # Unknown delegation tool
            return {"error": f"Unknown delegation tool: {tool_name}"}

    def _get_target_agent_for_tool(self, tool_name: str) -> str | None:
        """
        Parse tool name to determine target agent.

        Uses tool_to_agent_map built during initialization from agent configs.
        This ensures tool names defined in config files are properly mapped.

        Examples:
            consult_architect → architect
            delegate_to_executor → task_executor (NOT executor!)
            delegate_to_orchestrator → orchestrator

        Args:
            tool_name: Tool name from delegation tools

        Returns:
            Target agent name, or None if not a delegation tool
        """
        # Use explicit mapping built from config
        # This handles cases where tool name != agent name
        # e.g., delegate_to_executor → task_executor
        return self.tool_to_agent_map.get(tool_name)

    def _class_name_to_file(self, class_name: str) -> str:
        """
        Convert class name to Python file name.

        Uses same logic as AgentRegistry._class_to_module().

        Examples:
            ArchitectAgent → architect_agent.py
            TaskExecutorAgent → task_executor_agent.py
            OrchestratorAgent → orchestrator_agent.py

        Args:
            class_name: CamelCase class name

        Returns:
            snake_case file name with .py extension
        """
        import re

        # Insert underscores before capitals (except first char)
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)

        # Convert to lowercase and add .py extension
        return s2.lower() + ".py"

    def track_delegation(
        self,
        target_agent: str,
        task_description: str,
        result: dict[str, Any]
    ) -> None:
        """
        Track a delegation for reporting.

        Args:
            target_agent: Name of agent task was delegated to
            task_description: Description of delegated task
            result: Delegation result
        """
        self.delegated_tasks.append({
            "agent": target_agent,
            "task": task_description,
            "result": result,
        })

    def _delegate_via_subprocess(
        self,
        target_agent_name: str,
        tool_name: str,
        args: dict[str, Any],
        calling_agent: Any,
        registry: Any,
        server_manager: Any
    ) -> dict[str, Any]:
        """
        Delegate to agent via subprocess (orchestrator mode).

        This method runs agents as separate processes for:
        - Process isolation (crash resilience)
        - Independent execution context
        - Clean separation of concerns

        Fully generic - works with any agent type defined in agents.yaml.

        Args:
            target_agent_name: Name of target agent (e.g., "architect", "task_executor")
            tool_name: Name of delegation tool (for specialized handling if needed)
            args: Tool arguments
            calling_agent: The agent initiating delegation
            registry: AgentRegistry instance
            server_manager: ServerManager instance

        Returns:
            Delegation result dict
        """
        from pathlib import Path
        from agent_config import PROJECT_ROOT

        # Get agent config from registry
        if not registry or target_agent_name not in registry.config.get("agents", {}):
            return {
                "success": False,
                "error": f"Target agent '{target_agent_name}' not found in registry"
            }

        agent_config = registry.config["agents"][target_agent_name]
        agent_class_name = agent_config.get("class")

        if not agent_class_name:
            return {
                "success": False,
                "error": f"No class defined for agent '{target_agent_name}' in agents.yaml"
            }

        # Convert class name to Python file name
        # ArchitectAgent → architect_agent.py
        # TaskExecutorAgent → task_executor_agent.py
        agent_file = self._class_name_to_file(agent_class_name)

        # Verify file exists using absolute path (works from any CWD)
        agent_file_path = PROJECT_ROOT / "agents" / agent_file
        if not agent_file_path.exists():
            return {
                "success": False,
                "error": f"Agent file not found: {agent_file_path} (derived from class {agent_class_name})"
            }

        # Use generic subprocess delegation for ALL agents
        return self._generic_subprocess_delegation(
            target_agent_name,
            str(agent_file_path),  # Use absolute path so it works from any CWD
            args,
            calling_agent,
            registry
        )

    def _enforce_workspace_strategy(
        self,
        workspace_mode: str | None,
        workspace_path: str | None
    ) -> tuple[str | None, str | None]:
        """
        Enforce configured workspace strategy, overriding LLM choices if needed.

        This enforcement ensures predictable workspace behavior regardless of
        what the LLM decides. Critical for testing and production consistency.

        Args:
            workspace_mode: LLM-provided workspace_mode ('new', 'existing', or None)
            workspace_path: LLM-provided workspace_path

        Returns:
            Tuple of (enforced_workspace_mode, enforced_workspace_path)
        """
        if self.workspace_strategy == "enforce_inherit":
            # Force inheritance by clearing workspace params
            # This ensures sub-agents always continue work in same workspace
            return None, None

        elif self.workspace_strategy == "enforce_new":
            # Force new workspace for complete isolation
            # Useful for testing or when delegations must not interfere
            return "new", None

        elif self.workspace_strategy == "llm_chooses":
            # Respect LLM's decision (less predictable but allows flexibility)
            return workspace_mode, workspace_path

        else:
            raise ValueError(
                f"Unknown workspace_strategy: {self.workspace_strategy}. "
                f"Must be 'enforce_inherit', 'enforce_new', or 'llm_chooses'"
            )

    def _resolve_workspace_for_delegation(
        self,
        args: dict[str, Any],
        calling_agent: Any
    ) -> tuple[str | None, str]:
        """
        Resolve workspace path from delegation parameters.

        This method implements the workspace resolution priority order:
        1. workspace_mode="new" → None (create isolated workspace)
        2. workspace_mode="existing" + workspace_path → explicit path
        3. workspace_mode="existing" without path → calling agent's workspace
        4. workspace_path only (no mode) → explicit path
        5. No params but calling agent has workspace → calling agent's workspace
        6. No context → None (create isolated workspace)

        Args:
            args: Tool arguments containing workspace_mode and workspace_path
            calling_agent: The agent initiating delegation

        Returns:
            Tuple of (workspace_path, log_message)
            workspace_path is None for new workspaces, str/Path for existing

        Raises:
            ValueError: If workspace_mode="existing" but no workspace available
        """
        from pathlib import Path

        workspace_mode = args.get("workspace_mode", "")
        workspace_path = args.get("workspace_path", "")

        # Enforce workspace strategy (overrides LLM choices if configured)
        workspace_mode, workspace_path = self._enforce_workspace_strategy(
            workspace_mode, workspace_path
        )

        if workspace_mode == "new":
            return None, "[delegation] workspace_mode='new': subagent will create isolated workspace"

        elif workspace_mode == "existing":
            # Use explicit path if provided, otherwise use calling agent's workspace
            if workspace_path:
                return str(workspace_path), f"[delegation] workspace_mode='existing': using workspace_path={workspace_path}"
            elif hasattr(calling_agent, 'workspace') and calling_agent.workspace:
                return str(calling_agent.workspace), f"[delegation] workspace_mode='existing' without path: using calling agent's workspace: {calling_agent.workspace}"
            else:
                raise ValueError("workspace_mode='existing' requires either workspace_path parameter or calling agent must have workspace")

        elif workspace_path:
            # Explicit workspace path without mode (backward compat)
            return str(workspace_path), f"[delegation] Using explicit workspace_path: {workspace_path}"

        elif hasattr(calling_agent, 'workspace') and calling_agent.workspace:
            # No workspace_mode specified - reuse calling agent's workspace (backward compat)
            return str(calling_agent.workspace), f"[delegation] No workspace_mode: reusing calling agent's workspace: {calling_agent.workspace}"

        else:
            # No workspace context - let subagent create its own
            return None, "[delegation] No workspace context: subagent will create isolated workspace"

    def _extract_goal_description(self, args: dict[str, Any]) -> str | None:
        """
        Extract goal/task description from args using multiple common parameter names.

        Args:
            args: Tool arguments

        Returns:
            Goal description string, or None if not found
        """
        for key in ["task_description", "project_description", "goal", "description", "query"]:
            if key in args:
                return args[key]
        return None

    def _build_subprocess_command(
        self,
        agent_file: str,
        workspace_path: str | None,
        context: str,
        goal_description: str
    ) -> list[str]:
        """
        Build subprocess command with workspace, context, and goal parameters.

        Args:
            agent_file: Python file for target agent
            workspace_path: Resolved workspace path (None for new workspace)
            context: Optional context string
            goal_description: Task/goal description

        Returns:
            Command list for subprocess.run()
        """
        import sys

        cmd = [sys.executable, agent_file]

        # Add workspace parameter if resolved (None = new workspace, no --workspace flag)
        if workspace_path is not None:
            cmd.extend(["--workspace", workspace_path])

        # Add context parameter if provided
        if context:
            cmd.extend(["--context", context])

        # Add goal/task description as positional argument
        cmd.append(goal_description)

        return cmd

    def _print_delegation_header(
        self,
        target_agent_name: str,
        goal_description: str,
        workspace_path: str | None
    ) -> None:
        """
        Print delegation start header with task info.

        Args:
            target_agent_name: Name of target agent
            goal_description: Task description
            workspace_path: Workspace path (if any)
        """
        print("\n" + "=" * 60)
        print(f"DELEGATING TO {target_agent_name.upper()}")
        print("=" * 60)
        print(f"Task: {goal_description[:100]}...")
        if workspace_path:
            print(f"Workspace: {workspace_path}")
        print("=" * 60 + "\n")

    def _read_agent_messages(self, parent_name: str) -> list[dict[str, Any]]:
        """
        Read messages from subagent's messages_to_{parent}.jsonl file.

        Args:
            parent_name: Name of calling/parent agent

        Returns:
            List of message dicts from subagent
        """
        import json
        from pathlib import Path

        messages = []
        msg_file = Path(f".agent_context/messages_to_{parent_name}.jsonl")

        if msg_file.exists():
            try:
                with open(msg_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            messages.append(json.loads(line))
                # Clear the file after reading
                msg_file.unlink()
            except Exception as e:
                print(f"[delegation] Warning: Failed to read agent messages: {e}")

        return messages

    def _display_agent_messages(
        self,
        target_agent_name: str,
        messages: list[dict[str, Any]]
    ) -> None:
        """
        Display messages from subagent.

        Args:
            target_agent_name: Name of target agent
            messages: List of message dicts
        """
        if messages:
            print(f"Messages from {target_agent_name}:")
            for msg in messages:
                severity = msg.get("severity", "info").upper()
                content = msg.get("message", "")
                print(f"  [{severity}] {content}")
            print()

    def _verify_subprocess_completion(self, returncode: int) -> bool:
        """
        Verify subprocess completion by checking exit code and state.json.

        Args:
            returncode: Process exit code

        Returns:
            True if task completed successfully, False otherwise
        """
        import json
        from pathlib import Path

        if returncode != 0:
            return False

        # Double-check with state.json
        state_file = Path(".agent_context/state.json")
        if state_file.exists():
            try:
                with open(state_file, encoding="utf-8") as f:
                    state = json.load(f)
                    # Verify all tasks completed
                    if state.get("goal", {}).get("tasks"):
                        all_completed = all(
                            t.get("status") == "completed"
                            for t in state["goal"]["tasks"]
                        )
                        if not all_completed:
                            print("[delegation] Warning: Exit code 0 but not all tasks completed")
                            return False
            except Exception as e:
                print(f"[delegation] Warning: Could not verify state.json: {e}")

        return True

    def _build_subprocess_result(
        self,
        target_agent_name: str,
        task_completed: bool,
        returncode: int,
        messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Build delegation result dict with success/failure message.

        Args:
            target_agent_name: Name of target agent
            task_completed: Whether task completed successfully
            returncode: Process exit code
            messages: Messages from subagent

        Returns:
            Result dict with success, message, agent, and messages
        """
        if task_completed:
            result_msg = f"{target_agent_name} completed task successfully"

            # Include agent messages
            if messages:
                result_msg += f"\n\nMessages from {target_agent_name}:"
                for msg in messages:
                    result_msg += f"\n  [{msg.get('severity', 'info')}] {msg.get('message', '')}"

            return {
                "success": True,
                "message": result_msg,
                "agent": target_agent_name,
                "messages": messages,
            }
        else:
            error_msg = f"{target_agent_name} failed (exit code {returncode})"

            if messages:
                error_msg += f"\n\nMessages from {target_agent_name}:"
                for msg in messages:
                    error_msg += f"\n  [{msg.get('severity', 'info')}] {msg.get('message', '')}"

            return {
                "success": False,
                "message": error_msg,
                "agent": target_agent_name,
                "messages": messages,
            }

    def _generic_subprocess_delegation(
        self,
        target_agent_name: str,
        agent_file: str,
        args: dict[str, Any],
        calling_agent: Any,
        registry: Any
    ) -> dict[str, Any]:
        """
        Generic subprocess delegation for any agent type.

        This method provides a baseline subprocess delegation strategy that:
        1. Builds subprocess command from agent file and args
        2. Runs agent as isolated subprocess
        3. Checks exit code for success/failure
        4. Reads state.json for completion verification
        5. Reads messages from messages_to_{parent_name}.jsonl (auto-detected from calling agent)
        6. Returns structured result

        Args:
            target_agent_name: Name of target agent
            agent_file: Python file for target agent (e.g., "custom_agent.py")
            args: Tool arguments
            calling_agent: The agent initiating delegation
            registry: AgentRegistry instance

        Returns:
            Delegation result dict with success, message, and metadata
        """
        import subprocess

        # Extract goal description
        goal_description = self._extract_goal_description(args)
        if not goal_description:
            return {
                "success": False,
                "error": f"No task description found in args. Tried: task_description, project_description, goal, description, query. Got: {list(args.keys())}"
            }

        # Resolve workspace
        workspace_path, log_msg = self._resolve_workspace_for_delegation(args, calling_agent)
        print(log_msg)

        # Build subprocess command
        context = args.get("context", "")
        cmd = self._build_subprocess_command(agent_file, workspace_path, context, goal_description)

        # Print delegation header
        self._print_delegation_header(target_agent_name, goal_description, workspace_path)

        try:
            # Get timeout from agent (default 600 if not set)
            timeout = getattr(calling_agent, 'timeout_seconds', 600)

            # Run subprocess
            proc = subprocess.run(
                cmd,
                capture_output=False,  # Show output in real-time
                text=True,
                timeout=timeout,
            )

            # Print completion header
            print("\n" + "=" * 60)
            print(f"{target_agent_name.upper()} COMPLETED")
            print("=" * 60 + "\n")

            # Read and display messages from agent
            parent_name = calling_agent.name if calling_agent and hasattr(calling_agent, 'name') else "orchestrator"
            messages_from_agent = self._read_agent_messages(parent_name)
            self._display_agent_messages(target_agent_name, messages_from_agent)

            # Verify completion
            task_completed = self._verify_subprocess_completion(proc.returncode)

            # Build and return result
            return self._build_subprocess_result(
                target_agent_name,
                task_completed,
                proc.returncode,
                messages_from_agent
            )

        except subprocess.TimeoutExpired:
            timeout_minutes = timeout / 60
            return {
                "success": False,
                "message": f"{target_agent_name} execution timed out after {timeout_minutes:.1f} minutes (timeout: {timeout}s)",
                "agent": target_agent_name,
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"{target_agent_name} execution failed: {e}",
                "agent": target_agent_name,
            }

    def _get_workspace_info_from_task(self, task_description: str) -> dict | None:
        """
        Determine workspace location and files created from task description.

        Args:
            task_description: The task that was executed

        Returns:
            Dict with 'workspace' and 'files' keys, or None if not found
        """
        from pathlib import Path
        import re

        # Create workspace slug from task description (matches workspace_manager.py logic)
        slug = re.sub(r'[^a-z0-9]+', '-', task_description.lower())
        slug = slug.strip('-')[:60]

        workspace_path = Path.cwd() / ".agent_workspaces" / slug

        if not workspace_path.exists():
            return None

        # List all files in workspace (excluding directories and hidden files)
        try:
            files = []
            for item in workspace_path.iterdir():
                if item.is_file() and not item.name.startswith('.'):
                    files.append(item.name)

            return {
                "workspace": str(workspace_path),
                "files": sorted(files),
            }
        except Exception:
            return None

    def _import_agent_class(self, target_agent_name: str) -> tuple[Any, dict[str, Any] | None]:
        """
        Import agent class dynamically from agent relationships.

        Args:
            target_agent_name: Name of target agent

        Returns:
            Tuple of (agent_class, error_dict)
            If successful, returns (class, None)
            If failed, returns (None, error_dict)
        """
        # Get agent info from relationships
        agent_info = self.agent_relationships.get(target_agent_name, {})
        if not agent_info:
            return None, {
                "success": False,
                "error": f"Unknown target agent: {target_agent_name}"
            }

        agent_class_name = agent_info.get("class")
        if not agent_class_name:
            return None, {
                "success": False,
                "error": f"No class defined for agent: {target_agent_name}"
            }

        # Import agent class dynamically
        try:
            # Derive module name from class name (no hardcoded mapping)
            # Uses same logic as AgentRegistry._class_to_module()
            module_name = self._class_name_to_file(agent_class_name).replace(".py", "")

            # Import the class
            import importlib
            module = importlib.import_module(module_name)
            agent_class = getattr(module, agent_class_name)

            return agent_class, None

        except Exception as e:
            return None, {
                "success": False,
                "error": f"Failed to import {agent_class_name} from {module_name}: {e}"
            }

    def _build_direct_delegation_result(
        self,
        target_agent_name: str,
        status: str,
        success: bool,
        summary: str,
        workspace,
        files_created: list[str],
        goal_description: str
    ) -> dict[str, Any]:
        """
        Build result dict for direct (in-process) delegation.

        Args:
            target_agent_name: Name of target agent
            status: Execution status
            success: Whether task succeeded
            summary: Summary from subagent
            workspace: Workspace path
            files_created: List of created files
            goal_description: Original goal/task

        Returns:
            Result dict with success, status, message, and metadata
        """
        if success:
            message = f"""Task delegated to {target_agent_name} completed successfully.

SUMMARY:
{summary}

WORKSPACE: {workspace}
FILES CREATED: {', '.join(files_created) if files_created else 'none'}

This delegation phase is complete. Review the summary and determine what to do next based on your overall goal."""
        else:
            message = f"""Task delegated to {target_agent_name} did not complete successfully (status: {status}).

SUMMARY:
{summary}

WORKSPACE: {workspace}
FILES CREATED: {', '.join(files_created) if files_created else 'none'}

The delegated task did not complete. Consider:
- Breaking down into simpler subtasks
- Providing more specific requirements
- Delegating again with adjusted parameters"""

        return {
            "success": success,
            "status": status,
            "message": message,
            "target_agent": target_agent_name,
            "goal": goal_description,
            "workspace": str(workspace),
            "files_created": files_created,
            "summary": summary,
        }

    def _delegate_to_agent(
        self,
        target_agent_name: str,
        args: dict[str, Any],
        calling_agent: Any
    ) -> dict[str, Any]:
        """
        Generic delegation to any agent using BaseAgent + config.

        This method handles delegation to ANY agent type by:
        1. Loading agent config from agent_config.py
        2. Instantiating BaseAgent with config
        3. Setting goal and running agent
        4. Collecting results

        Args:
            target_agent_name: Name of target agent (e.g., "task_executor", "architect", "orchestrator")
            args: Tool arguments (varies by agent, but usually includes task/goal description)
            calling_agent: The agent initiating the delegation

        Returns:
            Delegation result dict
        """
        from pathlib import Path
        import os
        from agent_config import PROJECT_ROOT
        from base_agent import BaseAgent

        # Load agent config
        config_file = PROJECT_ROOT / f"config/agents/{target_agent_name}.yaml"
        if not config_file.exists():
            return {
                "success": False,
                "error": f"Agent config not found: {config_file}"
            }

        # Extract goal description
        goal_description = self._extract_goal_description(args)
        if not goal_description:
            return {
                "success": False,
                "error": f"No goal/task description found in args: {list(args.keys())}"
            }

        # Resolve workspace
        workspace_path, log_msg = self._resolve_workspace_for_delegation(args, calling_agent)
        print(log_msg)

        # Convert to Path object if not None (for agent instantiation)
        workspace = Path(workspace_path) if workspace_path is not None else None

        # Validate workspace path if provided
        if workspace is not None:
            # Ensure it's absolute (delegation should always use absolute paths)
            if not workspace.is_absolute():
                workspace = workspace.resolve()
                print(f"[delegation] Resolved relative workspace to absolute: {workspace}")

            # Check if parent directory exists (workspace itself will be created by BaseAgent)
            if not workspace.parent.exists():
                return {
                    "success": False,
                    "error": (
                        f"Cannot create workspace: parent directory does not exist.\n"
                        f"Workspace: {workspace}\n"
                        f"Parent: {workspace.parent}\n"
                        f"Ensure the parent directory exists before delegation."
                    )
                }

        # Instantiate and execute target agent
        try:
            print(f"\n[delegation] Delegating to {target_agent_name}: {goal_description[:60]}...")

            # Temporarily clear OLLAMA_MODEL env var so delegated agent uses its own config
            saved_model_override = os.environ.get("OLLAMA_MODEL")
            if saved_model_override:
                del os.environ["OLLAMA_MODEL"]
                print("[delegation] Cleared OLLAMA_MODEL override for delegated agent")

            try:
                # Instantiate BaseAgent with config
                # Note: BaseAgent requires name parameter
                target_agent = BaseAgent(
                    name=target_agent_name,
                    workspace=workspace,
                    config_file=str(config_file)
                )
                # Set goal after instantiation
                target_agent.set_goal(goal_description)
            finally:
                # Restore OLLAMA_MODEL for calling agent
                if saved_model_override:
                    os.environ["OLLAMA_MODEL"] = saved_model_override

            # Execute the agent synchronously
            print(f"[delegation] Executing {target_agent_name} with max_rounds=50...")
            execution_result = target_agent.run(max_rounds=50)

            # Extract execution status and summary
            status = execution_result.get('status', 'unknown')
            success = (status == 'success')
            summary = execution_result.get('summary') or execution_result.get('reason')

            if not summary:
                summary = f"Task execution {status}. No summary provided by subagent."

            # Get workspace and list files (inline simple listing)
            subagent_workspace = target_agent.workspace if hasattr(target_agent, 'workspace') else None
            files_created = []
            if subagent_workspace and subagent_workspace.exists():
                try:
                    files_created = [
                        item.name for item in subagent_workspace.iterdir()
                        if item.is_file() and not item.name.startswith('.')
                    ]
                except Exception:
                    pass

            # Build result
            result = self._build_direct_delegation_result(
                target_agent_name,
                status,
                success,
                summary,
                subagent_workspace,
                files_created,
                goal_description
            )

            # Track delegation
            self.track_delegation(target_agent_name, goal_description, result)

            print(f"[delegation] {target_agent_name} completed with status: {status}")
            print(f"[delegation] Files created: {len(files_created)}")

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Failed to delegate to {target_agent_name}: {e}"
            }

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject delegation information into initial context.

        Adds descriptions of delegatable agents after system prompt (ONCE at start).

        Args:
            agent: Agent instance
            context: Initial context (system prompt only)

        Returns:
            Modified context with delegation info
        """
        can_delegate_to = self.agent_relationships.get("can_delegate_to", [])
        if not can_delegate_to or len(context) == 0:
            return context

        # Build delegation info
        delegation_info = ["## Available Agents for Delegation\n"]
        for target_agent in can_delegate_to:
            agent_info = self.agent_relationships.get(target_agent, {})
            # Use blurb if available, fallback to description
            blurb = agent_info.get("blurb", agent_info.get("description", f"Agent: {target_agent}"))
            delegation_info.append(f"- **{target_agent}**: {blurb}")

        # Insert after system prompt (index 1)
        # Use role="user" since delegation options are presented to the delegating agent
            context = self.inject_message_after_system(context, "\n".join(delegation_info), role="user")

        return context

    def get_instructions(self) -> str:
        """
        Return delegation workflow instructions.

        Returns:
            Instructions for using delegation tools (config-driven from agent blurbs)
        """
        can_delegate_to = self.agent_relationships.get("can_delegate_to", [])
        if not can_delegate_to:
            return ""

        # Build guidelines from agent blurbs
        guidelines = []
        for target_agent in can_delegate_to:
            agent_info = self.agent_relationships.get(target_agent, {})
            blurb = agent_info.get("blurb", agent_info.get("description", ""))
            if blurb:
                # Extract key guidance from blurb (usually starts with "Best for...")
                # Take first sentence or find "Best for" clause
                blurb_lines = blurb.strip().split(". ")
                guidance = None
                for line in blurb_lines:
                    if "Best for" in line or "best for" in line:
                        guidance = line.strip()
                        break
                if guidance:
                    guidelines.append(f"- Use {target_agent} for: {guidance}")
                else:
                    # Fall back to description
                    guidelines.append(f"- Use {target_agent}: {agent_info.get('description', '')}")

        guidelines_text = "\n".join(guidelines) if guidelines else "- Assess task complexity and choose appropriate agent"

        return f"""
DELEGATION WORKFLOW:
You can delegate work to the following agents: {', '.join(can_delegate_to)}

Guidelines:
{guidelines_text}
- Always report delegation results back to user
"""
