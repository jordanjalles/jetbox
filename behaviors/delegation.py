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

This behavior does NOT handle being delegated to - use SubAgentModeBehavior for that.
"""

from typing import Any
from behaviors.base import AgentBehavior


class DelegationBehavior(AgentBehavior):
    """
    Behavior that provides delegation tools for agents that delegate work to others.

    Automatically creates delegation tools for each agent in can_delegate_to list.
    This is a delegator-only behavior - it does NOT provide mark_complete/mark_failed.
    Use SubAgentModeBehavior for agents that can BE delegated to.
    """

    def __init__(self, agent_relationships: dict[str, Any]):
        """
        Initialize delegation behavior.

        Args:
            agent_relationships: Dict mapping agent name → {class, description, can_delegate_to}
        """
        self.agent_relationships = agent_relationships
        self.delegation_tools = []
        self.delegated_tasks: list[dict[str, Any]] = []  # Track delegated tasks
        self.tool_to_agent_map: dict[str, str] = {}  # Map tool names to agent names
        self._build_delegation_tools()

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "delegation"

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
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Dispatch delegation tool calls.

        Args:
            tool_name: Tool name (consult_architect, delegate_to_executor, etc.)
            args: Tool arguments
            **kwargs: Additional context (agent, workspace, registry, server_manager, etc.)

        Returns:
            Delegation result
        """
        agent = kwargs.get("agent")
        registry = kwargs.get("registry")  # AgentRegistry (for subprocess delegation)
        server_manager = kwargs.get("server_manager")  # ServerManager (for subprocess delegation)

        # Parse tool name to get target agent name
        target_agent_name = self._get_target_agent_for_tool(tool_name)

        if target_agent_name:
            # Choose delegation strategy based on available context
            # If registry/server_manager provided, use subprocess delegation (orchestrator mode)
            # Otherwise, use direct instantiation (simpler mode)
            if registry:
                return self._delegate_via_subprocess(target_agent_name, tool_name, args, agent, registry, server_manager)
            else:
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
        import subprocess
        import sys
        import json

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

        # Verify file exists
        if not Path(agent_file).exists():
            return {
                "success": False,
                "error": f"Agent file not found: {agent_file} (derived from class {agent_class_name})"
            }

        # Use generic subprocess delegation for ALL agents
        return self._generic_subprocess_delegation(
            target_agent_name,
            agent_file,
            args,
            calling_agent,
            registry
        )

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
        5. Reads messages from messages_to_orchestrator.jsonl
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
        import sys
        import json
        from pathlib import Path

        # Extract task/goal description from args
        # Try multiple common parameter names
        goal_description = None
        for key in ["task_description", "project_description", "goal", "description", "query"]:
            if key in args:
                goal_description = args[key]
                break

        if not goal_description:
            return {
                "success": False,
                "error": f"No task description found in args. Tried: task_description, project_description, goal, description, query. Got: {list(args.keys())}"
            }

        # Handle workspace parameters
        workspace_mode = args.get("workspace_mode", "")
        workspace_path = args.get("workspace_path", "")

        # Build subprocess command
        cmd = [sys.executable, agent_file]

        # Add workspace parameter if provided
        if workspace_mode == "existing" and workspace_path:
            cmd.extend(["--workspace", workspace_path])
        elif workspace_mode == "new":
            # Don't pass workspace - let agent create its own
            pass

        # Add context parameter if provided
        context = args.get("context", "")
        if context:
            cmd.extend(["--context", context])

        # Add goal/task description as positional argument
        cmd.append(goal_description)

        # Print delegation header
        print("\n" + "=" * 60)
        print(f"DELEGATING TO {target_agent_name.upper()}")
        print("=" * 60)
        print(f"Task: {goal_description[:100]}...")
        if workspace_path:
            print(f"Workspace: {workspace_path}")
        print("=" * 60 + "\n")

        try:
            # Run subprocess
            proc = subprocess.run(
                cmd,
                capture_output=False,  # Show output in real-time
                text=True,
                timeout=600,  # 10 minute timeout
            )

            print("\n" + "=" * 60)
            print(f"{target_agent_name.upper()} COMPLETED")
            print("=" * 60 + "\n")

            # Read messages from agent if any
            messages_from_agent = []
            msg_file = Path(".agent_context/messages_to_orchestrator.jsonl")
            if msg_file.exists():
                try:
                    with open(msg_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                messages_from_agent.append(json.loads(line))
                    # Clear the file after reading
                    msg_file.unlink()
                except Exception as e:
                    print(f"[delegation] Warning: Failed to read agent messages: {e}")

            # Display messages
            if messages_from_agent:
                print(f"Messages from {target_agent_name}:")
                for msg in messages_from_agent:
                    severity = msg.get("severity", "info").upper()
                    content = msg.get("message", "")
                    print(f"  [{severity}] {content}")
                print()

            # Verify completion by checking exit code and state.json
            task_completed = proc.returncode == 0

            # Double-check with state.json
            if proc.returncode == 0:
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
                                    task_completed = False
                                    print(f"[delegation] Warning: Exit code 0 but not all tasks completed")
                    except Exception as e:
                        print(f"[delegation] Warning: Could not verify state.json: {e}")

            if task_completed:
                result_msg = f"{target_agent_name} completed task successfully"

                # Include agent messages
                if messages_from_agent:
                    result_msg += f"\n\nMessages from {target_agent_name}:"
                    for msg in messages_from_agent:
                        result_msg += f"\n  [{msg.get('severity', 'info')}] {msg.get('message', '')}"

                return {
                    "success": True,
                    "message": result_msg,
                    "agent": target_agent_name,
                    "messages": messages_from_agent,
                }
            else:
                error_msg = f"{target_agent_name} failed (exit code {proc.returncode})"

                if messages_from_agent:
                    error_msg += f"\n\nMessages from {target_agent_name}:"
                    for msg in messages_from_agent:
                        error_msg += f"\n  [{msg.get('severity', 'info')}] {msg.get('message', '')}"

                return {
                    "success": False,
                    "message": error_msg,
                    "agent": target_agent_name,
                    "messages": messages_from_agent,
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "message": f"{target_agent_name} execution timed out",
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

    def _delegate_to_agent(
        self,
        target_agent_name: str,
        args: dict[str, Any],
        calling_agent: Any
    ) -> dict[str, Any]:
        """
        Generic delegation to any agent.

        This method handles delegation to ANY agent type by:
        1. Looking up agent class from relationships
        2. Instantiating target agent
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

        # Get agent info from relationships
        agent_info = self.agent_relationships.get(target_agent_name, {})
        if not agent_info:
            return {
                "success": False,
                "error": f"Unknown target agent: {target_agent_name}"
            }

        agent_class_name = agent_info.get("class")
        if not agent_class_name:
            return {
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

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to import {agent_class_name} from {module_name}: {e}"
            }

        # Extract goal/task description from args
        # Different tools use different parameter names
        goal_description = None
        for key in ["task_description", "project_description", "goal", "query"]:
            if key in args:
                goal_description = args[key]
                break

        if not goal_description:
            return {
                "success": False,
                "error": f"No goal/task description found in args: {list(args.keys())}"
            }

        # WORKSPACE COORDINATION
        # Respect workspace_mode parameter if provided
        workspace = None
        workspace_mode = args.get("workspace_mode", None)

        # Priority order:
        # 1. workspace_mode="new" → create isolated workspace (workspace=None)
        # 2. workspace_mode="existing" + workspace_path → use explicit path
        # 3. No workspace_mode but workspace_path provided → use explicit path
        # 4. No workspace_mode, no workspace_path → reuse calling agent's workspace (backward compat)

        if workspace_mode == "new":
            # Create new isolated workspace - pass None so subagent creates its own
            workspace = None
            print(f"[delegation] workspace_mode='new': subagent will create isolated workspace")
        elif workspace_mode == "existing":
            # Use existing workspace - must have workspace_path
            if "workspace_path" in args and args["workspace_path"]:
                workspace = Path(args["workspace_path"])
                print(f"[delegation] workspace_mode='existing': using workspace_path={workspace}")
            else:
                raise ValueError("workspace_mode='existing' requires workspace_path parameter")
        elif "workspace_path" in args and args["workspace_path"]:
            # Explicit workspace path without mode (backward compat)
            workspace = Path(args["workspace_path"])
            print(f"[delegation] Using explicit workspace_path: {workspace}")
        elif hasattr(calling_agent, 'workspace') and calling_agent.workspace:
            # No workspace_mode specified - reuse calling agent's workspace (backward compat)
            workspace = calling_agent.workspace
            print(f"[delegation] No workspace_mode: reusing calling agent's workspace: {workspace}")
        else:
            # No workspace context - let subagent create its own
            workspace = None
            print(f"[delegation] No workspace context: subagent will create isolated workspace")

        # Instantiate target agent
        try:
            print(f"\n[delegation] Delegating to {target_agent_name}: {goal_description[:60]}...")

            # Temporarily clear OLLAMA_MODEL env var so delegated agent uses its own config
            # The calling agent's model choice shouldn't override the delegated agent's config
            import os
            saved_model_override = os.environ.get("OLLAMA_MODEL")
            if saved_model_override:
                del os.environ["OLLAMA_MODEL"]
                print(f"[delegation] Cleared OLLAMA_MODEL override for delegated agent")

            try:
                target_agent = agent_class(
                    workspace=workspace,
                    goal=goal_description
                )
            finally:
                # Restore OLLAMA_MODEL for calling agent
                if saved_model_override:
                    os.environ["OLLAMA_MODEL"] = saved_model_override

            # EXECUTE THE AGENT SYNCHRONOUSLY
            print(f"[delegation] Executing {target_agent_name} with max_rounds=50...")
            execution_result = target_agent.run(max_rounds=50)

            # Extract execution status and summary from subagent's completion signal
            status = execution_result.get('status', 'unknown')
            success = (status == 'success')

            # Get subagent workspace
            subagent_workspace = target_agent.workspace if hasattr(target_agent, 'workspace') else None

            # EXTRACT SUMMARY FROM SUBAGENT'S mark_complete/mark_failed CALL
            # The agent.run() returns summary/reason from SubAgentModeBehavior completion
            summary = execution_result.get('summary') or execution_result.get('reason')

            if not summary:
                # Fallback if no summary provided (shouldn't happen with SubAgentModeBehavior)
                summary = f"Task execution {status}. No summary provided by subagent."

            # LIST FILES CREATED BY SUBAGENT
            files_created = []
            if subagent_workspace and subagent_workspace.exists():
                try:
                    for item in subagent_workspace.iterdir():
                        if item.is_file() and not item.name.startswith('.'):
                            files_created.append(item.name)
                except Exception:
                    pass

            # BUILD CLEAR, ACTIONABLE RESULT MESSAGE
            if success:
                message = f"""Task delegated to {target_agent_name} completed successfully.

SUMMARY:
{summary}

WORKSPACE: {subagent_workspace}
FILES CREATED: {', '.join(files_created) if files_created else 'none'}

This delegation phase is complete. Review the summary and determine what to do next based on your overall goal."""
            else:
                message = f"""Task delegated to {target_agent_name} did not complete successfully (status: {status}).

SUMMARY:
{summary}

WORKSPACE: {subagent_workspace}
FILES CREATED: {', '.join(files_created) if files_created else 'none'}

The delegated task did not complete. Consider:
- Breaking down into simpler subtasks
- Providing more specific requirements
- Delegating again with adjusted parameters"""

            result = {
                "success": success,
                "status": status,
                "message": message,  # Clear, actionable summary for LLM
                "target_agent": target_agent_name,
                "goal": goal_description,
                "workspace": str(subagent_workspace),
                "files_created": files_created,
                "summary": summary,
            }

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

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Inject delegation information into context.

        Adds descriptions of delegatable agents after system prompt.

        Args:
            context: Current context
            **kwargs: Additional context

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
        delegation_message = {
            "role": "user",
            "content": "\n".join(delegation_info)
        }
        context.insert(1, delegation_message)

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
