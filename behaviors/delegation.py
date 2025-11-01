"""
DelegationBehavior - Bidirectional delegation (delegate TO and be delegated to).

This behavior handles BOTH directions of delegation:
1. Delegating TO other agents (orchestrator → task_executor)
2. Being delegated to (receiving work from another agent)

Features:
- Auto-generates delegation tools based on can_delegate_to relationships
- Provides mark_complete/mark_failed tools when acting as delegatee
- Handles goal initialization when delegated to
- Injects delegation context as appropriate
- Works for ALL agent types (TaskExecutor, Orchestrator, Architect)

Examples:
    Orchestrator (delegator):
        - Can delegate to architect, task_executor
        - Gets consult_architect, delegate_to_executor tools

    TaskExecutor (delegatee):
        - Can be delegated to by orchestrator
        - Gets mark_complete, mark_failed tools
        - Shows "DELEGATED GOAL" in context

    Orchestrator (both):
        - Can delegate to others (delegator mode)
        - Can be delegated to (delegatee mode)
        - Gets all tools for both directions
"""

from typing import Any
from behaviors.base import AgentBehavior


class DelegationBehavior(AgentBehavior):
    """
    Unified bidirectional delegation behavior.

    Handles both delegating TO agents and being delegated to BY agents.
    Auto-configures based on agent_relationships from agents.yaml.
    """

    def __init__(
        self,
        agent_relationships: dict[str, Any],
        is_delegatee: bool = False
    ):
        """
        Initialize delegation behavior.

        Args:
            agent_relationships: Dict mapping agent name → {class, description, can_delegate_to}
            is_delegatee: True if this agent can be delegated to (provides mark_complete tools)
        """
        self.agent_relationships = agent_relationships
        self.is_delegatee = is_delegatee  # Can be delegated to
        self.delegation_tools = []
        self.delegated_tasks: list[dict[str, Any]] = []  # Track delegated tasks
        self.goal = None  # Store goal for delegatee mode context injection

        # Build delegation tools based on mode
        self._build_delegation_tools()

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "delegation"

    def _build_delegation_tools(self) -> None:
        """
        Build delegation tools based on can_delegate_to relationships.

        Builds two types of tools:
        1. Delegator tools: consult_X, delegate_to_X (for delegating TO others)
        2. Delegatee tools: mark_complete, mark_failed (for being delegated to)
        """
        # 1. Build delegator tools (if this agent can delegate to others)
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
            else:
                # Fallback: generic delegation tool for agents without delegation_tool config
                description = agent_info.get("description", f"Delegate to {target_agent}")
                tool = {
                    "type": "function",
                    "function": {
                        "name": f"delegate_to_{target_agent}",
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

            self.delegation_tools.append(tool)

        # 2. Build delegatee tools (if this agent is a delegatee)
        if self.is_delegatee:
            self.delegation_tools.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "mark_complete",
                        "description": "Mark the delegated task as complete and report success to controlling agent. REQUIRED when work is finished.",
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
                        "description": "Mark the delegated task as failed and report reason to controlling agent. Use when you cannot complete the task.",
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
            ])

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return delegation tool definitions.

        Returns:
            List of dynamically generated delegation tools (delegator + delegatee)
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

        Handles both delegator tools (consult_X, delegate_to_X) and
        delegatee tools (mark_complete, mark_failed).

        Args:
            tool_name: Tool name
            args: Tool arguments
            **kwargs: Additional context (agent, workspace, context_manager, etc.)

        Returns:
            Delegation result
        """
        agent = kwargs.get("agent")
        context_manager = kwargs.get("context_manager")

        # Handle delegatee tools (mark_complete, mark_failed)
        if tool_name == "mark_complete":
            summary = args.get('summary', 'Task completed')

            # Mark goal as complete in context manager if available
            if context_manager and hasattr(context_manager, 'state') and context_manager.state.goal:
                context_manager.state.goal.mark_complete(success=True)

            return {
                "success": True,
                "result": f"Task marked complete: {summary}",
                "summary": summary
            }

        elif tool_name == "mark_failed":
            reason = args.get('reason', 'Task failed')

            # Mark goal as failed in context manager if available
            if context_manager and hasattr(context_manager, 'state') and context_manager.state.goal:
                context_manager.state.goal.mark_complete(success=False)

            return {
                "success": False,
                "result": f"Task marked failed: {reason}",
                "reason": reason
            }

        # Handle delegator tools (consult_X, delegate_to_X)
        # Map tool names to target agent names
        target_agent_name = None

        if tool_name == "consult_architect":
            target_agent_name = "architect"
        elif tool_name == "delegate_to_executor":
            target_agent_name = "task_executor"
        elif tool_name == "delegate_to_orchestrator":
            target_agent_name = "orchestrator"
        elif tool_name.startswith("delegate_to_"):
            # Generic delegation tool
            target_agent_name = tool_name.replace("delegate_to_", "")
        elif tool_name.startswith("consult_"):
            # Generic consultation tool
            target_agent_name = tool_name.replace("consult_", "")

        if target_agent_name:
            return self._delegate_to_agent(target_agent_name, args, agent)
        else:
            # Unknown delegation tool
            return {"error": f"Unknown delegation tool: {tool_name}"}

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
            # Map class names to module imports
            class_to_module = {
                "TaskExecutorAgent": "task_executor_agent",
                "OrchestratorAgent": "orchestrator_agent",
                "ArchitectAgent": "architect_agent",
            }

            module_name = class_to_module.get(agent_class_name)
            if not module_name:
                return {
                    "success": False,
                    "error": f"Unknown agent class: {agent_class_name}"
                }

            # Import the class
            import importlib
            module = importlib.import_module(module_name)
            agent_class = getattr(module, agent_class_name)

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to import {agent_class_name}: {e}"
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

        # Determine workspace
        workspace = None
        if "workspace_mode" in args:
            workspace_mode = args["workspace_mode"]
            if workspace_mode == "existing":
                workspace_path = args.get("workspace_path")
                if not workspace_path:
                    return {
                        "success": False,
                        "error": "workspace_path required when workspace_mode='existing'"
                    }
                workspace = Path(workspace_path)
            # else: workspace_mode == "new", workspace = None (agent creates new)
        elif "workspace_path" in args:
            # Direct workspace path provided
            workspace = Path(args["workspace_path"])
        else:
            # No workspace specified - create new workspace
            workspace = None

        # Instantiate target agent
        try:
            print(f"\n[delegation] Delegating to {target_agent_name}: {goal_description[:60]}...")

            target_agent = agent_class(
                workspace=workspace,
                goal=goal_description,
                use_behaviors=True
            )

            # NOTE: Actual execution happens in orchestrator_main.py via subprocess
            # This is just the setup phase - we return info for orchestrator to execute

            result = {
                "success": True,
                "message": f"Delegation to {target_agent_name} prepared",
                "target_agent": target_agent_name,
                "goal": goal_description,
                "workspace": str(target_agent.workspace) if hasattr(target_agent, 'workspace') and target_agent.workspace else None,
                "agent_class": agent_class_name,
            }

            # Track delegation
            self.track_delegation(target_agent_name, goal_description, result)

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

        Injects two types of context:
        1. Delegator context: Descriptions of agents this agent can delegate to
        2. Delegatee context: "DELEGATED GOAL" header when this agent was delegated to

        Args:
            context: Current context
            **kwargs: Additional context

        Returns:
            Modified context with delegation info
        """
        if len(context) == 0:
            return context

        context_manager = kwargs.get('context_manager')

        # 1. Inject delegatee context (if we're a delegatee with a goal)
        if self.is_delegatee and context_manager and context_manager.state.goal:
            delegatee_parts = [
                f"DELEGATED GOAL: {context_manager.state.goal.description}",
                "",
                "You are working on a task delegated by another agent.",
                "When complete, call mark_complete(summary) with what you accomplished.",
                "If you cannot complete it, call mark_failed(reason) explaining why."
            ]

            # Insert after system prompt (index 1)
            context.insert(1, {
                "role": "user",
                "content": "\n".join(delegatee_parts)
            })

        # 2. Inject delegator context (if we can delegate to others)
        can_delegate_to = self.agent_relationships.get("can_delegate_to", [])
        if can_delegate_to:
            # Build delegation info
            delegation_info = ["## Available Agents for Delegation\n"]
            for target_agent in can_delegate_to:
                agent_info = self.agent_relationships.get(target_agent, {})
                # Use blurb if available, fallback to description
                blurb = agent_info.get("blurb", agent_info.get("description", f"Agent: {target_agent}"))
                delegation_info.append(f"- **{target_agent}**: {blurb}")

            # Insert after system prompt and delegatee context (if present)
            insert_index = 2 if self.is_delegatee and context_manager and context_manager.state.goal else 1
            delegation_message = {
                "role": "user",
                "content": "\n".join(delegation_info)
            }
            context.insert(insert_index, delegation_message)

        return context

    def get_instructions(self) -> str:
        """
        Return delegation workflow instructions.

        Returns different instructions based on mode:
        - Delegator: How to delegate to other agents
        - Delegatee: How to complete delegated work
        - Both: Combined instructions

        Returns:
            Instructions for using delegation tools
        """
        instructions_parts = []

        # Delegatee instructions
        if self.is_delegatee:
            instructions_parts.append("""
SUB-AGENT WORKFLOW:
You are working on a task delegated by another agent.

Your job is to:
1. Complete the delegated goal using available tools
2. Report results back when done

Available tools:
- write_file: Create/modify files
- run_bash: Run commands (tests, linters, build tools, etc.)
- read_file: Check existing files
- list_dir: Explore directories
- mark_complete(summary): Signal successful completion with summary
- mark_failed(reason): Signal failure with explanation

IMPORTANT - YOU MUST SIGNAL COMPLETION:
- When work is done and tests pass: call mark_complete(summary="what you accomplished")
- If you cannot complete the task: call mark_failed(reason="why it failed")
- DO NOT just stop - always call one of these tools to report results
""")

        # Delegator instructions
        can_delegate_to = self.agent_relationships.get("can_delegate_to", [])
        if can_delegate_to:
            # Build guidelines from agent blurbs
            guidelines = []
            for target_agent in can_delegate_to:
                agent_info = self.agent_relationships.get(target_agent, {})
                blurb = agent_info.get("blurb", agent_info.get("description", ""))
                if blurb:
                    # Extract key guidance from blurb
                    blurb_lines = blurb.strip().split(". ")
                    guidance = None
                    for line in blurb_lines:
                        if "Best for" in line or "best for" in line:
                            guidance = line.strip()
                            break
                    if guidance:
                        guidelines.append(f"- Use {target_agent} for: {guidance}")
                    else:
                        guidelines.append(f"- Use {target_agent}: {agent_info.get('description', '')}")

            guidelines_text = "\n".join(guidelines) if guidelines else "- Assess task complexity and choose appropriate agent"

            instructions_parts.append(f"""
DELEGATION WORKFLOW:
You can delegate work to the following agents: {', '.join(can_delegate_to)}

Guidelines:
{guidelines_text}
- Always report delegation results back to user
""")

        return "\n".join(instructions_parts) if instructions_parts else ""

    def on_goal_set(self, agent: Any, goal: str, **kwargs: Any) -> None:
        """
        Handle goal setting event (delegatee mode).

        This method initializes all subsystems needed for delegated goal execution:
        - Context manager with goal
        - Workspace manager (new or existing)
        - Performance tracking
        - Status display

        Only runs if this behavior is configured as a delegatee.

        Args:
            agent: Agent instance
            goal: Goal description
            **kwargs: Additional context (workspace, etc.)
        """
        if not self.is_delegatee:
            return  # Only delegatees need goal setup

        # Store goal for context injection
        self.goal = goal

        # Initialize context manager with goal
        if not agent.context_manager:
            from context_manager import ContextManager
            agent.context_manager = ContextManager()
        agent.context_manager.load_or_init(goal)

        # Initialize workspace manager
        # workspace parameter: None = create new, Path = reuse existing
        workspace_param = kwargs.get('workspace')
        goal_slug = goal.lower()[:50].replace(" ", "-").replace("/", "-")

        if workspace_param:
            # Reuse mode: use existing workspace directory
            print(f"[delegation] Reusing workspace: {workspace_param}")
            agent.init_workspace_manager(goal_slug, workspace_path=workspace_param)
        else:
            # Create new mode: create isolated workspace
            print(f"[delegation] Creating new workspace for goal")
            agent.init_workspace_manager(goal_slug, workspace_path=None)

        # Initialize performance tracking
        agent.init_perf_stats()

        # Initialize status display if not already present
        if not hasattr(agent, 'status_display') or agent.status_display is None:
            from status_display import StatusDisplay
            agent.status_display = StatusDisplay(ctx=agent.context_manager, reset_stats=True)

        # Start wall-clock timer for goal
        import time
        agent.goal_start_time = time.time()

        print(f"[delegation] Goal set: {goal}")
