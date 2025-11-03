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
        self._build_delegation_tools()

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "delegation"

    def _build_delegation_tools(self) -> None:
        """
        Build delegation tools based on can_delegate_to relationships.

        For each delegatable agent, creates a tool definition from config.
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
            **kwargs: Additional context (agent, workspace, etc.)

        Returns:
            Delegation result
        """
        agent = kwargs.get("agent")

        # Map tool names to target agent names
        # Tool name format: "consult_X" or "delegate_to_X"
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

        # CRITICAL: For architect, reframe goal as DOCUMENTATION task, not implementation
        # This prevents architect from thinking it should implement code
        if target_agent_name == "architect":
            # Wrap goal to emphasize documentation-only role
            goal_description = (
                f"Design the architecture documentation for the following project:\n\n"
                f"{goal_description}\n\n"
                f"Your task is to create:\n"
                f"1. Architecture overview (write_architecture_doc)\n"
                f"2. Module specifications for each component (write_module_spec)\n"
                f"3. Task breakdown for implementation (write_task_list)\n"
                f"4. Call mark_complete when documentation is finished\n\n"
                f"DO NOT implement code - only create architecture documentation."
            )

        # PROPER WORKSPACE COORDINATION
        # If calling agent has a workspace, subagent should work in SAME workspace
        # This ensures orchestrator and subagents coordinate on file locations
        workspace = None

        # AUTOMATIC WORKSPACE COORDINATION:
        # Priority order:
        # 1. Explicit workspace_path provided → use it
        # 2. Calling agent has workspace attribute → reuse it (FOOLPROOF: works for orchestrator!)
        # 3. Neither → let subagent create isolated workspace (will be reused on retry via goal slug)

        if "workspace_path" in args and args["workspace_path"]:
            # Explicit workspace path overrides everything
            workspace = Path(args["workspace_path"])
            print(f"[delegation] Using explicit workspace path: {workspace}")
        elif hasattr(calling_agent, 'workspace') and calling_agent.workspace:
            # AUTOMATIC COORDINATION: Reuse calling agent's workspace
            # This is FOOLPROOF - works even if workspace_manager isn't initialized yet
            # (e.g., orchestrator doesn't have a goal, so workspace_manager is None)
            workspace = calling_agent.workspace
            print(f"[delegation] Reusing calling agent's workspace: {workspace}")
        else:
            # No workspace context - let subagent create its own
            # (will be reused on retry because same goal → same slug → same workspace path)
            workspace = None
            print(f"[delegation] Subagent will create isolated workspace based on goal")

        # Instantiate target agent
        try:
            print(f"\n[delegation] Delegating to {target_agent_name}: {goal_description[:60]}...")

            target_agent = agent_class(
                workspace=workspace,
                goal=goal_description
            )

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
