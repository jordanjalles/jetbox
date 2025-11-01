"""
DelegationBehavior - Auto-configured delegation tools from agent relationships.

This behavior dynamically creates delegation tools based on the agent's
can_delegate_to relationships defined in agents.yaml.

Features:
- Auto-generates delegation tools (consult_X, delegate_to_X)
- Injects delegatable agent descriptions into context
- Handles delegation tool dispatch
- No hardcoded agent relationships

Example:
    If agents.yaml defines:
        orchestrator:
            can_delegate_to: [architect, task_executor]

    Then DelegationBehavior will create:
        - consult_architect(project_description, requirements, constraints)
        - delegate_to_executor(task_description, workspace_mode, workspace_path)
"""

from typing import Any
from behaviors.base import AgentBehavior


class DelegationBehavior(AgentBehavior):
    """
    Behavior that provides dynamic delegation tools based on agent relationships.

    Automatically creates delegation tools for each agent in can_delegate_to list.
    """

    def __init__(self, agent_relationships: dict[str, Any]):
        """
        Initialize delegation behavior.

        Args:
            agent_relationships: Dict mapping agent name -> {class, description, can_delegate_to}
        """
        self.agent_relationships = agent_relationships
        self.delegation_tools = []
        self._build_delegation_tools()

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "delegation"

    def _build_delegation_tools(self) -> None:
        """
        Build delegation tools based on can_delegate_to relationships.

        For each delegatable agent, creates a tool definition.
        """
        can_delegate_to = self.agent_relationships.get("can_delegate_to", [])

        for target_agent in can_delegate_to:
            # Get agent info from relationships
            agent_info = self.agent_relationships.get(target_agent, {})
            description = agent_info.get("description", f"Delegate to {target_agent}")

            # Create tool based on agent type
            if target_agent == "architect":
                tool = {
                    "type": "function",
                    "function": {
                        "name": "consult_architect",
                        "description": "Consult the Architect agent for complex project architecture design",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "project_description": {
                                    "type": "string",
                                    "description": "Brief description of the project"
                                },
                                "requirements": {
                                    "type": "string",
                                    "description": "Functional and non-functional requirements"
                                },
                                "constraints": {
                                    "type": "string",
                                    "description": "Technical constraints (team size, tech stack, timeline, etc.)"
                                }
                            },
                            "required": ["project_description", "requirements", "constraints"]
                        }
                    }
                }
            elif target_agent == "task_executor":
                tool = {
                    "type": "function",
                    "function": {
                        "name": "delegate_to_executor",
                        "description": "Delegate a coding task to the TaskExecutor agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_description": {
                                    "type": "string",
                                    "description": "Clear description of the task to execute"
                                },
                                "workspace_mode": {
                                    "type": "string",
                                    "description": "Workspace mode: 'new' for new projects, 'existing' for updates",
                                    "enum": ["new", "existing"]
                                },
                                "workspace_path": {
                                    "type": "string",
                                    "description": "Path to existing workspace (required if workspace_mode='existing')"
                                }
                            },
                            "required": ["task_description", "workspace_mode"]
                        }
                    }
                }
            else:
                # Generic delegation tool for unknown agent types
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

        if tool_name == "consult_architect":
            return self._consult_architect(args, agent)
        elif tool_name == "delegate_to_executor":
            return self._delegate_to_executor(args, agent)
        else:
            # Generic delegation
            return {"error": f"Unknown delegation tool: {tool_name}"}

    def _consult_architect(self, args: dict[str, Any], agent: Any) -> dict[str, Any]:
        """
        Consult architect agent for project design.

        Args:
            args: Tool arguments (project_description, requirements, constraints)
            agent: Calling agent (orchestrator)

        Returns:
            Architect consultation result
        """
        from architect_agent import ArchitectAgent
        from pathlib import Path

        # Create architect agent
        workspace = Path(".agent_workspace") / "architecture_consultation"
        architect = ArchitectAgent(workspace=workspace, use_behaviors=True)

        # Build consultation prompt
        prompt = f"""Project: {args['project_description']}

Requirements:
{args['requirements']}

Constraints:
{args['constraints']}

Please design the architecture for this project and create:
1. Architecture overview document
2. Module specifications for each component
3. Task breakdown for implementation"""

        # Run architect consultation
        # This is a simplified placeholder - actual implementation would:
        # 1. Call architect.run() with the prompt
        # 2. Wait for completion
        # 3. Extract architecture docs and task list
        # 4. Return structured result

        return {
            "status": "success",
            "message": "Architect consultation would happen here",
            "workspace_path": str(workspace),
            "architecture_docs": ["architecture/overview.md"],
            "module_specs": ["architecture/modules/ingestion.md"],
            "tasks": [
                {
                    "task_id": "T1",
                    "description": "Implement ingestion module",
                    "module": "ingestion",
                    "priority": 1
                }
            ]
        }

    def _delegate_to_executor(self, args: dict[str, Any], agent: Any) -> dict[str, Any]:
        """
        Delegate task to executor agent.

        Args:
            args: Tool arguments (task_description, workspace_mode, workspace_path)
            agent: Calling agent (orchestrator)

        Returns:
            Executor delegation result
        """
        from task_executor_agent import TaskExecutorAgent
        from pathlib import Path

        # Determine workspace
        workspace_mode = args["workspace_mode"]
        if workspace_mode == "existing":
            workspace_path = args.get("workspace_path")
            if not workspace_path:
                return {"error": "workspace_path required when workspace_mode='existing'"}
            workspace = Path(workspace_path)
        else:
            # New workspace
            workspace = None  # TaskExecutor will create new isolated workspace

        # Create executor agent
        executor = TaskExecutorAgent(
            workspace=workspace,
            goal=args["task_description"],
            use_behaviors=True
        )

        # Run executor
        # This is a simplified placeholder - actual implementation would:
        # 1. Call executor.run() with the task
        # 2. Wait for completion
        # 3. Extract results
        # 4. Return structured result

        return {
            "status": "success",
            "message": "Task execution would happen here",
            "files_created": ["example.py"],
            "workspace_path": str(executor.workspace) if executor.workspace else ".agent_workspace/new"
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
            description = agent_info.get("description", f"Agent: {target_agent}")
            delegation_info.append(f"- **{target_agent}**: {description}")

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
            Instructions for using delegation tools
        """
        can_delegate_to = self.agent_relationships.get("can_delegate_to", [])
        if not can_delegate_to:
            return ""

        return f"""
DELEGATION WORKFLOW:
You can delegate work to the following agents: {', '.join(can_delegate_to)}

Guidelines:
- Assess task complexity before delegating
- Use architect for complex multi-component projects
- Use task_executor for coding implementation
- Always specify workspace_mode when delegating to executor
- Report delegation results back to user
"""
