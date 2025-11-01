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

        if tool_name == "consult_architect":
            return self._consult_architect(args, agent)
        elif tool_name == "delegate_to_executor":
            return self._delegate_to_executor(args, agent)
        else:
            # Generic delegation - tool found but handler not implemented
            return {"error": f"Delegation tool '{tool_name}' found but handler not implemented"}

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
