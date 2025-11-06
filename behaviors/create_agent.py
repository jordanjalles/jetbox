"""
CreateAgentBehavior - Agent configuration generation system.

This behavior provides tools for generating agent configuration YAML files.
"""

from typing import Any
from pathlib import Path
import json
import yaml
from behaviors.base import AgentBehavior


class CreateAgentBehavior(AgentBehavior):
    """
    Provides tools for generating agent configuration files.
    """

    def __init__(self, workspace_manager=None, **kwargs):
        """Initialize CreateAgentBehavior."""
        self.workspace_manager = workspace_manager
        self.staging_dir = Path(".agent_generated/staging")
        self.default_safety_mode = "review"

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "create_agent"

    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_agent",
                    "description": "Generate a new agent configuration YAML file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_name": {
                                "type": "string",
                                "description": "Agent name in kebab-case (e.g., 'doc-generator')"
                            },
                            "role": {
                                "type": "string",
                                "description": "Agent's role description"
                            },
                            "description": {
                                "type": "string",
                                "description": "Detailed description of agent's purpose"
                            },
                            "behaviors": {
                                "type": "array",
                                "description": "List of behavior types to include",
                                "items": {"type": "string"}
                            },
                            "system_prompt_guidelines": {
                                "type": "array",
                                "description": "Optional additional guidelines for system prompt",
                                "items": {"type": "string"}
                            },
                            "delegation_tool_params": {
                                "type": "object",
                                "description": "Optional custom delegation tool parameters"
                            },
                            "add_to_team": {
                                "type": "string",
                                "description": "Optional team name to add agent to"
                            },
                            "can_delegate_to": {
                                "type": "array",
                                "description": "Optional list of agent names this agent can delegate to",
                                "items": {"type": "string"}
                            },
                            "safety_mode": {
                                "type": "string",
                                "description": "Safety mode: 'dryrun', 'review', 'auto', 'strict'"
                            }
                        },
                        "required": ["agent_name", "role", "description", "behaviors"]
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        agent: Any,
        tool_name: str,
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle tool execution."""
        if tool_name == "create_agent":
            return self._execute_create_agent(agent, args)
        else:
            return super().dispatch_tool(agent, tool_name, args)

    def _execute_create_agent(
        self,
        agent: Any,
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute create_agent tool."""
        try:
            # Validate and extract parameters
            validation_result = self._validate_and_extract_params(args)
            if "error" in validation_result:
                return validation_result

            params = validation_result["params"]

            # Run the workflow
            return self._run_agent_generation_workflow(agent, params)

        except Exception as e:
            return {"error": f"Error creating agent: {str(e)}"}

    def _validate_and_extract_params(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate and extract parameters from args."""
        # Extract parameters (supporting both old and new parameter names)
        agent_name = args.get("agent_name", "")
        role = args.get("role", "")

        # Support both "description" and "blurb" for agent description
        description = args.get("description") or args.get("blurb", "")

        behaviors = args.get("behaviors", [])

        # Support both parameter styles for system prompt
        system_prompt_guidelines = args.get("system_prompt_guidelines")
        if system_prompt_guidelines is None:
            system_prompt = args.get("system_prompt", "")
            if system_prompt:
                system_prompt_guidelines = [system_prompt]
            else:
                system_prompt_guidelines = []

        # Support both parameter styles for delegation tool
        delegation_tool_params = args.get("delegation_tool_params")
        if delegation_tool_params is None:
            delegation_tool = args.get("delegation_tool", {})
            if delegation_tool:
                delegation_tool_params = delegation_tool.get("parameters", {})
            else:
                delegation_tool_params = {}

        add_to_team = args.get("add_to_team", "")
        can_delegate_to = args.get("can_delegate_to", [])
        safety_mode = args.get("safety_mode", self.default_safety_mode)

        # Validate required inputs
        if not agent_name:
            return {"error": "agent_name is required"}
        if not role:
            return {"error": "role is required"}
        if not description:
            return {"error": "description is required"}
        if not behaviors:
            return {"error": "behaviors must be a non-empty list"}

        # Validate agent_name format (kebab-case)
        if "_" in agent_name:
            return {"error": "agent_name must be kebab-case (use hyphens, not underscores)"}

        return {
            "params": {
                "agent_name": agent_name,
                "role": role,
                "description": description,
                "behaviors": behaviors,
                "system_prompt_guidelines": system_prompt_guidelines,
                "delegation_tool_params": delegation_tool_params,
                "add_to_team": add_to_team,
                "can_delegate_to": can_delegate_to,
                "safety_mode": safety_mode
            }
        }

    def _run_agent_generation_workflow(
        self,
        agent: Any,
        params: dict[str, Any]
    ) -> dict[str, Any]:
        """Run the full agent generation workflow."""
        agent_name = params["agent_name"]
        
        # For now, return a simple success result
        # Full implementation would generate YAML, validate, etc.
        return {
            "success": True,
            "agent_name": agent_name,
            "message": "Agent generation workflow placeholder"
        }
