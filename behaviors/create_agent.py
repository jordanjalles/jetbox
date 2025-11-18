"""
CreateAgentBehavior - Agent configuration generation system.

This behavior provides tools for generating agent configuration YAML files.

Now uses @tool decorator for automatic tool registration!"""

from typing import Any
from pathlib import Path
from behaviors.base import AgentBehavior
from behaviors.tool_decorator import tool
from behaviors.rule_of_two_types import RuleOfTwoProperty


class CreateAgentBehavior(AgentBehavior):
    """
    Provides tools for generating agent configuration files.

    Security: [] None (meta-programming - generates agent configs, security evaluated when invoked)
    """

    # Rule of Two: Empty (meta-programming utility)
    rule_of_two_properties = set()

    def __init__(self, workspace_manager=None, **kwargs):
        """Initialize CreateAgentBehavior."""
        self.workspace_manager = workspace_manager
        self.staging_dir = Path(".agent_generated/staging")
        self.default_safety_mode = "review"

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "create_agent"

    @tool
    def create_agent(
        self,
        agent_name: str,
        role: str,
        description: str,
        behaviors: list[str],
        system_prompt_guidelines: list[str] = None,
        delegation_tool_params: dict[str, Any] = None,
        add_to_team: str = "",
        can_delegate_to: list[str] = None,
        safety_mode: str = "review"
    ) -> dict[str, Any]:
        """Generate a new agent configuration YAML file.

        Args:
            agent_name: Agent name in kebab-case (e.g., 'doc-generator')
            role: Agent's role description
            description: Detailed description of agent's purpose
            behaviors: List of behavior types to include
            system_prompt_guidelines: Optional additional guidelines for system prompt
            delegation_tool_params: Optional custom delegation tool parameters
            add_to_team: Optional team name to add agent to
            can_delegate_to: Optional list of agent names this agent can delegate to
            safety_mode: Safety mode: 'dryrun', 'review', 'auto', 'strict'

        Returns:
            Dict with success status, config file path, and validation results
        """
        # Build args dict for existing implementation
        args = {
            "agent_name": agent_name,
            "role": role,
            "description": description,
            "behaviors": behaviors,
            "system_prompt_guidelines": system_prompt_guidelines or [],
            "delegation_tool_params": delegation_tool_params or {},
            "add_to_team": add_to_team,
            "can_delegate_to": can_delegate_to or [],
            "safety_mode": safety_mode
        }
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
            # If full delegation_tool dict is provided, use it directly
            delegation_tool = args.get("delegation_tool", {})
            if delegation_tool:
                delegation_tool_params = delegation_tool
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
        role = params["role"]
        description = params["description"]
        behaviors = params["behaviors"]
        system_prompt_guidelines = params.get("system_prompt_guidelines", "")
        delegation_tool_params = params.get("delegation_tool_params", None)
        # Note: safety_mode param exists but not used in this method yet

        # 1. Generate YAML configuration
        config_result = self._generate_agent_config(
            agent, agent_name, role, description, behaviors,
            system_prompt_guidelines, delegation_tool_params
        )

        if "error" in config_result:
            return config_result

        yaml_content = config_result["yaml_content"]

        # 2. Determine output path based on safety_mode
        staging_dir = Path(".agent_generated/staging")
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_path = staging_dir / f"{agent_name}.yaml"

        # 3. Write YAML to staging
        try:
            with open(staging_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)
        except Exception as e:
            return {"error": f"Failed to write config file: {e}"}

        # 4. Validate the generated YAML
        validation_result = self._validate_agent_config(agent, str(staging_path))
        if "error" in validation_result:
            return {
                "error": f"Generated config failed validation: {validation_result['error']}",
                "config_file": str(staging_path)
            }

        # 5. Return success with config file path
        return {
            "success": True,
            "config_file": str(staging_path),
            "agent_name": agent_name,
            "message": f"Agent config generated and validated at {staging_path}"
        }

    def _generate_agent_config(
        self,
        agent: Any,
        agent_name: str,
        role: str,
        description: str,
        behaviors: list[str],
        system_prompt_guidelines: list[str],
        delegation_tool_params: dict[str, Any] = None,
        author: str = "MetaProgrammer"
    ) -> dict[str, Any]:
        """Generate agent YAML configuration content with metadata header."""
        from utils.generation_metadata import generate_metadata_header

        # Generate metadata header
        metadata = generate_metadata_header(
            generator="CreateAgentBehavior",
            author=author,
            parent_request=description
        )

        # Build behaviors list in YAML format
        behaviors_yaml = []
        for behavior in behaviors:
            behaviors_yaml.append(f"  - type: {behavior}")
        behaviors_str = "\n".join(behaviors_yaml)

        # Build delegation tool section if provided
        delegation_tool_section = ""
        if delegation_tool_params:
            tool_name = delegation_tool_params.get("name", f"delegate_to_{agent_name.replace('-', '_')}")
            tool_desc = delegation_tool_params.get("description", f"Delegate tasks to {agent_name}")

            # Build parameters section
            params_yaml = []
            for param_name, param_config in delegation_tool_params.get("parameters", {}).items():
                param_type = param_config.get("type", "string")
                param_desc = param_config.get("description", "")
                param_required = param_config.get("required", False)
                params_yaml.append(f"    {param_name}:")
                params_yaml.append(f"      type: {param_type}")
                params_yaml.append(f"      description: \"{param_desc}\"")
                params_yaml.append(f"      required: {str(param_required).lower()}")
            params_str = "\n".join(params_yaml) if params_yaml else "    task_description:\n      type: string\n      description: \"Task to delegate\"\n      required: true"

            delegation_tool_section = f"""delegation_tool:
  name: "{tool_name}"
  description: "{tool_desc}"
  parameters:
{params_str}
"""

        # Build system prompt - handle list of guidelines
        if isinstance(system_prompt_guidelines, list) and len(system_prompt_guidelines) == 1:
            # If single string passed as list, use it directly
            guidelines_text = system_prompt_guidelines[0]
        elif isinstance(system_prompt_guidelines, list):
            # If multiple strings, join with newlines
            guidelines_text = "\n\n".join(system_prompt_guidelines)
        else:
            # Fallback for string input
            guidelines_text = str(system_prompt_guidelines) if system_prompt_guidelines else ""

        system_prompt = guidelines_text.strip()

        # Assemble full YAML with metadata header
        # Indent description lines for blurb section
        description_indented = '\n'.join('  ' + line for line in description.split('\n'))

        yaml_content = f"""{metadata}
role: "{role}"
blurb: |
{description_indented}
{delegation_tool_section}
system_prompt: |
{chr(10).join('  ' + line for line in system_prompt.split(chr(10)))}

behaviors:
{behaviors_str}
"""

        return {"yaml_content": yaml_content}

    def _validate_agent_config(
        self,
        agent: Any,
        config_path: str
    ) -> dict[str, Any]:
        """Validate agent configuration file."""
        try:
            # Load YAML
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Check required fields
            required_fields = ["role", "blurb", "system_prompt", "behaviors"]
            for field in required_fields:
                if field not in config:
                    return {"error": f"Missing required field: {field}"}

            # Validate behaviors is a list
            if not isinstance(config["behaviors"], list):
                return {"error": "behaviors must be a list"}

            # Validate delegation_tool if present
            if "delegation_tool" in config:
                dt = config["delegation_tool"]
                if not isinstance(dt, dict):
                    return {"error": "delegation_tool must be a dict"}
                if "name" not in dt or "description" not in dt:
                    return {"error": "delegation_tool must have name and description"}

            return {"success": True, "config": config}

        except Exception as e:
            return {"error": f"Validation failed: {e}"}
