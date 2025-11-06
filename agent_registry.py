"""
Agent registry - manages agent instances and delegation relationships.

Loads agent configuration from team config (config/teams/*.yaml) and provides:
- Agent instantiation (dynamic, no hardcoded agent classes)
- Delegation routing
- Agent lifecycle management

This registry is fully generic - it never refers to specific agent classes.
All agent instantiation is driven by team configuration.
"""
from __future__ import annotations
from typing import Any
from pathlib import Path
import yaml
import importlib

from base_agent import BaseAgent


class AgentRegistry:
    """
    Central registry for all agent instances.

    Responsibilities:
    - Load agent configuration from YAML
    - Instantiate agents on demand
    - Route delegation requests
    - Track active agents
    """

    def __init__(self, team_name: str = "default", workspace: Path = Path(".")):
        """
        Initialize agent registry.

        Args:
            team_name: Name of team config file (default: "default")
            workspace: Base workspace directory
        """
        self.team_name = team_name
        self.workspace = Path(workspace)
        self.agents: dict[str, BaseAgent] = {}
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """
        Load agent configuration from team config.

        Returns:
            Config dict with agent definitions
        """
        from agent_config import load_team_config

        config = load_team_config(self.team_name)

        # Default config if file doesn't exist
        if not config:
            return {
                "agents": {
                    "orchestrator": {
                        "class": "OrchestratorAgent",
                        "can_delegate_to": ["task_executor"],
                    },
                    "task_executor": {
                        "class": "TaskExecutorAgent",
                        "can_delegate_to": [],
                    },
                }
            }

        return config

    def get_agent(self, name: str) -> BaseAgent:
        """
        Get or create agent by name using dynamic instantiation.

        Dynamically imports and instantiates agent classes based on config.
        No hardcoded agent class references.

        Args:
            name: Agent name (e.g., "orchestrator", "task_executor")

        Returns:
            Agent instance

        Raises:
            ValueError: If agent not found in config
            ImportError: If agent module/class cannot be imported
        """
        # Return existing instance if already created
        if name in self.agents:
            return self.agents[name]

        # Check if agent exists in config
        if name not in self.config.get("agents", {}):
            raise ValueError(f"Agent '{name}' not found in config")

        # Get agent configuration
        agent_config = self.config["agents"][name]
        agent_class_name = agent_config["class"]

        # Determine module name from class name
        # Convention: OrchestratorAgent → orchestrator_agent
        # Convert CamelCase to snake_case and append .py convention
        module_name = self._class_to_module(agent_class_name)

        # Dynamically import agent class
        try:
            agent_module = importlib.import_module(module_name)
            agent_class = getattr(agent_module, agent_class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import {agent_class_name} from {module_name}: {e}"
            ) from e

        # Instantiate agent
        # If agent has a custom config, pass config_file parameter
        if "config" in agent_config:
            config_file = f"config/agents/{agent_config['config']}.yaml"
            agent = agent_class(workspace=self.workspace, config_file=config_file)
        else:
            agent = agent_class(workspace=self.workspace)

        # Store and return
        self.agents[name] = agent
        return agent

    def _class_to_module(self, class_name: str) -> str:
        """
        Convert class name to module name.

        Convention: OrchestratorAgent → orchestrator_agent

        Args:
            class_name: CamelCase class name

        Returns:
            snake_case module name
        """
        import re

        # Insert underscores before capitals (except first char)
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)

        # Convert to lowercase and remove trailing suffixes
        module_name = s2.lower()

        # Special case: if ends with "_agent", that's the module name
        # Otherwise might need adjustment
        return module_name

    def can_delegate(self, from_agent: str, to_agent: str) -> bool:
        """
        Check if one agent can delegate to another.

        Args:
            from_agent: Source agent name
            to_agent: Target agent name

        Returns:
            True if delegation is allowed
        """
        if from_agent not in self.config.get("agents", {}):
            return False

        allowed = self.config["agents"][from_agent].get("can_delegate_to", [])
        return to_agent in allowed

    def delegate_task(
        self,
        from_agent: str,
        to_agent: str,
        task_description: str,
        context: str = "",
        workspace: str = "",
    ) -> dict[str, Any]:
        """
        Delegate a task from one agent to another.

        Args:
            from_agent: Source agent name
            to_agent: Target agent name
            task_description: Task to delegate
            context: Additional context
            workspace: Optional workspace path for existing projects

        Returns:
            Result dict with success status and message

        Raises:
            ValueError: If delegation not allowed
        """
        # Check delegation permission
        if not self.can_delegate(from_agent, to_agent):
            raise ValueError(f"{from_agent} cannot delegate to {to_agent}")

        # Get target agent
        target = self.get_agent(to_agent)

        # Handle workspace override if specified
        workspace_path = None
        if workspace:
            workspace_path = Path(workspace)
            if not workspace_path.exists():
                return {
                    "success": False,
                    "message": f"Specified workspace does not exist: {workspace}",
                    "agent": to_agent,
                }

        # Trigger onGoalSet event (all agents support this as core functionality)
        # This is generic - works for any agent type
        target.trigger_behavior_event(
            "onGoalSet",
            goal=task_description,
            workspace=workspace_path,
        )

        return {
            "success": True,
            "message": f"Task delegated to {to_agent}",
            "agent": to_agent,
            "workspace": str(target.workspace),
        }

    def get_agent_status(self, name: str) -> dict[str, Any]:
        """
        Get status of an agent.

        Args:
            name: Agent name

        Returns:
            Status dict

        Raises:
            ValueError: If agent not found
        """
        if name not in self.agents:
            return {"status": "not_started", "agent": name}

        agent = self.agents[name]

        # Try agent-specific status methods (generic - no isinstance checks)
        if hasattr(agent, 'get_current_status') and callable(agent.get_current_status):
            return agent.get_current_status()

        if hasattr(agent, 'get_conversation_summary') and callable(agent.get_conversation_summary):
            return agent.get_conversation_summary()

        # Generic status fallback
        return {
            "status": "active",
            "agent": name,
            "rounds": agent.state.total_rounds if hasattr(agent, 'state') else 0,
        }

    def list_agents(self) -> list[str]:
        """
        List all available agent names.

        Returns:
            List of agent names from config
        """
        return list(self.config.get("agents", {}).keys())

    def get_delegation_graph(self) -> dict[str, list[str]]:
        """
        Get delegation relationships as a graph.

        Returns:
            Dict mapping agent names to list of agents they can delegate to
        """
        graph = {}
        for agent_name, agent_config in self.config.get("agents", {}).items():
            graph[agent_name] = agent_config.get("can_delegate_to", [])
        return graph
