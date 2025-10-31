"""
Agent registry - manages agent instances and delegation relationships.

Loads agent configuration from agents.yaml and provides:
- Agent instantiation
- Delegation routing
- Agent lifecycle management
"""
from __future__ import annotations
from typing import Any
from pathlib import Path
import yaml

from base_agent import BaseAgent
from orchestrator_agent import OrchestratorAgent
from task_executor_agent import TaskExecutorAgent
from architect_agent import ArchitectAgent


class AgentRegistry:
    """
    Central registry for all agent instances.

    Responsibilities:
    - Load agent configuration from YAML
    - Instantiate agents on demand
    - Route delegation requests
    - Track active agents
    """

    def __init__(self, config_path: str = "agents.yaml", workspace: Path = Path(".")):
        """
        Initialize agent registry.

        Args:
            config_path: Path to agents.yaml config file
            workspace: Base workspace directory
        """
        self.config_path = config_path
        self.workspace = Path(workspace)
        self.agents: dict[str, BaseAgent] = {}
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """
        Load agent configuration from YAML.

        Returns:
            Config dict with agent definitions
        """
        config_file = Path(self.config_path)

        # Default config if file doesn't exist
        if not config_file.exists():
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

        # Load from YAML
        with open(config_file) as f:
            return yaml.safe_load(f) or {}

    def get_agent(self, name: str) -> BaseAgent:
        """
        Get or create agent by name.

        Args:
            name: Agent name (e.g., "orchestrator", "task_executor")

        Returns:
            Agent instance

        Raises:
            ValueError: If agent not found in config
        """
        # Return existing instance if already created
        if name in self.agents:
            return self.agents[name]

        # Check if agent exists in config
        if name not in self.config.get("agents", {}):
            raise ValueError(f"Agent '{name}' not found in config")

        # Instantiate agent
        agent_config = self.config["agents"][name]
        agent_class = agent_config["class"]

        # Create agent based on class name
        if agent_class == "OrchestratorAgent":
            agent = OrchestratorAgent(workspace=self.workspace)
        elif agent_class == "TaskExecutorAgent":
            agent = TaskExecutorAgent(workspace=self.workspace)
        elif agent_class == "ArchitectAgent":
            agent = ArchitectAgent(workspace=self.workspace)
        else:
            raise ValueError(f"Unknown agent class: {agent_class}")

        # Store and return
        self.agents[name] = agent
        return agent

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

        # For TaskExecutor, set the goal
        if isinstance(target, TaskExecutorAgent):
            # If workspace specified, update TaskExecutor's workspace
            if workspace:
                workspace_path = Path(workspace)
                if workspace_path.exists():
                    target.workspace = workspace_path
                    # Also update workspace manager if it exists
                    if hasattr(target, 'workspace_manager') and target.workspace_manager:
                        target.workspace_manager.workspace_dir = workspace_path
                else:
                    return {
                        "success": False,
                        "message": f"Specified workspace does not exist: {workspace}",
                        "agent": to_agent,
                    }

            target.set_goal(task_description)
            return {
                "success": True,
                "message": f"Task delegated to {to_agent}",
                "agent": to_agent,
                "workspace": str(target.workspace),
            }

        return {
            "success": False,
            "message": f"Don't know how to delegate to {to_agent}",
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

        # TaskExecutor has get_current_status
        if isinstance(agent, TaskExecutorAgent):
            return agent.get_current_status()

        # Orchestrator has get_conversation_summary
        if isinstance(agent, OrchestratorAgent):
            return agent.get_conversation_summary()

        # Generic status
        return {
            "status": "active",
            "agent": name,
            "rounds": agent.state.total_rounds,
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
