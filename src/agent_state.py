"""
Agent state management module.

Handles state persistence, serialization, and file I/O for agent state.
This module is stateless - it manages state files but doesn't hold state itself.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import json


@dataclass
class AgentState:
    """Base state that all agents maintain."""
    name: str
    role: str
    messages: list[dict[str, Any]]
    start_time: float
    total_rounds: int

    def _serialize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert message to JSON-serializable format."""
        serialized = {}
        for key, value in message.items():
            if key == "tool_calls" and value is not None:
                # Convert ToolCall objects to dicts
                serialized_calls = []
                for tc in value:
                    if hasattr(tc, "model_dump"):
                        # Pydantic model
                        serialized_calls.append(tc.model_dump())
                    elif hasattr(tc, "to_dict"):
                        serialized_calls.append(tc.to_dict())
                    elif isinstance(tc, dict):
                        serialized_calls.append(tc)
                    else:
                        # Try to extract attributes manually
                        serialized_calls.append({
                            "id": getattr(tc, "id", None),
                            "type": getattr(tc, "type", "function"),
                            "function": {
                                "name": getattr(tc.function, "name", ""),
                                "arguments": getattr(tc.function, "arguments", {})
                            } if hasattr(tc, "function") else {}
                        })
                serialized[key] = serialized_calls
            else:
                serialized[key] = value
        return serialized

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "name": self.name,
            "role": self.role,
            "messages": [self._serialize_message(msg) for msg in self.messages],
            "start_time": self.start_time,
            "total_rounds": self.total_rounds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """Deserialize state from dictionary."""
        return cls(**data)


class StatePersistence:
    """
    Handles state file I/O operations.

    This class is stateless - it manages state files but doesn't hold state itself.
    All state is passed in as parameters or returned as results.
    """

    def __init__(self, workspace: Path):
        """
        Initialize state persistence manager.

        Args:
            workspace: Agent workspace directory
        """
        self.workspace = Path(workspace)

    def get_state_file_path(self, agent_name: str) -> Path:
        """
        Get the path to the state file for a given agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to the state file
        """
        state_file = self.workspace / ".agent_context" / f"{agent_name}_state.json"
        # Ensure directory exists
        state_file.parent.mkdir(parents=True, exist_ok=True)
        return state_file

    def persist(self, state: AgentState) -> None:
        """
        Save agent state to disk.

        Args:
            state: AgentState instance to persist
        """
        state_file = self.get_state_file_path(state.name)
        with open(state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def load_state(self, agent_name: str) -> AgentState | None:
        """
        Load agent state from disk if it exists.

        Args:
            agent_name: Name of the agent

        Returns:
            AgentState instance if file exists and is valid, None otherwise
        """
        state_file = self.get_state_file_path(agent_name)
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    return AgentState.from_dict(data)
            except Exception:
                # If load fails, return None (caller will create fresh state)
                return None
        return None

    def state_exists(self, agent_name: str) -> bool:
        """
        Check if a state file exists for the given agent.

        Args:
            agent_name: Name of the agent

        Returns:
            True if state file exists, False otherwise
        """
        state_file = self.get_state_file_path(agent_name)
        return state_file.exists()

    def delete_state(self, agent_name: str) -> bool:
        """
        Delete the state file for the given agent.

        Args:
            agent_name: Name of the agent

        Returns:
            True if file was deleted, False if it didn't exist
        """
        state_file = self.get_state_file_path(agent_name)
        if state_file.exists():
            state_file.unlink()
            return True
        return False
