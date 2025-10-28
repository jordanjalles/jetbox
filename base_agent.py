"""
Base agent class providing common functionality for all agent types.

All agents inherit from BaseAgent and implement:
- get_tools(): Returns list of tools available to this agent
- get_system_prompt(): Returns the system prompt for this agent
- get_context_strategy(): Returns context management strategy name
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import json
import time
from datetime import datetime


@dataclass
class AgentState:
    """Base state that all agents maintain."""
    name: str
    role: str
    messages: list[dict[str, Any]]
    start_time: float
    total_rounds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "messages": self.messages,
            "start_time": self.start_time,
            "total_rounds": self.total_rounds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        return cls(**data)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Provides common functionality:
    - LLM calling with tool support
    - Message history management
    - State persistence
    - Tool dispatch

    Subclasses must implement:
    - get_tools(): Tool definitions for this agent
    - get_system_prompt(): System prompt for this agent
    - get_context_strategy(): Context management strategy
    """

    def __init__(
        self,
        name: str,
        role: str,
        workspace: Path,
        config: Any = None,
    ):
        """
        Initialize base agent.

        Args:
            name: Agent identifier (e.g., "orchestrator", "task_executor")
            role: Human-readable role description
            workspace: Working directory for this agent
            config: Agent configuration (from agent_config.py)
        """
        self.name = name
        self.role = role
        self.workspace = Path(workspace)
        self.config = config

        # Create workspace if needed
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self.state = AgentState(
            name=name,
            role=role,
            messages=[],
            start_time=time.time(),
            total_rounds=0,
        )

        # State file location
        self.state_file = self.workspace / ".agent_context" / f"{name}_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Try to load existing state
        self.load_state()

    # ===========================
    # Abstract methods (must implement)
    # ===========================

    @abstractmethod
    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return tool definitions for this agent.

        Returns:
            List of tool definitions in Ollama format
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return system prompt for this agent.

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def get_context_strategy(self) -> str:
        """
        Return context management strategy name.

        Options:
            "hierarchical" - Keep last N exchanges (TaskExecutor)
            "append_until_full" - Append until token limit, then compact (Orchestrator)

        Returns:
            Strategy name
        """
        pass

    @abstractmethod
    def build_context(self) -> list[dict[str, Any]]:
        """
        Build context for LLM call using this agent's strategy.

        Returns:
            List of messages to send to LLM
        """
        pass

    # ===========================
    # Shared functionality
    # ===========================

    def call_llm(
        self,
        model: str,
        temperature: float,
        timeout: int = 120,
    ) -> dict[str, Any]:
        """
        Call LLM with current context and tools.

        Args:
            model: Model name (e.g., "gpt-oss:20b")
            temperature: Sampling temperature
            timeout: Timeout in seconds

        Returns:
            LLM response dict with 'message' key
        """
        from ollama import chat
        from llm_utils import chat_with_inactivity_timeout

        context = self.build_context()
        tools = self.get_tools()

        try:
            response = chat_with_inactivity_timeout(
                model=model,
                messages=context,
                options={"temperature": temperature},
                tools=tools,
                inactivity_timeout=timeout,
            )
            return response
        except Exception as e:
            return {
                "message": {
                    "role": "assistant",
                    "content": f"LLM call failed: {e}",
                }
            }

    def dispatch_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch a tool call to the appropriate handler.

        Args:
            tool_call: Tool call dict with function name and arguments

        Returns:
            Tool result dict
        """
        # This will be implemented by importing dispatch from agent.py
        # or by having each agent implement their own dispatch
        from agent import dispatch
        return dispatch(tool_call)

    def persist_state(self) -> None:
        """Save agent state to disk."""
        with open(self.state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def load_state(self) -> None:
        """Load agent state from disk if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    self.state = AgentState.from_dict(data)
            except Exception:
                # If load fails, keep fresh state
                pass

    def add_message(self, message: dict[str, Any]) -> None:
        """
        Add a message to history.

        Args:
            message: Message dict with role and content
        """
        self.state.messages.append(message)

    def get_message_history(self) -> list[dict[str, Any]]:
        """Get full message history."""
        return self.state.messages

    def clear_messages(self) -> None:
        """Clear message history (useful for fresh starts)."""
        self.state.messages = []

    def increment_round(self) -> None:
        """Increment round counter."""
        self.state.total_rounds += 1
