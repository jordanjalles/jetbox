"""
Display Interface - Abstract base class for all display implementations.

This defines the contract that all display systems must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum


class EventType(Enum):
    """Types of events that can be logged."""
    INFO = "info"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    MILESTONE = "milestone"
    STATUS_UPDATE = "status_update"


@dataclass
class AgentEvent:
    """
    Structured event for display.

    This is the universal format for all agent events, regardless of display type.
    """
    type: EventType
    message: str
    details: Optional[dict[str, Any]] = None
    timestamp: Optional[str] = None


class DisplayInterface(ABC):
    """
    Abstract interface for agent display systems.

    All display implementations (Plain, Textual, etc.) must implement these methods.
    """

    @abstractmethod
    def start(self) -> None:
        """
        Initialize and start the display.

        For TUI: Enter alternate screen, setup panels
        For Plain: No-op
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Clean up and stop the display.

        For TUI: Exit alternate screen, restore terminal
        For Plain: No-op
        """
        pass

    @abstractmethod
    def update_status(
        self,
        goal: str,
        agent_name: str,
        model: str,
        current_round: int,
        max_rounds: int,
        elapsed_time: float,
        status: str,
        tokens_used: Optional[int] = None,
        tokens_max: Optional[int] = None,
    ) -> None:
        """
        Update the main status display.

        Args:
            goal: Current goal text
            agent_name: Name of the agent (e.g., "task_executor")
            model: LLM model being used
            current_round: Current round number
            max_rounds: Maximum rounds allowed
            elapsed_time: Seconds elapsed
            status: Current status (e.g., "Running", "Paused", "Completed")
            tokens_used: Tokens used so far (optional)
            tokens_max: Maximum tokens allowed (optional)
        """
        pass

    @abstractmethod
    def log_event(self, event: AgentEvent) -> None:
        """
        Log an event to the display.

        Args:
            event: Structured event to log
        """
        pass

    @abstractmethod
    def show_completion(
        self,
        success: bool,
        summary: str,
        duration: float,
        files_created: list[str],
    ) -> None:
        """
        Show task completion screen.

        Args:
            success: Whether task succeeded
            summary: Completion summary text
            duration: Total duration in seconds
            files_created: List of files created
        """
        pass

    @abstractmethod
    def prompt_user(self, question: str) -> str:
        """
        Prompt user for input.

        Args:
            question: Question to ask

        Returns:
            User's response
        """
        pass

    # Optional: For interactive displays only

    def can_pause(self) -> bool:
        """Return True if this display supports pause/resume."""
        return False

    def is_paused(self) -> bool:
        """Return True if currently paused (interactive mode only)."""
        return False

    def wait_if_paused(self) -> None:
        """Block until resumed (interactive mode only)."""
        pass
