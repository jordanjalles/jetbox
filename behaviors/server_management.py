"""
ServerManagementBehavior - Manages development server lifecycle.

This behavior provides server management for agents that run user code
which might need dev servers (web apps, APIs, etc.).

Used by:
- Orchestrator: Coordinates servers across delegated tasks
- TaskExecutor: Runs code that might start servers

NOT used by:
- Architect: Only designs systems, doesn't run code
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from behaviors.base import AgentBehavior

if TYPE_CHECKING:
    pass


class ServerManagementBehavior(AgentBehavior):
    """
    Behavior that manages development server lifecycle.

    Provides:
    - Server initialization on demand
    - Pre-task cleanup (remove old server requests)
    - Post-execution cleanup (stop all servers)
    """

    def __init__(self):
        """Initialize server management behavior."""
        self.server_manager = None

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "server_management"

    def on_agent_init(self, agent: Any, **kwargs: Any) -> None:
        """
        Initialize server manager when agent starts.

        Args:
            agent: Agent instance
            **kwargs: Additional context
        """
        # Initialize server manager if agent supports it
        if hasattr(agent, 'init_server_manager') and not agent.server_manager:
            agent.init_server_manager()
            self.server_manager = agent.server_manager

    def pre_task_hook(self, agent: Any) -> None:
        """
        Hook called before each task in multi-task mode.

        Cleans up old server requests before starting new task.

        Args:
            agent: Agent instance
        """
        # Ensure server_manager is initialized
        if not self.server_manager and hasattr(agent, 'server_manager'):
            self.server_manager = agent.server_manager

        # Clean up old requests
        if self.server_manager:
            self.server_manager.cleanup_old_requests()

    def cleanup_hook(self, agent: Any) -> None:
        """
        Hook called at end of agent execution.

        Stops all servers and monitoring.

        Args:
            agent: Agent instance
        """
        # Ensure server_manager is initialized
        if not self.server_manager and hasattr(agent, 'server_manager'):
            self.server_manager = agent.server_manager

        # Stop all servers
        if self.server_manager:
            print("\n[Server Management] Stopping all servers...")
            self.server_manager.stop_all_servers()
            self.server_manager.stop_monitoring()
            print("All servers stopped.")
