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

    def on_goal_start(self, agent: Any, goal: str) -> None:
        """
        Initialize server manager when goal starts (STANDARD LIFECYCLE HOOK).

        Args:
            agent: Agent instance
            goal: Goal description
        """
        # Initialize server manager if agent supports it
        if hasattr(agent, 'init_server_manager'):
            if not hasattr(agent, 'server_manager') or not agent.server_manager:
                agent.init_server_manager()
            self.server_manager = agent.server_manager
        elif hasattr(agent, 'server_manager'):
            self.server_manager = agent.server_manager

        # Clean up old requests at start
        if self.server_manager:
            self.server_manager.cleanup_old_requests()

    def on_goal_complete(self, agent: Any, success: bool, summary: str) -> None:
        """
        Stop all servers when goal completes (STANDARD LIFECYCLE HOOK).

        Args:
            agent: Agent instance
            success: Whether goal succeeded
            summary: Completion summary
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
