"""
Test behavior for validating CLI injection system (Phase 2).

This behavior does nothing except print when it's loaded.
Used to verify that --TestCliInjector flag works.
"""
from behaviors.base import AgentBehavior


class TestCliInjectorBehavior(AgentBehavior):
    """
    Test behavior for CLI injection validation.

    Does nothing except announce itself when loaded.
    """

    def __init__(self, **params):
        """Initialize test behavior."""
        super().__init__()
        self.test_message = params.get("test_message", "CLI injection works!")

    def get_name(self) -> str:
        """Return behavior name."""
        return "test_cli_injector"

    def get_system_instructions(self) -> str:
        """Return system instructions."""
        return f"[TestCliInjectorBehavior loaded: {self.test_message}]"
