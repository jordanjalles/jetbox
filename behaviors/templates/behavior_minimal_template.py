"""
{BEHAVIOR_NAME}Behavior - {ONE_SENTENCE_DESCRIPTION}

Features:
- {FEATURE_1}
- {FEATURE_2}

This is a MINIMAL behavior showing the simplest possible implementation.
"""

from typing import Any
from behaviors.base import AgentBehavior


class {BEHAVIOR_CLASS_NAME}Behavior(AgentBehavior):
    """
    {DETAILED_DESCRIPTION}

    This behavior provides: {WHAT_IT_PROVIDES}
    This behavior does NOT: {WHAT_IT_DOES_NOT_DO}
    """

    def __init__(self, **kwargs):
        """
        Initialize {BEHAVIOR_NAME} behavior.

        Args:
            **kwargs: Additional parameters (for flexibility)
        """
        # Initialize any state here
        pass

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "{BEHAVIOR_NAME}"

    # Override ONLY the hooks you need below
    # Delete unused methods - keep it minimal

    # If this behavior provides tools, add:
    # 1. get_tools() - returns list of tool definitions
    # 2. dispatch_tool(agent, tool_name, args) - handles tool execution
    # 3. on_initial_context(agent, context) - injects tool documentation
    #    See behavior_with_tools_template.py for the full pattern
