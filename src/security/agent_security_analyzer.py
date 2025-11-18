"""
Agent Security Analyzer - Computes Rule of Two properties for agents.

This module provides security analysis functionality for analyzing agent
configurations and computing their aggregate security properties based on
the Rule of Two security model.

Extracted from DelegationBehavior to separate security analysis concerns
from delegation functionality.
"""

import importlib
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class AgentSecurityAnalyzer:
    """
    Analyzes agent configurations to compute Rule of Two security properties.

    This analyzer is responsible for computing the aggregate security properties
    of an agent based on its configured behaviors. It dynamically instantiates
    behaviors and queries their Rule of Two properties to determine the overall
    security posture of an agent.

    The analyzer is stateless and can be used as a utility for computing
    properties for any agent configuration.

    Key Responsibilities:
    - Convert behavior type names to module paths
    - Dynamically import and instantiate behavior classes
    - Query behaviors for their security properties
    - Aggregate properties across all behaviors in a configuration

    Usage:
        analyzer = AgentSecurityAnalyzer()
        props = analyzer.compute_aggregate_properties(
            "task_executor",
            behaviors_config,
            security_context
        )
    """

    def compute_aggregate_properties(
        self,
        agent_name: str,
        behaviors_config: list[dict[str, Any]],
        security_context: Any | None  # SecurityContext | None
    ) -> set:
        """
        Compute aggregate security properties for a target agent.

        Creates behavior instances from the provided configuration and calls
        their dynamic get_rule_of_two_properties() method to compute
        context-aware properties.

        This method handles behavior instantiation errors gracefully, skipping
        behaviors that cannot be loaded or instantiated. This is expected for
        behaviors that may not be present in all configurations.

        Args:
            agent_name: Name of target agent (e.g., "task_executor", "architect")
            behaviors_config: List of behavior specs from agent config
                Each spec should have format:
                {
                    "type": "BehaviorClassName",
                    "params": {...}  # Optional
                }
            security_context: SecurityContext to pass to behaviors for
                context-aware property resolution. If None, a default context
                is created with no untrusted files, no sensitive files, and
                no network access (safe defaults for Jetbox environment).

        Returns:
            Set of aggregated Rule of Two properties across all behaviors.
            Each property is a RuleOfTwoProperty enum value (A, B, or C).

        Example:
            >>> analyzer = AgentSecurityAnalyzer()
            >>> context = SecurityContext(
            ...     workspace_has_untrusted_files=False,
            ...     workspace_has_sensitive_files=False,
            ...     workspace_has_network_access=True
            ... )
            >>> behaviors = [
            ...     {"type": "FileToolsBehavior", "params": {}},
            ...     {"type": "CommandToolsBehavior", "params": {}}
            ... ]
            >>> props = analyzer.compute_aggregate_properties(
            ...     "task_executor", behaviors, context
            ... )
            >>> # props might be {RuleOfTwoProperty.EXTERNAL_ACTION}
        """
        props = set()

        # Use provided context, or create default for Jetbox environment if None
        # Default context: no untrusted files, no sensitive files, no network access
        # This is the safe default for Jetbox's own codebase
        if security_context is None:
            # Import SecurityContext lazily to avoid circular imports
            from behaviors.security_context import SecurityContext
            security_context = SecurityContext(
                workspace_has_untrusted_files=False,
                workspace_has_sensitive_files=False,
                workspace_has_network_access=False
            )

        for behavior_spec in behaviors_config:
            behavior_type = behavior_spec.get("type")
            if not behavior_type:
                continue

            try:
                # Import behavior class
                module_name = self.behavior_type_to_module(behavior_type)
                module = importlib.import_module(module_name)
                behavior_class = getattr(module, behavior_type)

                # Instantiate behavior with params if provided
                params = behavior_spec.get("params", {})
                try:
                    behavior_instance = behavior_class(**params)
                except TypeError:
                    # Some behaviors don't accept params, try without
                    behavior_instance = behavior_class()

                # Call dynamic method to get properties
                if hasattr(behavior_instance, "get_rule_of_two_properties"):
                    behavior_props = behavior_instance.get_rule_of_two_properties(
                        None,  # agent parameter (not needed for static analysis)
                        security_context
                    )
                    if isinstance(behavior_props, set):
                        props.update(behavior_props)

            except (ImportError, AttributeError, TypeError):
                # Behavior class not found or instantiation failed
                # Silently skip (this is expected for some behaviors)
                # Not all behaviors have security properties or may not be installed
                pass

        return props

    def behavior_type_to_module(self, behavior_type: str) -> str:
        """
        Convert behavior type name to module path.

        Transforms CamelCase behavior class names to snake_case module paths
        following Jetbox naming conventions. Removes "Behavior" suffix and
        converts to snake_case.

        Args:
            behavior_type: CamelCase behavior class name
                Example: "ReadFileToolsBehavior"

        Returns:
            Module path in behaviors package
                Example: "behaviors.read_file_tools"

        Examples:
            >>> analyzer = AgentSecurityAnalyzer()
            >>> analyzer.behavior_type_to_module("ReadFileToolsBehavior")
            'behaviors.read_file_tools'
            >>> analyzer.behavior_type_to_module("DelegationBehavior")
            'behaviors.delegation'
            >>> analyzer.behavior_type_to_module("CommandToolsBehavior")
            'behaviors.command_tools'
        """
        # Remove "Behavior" suffix
        name = behavior_type.replace("Behavior", "")

        # Convert CamelCase to snake_case
        # Insert underscore before uppercase letters (except at start)
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

        return f"behaviors.{snake}"
