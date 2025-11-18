"""Tests for AgentSecurityAnalyzer module."""

import pytest
from src.security.agent_security_analyzer import AgentSecurityAnalyzer
from behaviors.security_context import SecurityContext
from behaviors.rule_of_two_types import RuleOfTwoProperty


class TestAgentSecurityAnalyzer:
    """Test suite for AgentSecurityAnalyzer."""

    def test_analyzer_instantiation(self):
        """Test that analyzer can be instantiated."""
        analyzer = AgentSecurityAnalyzer()
        assert analyzer is not None

    def test_behavior_type_to_module_conversion(self):
        """Test conversion of behavior type names to module paths."""
        analyzer = AgentSecurityAnalyzer()

        # Test various behavior name conversions
        assert analyzer.behavior_type_to_module("ReadFileToolsBehavior") == "behaviors.read_file_tools"
        assert analyzer.behavior_type_to_module("DelegationBehavior") == "behaviors.delegation"
        assert analyzer.behavior_type_to_module("CommandToolsBehavior") == "behaviors.command_tools"
        assert analyzer.behavior_type_to_module("FileToolsBehavior") == "behaviors.file_tools"

    def test_compute_aggregate_properties_empty_config(self):
        """Test computing properties with empty behaviors config."""
        analyzer = AgentSecurityAnalyzer()
        context = SecurityContext()

        props = analyzer.compute_aggregate_properties(
            "test_agent",
            [],
            context
        )

        assert isinstance(props, set)
        assert len(props) == 0

    def test_compute_aggregate_properties_with_file_behavior(self):
        """Test computing properties with file tools behavior."""
        analyzer = AgentSecurityAnalyzer()
        context = SecurityContext(
            workspace_has_untrusted_files=True,  # Enable untrusted files
            workspace_has_sensitive_files=False,
            workspace_has_network_access=False
        )

        behaviors_config = [
            {
                "type": "ReadFileToolsBehavior",  # ReadFile has dynamic properties
                "params": {}
            }
        ]

        props = analyzer.compute_aggregate_properties(
            "test_agent",
            behaviors_config,
            context
        )

        # ReadFileToolsBehavior should provide UNTRUSTED_INPUT when untrusted files present
        assert isinstance(props, set)
        # Properties may include UNTRUSTED_INPUT due to untrusted workspace
        assert RuleOfTwoProperty.UNTRUSTED_INPUT in props

    def test_compute_aggregate_properties_with_default_context(self):
        """Test that None security_context creates safe defaults."""
        analyzer = AgentSecurityAnalyzer()

        behaviors_config = [
            {
                "type": "ReadFileToolsBehavior",
                "params": {}
            }
        ]

        # Pass None - should create default SecurityContext
        props = analyzer.compute_aggregate_properties(
            "test_agent",
            behaviors_config,
            None  # Should create default context with no untrusted files
        )

        # With default context (no untrusted files), ReadFile should not have UNTRUSTED_INPUT
        assert isinstance(props, set)
        assert RuleOfTwoProperty.UNTRUSTED_INPUT not in props

    def test_compute_aggregate_properties_invalid_behavior(self):
        """Test that invalid behavior types are handled gracefully."""
        analyzer = AgentSecurityAnalyzer()
        context = SecurityContext()

        behaviors_config = [
            {
                "type": "NonExistentBehavior",
                "params": {}
            }
        ]

        # Should not crash, just return empty set
        props = analyzer.compute_aggregate_properties(
            "test_agent",
            behaviors_config,
            context
        )

        assert isinstance(props, set)

    def test_compute_aggregate_properties_mixed_behaviors(self):
        """Test computing properties with multiple behaviors."""
        analyzer = AgentSecurityAnalyzer()
        context = SecurityContext(
            workspace_has_untrusted_files=False,
            workspace_has_sensitive_files=False,
            workspace_has_network_access=True
        )

        behaviors_config = [
            {"type": "ReadFileToolsBehavior", "params": {}},
            {"type": "CommandToolsBehavior", "params": {}},
        ]

        props = analyzer.compute_aggregate_properties(
            "test_agent",
            behaviors_config,
            context
        )

        # Should aggregate properties from both behaviors
        assert isinstance(props, set)
        # With network access, CommandTools provides EXTERNAL_ACTION
        assert RuleOfTwoProperty.EXTERNAL_ACTION in props

    def test_analyzer_is_stateless(self):
        """Test that analyzer maintains no state between calls."""
        analyzer = AgentSecurityAnalyzer()
        context = SecurityContext()

        behaviors_config = [
            {"type": "ReadFileToolsBehavior", "params": {}}
        ]

        # Call twice - should get same results
        props1 = analyzer.compute_aggregate_properties("agent1", behaviors_config, context)
        props2 = analyzer.compute_aggregate_properties("agent2", behaviors_config, context)

        assert props1 == props2
        assert props1 is not props2  # Different set objects


class TestIntegrationWithDelegationBehavior:
    """Test that DelegationBehavior correctly uses AgentSecurityAnalyzer."""

    def test_delegation_behavior_uses_analyzer(self):
        """Test that DelegationBehavior computes properties using analyzer."""
        from behaviors.delegation import DelegationBehavior

        # Create delegation behavior with mock agent relationships
        agent_relationships = {
            "can_delegate_to": ["task_executor"],
            "task_executor": {
                "class": "TaskExecutorAgent",
                "description": "Executes tasks",
                "behaviors": [
                    {"type": "CommandToolsBehavior", "params": {}}
                ]
            }
        }

        behavior = DelegationBehavior(agent_relationships)
        context = SecurityContext(workspace_has_network_access=True)

        # Get Rule of Two properties
        props = behavior.get_rule_of_two_properties(None, context)

        # Should inherit properties from task_executor
        assert isinstance(props, set)
        # task_executor has CommandToolsBehavior which provides EXTERNAL_ACTION with network
        assert RuleOfTwoProperty.EXTERNAL_ACTION in props

    def test_delegation_behavior_aggregates_multiple_agents(self):
        """Test that DelegationBehavior aggregates properties from all delegatable agents."""
        from behaviors.delegation import DelegationBehavior

        agent_relationships = {
            "can_delegate_to": ["agent1", "agent2"],
            "agent1": {
                "behaviors": [
                    {"type": "ReadFileToolsBehavior", "params": {}}
                ]
            },
            "agent2": {
                "behaviors": [
                    {"type": "CommandToolsBehavior", "params": {}}
                ]
            }
        }

        behavior = DelegationBehavior(agent_relationships)
        context = SecurityContext(
            workspace_has_untrusted_files=True,
            workspace_has_network_access=True
        )

        props = behavior.get_rule_of_two_properties(None, context)

        # Should aggregate from both agents
        assert isinstance(props, set)
        # Should have both UNTRUSTED_INPUT (from agent1) and EXTERNAL_ACTION (from agent2)
        assert RuleOfTwoProperty.UNTRUSTED_INPUT in props
        assert RuleOfTwoProperty.EXTERNAL_ACTION in props
