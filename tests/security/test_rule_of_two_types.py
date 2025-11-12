"""
Unit tests for Rule of Two type system.

Tests:
- RuleOfTwoProperty enum operations
- AgentBehavior default properties
- Behavior property overrides
- Property aggregation logic
"""

import pytest
from behaviors.rule_of_two_types import RuleOfTwoProperty
from behaviors.base import AgentBehavior
from behaviors.write_file_tools import WriteFileToolsBehavior
from behaviors.read_file_tools import ReadFileToolsBehavior
from behaviors.command_tools import CommandToolsBehavior
from behaviors.directory_tools import DirectoryToolsBehavior
from behaviors.delegation import DelegationBehavior
from behaviors.loop_detection import LoopDetectionBehavior


class TestRuleOfTwoPropertyEnum:
    """Test the RuleOfTwoProperty enum."""

    def test_enum_values(self):
        """Test enum has correct values."""
        assert RuleOfTwoProperty.UNTRUSTED_INPUT.value == "A"
        assert RuleOfTwoProperty.SENSITIVE_ACCESS.value == "B"
        assert RuleOfTwoProperty.EXTERNAL_ACTION.value == "C"

    def test_enum_str(self):
        """Test enum __str__ returns single letter."""
        assert str(RuleOfTwoProperty.UNTRUSTED_INPUT) == "A"
        assert str(RuleOfTwoProperty.SENSITIVE_ACCESS) == "B"
        assert str(RuleOfTwoProperty.EXTERNAL_ACTION) == "C"

    def test_enum_repr(self):
        """Test enum __repr__ is readable."""
        assert "UNTRUSTED_INPUT" in repr(RuleOfTwoProperty.UNTRUSTED_INPUT)
        assert "SENSITIVE_ACCESS" in repr(RuleOfTwoProperty.SENSITIVE_ACCESS)
        assert "EXTERNAL_ACTION" in repr(RuleOfTwoProperty.EXTERNAL_ACTION)

    def test_enum_membership(self):
        """Test enum can be used in sets."""
        props = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS
        }
        assert RuleOfTwoProperty.UNTRUSTED_INPUT in props
        assert RuleOfTwoProperty.EXTERNAL_ACTION not in props

    def test_enum_equality(self):
        """Test enum equality."""
        a1 = RuleOfTwoProperty.UNTRUSTED_INPUT
        a2 = RuleOfTwoProperty.UNTRUSTED_INPUT
        b = RuleOfTwoProperty.SENSITIVE_ACCESS
        assert a1 == a2
        assert a1 != b


class TestAgentBehaviorDefaults:
    """Test AgentBehavior default properties."""

    def test_default_properties_abc(self):
        """Test base class has {A, B, C} default."""
        expected = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }
        assert AgentBehavior.rule_of_two_properties == expected

    def test_default_is_set(self):
        """Test default is a set type."""
        assert isinstance(AgentBehavior.rule_of_two_properties, set)

    def test_default_is_safest(self):
        """Test default includes all three properties (safest)."""
        assert len(AgentBehavior.rule_of_two_properties) == 3


class TestBehaviorPropertyOverrides:
    """Test that behaviors override properties correctly."""

    def test_write_file_is_empty(self):
        """WriteFileToolsBehavior should be [] (local writes, not network)."""
        assert WriteFileToolsBehavior.rule_of_two_properties == set()

    def test_read_file_is_dynamic(self):
        """ReadFileToolsBehavior properties are dynamic based on workspace config."""
        from behaviors.security_context import SecurityContext

        behavior = ReadFileToolsBehavior()

        # Test 1: Jetbox workspace (no untrusted, no sensitive) = []
        context = SecurityContext(
            workspace_has_untrusted_files=False,
            workspace_has_sensitive_files=False
        )
        assert behavior.get_rule_of_two_properties(None, context) == set()

        # Test 2: User workspace (untrusted + sensitive) = [AB]
        context_ab = SecurityContext(
            workspace_has_untrusted_files=True,
            workspace_has_sensitive_files=True
        )
        expected_ab = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS
        }
        assert behavior.get_rule_of_two_properties(None, context_ab) == expected_ab

    def test_command_is_dynamic(self):
        """CommandToolsBehavior is dynamic based on whitelist and workspace config."""
        from behaviors.security_context import SecurityContext

        # Test 1: Jetbox workspace with default whitelist (python, pytest, ruff) = [] (no sensitive files)
        context_jetbox = SecurityContext(
            workspace_has_untrusted_files=False,
            workspace_has_sensitive_files=False,
            workspace_has_network_access=False
        )
        behavior_jetbox = CommandToolsBehavior(whitelist={'python', 'pytest', 'ruff', 'pip'})
        props_jetbox = behavior_jetbox.get_rule_of_two_properties(None, context_jetbox)
        # Should have [] (no properties) because workspace has no sensitive files
        assert props_jetbox == set()

        # Test 1b: Same whitelist but WITH sensitive files = [B]
        context_jetbox_sensitive = SecurityContext(
            workspace_has_untrusted_files=False,
            workspace_has_sensitive_files=True,
            workspace_has_network_access=False
        )
        props_jetbox_sensitive = behavior_jetbox.get_rule_of_two_properties(None, context_jetbox_sensitive)
        # Should have [B] (sensitive access) but not [A] or [C]
        assert RuleOfTwoProperty.SENSITIVE_ACCESS in props_jetbox_sensitive
        assert RuleOfTwoProperty.UNTRUSTED_INPUT not in props_jetbox_sensitive
        assert RuleOfTwoProperty.EXTERNAL_ACTION not in props_jetbox_sensitive

        # Test 2: User workspace with file-reading commands + untrusted files = [AB]
        context_user = SecurityContext(
            workspace_has_untrusted_files=True,
            workspace_has_sensitive_files=True,
            workspace_has_network_access=False
        )
        behavior_read = CommandToolsBehavior(whitelist={'cat', 'grep', 'python'})
        props_read = behavior_read.get_rule_of_two_properties(None, context_user)
        assert RuleOfTwoProperty.UNTRUSTED_INPUT in props_read
        assert RuleOfTwoProperty.SENSITIVE_ACCESS in props_read
        assert RuleOfTwoProperty.EXTERNAL_ACTION not in props_read

        # Test 3: Network-enabled workspace with network commands = [BC]
        context_network = SecurityContext(
            workspace_has_untrusted_files=False,
            workspace_has_sensitive_files=True,  # Added for [B]
            workspace_has_network_access=True
        )
        behavior_network = CommandToolsBehavior(whitelist={'git', 'curl', 'python'})
        props_network = behavior_network.get_rule_of_two_properties(None, context_network)
        assert RuleOfTwoProperty.SENSITIVE_ACCESS in props_network
        assert RuleOfTwoProperty.EXTERNAL_ACTION in props_network
        assert RuleOfTwoProperty.UNTRUSTED_INPUT not in props_network

    def test_directory_is_dynamic(self):
        """DirectoryToolsBehavior is dynamic based on workspace config."""
        from behaviors.security_context import SecurityContext

        behavior = DirectoryToolsBehavior()

        # Test 1: Jetbox workspace (no sensitive files) = []
        context_jetbox = SecurityContext(
            workspace_has_sensitive_files=False
        )
        assert behavior.get_rule_of_two_properties(None, context_jetbox) == set()

        # Test 2: User workspace (has sensitive files) = [B]
        context_user = SecurityContext(
            workspace_has_sensitive_files=True
        )
        assert behavior.get_rule_of_two_properties(None, context_user) == {
            RuleOfTwoProperty.SENSITIVE_ACCESS
        }

    def test_delegation_is_dynamic(self):
        """DelegationBehavior properties are dynamic based on delegatable agents."""
        from behaviors.security_context import SecurityContext

        # Static fallback is empty (computed dynamically at runtime)
        assert DelegationBehavior.rule_of_two_properties == set()

        # Test dynamic computation with mock agent relationships
        agent_relationships = {
            'can_delegate_to': ['task_executor'],
            'task_executor': {
                'behaviors': [
                    {'type': 'ReadFileToolsBehavior'},
                    {'type': 'WriteFileToolsBehavior'},
                    {'type': 'CommandToolsBehavior', 'params': {'whitelist': ['python', 'pytest']}},
                ]
            }
        }

        # Test 1: No sensitive files = []
        delegation = DelegationBehavior(agent_relationships=agent_relationships)
        context_no_sensitive = SecurityContext(
            workspace_has_untrusted_files=False,
            workspace_has_sensitive_files=False,
            workspace_has_network_access=False
        )
        props_no_sensitive = delegation.get_rule_of_two_properties(None, context_no_sensitive)
        # Jetbox workspace: ReadFile=[], WriteFile=[], Command=[] (no sensitive files)
        # Should inherit [] from task_executor
        assert props_no_sensitive == set()

        # Test 2: WITH sensitive files = [B]
        context_sensitive = SecurityContext(
            workspace_has_untrusted_files=False,
            workspace_has_sensitive_files=True,
            workspace_has_network_access=False
        )
        props_sensitive = delegation.get_rule_of_two_properties(None, context_sensitive)
        # User workspace with sensitive files: ReadFile=[B], WriteFile=[], Command=[B]
        # Should inherit [B] from task_executor
        assert RuleOfTwoProperty.SENSITIVE_ACCESS in props_sensitive
        assert RuleOfTwoProperty.UNTRUSTED_INPUT not in props_sensitive
        assert RuleOfTwoProperty.EXTERNAL_ACTION not in props_sensitive

    def test_loop_detection_is_empty(self):
        """LoopDetectionBehavior should have no properties (utility only)."""
        assert LoopDetectionBehavior.rule_of_two_properties == set()


class TestPropertyAggregation:
    """Test aggregating properties from multiple behaviors."""

    def test_aggregate_no_overlap(self):
        """Test aggregating behaviors with no overlapping properties."""
        from behaviors.security_context import SecurityContext

        # Write [] + Directory [B] (in user workspace) = [B]
        context = SecurityContext(workspace_has_sensitive_files=True)
        write_props = WriteFileToolsBehavior.rule_of_two_properties
        dir_behavior = DirectoryToolsBehavior()
        dir_props = dir_behavior.get_rule_of_two_properties(None, context)
        combined = write_props | dir_props

        assert combined == {
            RuleOfTwoProperty.SENSITIVE_ACCESS
        }

    def test_aggregate_with_overlap(self):
        """Test aggregating behaviors with overlapping properties."""
        # Read [AB] + Command [BC] = [ABC] (in user workspace with untrusted/sensitive files + network)
        from behaviors.security_context import SecurityContext

        context = SecurityContext(
            workspace_has_untrusted_files=True,
            workspace_has_sensitive_files=True,
            workspace_has_network_access=True
        )
        read_behavior = ReadFileToolsBehavior()
        read_props = read_behavior.get_rule_of_two_properties(None, context)
        cmd_behavior = CommandToolsBehavior(whitelist={'cat', 'git', 'python'})
        cmd_props = cmd_behavior.get_rule_of_two_properties(None, context)
        combined = read_props | cmd_props

        assert combined == {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }

    def test_aggregate_with_utility(self):
        """Test aggregating with utility behavior (no properties)."""
        # Write [] + LoopDetection [] = []
        write_props = WriteFileToolsBehavior.rule_of_two_properties
        loop_props = LoopDetectionBehavior.rule_of_two_properties
        combined = write_props | loop_props

        assert combined == set()

    def test_aggregate_multiple_behaviors(self):
        """Test aggregating 3+ behaviors."""
        from behaviors.security_context import SecurityContext

        # Write [] + Directory [B] + Loop [] = [B] (in user workspace)
        context = SecurityContext(workspace_has_sensitive_files=True)
        write_props = WriteFileToolsBehavior.rule_of_two_properties
        dir_behavior = DirectoryToolsBehavior()
        dir_props = dir_behavior.get_rule_of_two_properties(None, context)
        loop_props = LoopDetectionBehavior.rule_of_two_properties

        combined = write_props | dir_props | loop_props

        assert combined == {
            RuleOfTwoProperty.SENSITIVE_ACCESS
        }

    def test_detect_abc_trifecta(self):
        """Test detecting [ABC] trifecta."""
        # Read [AB] + Command [BC] = [ABC] (in user workspace with network)
        from behaviors.security_context import SecurityContext

        context = SecurityContext(
            workspace_has_untrusted_files=True,
            workspace_has_sensitive_files=True,
            workspace_has_network_access=True
        )
        read_behavior = ReadFileToolsBehavior()
        read_props = read_behavior.get_rule_of_two_properties(None, context)
        cmd_behavior = CommandToolsBehavior(whitelist={'cat', 'git', 'python'})
        cmd_props = cmd_behavior.get_rule_of_two_properties(None, context)
        combined = read_props | cmd_props

        # Check if all three present
        is_abc = len(combined) == 3
        assert is_abc

        # Alternative check
        all_props = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }
        assert combined == all_props

    def test_detect_compliant_ab(self):
        """Test detecting compliant [AB] config (not ABC)."""
        from behaviors.security_context import SecurityContext

        context = SecurityContext(
            workspace_has_untrusted_files=True,
            workspace_has_sensitive_files=True,
            workspace_has_network_access=False
        )
        read_behavior = ReadFileToolsBehavior()
        read_props = read_behavior.get_rule_of_two_properties(None, context)
        dir_behavior = DirectoryToolsBehavior()
        dir_props = dir_behavior.get_rule_of_two_properties(None, context)
        combined = read_props | dir_props

        # [AB] + [B] = [AB] (compliant, <3 properties)
        assert len(combined) == 2
        assert RuleOfTwoProperty.EXTERNAL_ACTION not in combined

    def test_detect_compliant_bc(self):
        """Test detecting compliant [BC] config (not ABC)."""
        from behaviors.security_context import SecurityContext

        context = SecurityContext(
            workspace_has_untrusted_files=False,
            workspace_has_sensitive_files=True,  # Added for [B]
            workspace_has_network_access=True
        )
        cmd_behavior = CommandToolsBehavior(whitelist={'git', 'python'})
        cmd_props = cmd_behavior.get_rule_of_two_properties(None, context)
        write_props = WriteFileToolsBehavior.rule_of_two_properties
        combined = cmd_props | write_props

        # [BC] + [] = [BC] (compliant, <3 properties)
        assert len(combined) == 2
        assert RuleOfTwoProperty.UNTRUSTED_INPUT not in combined

    def test_detect_compliant_ac(self):
        """Test detecting compliant [AC] config (not ABC)."""
        # Hypothetical: UNTRUSTED_INPUT + EXTERNAL_ACTION (no sensitive access)
        # We don't have this combo in real behaviors yet, test the logic
        props_ac = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }
        assert len(props_ac) == 2
        assert RuleOfTwoProperty.SENSITIVE_ACCESS not in props_ac


class TestPropertyInspection:
    """Test inspecting properties for validation."""

    def test_get_property_count(self):
        """Test counting unique properties."""
        props = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS
        }
        assert len(props) == 2

    def test_check_specific_property(self):
        """Test checking if specific property present."""
        from behaviors.security_context import SecurityContext

        # ReadFile in user workspace with untrusted/sensitive
        context = SecurityContext(
            workspace_has_untrusted_files=True,
            workspace_has_sensitive_files=True
        )
        read_behavior = ReadFileToolsBehavior()
        props = read_behavior.get_rule_of_two_properties(None, context)

        assert RuleOfTwoProperty.UNTRUSTED_INPUT in props
        assert RuleOfTwoProperty.SENSITIVE_ACCESS in props
        assert RuleOfTwoProperty.EXTERNAL_ACTION not in props

    def test_format_properties_for_display(self):
        """Test formatting properties as [BC] string for CommandTools."""
        from behaviors.security_context import SecurityContext

        # CommandTools with network access and sensitive files gets [BC]
        context = SecurityContext(
            workspace_has_network_access=True,
            workspace_has_sensitive_files=True
        )
        cmd_behavior = CommandToolsBehavior(whitelist={'git', 'python'})
        props = cmd_behavior.get_rule_of_two_properties(None, context)
        letters = sorted([str(p) for p in props])
        formatted = "".join(letters)
        assert formatted == "BC"

    def test_format_partial_properties(self):
        """Test formatting partial property sets."""
        from behaviors.security_context import SecurityContext

        # ReadFile in user workspace
        context = SecurityContext(
            workspace_has_untrusted_files=True,
            workspace_has_sensitive_files=True
        )
        read_behavior = ReadFileToolsBehavior()
        props = read_behavior.get_rule_of_two_properties(None, context)
        letters = sorted([str(p) for p in props])
        formatted = "".join(letters)
        assert formatted == "AB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
