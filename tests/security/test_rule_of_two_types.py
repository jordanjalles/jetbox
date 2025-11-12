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

    def test_write_file_is_c_only(self):
        """WriteFileToolsBehavior should be [C] only."""
        expected = {RuleOfTwoProperty.EXTERNAL_ACTION}
        assert WriteFileToolsBehavior.rule_of_two_properties == expected

    def test_read_file_is_ab(self):
        """ReadFileToolsBehavior should be [AB]."""
        expected = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS
        }
        assert ReadFileToolsBehavior.rule_of_two_properties == expected

    def test_command_is_bc(self):
        """CommandToolsBehavior should be [BC]."""
        expected = {
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }
        assert CommandToolsBehavior.rule_of_two_properties == expected

    def test_directory_is_b_only(self):
        """DirectoryToolsBehavior should be [B] only."""
        expected = {RuleOfTwoProperty.SENSITIVE_ACCESS}
        assert DirectoryToolsBehavior.rule_of_two_properties == expected

    def test_delegation_is_abc(self):
        """DelegationBehavior should be [ABC] (high risk)."""
        expected = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }
        assert DelegationBehavior.rule_of_two_properties == expected

    def test_loop_detection_is_empty(self):
        """LoopDetectionBehavior should have no properties (utility only)."""
        assert LoopDetectionBehavior.rule_of_two_properties == set()


class TestPropertyAggregation:
    """Test aggregating properties from multiple behaviors."""

    def test_aggregate_no_overlap(self):
        """Test aggregating behaviors with no overlapping properties."""
        # Write [C] + Directory [B] = [BC]
        write_props = WriteFileToolsBehavior.rule_of_two_properties
        dir_props = DirectoryToolsBehavior.rule_of_two_properties
        combined = write_props | dir_props

        assert combined == {
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }

    def test_aggregate_with_overlap(self):
        """Test aggregating behaviors with overlapping properties."""
        # Read [AB] + Command [BC] = [ABC]
        read_props = ReadFileToolsBehavior.rule_of_two_properties
        cmd_props = CommandToolsBehavior.rule_of_two_properties
        combined = read_props | cmd_props

        assert combined == {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }

    def test_aggregate_with_utility(self):
        """Test aggregating with utility behavior (no properties)."""
        # Write [C] + LoopDetection [] = [C]
        write_props = WriteFileToolsBehavior.rule_of_two_properties
        loop_props = LoopDetectionBehavior.rule_of_two_properties
        combined = write_props | loop_props

        assert combined == {RuleOfTwoProperty.EXTERNAL_ACTION}

    def test_aggregate_multiple_behaviors(self):
        """Test aggregating 3+ behaviors."""
        # Write [C] + Directory [B] + Loop [] = [BC]
        write_props = WriteFileToolsBehavior.rule_of_two_properties
        dir_props = DirectoryToolsBehavior.rule_of_two_properties
        loop_props = LoopDetectionBehavior.rule_of_two_properties

        combined = write_props | dir_props | loop_props

        assert combined == {
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }

    def test_detect_abc_trifecta(self):
        """Test detecting [ABC] trifecta."""
        # Read [AB] + Write [C] = [ABC]
        read_props = ReadFileToolsBehavior.rule_of_two_properties
        write_props = WriteFileToolsBehavior.rule_of_two_properties
        combined = read_props | write_props

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
        read_props = ReadFileToolsBehavior.rule_of_two_properties
        dir_props = DirectoryToolsBehavior.rule_of_two_properties
        combined = read_props | dir_props

        # [AB] + [B] = [AB] (compliant, <3 properties)
        assert len(combined) == 2
        assert RuleOfTwoProperty.EXTERNAL_ACTION not in combined

    def test_detect_compliant_bc(self):
        """Test detecting compliant [BC] config (not ABC)."""
        cmd_props = CommandToolsBehavior.rule_of_two_properties
        write_props = WriteFileToolsBehavior.rule_of_two_properties
        combined = cmd_props | write_props

        # [BC] + [C] = [BC] (compliant, <3 properties)
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
        props = ReadFileToolsBehavior.rule_of_two_properties
        assert RuleOfTwoProperty.UNTRUSTED_INPUT in props
        assert RuleOfTwoProperty.SENSITIVE_ACCESS in props
        assert RuleOfTwoProperty.EXTERNAL_ACTION not in props

    def test_format_properties_for_display(self):
        """Test formatting properties as [ABC] string."""
        props = DelegationBehavior.rule_of_two_properties
        letters = sorted([str(p) for p in props])
        formatted = "".join(letters)
        assert formatted == "ABC"

    def test_format_partial_properties(self):
        """Test formatting partial property sets."""
        props = ReadFileToolsBehavior.rule_of_two_properties
        letters = sorted([str(p) for p in props])
        formatted = "".join(letters)
        assert formatted == "AB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
