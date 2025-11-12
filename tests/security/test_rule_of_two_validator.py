"""
Unit tests for RuleOfTwoValidator behavior.

Tests:
- Property collection from behaviors (context-aware)
- [ABC] trifecta detection
- Dynamic property resolution based on workspace trust
- Risk acknowledgment checking
- Error message generation
- Enforcement levels (off/warn/block)
"""

import pytest
from unittest.mock import Mock, MagicMock
from behaviors.rule_of_two_validator import RuleOfTwoValidator, SecurityViolationError
from behaviors.rule_of_two_types import RuleOfTwoProperty
from behaviors.security_context import SecurityContext
from behaviors.base import AgentBehavior


# Mock behaviors for testing
class MockReadBehavior(AgentBehavior):
    """Mock read behavior with [AB] properties."""
    rule_of_two_properties = {
        RuleOfTwoProperty.UNTRUSTED_INPUT,
        RuleOfTwoProperty.SENSITIVE_ACCESS
    }

    def get_name(self):
        return "mock_read"


class MockWriteBehavior(AgentBehavior):
    """Mock write behavior with [C] property."""
    rule_of_two_properties = {RuleOfTwoProperty.EXTERNAL_ACTION}

    def get_name(self):
        return "mock_write"


class MockDynamicBehavior(AgentBehavior):
    """Mock behavior with dynamic properties based on context."""
    rule_of_two_properties = {
        RuleOfTwoProperty.UNTRUSTED_INPUT,
        RuleOfTwoProperty.SENSITIVE_ACCESS
    }

    def get_name(self):
        return "mock_dynamic"

    def get_rule_of_two_properties(self, agent, security_context):
        # Dynamic: remove [A] in isolated workspace
        props = {RuleOfTwoProperty.SENSITIVE_ACCESS}
        if security_context and security_context.workspace_trust_level == "user":
            props.add(RuleOfTwoProperty.UNTRUSTED_INPUT)
        return props


class TestPropertyCollection:
    """Test collecting properties from behaviors."""

    def test_collect_single_behavior(self):
        """Test collecting properties from single behavior."""
        validator = RuleOfTwoValidator(security_config={})
        agent = Mock()
        agent.name = "test"
        agent.behaviors = [MockWriteBehavior()]
        context = SecurityContext()

        props = validator._collect_properties(agent, context)

        assert props == {RuleOfTwoProperty.EXTERNAL_ACTION}

    def test_collect_multiple_behaviors(self):
        """Test aggregating properties from multiple behaviors."""
        validator = RuleOfTwoValidator(security_config={})
        agent = Mock()
        agent.name = "test"
        agent.behaviors = [MockReadBehavior(), MockWriteBehavior()]
        context = SecurityContext()

        props = validator._collect_properties(agent, context)

        # Should aggregate [AB] + [C] = [ABC]
        assert props == {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }

    def test_collect_skips_validator_itself(self):
        """Test that validator doesn't include its own properties."""
        validator = RuleOfTwoValidator(security_config={})
        agent = Mock()
        agent.name = "test"
        agent.behaviors = [MockWriteBehavior(), validator]
        context = SecurityContext()

        props = validator._collect_properties(agent, context)

        # Should only have [C] from MockWriteBehavior, not validator's empty set
        assert props == {RuleOfTwoProperty.EXTERNAL_ACTION}

    def test_collect_dynamic_properties_user_workspace(self):
        """Test dynamic properties in user workspace."""
        validator = RuleOfTwoValidator(security_config={})
        agent = Mock()
        agent.name = "test"
        agent.behaviors = [MockDynamicBehavior()]
        context = SecurityContext(workspace_trust_level="user")

        props = validator._collect_properties(agent, context)

        # In user workspace: [AB]
        assert props == {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS
        }

    def test_collect_dynamic_properties_isolated_workspace(self):
        """Test dynamic properties in isolated workspace."""
        validator = RuleOfTwoValidator(security_config={})
        agent = Mock()
        agent.name = "test"
        agent.behaviors = [MockDynamicBehavior()]
        context = SecurityContext(workspace_trust_level="isolated")

        props = validator._collect_properties(agent, context)

        # In isolated workspace: [B] only (no [A])
        assert props == {RuleOfTwoProperty.SENSITIVE_ACCESS}


class TestABCTrifectaDetection:
    """Test detecting [ABC] trifecta."""

    def test_detect_abc_trifecta_true(self):
        """Test detection when all three properties present."""
        validator = RuleOfTwoValidator(security_config={})
        props = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }

        is_abc = validator._detect_abc_trifecta(props)

        assert is_abc is True

    def test_detect_abc_trifecta_false_ab(self):
        """Test detection with only [AB]."""
        validator = RuleOfTwoValidator(security_config={})
        props = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS
        }

        is_abc = validator._detect_abc_trifecta(props)

        assert is_abc is False

    def test_detect_abc_trifecta_false_bc(self):
        """Test detection with only [BC]."""
        validator = RuleOfTwoValidator(security_config={})
        props = {
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }

        is_abc = validator._detect_abc_trifecta(props)

        assert is_abc is False

    def test_detect_abc_trifecta_false_empty(self):
        """Test detection with no properties."""
        validator = RuleOfTwoValidator(security_config={})
        props = set()

        is_abc = validator._detect_abc_trifecta(props)

        assert is_abc is False


class TestPropertyFormatting:
    """Test formatting properties for display."""

    def test_format_abc(self):
        """Test formatting [ABC]."""
        validator = RuleOfTwoValidator(security_config={})
        props = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }

        formatted = validator._format_properties_display(props)

        assert formatted == "[ABC]"

    def test_format_ab(self):
        """Test formatting [AB]."""
        validator = RuleOfTwoValidator(security_config={})
        props = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS
        }

        formatted = validator._format_properties_display(props)

        assert formatted == "[AB]"

    def test_format_single(self):
        """Test formatting single property."""
        validator = RuleOfTwoValidator(security_config={})
        props = {RuleOfTwoProperty.EXTERNAL_ACTION}

        formatted = validator._format_properties_display(props)

        assert formatted == "[C]"

    def test_format_empty(self):
        """Test formatting empty set."""
        validator = RuleOfTwoValidator(security_config={})
        props = set()

        formatted = validator._format_properties_display(props)

        assert formatted == "[]"


class TestRiskAcknowledgment:
    """Test risk acknowledgment checking."""

    def test_risk_acknowledged_in_config(self):
        """Test detecting risk acknowledgment in security config."""
        security_config = {
            "rule_of_two": {
                "acknowledge_abc_risk": True
            }
        }
        validator = RuleOfTwoValidator(security_config=security_config)
        agent = Mock()
        agent.config = None

        acknowledged = validator._check_risk_acknowledged(agent)

        assert acknowledged is True

    def test_risk_not_acknowledged(self):
        """Test no acknowledgment in config."""
        security_config = {
            "rule_of_two": {
                "acknowledge_abc_risk": False
            }
        }
        validator = RuleOfTwoValidator(security_config=security_config)
        agent = Mock()
        agent.config = None

        acknowledged = validator._check_risk_acknowledged(agent)

        assert acknowledged is False

    def test_risk_acknowledged_missing_key(self):
        """Test missing acknowledgment key defaults to False."""
        security_config = {
            "rule_of_two": {}
        }
        validator = RuleOfTwoValidator(security_config=security_config)
        agent = Mock()
        agent.config = None

        acknowledged = validator._check_risk_acknowledged(agent)

        assert acknowledged is False


class TestDefenseLayerSkip:
    """Test checking if defense layers should be skipped."""

    def test_skip_defense_layers_true(self):
        """Test skipping defense layers when configured."""
        security_config = {
            "rule_of_two": {
                "skip_defense_in_depth": True
            }
        }
        validator = RuleOfTwoValidator(security_config=security_config)
        agent = Mock()
        agent.config = None

        should_skip = validator._should_skip_defense_layers(agent)

        assert should_skip is True

    def test_skip_defense_layers_false(self):
        """Test not skipping when not configured."""
        security_config = {
            "rule_of_two": {
                "skip_defense_in_depth": False
            }
        }
        validator = RuleOfTwoValidator(security_config=security_config)
        agent = Mock()
        agent.config = None

        should_skip = validator._should_skip_defense_layers(agent)

        assert should_skip is False


class TestErrorMessageGeneration:
    """Test rich error message generation."""

    def test_error_message_includes_properties(self):
        """Test error message includes property display."""
        validator = RuleOfTwoValidator(security_config={})
        agent = Mock()
        agent.name = "test_agent"
        agent.security_context = SecurityContext(workspace_trust_level="user")
        agent.behaviors = [MockReadBehavior(), MockWriteBehavior()]

        props = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }

        message = validator._get_rich_error_message(props, agent)

        assert "[ABC]" in message
        assert "RULE OF TWO SECURITY VIOLATION" in message
        assert "SOLUTION OPTIONS" in message

    def test_error_message_includes_behaviors(self):
        """Test error message lists behaviors with properties."""
        validator = RuleOfTwoValidator(security_config={})
        agent = Mock()
        agent.name = "test_agent"
        agent.security_context = SecurityContext()
        agent.behaviors = [MockReadBehavior(), MockWriteBehavior()]

        props = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }

        message = validator._get_rich_error_message(props, agent)

        assert "mock_read" in message
        assert "mock_write" in message


class TestValidationWorkflow:
    """Test end-to-end validation workflows."""

    def test_validation_passes_for_ab(self):
        """Test validation passes for [AB] agent."""
        validator = RuleOfTwoValidator(security_config={"enabled": True})
        agent = Mock()
        agent.name = "test_agent"
        agent.security_context = SecurityContext()
        agent.behaviors = [MockReadBehavior(), validator]

        # Should not raise
        validator.validate_agent_configuration(agent)

        assert validator.has_abc_trifecta is False

    def test_validation_passes_for_bc(self):
        """Test validation passes for [BC] agent."""
        validator = RuleOfTwoValidator(security_config={"enabled": True})
        agent = Mock()
        agent.name = "test_agent"
        agent.security_context = SecurityContext()
        # Create BC behavior
        bc_behavior = MockWriteBehavior()
        bc_behavior.rule_of_two_properties = {
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }
        agent.behaviors = [bc_behavior, validator]

        # Should not raise
        validator.validate_agent_configuration(agent)

        assert validator.has_abc_trifecta is False

    def test_validation_raises_for_abc_without_ack(self):
        """Test validation raises SecurityViolationError for [ABC] without acknowledgment."""
        security_config = {
            "enabled": True,
            "rule_of_two": {
                "enforcement": "block",
                "acknowledge_abc_risk": False
            }
        }
        validator = RuleOfTwoValidator(security_config=security_config)
        agent = Mock()
        agent.name = "test_agent"
        agent.security_context = SecurityContext()
        agent.config = None
        agent.behaviors = [MockReadBehavior(), MockWriteBehavior(), validator]

        with pytest.raises(SecurityViolationError) as exc_info:
            validator.validate_agent_configuration(agent)

        assert "[ABC]" in str(exc_info.value)
        assert validator.has_abc_trifecta is True

    def test_validation_warns_for_abc_with_warn_enforcement(self):
        """Test validation warns but doesn't raise with 'warn' enforcement."""
        security_config = {
            "enabled": True,
            "rule_of_two": {
                "enforcement": "warn",
                "acknowledge_abc_risk": False
            }
        }
        validator = RuleOfTwoValidator(security_config=security_config)
        agent = Mock()
        agent.name = "test_agent"
        agent.security_context = SecurityContext()
        agent.config = None
        agent.behaviors = [MockReadBehavior(), MockWriteBehavior(), validator]

        # Should not raise (warns instead)
        validator.validate_agent_configuration(agent)

        assert validator.has_abc_trifecta is True

    def test_validation_passes_for_abc_with_ack(self):
        """Test validation passes for [ABC] with risk acknowledged."""
        security_config = {
            "enabled": True,
            "rule_of_two": {
                "enforcement": "block",
                "acknowledge_abc_risk": True
            }
        }
        validator = RuleOfTwoValidator(security_config=security_config)
        agent = Mock()
        agent.name = "test_agent"
        agent.security_context = SecurityContext()
        agent.config = None
        agent.behaviors = [MockReadBehavior(), MockWriteBehavior(), validator]

        # Should not raise (risk acknowledged)
        validator.validate_agent_configuration(agent)

        assert validator.has_abc_trifecta is True

    def test_validation_skips_when_enforcement_off(self):
        """Test validation skips when enforcement is 'off'."""
        security_config = {
            "enabled": True,
            "rule_of_two": {
                "enforcement": "off",
                "acknowledge_abc_risk": False
            }
        }
        validator = RuleOfTwoValidator(security_config=security_config)
        agent = Mock()
        agent.name = "test_agent"
        agent.security_context = SecurityContext()
        agent.config = None
        agent.behaviors = [MockReadBehavior(), MockWriteBehavior(), validator]

        # Should not raise (enforcement off)
        validator.validate_agent_configuration(agent)

        assert validator.has_abc_trifecta is True


class TestDynamicPropertiesIntegration:
    """Test integration with dynamic property resolution."""

    def test_abc_in_user_workspace_bc_in_isolated(self):
        """Test same behaviors yield [ABC] in user workspace, [BC] in isolated."""
        # User workspace: [AB] + [C] = [ABC]
        user_config = {"enabled": True, "rule_of_two": {"enforcement": "warn"}}
        validator_user = RuleOfTwoValidator(security_config=user_config)
        agent_user = Mock()
        agent_user.name = "user_agent"
        agent_user.security_context = SecurityContext(workspace_trust_level="user")
        agent_user.config = None
        agent_user.behaviors = [MockDynamicBehavior(), MockWriteBehavior(), validator_user]

        validator_user.validate_agent_configuration(agent_user)
        assert validator_user.has_abc_trifecta is True  # [AB] + [C] = [ABC]

        # Isolated workspace: [B] + [C] = [BC]
        isolated_config = {"enabled": True}
        validator_isolated = RuleOfTwoValidator(security_config=isolated_config)
        agent_isolated = Mock()
        agent_isolated.name = "isolated_agent"
        agent_isolated.security_context = SecurityContext(workspace_trust_level="isolated")
        agent_isolated.behaviors = [MockDynamicBehavior(), MockWriteBehavior(), validator_isolated]

        validator_isolated.validate_agent_configuration(agent_isolated)
        assert validator_isolated.has_abc_trifecta is False  # [B] + [C] = [BC] (compliant)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
