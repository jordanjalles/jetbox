"""
Integration tests for Rule of Two security system.

Tests end-to-end scenarios with full agent lifecycle and defense layer auto-injection.
Phase 5: Integration & Auto-Injection
"""
import pytest
import os
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from base_agent import BaseAgent
from behaviors.rule_of_two_validator import SecurityViolationError
from behaviors.security_context import SecurityContext


def create_test_agent(tmp_path, name, behaviors_config, security_config):
    """Helper to create agent with config files."""
    # Write agent config to temp file
    config = {"behaviors": behaviors_config}
    config_file = tmp_path / f"{name}_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    # Patch security config and create agent
    with patch("agent_config.load_security_config", return_value=security_config):
        return MockAgent(
            name=name,
            workspace=tmp_path / name,
            config_file=str(config_file)
        )


class MockAgent(BaseAgent):
    """Minimal mock agent for testing."""

    def get_context_strategy(self) -> str:
        return "append"

    def get_system_prompt(self) -> str:
        return "Test agent"

    def should_continue(self, context: list[dict], perf_stats: dict | None) -> tuple[bool, str | None]:
        return True, None


class TestRuleOfTwoCompliance:
    """Test that compliant agents ([AB], [AC], [BC]) pass validation."""

    def test_ab_agent_no_defense_layers(self, tmp_path):
        """Test [AB] agent doesn't trigger defense layers."""
        behaviors_config = [
            {"type": "ReadFileToolsBehavior"},  # [AB]
        ]

        security_config = {
            "enabled": True,
            "rule_of_two": {
                "enforcement": "block",
                "acknowledge_abc_risk": False
            }
        }

        agent = create_test_agent(tmp_path, "test_ab", behaviors_config, security_config)

        # Check validator was injected
        validator_found = any(
            b.__class__.__name__ == "RuleOfTwoValidator"
            for b in agent.behaviors
        )
        assert validator_found, "RuleOfTwoValidator should be auto-injected"

        # Check defense layers NOT injected (compliant [AB] agent)
        defense_layers = [
            "PromptInjectionDetectorBehavior",
            "SensitiveAccessAuditorBehavior",
            "NetworkAuditBehavior"
        ]
        for layer_name in defense_layers:
            assert not any(
                b.__class__.__name__ == layer_name
                for b in agent.behaviors
            ), f"{layer_name} should NOT be injected for [AB] agent"

    def test_bc_agent_no_defense_layers(self, tmp_path):
        """Test [BC] agent doesn't trigger defense layers."""
        behaviors_config = [
            {"type": "CommandToolsBehavior"},  # [BC]
        ]

        security_config = {
            "enabled": True,
            "rule_of_two": {
                "enforcement": "block",
                "acknowledge_abc_risk": False
            }
        }

        agent = create_test_agent(tmp_path, "test_bc", behaviors_config, security_config)

        # Check no defense layers
        defense_layers = [
            "PromptInjectionDetectorBehavior",
            "SensitiveAccessAuditorBehavior",
            "NetworkAuditBehavior"
        ]
        for layer_name in defense_layers:
            assert not any(
                b.__class__.__name__ == layer_name
                for b in agent.behaviors
            )


class TestABCAgentBlocking:
    """Test that [ABC] agents without acknowledgment are blocked."""

    def test_abc_agent_blocks_without_ack(self, tmp_path, monkeypatch):
        """Test [ABC] agent raises SecurityViolationError without acknowledgment."""
        # Set user workspace (not isolated) so ReadFile is [AB] not just [B]
        monkeypatch.setenv("IS_SANDBOX", "0")

        behaviors_config = [
            {"type": "ReadFileToolsBehavior"},  # [AB] in user workspace
            {"type": "CommandToolsBehavior"},  # [C] with network access
        ]

        security_config = {
            "enabled": True,
            "workspace": {
                "has_untrusted_files": True,  # Workspace has untrusted files
                "has_sensitive_files": True,  # Workspace has sensitive files
                "has_network_access": True     # Network enabled for [C]
            },
            "rule_of_two": {
                "enforcement": "block",
                "acknowledge_abc_risk": False,  # No acknowledgment
                "skip_defense_in_depth": False
            }
        }

        agent = create_test_agent(tmp_path, "test_abc", behaviors_config, security_config)

        # Trigger validation by calling trigger_goal_start
        with pytest.raises(SecurityViolationError) as exc_info:
            agent.event_system.trigger_goal_start("test goal")

        # Verify error message mentions [ABC]
        assert "[ABC]" in str(exc_info.value)
        assert "RULE OF TWO" in str(exc_info.value)

    def test_abc_agent_warns_with_warn_enforcement(self, tmp_path, monkeypatch):
        """Test [ABC] agent warns but doesn't block with 'warn' enforcement."""
        # Set user workspace (not isolated) so ReadFile is [AB] not just [B]
        monkeypatch.setenv("IS_SANDBOX", "0")

        behaviors_config = [
            {"type": "ReadFileToolsBehavior"},  # [AB] in user workspace
            {"type": "WriteFileToolsBehavior"},  # [C]
        ]

        security_config = {
            "enabled": True,
            "workspace": {
                "has_untrusted_files": True,
                "has_sensitive_files": True
            },
            "rule_of_two": {
                "enforcement": "warn",  # Warn instead of block
                "acknowledge_abc_risk": False
            }
        }

        agent = create_test_agent(tmp_path, "test_abc", behaviors_config, security_config)

        # Should NOT raise (warns instead)
        agent.event_system.trigger_goal_start("test goal")  # Should succeed


class TestDefenseLayerInjection:
    """Test that defense layers are auto-injected for acknowledged [ABC] agents."""

    def test_abc_agent_injects_all_defense_layers(self, tmp_path, monkeypatch):
        """Test [ABC] agent with acknowledgment injects all 3 defense layers."""
        # Set user workspace (not isolated) so ReadFile is [AB] not just [B]
        monkeypatch.setenv("IS_SANDBOX", "0")

        behaviors_config = [
            {"type": "ReadFileToolsBehavior"},  # [AB] in user workspace
            {"type": "CommandToolsBehavior"},  # [C] with network access
        ]

        security_config = {
            "enabled": True,
            "workspace": {
                "has_untrusted_files": True,
                "has_sensitive_files": True,
                "has_network_access": True
            },
            "rule_of_two": {
                "enforcement": "block",
                "acknowledge_abc_risk": True,  # Acknowledged
                "skip_defense_in_depth": False,
                "defense_in_depth": {
                    "input_validation": {"enabled": True},
                    "access_auditing": {"enabled": True},
                    "network_audit": {"enabled": True}
                }
            }
        }

        agent = create_test_agent(tmp_path, "test_abc_ack", behaviors_config, security_config)

        # Trigger validation
        agent.event_system.trigger_goal_start("test goal")

        # Check all 3 defense layers were injected
        layer_names = [b.__class__.__name__ for b in agent.behaviors]
        assert "PromptInjectionDetectorBehavior" in layer_names, "Layer 1 should be injected"
        assert "SensitiveAccessAuditorBehavior" in layer_names, "Layer 2 should be injected"
        assert "NetworkAuditBehavior" in layer_names, "Layer 3 should be injected"

    def test_abc_agent_skips_defense_layers_when_configured(self, tmp_path, monkeypatch):
        """Test [ABC] agent with skip_defense_in_depth doesn't inject layers."""
        # Set user workspace (not isolated) so ReadFile is [AB] not just [B]
        monkeypatch.setenv("IS_SANDBOX", "0")

        behaviors_config = [
            {"type": "ReadFileToolsBehavior"},  # [AB] in user workspace
            {"type": "CommandToolsBehavior"},  # [C] with network
        ]

        security_config = {
            "enabled": True,
            "workspace": {
                "has_untrusted_files": True,
                "has_sensitive_files": True,
                "has_network_access": True
            },
            "rule_of_two": {
                "enforcement": "block",
                "acknowledge_abc_risk": True,  # Acknowledged
                "skip_defense_in_depth": True,  # Skip layers
            }
        }

        agent = create_test_agent(tmp_path, "test_abc_skip", behaviors_config, security_config)

        # Trigger validation
        agent.event_system.trigger_goal_start("test goal")

        # Check defense layers were NOT injected
        layer_names = [b.__class__.__name__ for b in agent.behaviors]
        assert "PromptInjectionDetectorBehavior" not in layer_names
        assert "SensitiveAccessAuditorBehavior" not in layer_names
        assert "NetworkAuditBehavior" not in layer_names

    def test_individual_defense_layers_can_be_disabled(self, tmp_path, monkeypatch):
        """Test individual defense layers can be disabled via config."""
        # Set user workspace (not isolated) so ReadFile is [AB] not just [B]
        monkeypatch.setenv("IS_SANDBOX", "0")

        behaviors_config = [
            {"type": "ReadFileToolsBehavior"},  # [AB] in user workspace
            {"type": "CommandToolsBehavior"},  # [C] with network
        ]

        security_config = {
            "enabled": True,
            "workspace": {
                "has_untrusted_files": True,
                "has_sensitive_files": True,
                "has_network_access": True
            },
            "rule_of_two": {
                "enforcement": "block",
                "acknowledge_abc_risk": True,
                "skip_defense_in_depth": False,
                "defense_in_depth": {
                    "input_validation": {"enabled": True},
                    "access_auditing": {"enabled": False},  # Disabled
                    "network_audit": {"enabled": True}
                }
            }
        }

        agent = create_test_agent(tmp_path, "test_abc_partial", behaviors_config, security_config)

        agent.event_system.trigger_goal_start("test goal")

        layer_names = [b.__class__.__name__ for b in agent.behaviors]
        assert "PromptInjectionDetectorBehavior" in layer_names
        assert "SensitiveAccessAuditorBehavior" not in layer_names  # Disabled
        assert "NetworkAuditBehavior" in layer_names


class TestSecurityDisabled:
    """Test that security can be disabled globally."""

    def test_security_disabled_no_validator_injected(self, tmp_path):
        """Test validator not injected when security disabled."""
        behaviors_config = [
            {"type": "ReadFileToolsBehavior"},
            {"type": "WriteFileToolsBehavior"},
        ]

        security_config = {
            "enabled": False,  # Security disabled
        }

        agent = create_test_agent(tmp_path, "test_disabled", behaviors_config, security_config)

        # Check validator NOT injected
        validator_found = any(
            b.__class__.__name__ == "RuleOfTwoValidator"
            for b in agent.behaviors
        )
        assert not validator_found, "Validator should NOT be injected when security disabled"

    def test_env_var_override_disables_security(self, tmp_path, monkeypatch):
        """Test JETBOX_DISABLE_SECURITY=1 overrides config."""
        monkeypatch.setenv("JETBOX_DISABLE_SECURITY", "1")

        behaviors_config = [
            {"type": "ReadFileToolsBehavior"},
            {"type": "WriteFileToolsBehavior"},
        ]

        security_config = {
            "enabled": True,  # Enabled in config
        }

        agent = create_test_agent(tmp_path, "test_env_override", behaviors_config, security_config)

        # Validator should NOT be injected (env var override)
        validator_found = any(
            b.__class__.__name__ == "RuleOfTwoValidator"
            for b in agent.behaviors
        )
        assert not validator_found


class TestEndToEndAttackScenario:
    """Test full attack scenario with all 3 defense layers."""

    def test_full_attack_chain_detected(self, tmp_path, monkeypatch):
        """
        Test complete attack chain:
        1. Read malicious file (Layer 1: Input Validation)
        2. Read credentials (Layer 2: Access Auditing)
        3. Git push (Layer 3: Network Audit)
        """
        # Set user workspace (not isolated) so ReadFile is [AB] not just [B]
        monkeypatch.setenv("IS_SANDBOX", "0")

        behaviors_config = [
            {"type": "ReadFileToolsBehavior"},  # [AB] in user workspace
            {"type": "CommandToolsBehavior"},  # [ABC] with all three
        ]

        security_config = {
            "enabled": True,
            "workspace": {
                "has_untrusted_files": True,
                "has_sensitive_files": True,
                "has_network_access": True
            },
            "rule_of_two": {
                "enforcement": "block",
                "acknowledge_abc_risk": True,  # Acknowledged
                "skip_defense_in_depth": False,
                "defense_in_depth": {
                    "input_validation": {"enabled": True},
                    "access_auditing": {"enabled": True},
                    "network_audit": {"enabled": True}
                }
            }
        }

        agent = create_test_agent(tmp_path, "test_attack", behaviors_config, security_config)

        agent.event_system.trigger_goal_start("test goal")

        # Verify all 3 defense layers present
        layer_names = [b.__class__.__name__ for b in agent.behaviors]
        assert "PromptInjectionDetectorBehavior" in layer_names
        assert "SensitiveAccessAuditorBehavior" in layer_names
        assert "NetworkAuditBehavior" in layer_names

        # Each layer will independently detect/block their part
        # This test verifies they're all present and initialized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
