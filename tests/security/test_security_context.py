"""
Unit tests for SecurityContext.

Tests:
- Initialization with different trust levels
- Sensitive file detection
- Prompt injection tracking
- Trust level updates
- Context-aware property resolution (fallback to static)
"""

import pytest
import os
from behaviors.security_context import SecurityContext
from behaviors.base import AgentBehavior
from behaviors.rule_of_two_types import RuleOfTwoProperty


class TestSecurityContextInitialization:
    """Test SecurityContext initialization and defaults."""

    def test_default_initialization(self):
        """Test SecurityContext defaults to 'user' trust level."""
        context = SecurityContext()
        assert context.workspace_trust_level == "user"
        assert context.sensitive_data_detected is False
        assert len(context.sensitive_files_seen) == 0
        assert context.prompt_injection_detected is False
        assert len(context.injection_sources) == 0

    def test_isolated_trust_level(self):
        """Test SecurityContext with 'isolated' trust level."""
        context = SecurityContext(workspace_trust_level="isolated")
        assert context.workspace_trust_level == "isolated"

    def test_mixed_trust_level(self):
        """Test SecurityContext with 'mixed' trust level."""
        context = SecurityContext(workspace_trust_level="mixed")
        assert context.workspace_trust_level == "mixed"

    def test_default_expansion_fields(self):
        """Test future expansion fields have sensible defaults."""
        context = SecurityContext()
        assert context.network_policy == "full"
        assert context.enforcement_level == "warn"


class TestSensitiveFileDetection:
    """Test sensitive file pattern matching."""

    def test_env_files_detected(self):
        """Test .env file patterns are detected."""
        context = SecurityContext()
        assert context.is_sensitive_file(".env")
        assert context.is_sensitive_file(".env.local")
        assert context.is_sensitive_file(".env.production")
        assert context.is_sensitive_file("config/.env")

    def test_key_files_detected(self):
        """Test key/certificate files are detected."""
        context = SecurityContext()
        assert context.is_sensitive_file("id_rsa.key")
        assert context.is_sensitive_file("cert.pem")
        assert context.is_sensitive_file("keystore.p12")
        assert context.is_sensitive_file("cert.pfx")

    def test_credential_files_detected(self):
        """Test credential files are detected."""
        context = SecurityContext()
        assert context.is_sensitive_file("credentials.json")
        assert context.is_sensitive_file("aws_credentials")
        assert context.is_sensitive_file("gcloud-credentials.json")

    def test_aws_ssh_directories_detected(self):
        """Test AWS/SSH config directories are detected."""
        context = SecurityContext()
        assert context.is_sensitive_file("~/.aws/credentials")
        assert context.is_sensitive_file("/home/user/.ssh/id_rsa")

    def test_secret_keywords_detected(self):
        """Test files with secret keywords are detected."""
        context = SecurityContext()
        assert context.is_sensitive_file("api_secret.json")
        assert context.is_sensitive_file("auth_token.txt")
        assert context.is_sensitive_file("db_password.conf")
        assert context.is_sensitive_file("github_apikey.txt")

    def test_normal_files_not_detected(self):
        """Test normal files are NOT detected as sensitive."""
        context = SecurityContext()
        assert not context.is_sensitive_file("README.md")
        assert not context.is_sensitive_file("package.json")
        assert not context.is_sensitive_file("main.py")
        assert not context.is_sensitive_file("test_data.json")

    def test_case_insensitive_detection(self):
        """Test sensitive file detection is case-insensitive."""
        context = SecurityContext()
        assert context.is_sensitive_file(".ENV")
        assert context.is_sensitive_file("CREDENTIALS.json")
        assert context.is_sensitive_file("Secret.txt")


class TestSensitiveFileTracking:
    """Test tracking of accessed sensitive files."""

    def test_mark_sensitive_file_accessed(self):
        """Test marking a sensitive file as accessed."""
        context = SecurityContext()
        context.mark_sensitive_file_accessed(".env")

        assert context.sensitive_data_detected is True
        assert ".env" in context.sensitive_files_seen

    def test_multiple_sensitive_files_tracked(self):
        """Test tracking multiple sensitive files."""
        context = SecurityContext()
        context.mark_sensitive_file_accessed(".env")
        context.mark_sensitive_file_accessed("id_rsa.key")
        context.mark_sensitive_file_accessed("credentials.json")

        assert len(context.sensitive_files_seen) == 3
        assert ".env" in context.sensitive_files_seen
        assert "id_rsa.key" in context.sensitive_files_seen
        assert "credentials.json" in context.sensitive_files_seen

    def test_duplicate_files_not_duplicated(self):
        """Test accessing same file multiple times doesn't duplicate tracking."""
        context = SecurityContext()
        context.mark_sensitive_file_accessed(".env")
        context.mark_sensitive_file_accessed(".env")
        context.mark_sensitive_file_accessed(".env")

        assert len(context.sensitive_files_seen) == 1


class TestPromptInjectionTracking:
    """Test prompt injection detection tracking."""

    def test_mark_prompt_injection_detected(self):
        """Test marking prompt injection detected in a file."""
        context = SecurityContext()
        context.mark_prompt_injection_detected("README.md")

        assert context.prompt_injection_detected is True
        assert "README.md" in context.injection_sources

    def test_multiple_injection_sources_tracked(self):
        """Test tracking multiple injection sources."""
        context = SecurityContext()
        context.mark_prompt_injection_detected("README.md")
        context.mark_prompt_injection_detected("CONTRIBUTING.md")

        assert len(context.injection_sources) == 2
        assert "README.md" in context.injection_sources
        assert "CONTRIBUTING.md" in context.injection_sources

    def test_duplicate_sources_not_duplicated(self):
        """Test same file multiple times doesn't duplicate."""
        context = SecurityContext()
        context.mark_prompt_injection_detected("README.md")
        context.mark_prompt_injection_detected("README.md")

        assert len(context.injection_sources) == 1


class TestTrustLevelUpdates:
    """Test trust level updates."""

    def test_update_trust_level(self):
        """Test updating trust level."""
        context = SecurityContext(workspace_trust_level="isolated")
        assert context.workspace_trust_level == "isolated"

        context.update_trust_level("user")
        assert context.workspace_trust_level == "user"

    def test_update_to_mixed(self):
        """Test updating to mixed trust level."""
        context = SecurityContext(workspace_trust_level="user")
        context.update_trust_level("mixed")
        assert context.workspace_trust_level == "mixed"


class TestSecurityContextRepr:
    """Test string representation."""

    def test_repr_format(self):
        """Test __repr__ includes key fields."""
        context = SecurityContext(workspace_trust_level="user")
        repr_str = repr(context)

        assert "SecurityContext" in repr_str
        assert "trust=user" in repr_str
        assert "sensitive=False" in repr_str
        assert "injection=False" in repr_str

    def test_repr_with_detection(self):
        """Test __repr__ with detections."""
        context = SecurityContext()
        context.mark_sensitive_file_accessed(".env")
        context.mark_prompt_injection_detected("README.md")
        repr_str = repr(context)

        assert "sensitive=True" in repr_str
        assert "injection=True" in repr_str


class TestDynamicPropertyMethod:
    """Test dynamic property resolution method on AgentBehavior."""

    def test_default_returns_static_properties(self):
        """Test default implementation returns static attribute."""
        # Create mock behavior with static properties
        class MockBehavior(AgentBehavior):
            rule_of_two_properties = {RuleOfTwoProperty.EXTERNAL_ACTION}

            def get_name(self) -> str:
                return "mock"

        behavior = MockBehavior()
        context = SecurityContext()

        # Should return static properties by default
        props = behavior.get_rule_of_two_properties(agent=None, context=context)
        assert props == {RuleOfTwoProperty.EXTERNAL_ACTION}

    def test_can_override_with_dynamic_logic(self):
        """Test behaviors can override for dynamic properties."""
        class DynamicBehavior(AgentBehavior):
            rule_of_two_properties = {
                RuleOfTwoProperty.UNTRUSTED_INPUT,
                RuleOfTwoProperty.SENSITIVE_ACCESS
            }

            def get_name(self) -> str:
                return "dynamic"

            def get_rule_of_two_properties(self, agent, context):
                # Dynamic: remove [A] in isolated workspace
                props = {RuleOfTwoProperty.SENSITIVE_ACCESS}
                if context and context.workspace_trust_level == "user":
                    props.add(RuleOfTwoProperty.UNTRUSTED_INPUT)
                return props

        behavior = DynamicBehavior()

        # In user workspace: [AB]
        user_context = SecurityContext(workspace_trust_level="user")
        props = behavior.get_rule_of_two_properties(agent=None, context=user_context)
        assert props == {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS
        }

        # In isolated workspace: [B] only
        isolated_context = SecurityContext(workspace_trust_level="isolated")
        props = behavior.get_rule_of_two_properties(agent=None, context=isolated_context)
        assert props == {RuleOfTwoProperty.SENSITIVE_ACCESS}

    def test_backwards_compatible_with_no_context(self):
        """Test method works even if context=None (backwards compatibility)."""
        class MockBehavior(AgentBehavior):
            rule_of_two_properties = {RuleOfTwoProperty.EXTERNAL_ACTION}

            def get_name(self) -> str:
                return "mock"

        behavior = MockBehavior()

        # Should work with context=None
        props = behavior.get_rule_of_two_properties(agent=None, context=None)
        assert props == {RuleOfTwoProperty.EXTERNAL_ACTION}


class TestBaseAgentSecurityContextInit:
    """Test BaseAgent initializes security_context from IS_SANDBOX env var."""

    def test_is_sandbox_1_sets_isolated(self, monkeypatch, tmp_path):
        """Test IS_SANDBOX=1 sets trust level to isolated."""
        monkeypatch.setenv("IS_SANDBOX", "1")

        # Import after setting env var
        from base_agent import BaseAgent

        agent = BaseAgent(
            name="test",
            workspace=tmp_path,
            config_file="config/agents/task_executor.yaml"
        )

        assert agent.security_context.workspace_trust_level == "isolated"

    def test_is_sandbox_true_sets_isolated(self, monkeypatch, tmp_path):
        """Test IS_SANDBOX=true sets trust level to isolated."""
        monkeypatch.setenv("IS_SANDBOX", "true")

        from base_agent import BaseAgent

        agent = BaseAgent(
            name="test",
            workspace=tmp_path,
            config_file="config/agents/task_executor.yaml"
        )

        assert agent.security_context.workspace_trust_level == "isolated"

    def test_is_sandbox_yes_sets_isolated(self, monkeypatch, tmp_path):
        """Test IS_SANDBOX=yes sets trust level to isolated."""
        monkeypatch.setenv("IS_SANDBOX", "yes")

        from base_agent import BaseAgent

        agent = BaseAgent(
            name="test",
            workspace=tmp_path,
            config_file="config/agents/task_executor.yaml"
        )

        assert agent.security_context.workspace_trust_level == "isolated"

    def test_is_sandbox_0_sets_user(self, monkeypatch, tmp_path):
        """Test IS_SANDBOX=0 sets trust level to user."""
        monkeypatch.setenv("IS_SANDBOX", "0")

        from base_agent import BaseAgent

        agent = BaseAgent(
            name="test",
            workspace=tmp_path,
            config_file="config/agents/task_executor.yaml"
        )

        assert agent.security_context.workspace_trust_level == "user"

    def test_is_sandbox_false_sets_user(self, monkeypatch, tmp_path):
        """Test IS_SANDBOX=false sets trust level to user."""
        monkeypatch.setenv("IS_SANDBOX", "false")

        from base_agent import BaseAgent

        agent = BaseAgent(
            name="test",
            workspace=tmp_path,
            config_file="config/agents/task_executor.yaml"
        )

        assert agent.security_context.workspace_trust_level == "user"

    def test_no_env_var_defaults_to_user(self, monkeypatch, tmp_path):
        """Test no IS_SANDBOX defaults to 'user'."""
        monkeypatch.delenv("IS_SANDBOX", raising=False)

        from base_agent import BaseAgent

        agent = BaseAgent(
            name="test",
            workspace=tmp_path,
            config_file="config/agents/task_executor.yaml"
        )

        assert agent.security_context.workspace_trust_level == "user"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
