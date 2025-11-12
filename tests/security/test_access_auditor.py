"""
Unit tests for SensitiveAccessAuditorBehavior (Defense Layer 2).

Tests:
- Sensitive file pattern matching (20+ patterns)
- Session tracking (multiple reads)
- Threshold detection (1 file OK, 3 files = anomaly)
- User approval flow (mock input)
- SecurityContext integration
- Workspace trust level consideration
- File type classification
"""

from unittest.mock import Mock, patch

from behaviors.security_access_auditor import (
    SensitiveAccessAuditorBehavior
)
from behaviors.security_context import SecurityContext


class TestSensitiveFilePatternMatching:
    """Test sensitive file pattern detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auditor = SensitiveAccessAuditorBehavior()

    def test_env_files_detected(self):
        """Test .env file patterns are detected."""
        assert self.auditor._is_sensitive_file(".env")
        assert self.auditor._is_sensitive_file(".env.local")
        assert self.auditor._is_sensitive_file(".env.production")
        assert self.auditor._is_sensitive_file(".env.development")
        assert self.auditor._is_sensitive_file(".env.test")
        assert self.auditor._is_sensitive_file("config/.env.staging")

    def test_credential_files_detected(self):
        """Test credential file patterns are detected."""
        assert self.auditor._is_sensitive_file("credentials.json")
        assert self.auditor._is_sensitive_file("credentials.yaml")
        assert self.auditor._is_sensitive_file("credentials.yml")
        assert self.auditor._is_sensitive_file("aws_credentials")
        assert self.auditor._is_sensitive_file("gcloud-credentials.json")
        assert self.auditor._is_sensitive_file("gcp-credentials")
        assert self.auditor._is_sensitive_file("azure-credentials.json")

    def test_key_files_detected(self):
        """Test key and certificate files are detected."""
        assert self.auditor._is_sensitive_file("private.key")
        assert self.auditor._is_sensitive_file("cert.pem")
        assert self.auditor._is_sensitive_file("keystore.p12")
        assert self.auditor._is_sensitive_file("certificate.pfx")
        assert self.auditor._is_sensitive_file("id_rsa")
        assert self.auditor._is_sensitive_file("id_dsa")
        assert self.auditor._is_sensitive_file("id_ecdsa")
        assert self.auditor._is_sensitive_file("id_ed25519")

    def test_config_directory_files_detected(self):
        """Test files in sensitive config directories are detected."""
        assert self.auditor._is_sensitive_file("/home/user/.aws/credentials")
        assert self.auditor._is_sensitive_file("/home/user/.ssh/id_rsa")
        assert self.auditor._is_sensitive_file("/root/.config/gcloud/credentials")
        assert self.auditor._is_sensitive_file("~/.kube/config")
        assert self.auditor._is_sensitive_file("~/.docker/config.json")

    def test_secret_keyword_files_detected(self):
        """Test files with secret keywords are detected."""
        assert self.auditor._is_sensitive_file("api_secret.json")
        assert self.auditor._is_sensitive_file("auth_token.txt")
        assert self.auditor._is_sensitive_file("db_password.conf")
        assert self.auditor._is_sensitive_file("github_apikey.txt")
        assert self.auditor._is_sensitive_file("api_key.json")
        assert self.auditor._is_sensitive_file("private_key.pem")
        assert self.auditor._is_sensitive_file("passwd")

    def test_normal_files_not_detected(self):
        """Test normal files are NOT detected as sensitive."""
        assert not self.auditor._is_sensitive_file("README.md")
        assert not self.auditor._is_sensitive_file("package.json")
        assert not self.auditor._is_sensitive_file("main.py")
        assert not self.auditor._is_sensitive_file("test_data.json")
        assert not self.auditor._is_sensitive_file("config.yaml")
        assert not self.auditor._is_sensitive_file("settings.toml")

    def test_case_insensitive_detection(self):
        """Test sensitive file detection is case-insensitive."""
        assert self.auditor._is_sensitive_file(".ENV")
        assert self.auditor._is_sensitive_file("CREDENTIALS.json")
        assert self.auditor._is_sensitive_file("API_SECRET.txt")
        assert self.auditor._is_sensitive_file("ID_RSA")


class TestFileTypeClassification:
    """Test file type classification for reporting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auditor = SensitiveAccessAuditorBehavior()

    def test_env_file_classification(self):
        """Test .env files are classified correctly."""
        assert self.auditor._classify_file_type(".env") == "environment variables"
        assert self.auditor._classify_file_type(".env.local") == "environment variables"

    def test_credential_file_classification(self):
        """Test credential files are classified correctly."""
        assert self.auditor._classify_file_type("credentials.json") == "credentials file"
        assert self.auditor._classify_file_type("aws_credentials") == "credentials file"

    def test_ssh_key_classification(self):
        """Test SSH keys are classified correctly."""
        assert self.auditor._classify_file_type("id_rsa") == "SSH private key"
        assert self.auditor._classify_file_type("id_dsa") == "SSH private key"
        assert self.auditor._classify_file_type("id_ecdsa") == "SSH private key"
        assert self.auditor._classify_file_type("id_ed25519") == "SSH private key"

    def test_generic_key_classification(self):
        """Test generic keys are classified correctly."""
        assert self.auditor._classify_file_type("private.key") == "private key"
        assert self.auditor._classify_file_type("cert.pem") == "private key"

    def test_certificate_classification(self):
        """Test certificates are classified correctly."""
        assert self.auditor._classify_file_type("keystore.p12") == "certificate"
        assert self.auditor._classify_file_type("cert.pfx") == "certificate"

    def test_aws_classification(self):
        """Test AWS files are classified correctly."""
        assert self.auditor._classify_file_type("/home/user/.aws/credentials") == "AWS credentials"

    def test_secret_classification(self):
        """Test secret files are classified correctly."""
        assert self.auditor._classify_file_type("api_secret.json") == "secret"
        assert self.auditor._classify_file_type("auth_token.txt") == "API token"
        assert self.auditor._classify_file_type("github_apikey.json") == "API token"
        assert self.auditor._classify_file_type("db_password.conf") == "password file"


class TestSessionTracking:
    """Test session-level tracking of sensitive file access."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auditor = SensitiveAccessAuditorBehavior()
        self.agent = Mock()
        self.agent.security_context = SecurityContext()

    def test_initial_session_stats_empty(self):
        """Test session stats start empty."""
        assert len(self.auditor.session_stats.files_accessed) == 0
        assert not self.auditor.session_stats.user_approved
        assert not self.auditor.session_stats.approval_session

    def test_reset_on_goal_start(self):
        """Test session stats reset when goal starts."""
        # Populate with fake data
        self.auditor.session_stats.files_accessed.append((".env", "env"))
        self.auditor.session_stats.user_approved = True

        # Reset on goal start
        self.auditor.on_goal_start(self.agent, "test goal")

        # Verify reset
        assert len(self.auditor.session_stats.files_accessed) == 0
        assert not self.auditor.session_stats.user_approved

    def test_track_single_file_access(self):
        """Test tracking single sensitive file access."""
        # Simulate read_file call
        args = {"path": ".env"}
        result = {"success": True, "content": "SECRET=value"}

        self.auditor.on_tool_call(self.agent, "read_file", args, result)

        # Verify tracked
        assert len(self.auditor.session_stats.files_accessed) == 1
        path, file_type = self.auditor.session_stats.files_accessed[0]
        assert path == ".env"
        assert file_type == "environment variables"

    @patch.object(SensitiveAccessAuditorBehavior, "_handle_anomaly")
    def test_track_multiple_file_accesses(self, mock_handle):
        """Test tracking multiple sensitive file accesses."""
        files = [
            (".env", {"success": True, "content": "SECRET=value"}),
            ("credentials.json", {"success": True, "content": "{}"}),
            ("id_rsa", {"success": True, "content": "-----BEGIN RSA PRIVATE KEY-----"})
        ]

        for path, result in files:
            self.auditor.on_tool_call(self.agent, "read_file", {"path": path}, result)

        # Verify all tracked
        assert len(self.auditor.session_stats.files_accessed) == 3

    def test_ignore_non_read_file_calls(self):
        """Test non-read_file tool calls are ignored."""
        args = {"path": ".env"}
        result = {"success": True}

        # Call different tools
        self.auditor.on_tool_call(self.agent, "write_file", args, result)
        self.auditor.on_tool_call(self.agent, "list_dir", args, result)

        # Verify not tracked
        assert len(self.auditor.session_stats.files_accessed) == 0

    def test_ignore_failed_reads(self):
        """Test failed reads are not tracked."""
        args = {"path": ".env"}
        result = {"error": "File not found"}

        self.auditor.on_tool_call(self.agent, "read_file", args, result)

        # Verify not tracked
        assert len(self.auditor.session_stats.files_accessed) == 0

    def test_ignore_non_sensitive_files(self):
        """Test non-sensitive files are not tracked."""
        args = {"path": "README.md"}
        result = {"success": True, "content": "# Hello"}

        self.auditor.on_tool_call(self.agent, "read_file", args, result)

        # Verify not tracked
        assert len(self.auditor.session_stats.files_accessed) == 0


class TestSecurityContextIntegration:
    """Test integration with SecurityContext."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auditor = SensitiveAccessAuditorBehavior()
        self.agent = Mock()
        self.agent.security_context = SecurityContext()

    def test_updates_security_context_on_access(self):
        """Test SecurityContext is updated when sensitive file accessed."""
        args = {"path": ".env"}
        result = {"success": True, "content": "SECRET=value"}

        self.auditor.on_tool_call(self.agent, "read_file", args, result)

        # Verify SecurityContext updated
        assert self.agent.security_context.sensitive_data_detected
        assert ".env" in self.agent.security_context.sensitive_files_seen

    def test_handles_missing_security_context(self):
        """Test behavior when agent lacks security_context."""
        agent = Mock(spec=[])  # No security_context attribute

        args = {"path": ".env"}
        result = {"success": True, "content": "SECRET=value"}

        # Should not crash
        self.auditor.on_tool_call(agent, "read_file", args, result)

        # Verify still tracked locally
        assert len(self.auditor.session_stats.files_accessed) == 1


class TestThresholdDetection:
    """Test anomaly detection based on thresholds."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auditor = SensitiveAccessAuditorBehavior()
        self.agent = Mock()
        self.agent.security_context = SecurityContext()

    def test_single_file_no_anomaly(self):
        """Test single file access does not trigger anomaly."""
        # Add one file
        self.auditor.session_stats.files_accessed.append((".env", "env"))

        # Check anomaly
        assert not self.auditor._check_anomaly(self.agent)

    def test_two_files_no_anomaly(self):
        """Test two files do not trigger anomaly."""
        # Add two files
        self.auditor.session_stats.files_accessed.append((".env", "env"))
        self.auditor.session_stats.files_accessed.append(("credentials.json", "credentials"))

        # Check anomaly
        assert not self.auditor._check_anomaly(self.agent)

    def test_three_files_triggers_anomaly(self):
        """Test three files trigger anomaly (threshold = 2)."""
        # Add three files
        self.auditor.session_stats.files_accessed.append((".env", "env"))
        self.auditor.session_stats.files_accessed.append(("credentials.json", "credentials"))
        self.auditor.session_stats.files_accessed.append(("id_rsa", "ssh_key"))

        # Check anomaly
        assert self.auditor._check_anomaly(self.agent)

    def test_duplicate_files_not_counted_multiple_times(self):
        """Test duplicate file accesses only count once."""
        # Add same file 3 times
        self.auditor.session_stats.files_accessed.append((".env", "env"))
        self.auditor.session_stats.files_accessed.append((".env", "env"))
        self.auditor.session_stats.files_accessed.append((".env", "env"))

        # Check anomaly (should not trigger - only 1 unique file)
        assert not self.auditor._check_anomaly(self.agent)

    def test_isolated_workspace_higher_threshold(self):
        """Test isolated workspace has higher threshold."""
        # Set isolated workspace
        self.agent.security_context.workspace_trust_level = "isolated"

        # Add 3 files (would trigger in user workspace)
        self.auditor.session_stats.files_accessed.append((".env", "env"))
        self.auditor.session_stats.files_accessed.append(("credentials.json", "credentials"))
        self.auditor.session_stats.files_accessed.append(("id_rsa", "ssh_key"))

        # Check anomaly (should NOT trigger - threshold is 4 for isolated)
        assert not self.auditor._check_anomaly(self.agent)

    def test_isolated_workspace_exceeds_higher_threshold(self):
        """Test isolated workspace anomaly when threshold exceeded."""
        # Set isolated workspace
        self.agent.security_context.workspace_trust_level = "isolated"

        # Add 5 files (exceeds threshold of 4)
        for i, path in enumerate([".env", "creds.json", "id_rsa", "token.txt", "secret.key"]):
            file_type = "file" + str(i)
            self.auditor.session_stats.files_accessed.append((path, file_type))

        # Check anomaly (should trigger - threshold is 4 for isolated)
        assert self.auditor._check_anomaly(self.agent)

    def test_user_workspace_default_threshold(self):
        """Test user workspace uses default threshold."""
        # Set user workspace (default)
        self.agent.security_context.workspace_trust_level = "user"

        # Add 3 files
        self.auditor.session_stats.files_accessed.append((".env", "env"))
        self.auditor.session_stats.files_accessed.append(("credentials.json", "credentials"))
        self.auditor.session_stats.files_accessed.append(("id_rsa", "ssh_key"))

        # Check anomaly (should trigger - threshold is 2 for user)
        assert self.auditor._check_anomaly(self.agent)


class TestUserApprovalFlow:
    """Test user approval workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auditor = SensitiveAccessAuditorBehavior()
        self.agent = Mock()
        self.agent.security_context = SecurityContext()

    @patch("builtins.input", return_value="a")
    @patch("builtins.print")
    def test_approve_session(self, mock_print, mock_input):
        """Test user approving access for session."""
        files = [(".env", "env"), ("creds.json", "creds")]

        self.auditor._handle_anomaly(self.agent, files)

        # Verify approval
        assert self.auditor.session_stats.user_approved
        assert self.auditor.session_stats.approval_session

    @patch("builtins.input", return_value="o")
    @patch("builtins.print")
    def test_approve_once(self, mock_print, mock_input):
        """Test user approving access once."""
        files = [(".env", "env"), ("creds.json", "creds")]

        self.auditor._handle_anomaly(self.agent, files)

        # Verify approval (but not session)
        assert self.auditor.session_stats.user_approved
        assert not self.auditor.session_stats.approval_session

    @patch("builtins.input", return_value="d")
    @patch("sys.exit")
    @patch("builtins.print")
    def test_deny_exits(self, mock_print, mock_exit, mock_input):
        """Test user denying access exits."""
        files = [(".env", "env"), ("creds.json", "creds")]

        self.auditor._handle_anomaly(self.agent, files)

        # Verify exit called
        mock_exit.assert_called_once_with(1)

    @patch("builtins.input", return_value="invalid")
    @patch("sys.exit")
    @patch("builtins.print")
    def test_invalid_choice_defaults_to_deny(self, mock_print, mock_exit, mock_input):
        """Test invalid choice defaults to deny."""
        files = [(".env", "env"), ("creds.json", "creds")]

        self.auditor._handle_anomaly(self.agent, files)

        # Verify exit called (default behavior)
        mock_exit.assert_called_once_with(1)

    @patch("builtins.input", side_effect=EOFError)
    @patch("sys.exit")
    @patch("builtins.print")
    def test_eof_defaults_to_deny(self, mock_print, mock_exit, mock_input):
        """Test EOF (Ctrl+D) defaults to deny."""
        files = [(".env", "env"), ("creds.json", "creds")]

        self.auditor._handle_anomaly(self.agent, files)

        # Verify exit called
        mock_exit.assert_called_once_with(1)

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    @patch("sys.exit")
    @patch("builtins.print")
    def test_keyboard_interrupt_defaults_to_deny(self, mock_print, mock_exit, mock_input):
        """Test Ctrl+C defaults to deny."""
        files = [(".env", "env"), ("creds.json", "creds")]

        self.auditor._handle_anomaly(self.agent, files)

        # Verify exit called
        mock_exit.assert_called_once_with(1)


class TestRichAnomalyReporting:
    """Test anomaly warning message content."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auditor = SensitiveAccessAuditorBehavior()
        self.agent = Mock()
        self.agent.security_context = SecurityContext()

    @patch("builtins.input", return_value="a")
    @patch("builtins.print")
    def test_warning_shows_all_files(self, mock_print, mock_input):
        """Test warning lists all accessed files."""
        files = [
            (".env", "environment variables"),
            ("credentials.json", "credentials file"),
            ("id_rsa", "SSH private key")
        ]

        self.auditor._handle_anomaly(self.agent, files)

        # Check print was called with file paths
        printed_output = " ".join(str(call) for call in mock_print.call_args_list)
        assert ".env" in printed_output
        assert "credentials.json" in printed_output
        assert "id_rsa" in printed_output

    @patch("builtins.input", return_value="a")
    @patch("builtins.print")
    def test_warning_shows_file_types(self, mock_print, mock_input):
        """Test warning shows file types."""
        files = [
            (".env", "environment variables"),
            ("credentials.json", "credentials file"),
            ("id_rsa", "SSH private key")
        ]

        self.auditor._handle_anomaly(self.agent, files)

        # Check print was called with file types
        printed_output = " ".join(str(call) for call in mock_print.call_args_list)
        assert "environment variables" in printed_output
        assert "credentials file" in printed_output
        assert "SSH private key" in printed_output

    @patch("builtins.input", return_value="a")
    @patch("builtins.print")
    def test_warning_explains_risk(self, mock_print, mock_input):
        """Test warning explains why flagged."""
        files = [(".env", "env"), ("creds.json", "creds")]

        self.auditor._handle_anomaly(self.agent, files)

        # Check explanation present
        printed_output = " ".join(str(call) for call in mock_print.call_args_list)
        assert "credential harvesting" in printed_output.lower() or "harvesting" in printed_output.lower()


class TestRuleOfTwoProperties:
    """Test Rule of Two integration."""

    def test_behavior_name(self):
        """Test behavior returns correct name."""
        auditor = SensitiveAccessAuditorBehavior()
        assert auditor.get_name() == "security_access_auditor"

    def test_defense_layer_has_no_properties(self):
        """Test defense layer declares no Rule of Two properties."""
        auditor = SensitiveAccessAuditorBehavior()
        assert len(auditor.rule_of_two_properties) == 0

    def test_get_rule_of_two_properties_returns_empty(self):
        """Test get_rule_of_two_properties returns empty set."""
        auditor = SensitiveAccessAuditorBehavior()
        agent = Mock()
        context = SecurityContext()

        props = auditor.get_rule_of_two_properties(agent, context)
        assert len(props) == 0


class TestIntegrationScenario:
    """Test end-to-end credential harvesting detection scenario."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auditor = SensitiveAccessAuditorBehavior()
        self.agent = Mock()
        self.agent.security_context = SecurityContext()

    @patch("builtins.input", return_value="a")
    @patch("builtins.print")
    def test_credential_harvesting_detection(self, mock_print, mock_input):
        """
        Integration test: Simulate agent reading .env, id_rsa, credentials.json.

        Expected:
        1. First two files tracked but no anomaly
        2. Third file triggers anomaly
        3. User prompted for approval
        4. Approval allows continued access
        """
        # Read first file (.env)
        self.auditor.on_tool_call(
            self.agent,
            "read_file",
            {"path": ".env"},
            {"success": True, "content": "SECRET=value"}
        )

        # Verify tracked, no anomaly yet
        assert len(self.auditor.session_stats.files_accessed) == 1
        assert not self.auditor._check_anomaly(self.agent)

        # Read second file (id_rsa)
        self.auditor.on_tool_call(
            self.agent,
            "read_file",
            {"path": "id_rsa"},
            {"success": True, "content": "-----BEGIN RSA PRIVATE KEY-----"}
        )

        # Verify tracked, still no anomaly
        assert len(self.auditor.session_stats.files_accessed) == 2
        assert not self.auditor._check_anomaly(self.agent)

        # Read third file (credentials.json) - should trigger anomaly
        self.auditor.on_tool_call(
            self.agent,
            "read_file",
            {"path": "credentials.json"},
            {"success": True, "content": "{}"}
        )

        # Verify anomaly triggered and user approved
        assert len(self.auditor.session_stats.files_accessed) == 3
        assert self.auditor._check_anomaly(self.agent)
        assert self.auditor.session_stats.user_approved

        # Verify SecurityContext updated
        assert self.agent.security_context.sensitive_data_detected
        assert len(self.agent.security_context.sensitive_files_seen) == 3

    @patch("builtins.input", return_value="a")
    @patch("builtins.print")
    def test_approval_prevents_repeat_prompts(self, mock_print, mock_input):
        """Test that approval prevents repeated prompts."""
        # Trigger anomaly and approve
        for path in [".env", "id_rsa", "credentials.json"]:
            self.auditor.on_tool_call(
                self.agent,
                "read_file",
                {"path": path},
                {"success": True, "content": "data"}
            )

        # User was prompted once
        assert mock_input.call_count == 1
        assert self.auditor.session_stats.user_approved

        # Read another file - should not prompt again
        self.auditor.on_tool_call(
            self.agent,
            "read_file",
            {"path": "secret.key"},
            {"success": True, "content": "data"}
        )

        # Still only prompted once
        assert mock_input.call_count == 1
