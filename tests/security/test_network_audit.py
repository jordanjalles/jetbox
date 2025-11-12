"""
Unit tests for NetworkAuditBehavior (Defense Layer 3).

Tests:
- Network command detection (15+ patterns)
- Audit log append and immutability
- Risk analysis (CRITICAL/HIGH/MEDIUM/LOW scenarios)
- Git staging detection
- User approval flow
- BEFORE execution interception
- Integration test (full workflow)

Target: 90%+ coverage, 100% audit coverage
"""

import pytest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

from behaviors.security_network_audit import (
    NetworkAuditBehavior,
    SecurityViolationError
)
from behaviors.security_context import SecurityContext


class TestNetworkCommandDetection:
    """Test network command detection patterns."""

    def test_detect_git_push(self):
        """Detect git push command."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("git push origin main")

    def test_detect_git_pull(self):
        """Detect git pull command."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("git pull origin main")

    def test_detect_git_fetch(self):
        """Detect git fetch command."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("git fetch --all")

    def test_detect_git_clone(self):
        """Detect git clone command."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("git clone https://github.com/user/repo.git")

    def test_detect_curl(self):
        """Detect curl command."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("curl https://api.example.com/data")

    def test_detect_wget(self):
        """Detect wget command."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("wget https://example.com/file.zip")

    def test_detect_ssh(self):
        """Detect ssh command."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("ssh user@server.com")

    def test_detect_scp(self):
        """Detect scp command."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("scp file.txt user@server.com:/path/")

    def test_detect_rsync(self):
        """Detect rsync command."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("rsync -avz /local/ user@server:/remote/")

    def test_detect_nc(self):
        """Detect nc (netcat) command."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("nc -l 8080")

    def test_detect_python_requests(self):
        """Detect python with requests library."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("python script.py requests.get('http://example.com')")

    def test_detect_npm_publish(self):
        """Detect npm publish command."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("npm publish")

    def test_detect_pip_custom_index(self):
        """Detect pip install with custom index."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("pip install package --index-url https://custom.pypi.org")

    def test_detect_aws_s3_cp(self):
        """Detect AWS S3 copy command."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("aws s3 cp file.txt s3://bucket/")

    def test_detect_gcloud_storage(self):
        """Detect gcloud storage command."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("gcloud storage cp file.txt gs://bucket/")

    def test_no_detection_safe_commands(self):
        """Don't detect safe commands."""
        behavior = NetworkAuditBehavior()
        assert not behavior._is_network_command("ls -la")
        assert not behavior._is_network_command("pytest tests/")
        assert not behavior._is_network_command("python script.py")
        assert not behavior._is_network_command("cat file.txt")
        assert not behavior._is_network_command("git status")
        assert not behavior._is_network_command("git log")
        assert not behavior._is_network_command("git diff")

    def test_case_insensitive_detection(self):
        """Detection is case-insensitive."""
        behavior = NetworkAuditBehavior()
        assert behavior._is_network_command("GIT PUSH")
        assert behavior._is_network_command("Curl https://example.com")
        assert behavior._is_network_command("WGET file.zip")


class TestAuditLogging:
    """Test audit log append and immutability."""

    def test_audit_log_append(self):
        """Test audit log entry is appended."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "network_audit.log"
            behavior = NetworkAuditBehavior(audit_log_path=log_path)

            entry = {
                "timestamp": "2025-01-12T10:30:00",
                "command": "git push",
                "risk_level": "MEDIUM",
                "user_decision": "APPROVED",
                "staged_files": [],
                "sensitive_reads": []
            }

            behavior._audit_log_append(entry)

            # Verify log file exists and contains entry
            assert log_path.exists()
            content = log_path.read_text()
            assert "git push" in content
            assert "MEDIUM" in content
            assert "APPROVED" in content

    def test_audit_log_format(self):
        """Test audit log format is pipe-delimited."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "network_audit.log"
            behavior = NetworkAuditBehavior(audit_log_path=log_path)

            entry = {
                "timestamp": "2025-01-12T10:30:00",
                "command": "git push origin main",
                "risk_level": "HIGH",
                "user_decision": "DENIED",
                "staged_files": [".env"],
                "sensitive_reads": ["config.key"]
            }

            behavior._audit_log_append(entry)

            # Verify format
            content = log_path.read_text()
            parts = content.strip().split(" | ")
            assert len(parts) == 6
            assert parts[0] == "2025-01-12T10:30:00"
            assert parts[1] == "git push origin main"
            assert parts[2] == "HIGH"
            assert parts[3] == "DENIED"
            assert parts[4] == ".env"
            assert parts[5] == "config.key"

    def test_audit_log_append_only(self):
        """Test audit log is append-only (doesn't overwrite)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "network_audit.log"
            behavior = NetworkAuditBehavior(audit_log_path=log_path)

            # Append first entry
            entry1 = {
                "timestamp": "2025-01-12T10:30:00",
                "command": "git push",
                "risk_level": "MEDIUM",
                "user_decision": "APPROVED",
                "staged_files": [],
                "sensitive_reads": []
            }
            behavior._audit_log_append(entry1)

            # Append second entry
            entry2 = {
                "timestamp": "2025-01-12T10:35:00",
                "command": "curl https://api.example.com",
                "risk_level": "LOW",
                "user_decision": "AUTO_APPROVED",
                "staged_files": [],
                "sensitive_reads": []
            }
            behavior._audit_log_append(entry2)

            # Verify both entries exist
            content = log_path.read_text()
            lines = content.strip().split("\n")
            assert len(lines) == 2
            assert "git push" in lines[0]
            assert "curl" in lines[1]

    def test_audit_log_immutability(self):
        """Test audit log entries cannot be modified after writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "network_audit.log"
            behavior = NetworkAuditBehavior(audit_log_path=log_path)

            # Append entry
            entry = {
                "timestamp": "2025-01-12T10:30:00",
                "command": "git push",
                "risk_level": "MEDIUM",
                "user_decision": "APPROVED",
                "staged_files": [],
                "sensitive_reads": []
            }
            behavior._audit_log_append(entry)

            # Read original content
            original_content = log_path.read_text()

            # Append another entry
            entry2 = {
                "timestamp": "2025-01-12T10:35:00",
                "command": "curl api",
                "risk_level": "LOW",
                "user_decision": "AUTO_APPROVED",
                "staged_files": [],
                "sensitive_reads": []
            }
            behavior._audit_log_append(entry2)

            # Verify original entry is still there and unchanged
            new_content = log_path.read_text()
            assert original_content in new_content  # Original preserved
            assert new_content.count("\n") == 2  # Two entries

    def test_audit_log_truncates_long_commands(self):
        """Test audit log truncates very long commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "network_audit.log"
            behavior = NetworkAuditBehavior(audit_log_path=log_path)

            # Create entry with long command
            long_command = "curl " + "a" * 500
            entry = {
                "timestamp": "2025-01-12T10:30:00",
                "command": long_command[:200],  # Already truncated by _handle_network_operation
                "risk_level": "LOW",
                "user_decision": "AUTO_APPROVED",
                "staged_files": [],
                "sensitive_reads": []
            }

            behavior._audit_log_append(entry)

            # Verify command is truncated
            content = log_path.read_text()
            assert len(content.split(" | ")[1]) <= 200


class TestRiskAnalysis:
    """Test risk analysis and level computation."""

    def test_risk_critical_upload_with_staged_credentials(self):
        """CRITICAL: Uploading with credentials in git staging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            behavior = NetworkAuditBehavior()

            # Mock agent with security context
            agent = Mock()
            agent.workspace = Path(tmpdir)
            agent.security_context = SecurityContext()

            # Mock git staging with sensitive files
            with patch.object(behavior, '_get_staged_files', return_value=[".env", "config.key"]):
                risk = behavior._analyze_risk_factors(agent, "git push origin main")

                assert risk["risk_level"] == "CRITICAL"
                assert ".env" in risk["staged_files"]
                assert "config.key" in risk["staged_files"]
                assert "staging" in risk["reason"].lower()

    def test_risk_high_upload_with_recent_reads(self):
        """HIGH: Uploading after reading sensitive files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            behavior = NetworkAuditBehavior()
            behavior.recent_sensitive_reads = [".env", "credentials.json"]

            agent = Mock()
            agent.workspace = Path(tmpdir)
            agent.security_context = SecurityContext()

            # No staged files
            with patch.object(behavior, '_get_staged_files', return_value=[]):
                risk = behavior._analyze_risk_factors(agent, "git push origin main")

                assert risk["risk_level"] == "HIGH"
                assert risk["recent_sensitive_reads"] == [".env", "credentials.json"]
                assert "reading sensitive" in risk["reason"].lower()

    def test_risk_medium_upload_operation(self):
        """MEDIUM: Any upload operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            behavior = NetworkAuditBehavior()

            agent = Mock()
            agent.workspace = Path(tmpdir)
            agent.security_context = SecurityContext()

            with patch.object(behavior, '_get_staged_files', return_value=[]):
                risk = behavior._analyze_risk_factors(agent, "git push origin main")

                assert risk["risk_level"] == "MEDIUM"
                assert "upload" in risk["reason"].lower()

    def test_risk_low_download_operation(self):
        """LOW: Download operations only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            behavior = NetworkAuditBehavior()

            agent = Mock()
            agent.workspace = Path(tmpdir)
            agent.security_context = SecurityContext()

            with patch.object(behavior, '_get_staged_files', return_value=[]):
                risk = behavior._analyze_risk_factors(agent, "git pull origin main")

                assert risk["risk_level"] == "LOW"
                assert "download" in risk["reason"].lower()

    def test_risk_analysis_detects_upload_commands(self):
        """Test upload detection for various commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            behavior = NetworkAuditBehavior()

            agent = Mock()
            agent.workspace = Path(tmpdir)
            agent.security_context = SecurityContext()

            with patch.object(behavior, '_get_staged_files', return_value=[]):
                # Upload commands
                assert behavior._analyze_risk_factors(agent, "git push")["risk_level"] in ["MEDIUM", "HIGH", "CRITICAL"]
                assert behavior._analyze_risk_factors(agent, "npm publish")["risk_level"] in ["MEDIUM", "HIGH", "CRITICAL"]
                assert behavior._analyze_risk_factors(agent, "scp file.txt server:/path/")["risk_level"] in ["MEDIUM", "HIGH", "CRITICAL"]
                assert behavior._analyze_risk_factors(agent, "rsync -avz /local/ server:/remote/")["risk_level"] in ["MEDIUM", "HIGH", "CRITICAL"]

    def test_risk_analysis_detects_download_commands(self):
        """Test download detection for various commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            behavior = NetworkAuditBehavior()

            agent = Mock()
            agent.workspace = Path(tmpdir)
            agent.security_context = SecurityContext()

            with patch.object(behavior, '_get_staged_files', return_value=[]):
                # Download commands
                assert behavior._analyze_risk_factors(agent, "git pull")["risk_level"] == "LOW"
                assert behavior._analyze_risk_factors(agent, "git fetch")["risk_level"] == "LOW"
                assert behavior._analyze_risk_factors(agent, "wget file.zip")["risk_level"] == "LOW"
                assert behavior._analyze_risk_factors(agent, "curl https://api.example.com")["risk_level"] == "LOW"


class TestGitStagingDetection:
    """Test git staging area detection."""

    def test_get_staged_files_detects_sensitive_files(self):
        """Detect sensitive files in git staging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmpdir, capture_output=True)

            # Create and stage sensitive files
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("SECRET=123")
            subprocess.run(["git", "add", ".env"], cwd=tmpdir, capture_output=True)

            key_file = Path(tmpdir) / "config.key"
            key_file.write_text("private_key")
            subprocess.run(["git", "add", "config.key"], cwd=tmpdir, capture_output=True)

            # Create and stage non-sensitive file
            safe_file = Path(tmpdir) / "README.md"
            safe_file.write_text("# Project")
            subprocess.run(["git", "add", "README.md"], cwd=tmpdir, capture_output=True)

            # Test detection
            behavior = NetworkAuditBehavior()
            agent = Mock()
            agent.workspace = Path(tmpdir)
            agent.security_context = SecurityContext()

            staged = behavior._get_staged_files(agent)

            # Should detect .env and config.key, but not README.md
            assert ".env" in staged
            assert "config.key" in staged
            assert "README.md" not in staged

    def test_get_staged_files_no_git_repo(self):
        """Handle case where no git repo exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            behavior = NetworkAuditBehavior()
            agent = Mock()
            agent.workspace = Path(tmpdir)
            agent.security_context = SecurityContext()

            staged = behavior._get_staged_files(agent)

            # Should return empty list, not crash
            assert staged == []

    def test_get_staged_files_empty_staging(self):
        """Handle case where staging area is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)

            behavior = NetworkAuditBehavior()
            agent = Mock()
            agent.workspace = Path(tmpdir)
            agent.security_context = SecurityContext()

            staged = behavior._get_staged_files(agent)

            # Should return empty list
            assert staged == []


class TestUserApprovalFlow:
    """Test user approval workflow."""

    @patch("builtins.input", return_value="yes")
    def test_request_approval_approved(self, mock_input):
        """Test user approves network operation."""
        behavior = NetworkAuditBehavior()
        agent = Mock()

        risk_analysis = {
            "risk_level": "MEDIUM",
            "reason": "Upload operation",
            "staged_files": [],
            "recent_sensitive_reads": []
        }

        approved = behavior._request_network_approval(agent, "git push", risk_analysis)
        assert approved is True

    @patch("builtins.input", return_value="no")
    def test_request_approval_denied(self, mock_input):
        """Test user denies network operation."""
        behavior = NetworkAuditBehavior()
        agent = Mock()

        risk_analysis = {
            "risk_level": "CRITICAL",
            "reason": "Credentials in staging",
            "staged_files": [".env"],
            "recent_sensitive_reads": []
        }

        approved = behavior._request_network_approval(agent, "git push", risk_analysis)
        assert approved is False

    @patch("builtins.input", return_value="y")
    def test_request_approval_short_yes(self, mock_input):
        """Test user approves with 'y'."""
        behavior = NetworkAuditBehavior()
        agent = Mock()

        risk_analysis = {
            "risk_level": "MEDIUM",
            "reason": "Upload operation",
            "staged_files": [],
            "recent_sensitive_reads": []
        }

        approved = behavior._request_network_approval(agent, "git push", risk_analysis)
        assert approved is True

    @patch("builtins.input", side_effect=EOFError)
    def test_request_approval_input_error(self, mock_input):
        """Test input error is treated as denial."""
        behavior = NetworkAuditBehavior()
        agent = Mock()

        risk_analysis = {
            "risk_level": "MEDIUM",
            "reason": "Upload operation",
            "staged_files": [],
            "recent_sensitive_reads": []
        }

        approved = behavior._request_network_approval(agent, "git push", risk_analysis)
        assert approved is False

    @patch("builtins.input", return_value="maybe")
    def test_request_approval_invalid_response(self, mock_input):
        """Test invalid response is treated as denial."""
        behavior = NetworkAuditBehavior()
        agent = Mock()

        risk_analysis = {
            "risk_level": "MEDIUM",
            "reason": "Upload operation",
            "staged_files": [],
            "recent_sensitive_reads": []
        }

        approved = behavior._request_network_approval(agent, "git push", risk_analysis)
        assert approved is False


class TestToolCallInterception:
    """Test on_tool_call interception (BEFORE execution)."""

    def test_intercepts_run_bash_network_command(self):
        """Test run_bash with network command is intercepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "network_audit.log"
            behavior = NetworkAuditBehavior(
                audit_log_path=log_path,
                require_approval=False  # Auto-approve for testing
            )

            agent = Mock()
            agent.workspace = Path(tmpdir)
            agent.security_context = SecurityContext()

            # Mock _get_staged_files
            with patch.object(behavior, '_get_staged_files', return_value=[]):
                # Should not raise (auto-approved)
                behavior.on_tool_call(agent, "run_bash", {"command": "git pull"}, {})

            # Verify audit log entry
            assert log_path.exists()
            content = log_path.read_text()
            assert "git pull" in content
            assert "AUTO_APPROVED" in content

    @patch("builtins.input", return_value="no")
    def test_intercepts_and_denies_network_command(self, mock_input):
        """Test network command is denied before execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "network_audit.log"
            behavior = NetworkAuditBehavior(
                audit_log_path=log_path,
                require_approval=True
            )

            agent = Mock()
            agent.workspace = Path(tmpdir)
            agent.security_context = SecurityContext()

            # Mock _get_staged_files
            with patch.object(behavior, '_get_staged_files', return_value=[]):
                # Should raise SecurityViolationError
                with pytest.raises(SecurityViolationError):
                    behavior.on_tool_call(agent, "run_bash", {"command": "git push"}, {})

            # Verify audit log shows DENIED
            content = log_path.read_text()
            assert "DENIED" in content

    def test_tracks_sensitive_file_reads(self):
        """Test read_file calls are tracked for risk analysis."""
        behavior = NetworkAuditBehavior()

        agent = Mock()
        agent.security_context = SecurityContext()

        # Simulate reading sensitive files
        behavior.on_tool_call(agent, "read_file", {"path": ".env"}, {})
        behavior.on_tool_call(agent, "read_file", {"path": "credentials.json"}, {})

        # Verify tracked
        assert ".env" in behavior.recent_sensitive_reads
        assert "credentials.json" in behavior.recent_sensitive_reads

    def test_ignores_safe_commands(self):
        """Test safe bash commands are not intercepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "network_audit.log"
            behavior = NetworkAuditBehavior(audit_log_path=log_path)

            agent = Mock()
            agent.workspace = Path(tmpdir)
            agent.security_context = SecurityContext()

            # Should not raise or log
            behavior.on_tool_call(agent, "run_bash", {"command": "ls -la"}, {})
            behavior.on_tool_call(agent, "run_bash", {"command": "pytest tests/"}, {})

            # Verify no audit log entries
            assert not log_path.exists()


class TestIntegration:
    """Integration test: Full workflow."""

    @patch("builtins.input", return_value="no")
    def test_full_workflow_with_credentials_staged(self, mock_input):
        """
        Integration: Agent reads .env, runs git add .env, runs git push.

        Expected:
        - Audit log shows staging detection
        - Risk analysis shows CRITICAL
        - User prompted with clear risk explanation
        - If denied, git push never executes
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmpdir, capture_output=True)

            # Create .env file
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("SECRET_KEY=abc123")

            # Stage .env
            subprocess.run(["git", "add", ".env"], cwd=tmpdir, capture_output=True)

            # Initialize behavior
            log_path = Path(tmpdir) / "network_audit.log"
            behavior = NetworkAuditBehavior(
                audit_log_path=log_path,
                require_approval=True,
                check_git_staging=True
            )

            # Initialize behavior for goal
            agent = Mock()
            agent.workspace = Path(tmpdir)
            agent.security_context = SecurityContext()
            behavior.on_goal_start(agent, "test goal")

            # Simulate reading .env
            behavior.on_tool_call(agent, "read_file", {"path": ".env"}, {})

            # Verify .env is tracked
            assert ".env" in behavior.recent_sensitive_reads

            # Attempt git push (should be denied)
            with pytest.raises(SecurityViolationError) as exc_info:
                behavior.on_tool_call(agent, "run_bash", {"command": "git push origin main"}, {})

            # Verify error message mentions risk level and reason
            error_msg = str(exc_info.value)
            assert "CRITICAL" in error_msg
            assert "denied" in error_msg.lower()

            # Verify audit log shows CRITICAL risk and DENIED
            content = log_path.read_text()
            assert "CRITICAL" in content
            assert "DENIED" in content
            assert ".env" in content


class TestBehaviorProperties:
    """Test Rule of Two properties."""

    def test_no_properties_defense_layer(self):
        """NetworkAuditBehavior is a defense layer (no properties)."""
        behavior = NetworkAuditBehavior()
        assert behavior.rule_of_two_properties == set()

    def test_behavior_name(self):
        """Test behavior name."""
        behavior = NetworkAuditBehavior()
        assert behavior.get_name() == "network_audit"
