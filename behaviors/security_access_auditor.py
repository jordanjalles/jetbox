"""
Defense Layer 2: Access Auditing - Detect credential harvesting patterns.

This behavior monitors sensitive file access patterns and detects anomalies
that may indicate credential harvesting attacks via prompt injection.

Key Features:
- Session tracking of all sensitive file accesses
- Threshold-based anomaly detection (>2 credentials = suspicious)
- User approval workflow with clear explanation
- Integration with SecurityContext for audit trail
- Context-aware thresholds (isolated workspace = more lenient)

Design Philosophy:
- Conservative threshold to avoid blocking legitimate use
- User always in control (can override)
- Clear explanation of why flagged
- Respects workspace trust (isolated = more trusted)
"""

from typing import Any, TYPE_CHECKING
from dataclasses import dataclass, field
import sys

from behaviors.base import AgentBehavior

if TYPE_CHECKING:
    from behaviors.base_agent import BaseAgent


@dataclass
class SessionStats:
    """
    Track sensitive file accesses during current session.

    Fields:
        files_accessed: List of (path, file_type) tuples
        user_approved: Whether user has approved continued access
        approval_session: True if user granted "always for this session"
    """
    files_accessed: list[tuple[str, str]] = field(default_factory=list)
    user_approved: bool = False
    approval_session: bool = False


class SensitiveAccessAuditorBehavior(AgentBehavior):
    """
    Defense Layer 2: Monitors sensitive file access and detects anomalies.

    This behavior intercepts read_file tool calls and tracks access to
    sensitive files (.env, credentials, keys, etc.). When an anomaly is
    detected (>2 different sensitive files in short time), it warns the
    user and requires approval before continuing.

    Security Properties:
    - No Rule of Two properties (defense layer, not capability)
    - Acts as safety layer on top of ReadFileToolsBehavior

    Detection Logic:
    - Track all sensitive file reads in session_stats
    - Threshold: >2 different sensitive files = anomaly
    - Context-aware: isolated workspace = higher threshold
    - User approval: Block further access until user responds

    Integration:
    - Updates agent.security_context.sensitive_files_seen
    - Logs all sensitive accesses for audit trail
    - Respects workspace trust level
    """

    # Rule of Two: Defense layer (no properties)
    rule_of_two_properties = set()

    # Sensitive file patterns (aligned with SecurityContext)
    SENSITIVE_PATTERNS = {
        # Environment files
        "env": [".env", ".env.local", ".env.production", ".env.development",
                ".env.test", ".env.staging"],
        # Credentials
        "credentials": ["credentials.json", "credentials.yaml", "credentials.yml",
                       "aws_credentials", "gcloud-credentials", "gcp-credentials",
                       "azure-credentials"],
        # Keys and certificates
        "keys": [".key", ".pem", ".p12", ".pfx", "id_rsa", "id_dsa", "id_ecdsa",
                "id_ed25519", "*.key", "*.pem"],
        # Config directories
        "config_dirs": ["/.aws/", "/.ssh/", "/.config/gcloud/", "/.kube/",
                       "/.docker/"],
        # Secret keywords
        "secrets": ["secret", "token", "apikey", "api_key", "password",
                   "passwd", "auth_key", "private_key"]
    }

    # Anomaly thresholds
    THRESHOLD_DEFAULT = 2  # >2 files = anomaly
    THRESHOLD_ISOLATED = 4  # Higher threshold for isolated workspaces

    def __init__(self, **kwargs):
        """
        Initialize SensitiveAccessAuditorBehavior.

        Args:
            **kwargs: Additional parameters (ignored)
        """
        self.session_stats = SessionStats()

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "security_access_auditor"

    def on_goal_start(self, agent: "BaseAgent", goal: str) -> None:
        """
        Reset session stats at goal start.

        Args:
            agent: Agent instance
            goal: Goal string
        """
        self.session_stats = SessionStats()

    def on_tool_call(
        self,
        agent: "BaseAgent",
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any]
    ) -> None:
        """
        Intercept read_file calls and track sensitive file access.

        This is the main hook where we detect and respond to sensitive
        file access patterns.

        Args:
            agent: Agent instance
            tool_name: Name of tool that was called
            args: Tool arguments
            result: Tool result
        """
        # Only monitor read_file calls
        if tool_name != "read_file":
            return

        # Only track successful reads
        if "error" in result or not result.get("success", False):
            return

        # Get file path from args
        file_path = args.get("path", "")
        if not file_path:
            return

        # Check if file is sensitive
        if not self._is_sensitive_file(file_path):
            return

        # Classify file type
        file_type = self._classify_file_type(file_path)

        # Track in session stats
        self.session_stats.files_accessed.append((file_path, file_type))

        # Update SecurityContext
        if hasattr(agent, "security_context"):
            agent.security_context.mark_sensitive_file_accessed(file_path)

        # Check for anomaly (skip if already approved)
        if not self.session_stats.user_approved:
            if self._check_anomaly(agent):
                # Anomaly detected - require user approval
                self._handle_anomaly(agent, self.session_stats.files_accessed)

    def _is_sensitive_file(self, file_path: str) -> bool:
        """
        Check if file matches sensitive patterns.

        Uses same patterns as SecurityContext.is_sensitive_file() for
        consistency.

        Args:
            file_path: Path to check

        Returns:
            True if file is sensitive, False otherwise
        """
        path_lower = file_path.lower()

        # Environment files
        for pattern in self.SENSITIVE_PATTERNS["env"]:
            if pattern in path_lower:
                return True

        # Credentials
        for pattern in self.SENSITIVE_PATTERNS["credentials"]:
            if pattern in path_lower:
                return True

        # Keys/certificates
        key_extensions = tuple(
            p.replace("*", "")
            for p in self.SENSITIVE_PATTERNS["keys"]
            if not p.startswith("*")
        )
        for pattern in self.SENSITIVE_PATTERNS["keys"]:
            if path_lower.endswith(key_extensions):
                return True
            if pattern in path_lower:
                return True

        # Config directories
        for pattern in self.SENSITIVE_PATTERNS["config_dirs"]:
            if pattern in path_lower:
                return True

        # Secret keywords
        for pattern in self.SENSITIVE_PATTERNS["secrets"]:
            if pattern in path_lower:
                return True

        return False

    def _classify_file_type(self, file_path: str) -> str:
        """
        Classify sensitive file type for reporting.

        Args:
            file_path: Path to classify

        Returns:
            Human-readable file type (e.g., "environment variable", "SSH key")
        """
        path_lower = file_path.lower()

        # AWS (check before generic credentials)
        if "/.aws/" in path_lower:
            return "AWS credentials"

        # Environment files
        if ".env" in path_lower:
            return "environment variables"

        # SSH keys (check before generic keys)
        ssh_keys = ["id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"]
        if any(key in path_lower for key in ssh_keys):
            return "SSH private key"

        # Generic keys
        if path_lower.endswith((".key", ".pem")):
            return "private key"

        # Certificates
        if path_lower.endswith((".p12", ".pfx")):
            return "certificate"

        # Credentials
        if "credential" in path_lower:
            return "credentials file"

        # Secrets
        if "secret" in path_lower:
            return "secret"

        if "token" in path_lower or "apikey" in path_lower or "api_key" in path_lower:
            return "API token"

        if "password" in path_lower or "passwd" in path_lower:
            return "password file"

        return "sensitive file"

    def _check_anomaly(self, agent: "BaseAgent") -> bool:
        """
        Check if access pattern indicates anomaly.

        Threshold:
        - Default (user workspace): >2 different files = anomaly
        - Isolated workspace: >4 different files = anomaly

        Args:
            agent: Agent instance

        Returns:
            True if anomaly detected, False otherwise
        """
        # Get threshold based on workspace trust
        threshold = self.THRESHOLD_DEFAULT
        if hasattr(agent, "security_context"):
            if agent.security_context.workspace_trust_level == "isolated":
                threshold = self.THRESHOLD_ISOLATED

        # Count unique files accessed
        unique_files = len(set(path for path, _ in self.session_stats.files_accessed))

        return unique_files > threshold

    def _handle_anomaly(
        self,
        agent: "BaseAgent",
        files_accessed: list[tuple[str, str]]
    ) -> None:
        """
        Handle detected anomaly - warn user and get approval.

        Displays:
        - List of all sensitive files accessed
        - File types (SSH keys, credentials, etc.)
        - Explanation of why this is suspicious
        - Approval prompt

        Args:
            agent: Agent instance
            files_accessed: List of (path, file_type) tuples
        """
        print("\n" + "=" * 70)
        print("⚠️  SECURITY WARNING: Suspicious Credential Access Pattern Detected")
        print("=" * 70)
        print()
        print("The agent has accessed multiple sensitive files in this session:")
        print()

        # List all files with types
        for i, (path, file_type) in enumerate(files_accessed, 1):
            print(f"  {i}. {path}")
            print(f"     Type: {file_type}")
            print()

        # Explain why suspicious
        print("Why is this flagged?")
        print("  • Accessing multiple credential files is unusual")
        print("  • This pattern may indicate credential harvesting")
        print("  • Could be legitimate OR prompt injection attack")
        print()

        # Get user decision
        print("What would you like to do?")
        print("  [a] Allow continued access for this session")
        print("  [d] Deny and stop agent")
        print("  [o] Allow this once, prompt again next time")
        print()

        # Read user input
        try:
            choice = input("Choice [a/d/o]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "d"

        print()

        if choice == "a":
            # Approve for session
            self.session_stats.user_approved = True
            self.session_stats.approval_session = True
            print("✓ Access approved for this session")
        elif choice == "o":
            # Approve once
            self.session_stats.user_approved = True
            self.session_stats.approval_session = False
            print("✓ Access approved once")
        else:
            # Deny (default)
            print("✗ Access denied - stopping agent")
            print("=" * 70)
            print()
            sys.exit(1)

        print("=" * 70)
        print()
