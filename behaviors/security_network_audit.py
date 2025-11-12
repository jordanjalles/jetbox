"""
Defense Layer 3: Network Audit - Exfiltration Prevention.

Prevents unauthorized network operations that could leak sensitive data.
Detects network-capable commands BEFORE execution and requires user approval
with risk analysis (credentials in git staging, recent sensitive reads).

Key Features:
- Network command detection (15+ patterns: git push, curl, wget, etc.)
- Immutable audit log (write-only, append-only for forensics)
- Risk analysis (CRITICAL/HIGH/MEDIUM/LOW based on context)
- Git staging detection (flag commits containing credentials)
- User approval flow (block MEDIUM+ risk operations)

This is a defense layer, not a capability behavior:
- No tool registration (monitors existing run_bash tool)
- No properties added (pure defense)
- Intercepts on_tool_call BEFORE execution
"""
from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from behaviors.base import AgentBehavior

if TYPE_CHECKING:
    from behaviors.base_agent import BaseAgent


class SecurityViolationError(Exception):
    """Raised when a network operation is denied by security policy."""
    pass


class NetworkAuditBehavior(AgentBehavior):
    """
    Defense Layer 3: Network audit and exfiltration prevention.

    Monitors all bash commands for network operations and requires user
    approval before execution. Analyzes risk factors (credentials in git
    staging, recent sensitive file reads) and logs all decisions to an
    immutable audit log.

    Security: Defense layer (no properties)
    - Does not add capabilities
    - Only monitors and blocks existing capabilities
    - Pure defensive behavior

    Hook: on_tool_call (BEFORE execution via early return)
    - Intercepts run_bash calls
    - Detects network commands
    - Performs risk analysis
    - Requests user approval if risky
    - Logs to immutable audit log
    - Raises SecurityViolationError if denied
    """

    # Rule of Two: No properties (defense layer, not capability)
    rule_of_two_properties = set()

    # Network command patterns (15+ commands)
    NETWORK_COMMANDS = {
        # Git operations (most common exfiltration vector)
        "git push", "git pull", "git fetch", "git clone",

        # HTTP clients
        "curl", "wget", "http", "httpie",

        # Network tools
        "nc", "netcat", "telnet", "ssh", "scp", "rsync", "sftp",

        # Python networking (detect python scripts with network libs)
        "python.*requests", "python.*urllib", "python.*socket",

        # Package managers (can upload to registries)
        "pip install.*--index-url", "npm publish", "poetry publish",

        # Cloud CLIs (upload to cloud storage)
        "aws s3 cp", "gcloud storage cp", "az storage", "gsutil cp",
    }

    def __init__(
        self,
        audit_log_path: Path | None = None,
        require_approval: bool = True,
        check_git_staging: bool = True,
        **kwargs
    ):
        """
        Initialize NetworkAuditBehavior.

        Args:
            audit_log_path: Path to immutable audit log (default: .agent_context/network_audit.log)
            require_approval: Whether to require user approval for network ops (default: True)
            check_git_staging: Whether to check git staging for sensitive files (default: True)
            **kwargs: Additional parameters (ignored)
        """
        self.audit_log_path = audit_log_path
        self.require_approval = require_approval
        self.check_git_staging = check_git_staging

        # Track recent file reads for risk analysis
        self.recent_sensitive_reads: list[str] = []

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "network_audit"

    def on_goal_start(self, agent: "BaseAgent", goal: str) -> None:
        """
        Initialize audit log on goal start.

        Args:
            agent: Agent instance
            goal: Goal string
        """
        # Set up audit log path if not provided
        if self.audit_log_path is None:
            context_dir = agent.workspace / ".agent_context"
            context_dir.mkdir(exist_ok=True)
            self.audit_log_path = context_dir / "network_audit.log"

        # Reset recent reads tracker
        self.recent_sensitive_reads = []

    def on_tool_call(
        self,
        agent: "BaseAgent",
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any]
    ) -> None:
        """
        Intercept tool calls to detect and audit network operations.

        TIMING: Called BEFORE tool execution (early interception).

        For run_bash:
        - Detect network commands
        - Perform risk analysis
        - Request user approval if needed
        - Log decision to audit log
        - Raise SecurityViolationError if denied

        For read_file:
        - Track sensitive file reads for risk analysis

        Args:
            agent: Agent instance
            tool_name: Name of tool being called
            args: Tool arguments
            result: Tool result (unused for pre-execution)

        Raises:
            SecurityViolationError: If network operation is denied
        """
        # Track sensitive file reads
        if tool_name == "read_file":
            path = args.get("path", "")
            context = getattr(agent, "security_context", None)
            if context and context.is_sensitive_file(path):
                self.recent_sensitive_reads.append(path)

        # Intercept run_bash for network operations
        if tool_name == "run_bash":
            command = args.get("command", "")
            if self._is_network_command(command):
                self._handle_network_operation(agent, command)

    def _is_network_command(self, command: str) -> bool:
        """
        Detect if a command is network-capable.

        Patterns:
        - Git operations: "git push", "git pull", etc.
        - HTTP clients: "curl", "wget", etc.
        - Network tools: "nc", "ssh", "scp", etc.
        - Python networking: "python.*requests", etc.
        - Package managers: "pip install.*--index-url", "npm publish", etc.
        - Cloud CLIs: "aws s3 cp", "gcloud storage cp", etc.

        Args:
            command: Bash command string

        Returns:
            True if command is network-capable, False otherwise
        """
        command_lower = command.lower()

        # Check each pattern
        for pattern in self.NETWORK_COMMANDS:
            # Convert pattern to regex (handle wildcards)
            regex_pattern = pattern.replace(".*", ".*?")
            if re.search(regex_pattern, command_lower):
                return True

        return False

    def _handle_network_operation(
        self,
        agent: "BaseAgent",
        command: str
    ) -> None:
        """
        Handle detected network operation.

        Workflow:
        1. Analyze risk factors (git staging, recent reads)
        2. Compute risk level (CRITICAL/HIGH/MEDIUM/LOW)
        3. If MEDIUM+: Request user approval
        4. Log decision to audit log
        5. If denied: Raise SecurityViolationError

        Args:
            agent: Agent instance
            command: Network command being executed

        Raises:
            SecurityViolationError: If operation is denied
        """
        # Analyze risk factors
        risk_analysis = self._analyze_risk_factors(agent, command)

        # Request approval if required and risk is MEDIUM+
        if self.require_approval and risk_analysis["risk_level"] in ["MEDIUM", "HIGH", "CRITICAL"]:
            approved = self._request_network_approval(agent, command, risk_analysis)

            # Log decision
            self._audit_log_append({
                "timestamp": datetime.now().isoformat(),
                "command": command[:200],  # Truncate long commands
                "risk_level": risk_analysis["risk_level"],
                "user_decision": "APPROVED" if approved else "DENIED",
                "staged_files": risk_analysis["staged_files"],
                "sensitive_reads": risk_analysis["recent_sensitive_reads"]
            })

            # Raise error if denied
            if not approved:
                raise SecurityViolationError(
                    f"Network operation denied by user: {command[:100]}\n"
                    f"Risk level: {risk_analysis['risk_level']}\n"
                    f"Reason: {risk_analysis['reason']}"
                )
        else:
            # Log LOW risk operations (no approval needed)
            self._audit_log_append({
                "timestamp": datetime.now().isoformat(),
                "command": command[:200],
                "risk_level": risk_analysis["risk_level"],
                "user_decision": "AUTO_APPROVED",
                "staged_files": risk_analysis["staged_files"],
                "sensitive_reads": risk_analysis["recent_sensitive_reads"]
            })

    def _analyze_risk_factors(
        self,
        agent: "BaseAgent",
        command: str
    ) -> dict[str, Any]:
        """
        Analyze risk factors for a network operation.

        Risk Levels:
        - CRITICAL: Uploading with credentials in git staging
        - HIGH: Uploading with recent sensitive reads
        - MEDIUM: Any upload operation
        - LOW: Download operations only

        Factors:
        - Git staging: Any .env, *.key, credentials.json staged?
        - Recent reads: Did agent just read sensitive files?
        - Operation type: Upload (push/cp) or download (pull/fetch)?

        Args:
            agent: Agent instance
            command: Network command being executed

        Returns:
            Dict with:
            - risk_level: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
            - reason: Human-readable explanation
            - staged_files: List of staged sensitive files
            - recent_sensitive_reads: List of recently read sensitive files
        """
        # Check git staging if enabled
        staged_files = []
        if self.check_git_staging:
            staged_files = self._get_staged_files(agent)

        # Detect operation type
        is_upload = any(op in command.lower() for op in [
            "push", "publish", "cp", "upload", "scp", "rsync"
        ])
        is_download = any(op in command.lower() for op in [
            "pull", "fetch", "clone", "download", "wget", "curl"
        ])

        # Compute risk level
        if is_upload and staged_files:
            # CRITICAL: Uploading with credentials in staging
            return {
                "risk_level": "CRITICAL",
                "reason": f"Attempting to upload with sensitive files in git staging: {staged_files}",
                "staged_files": staged_files,
                "recent_sensitive_reads": self.recent_sensitive_reads.copy()
            }
        elif is_upload and self.recent_sensitive_reads:
            # HIGH: Uploading with recent sensitive reads
            return {
                "risk_level": "HIGH",
                "reason": f"Attempting to upload after reading sensitive files: {self.recent_sensitive_reads}",
                "staged_files": staged_files,
                "recent_sensitive_reads": self.recent_sensitive_reads.copy()
            }
        elif is_upload:
            # MEDIUM: Any upload operation
            return {
                "risk_level": "MEDIUM",
                "reason": "Upload operation detected",
                "staged_files": staged_files,
                "recent_sensitive_reads": self.recent_sensitive_reads.copy()
            }
        elif is_download:
            # LOW: Download only
            return {
                "risk_level": "LOW",
                "reason": "Download operation (low risk)",
                "staged_files": staged_files,
                "recent_sensitive_reads": self.recent_sensitive_reads.copy()
            }
        else:
            # MEDIUM: Unknown network operation
            return {
                "risk_level": "MEDIUM",
                "reason": "Unknown network operation",
                "staged_files": staged_files,
                "recent_sensitive_reads": self.recent_sensitive_reads.copy()
            }

    def _get_staged_files(self, agent: "BaseAgent") -> list[str]:
        """
        Get list of sensitive files in git staging area.

        Checks:
        - .env* files
        - *.key, *.pem files
        - credentials.json
        - Any file matching sensitive patterns

        Args:
            agent: Agent instance

        Returns:
            List of staged sensitive file paths
        """
        try:
            # Get staged files using git diff --cached --name-only
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(agent.workspace)
            )

            if result.returncode != 0:
                return []

            # Filter for sensitive files
            staged_files = result.stdout.strip().split("\n")
            sensitive_staged = []

            context = getattr(agent, "security_context", None)
            for file in staged_files:
                if file and context and context.is_sensitive_file(file):
                    sensitive_staged.append(file)

            return sensitive_staged

        except Exception:
            # Git not available or not in git repo
            return []

    def _request_network_approval(
        self,
        agent: "BaseAgent",
        command: str,
        risk_analysis: dict[str, Any]
    ) -> bool:
        """
        Request user approval for network operation.

        Displays:
        - Command being executed
        - Risk level (CRITICAL/HIGH/MEDIUM/LOW)
        - Risk factors (staged files, recent reads)
        - Clear explanation of why this is risky

        Args:
            agent: Agent instance
            command: Network command being executed
            risk_analysis: Risk analysis dict

        Returns:
            True if approved, False if denied
        """
        # Build approval prompt
        print("\n" + "=" * 70)
        print("NETWORK OPERATION DETECTED - USER APPROVAL REQUIRED")
        print("=" * 70)
        print(f"\nCommand: {command}\n")
        print(f"Risk Level: {risk_analysis['risk_level']}")
        print(f"Reason: {risk_analysis['reason']}\n")

        # Show risk factors
        if risk_analysis['staged_files']:
            print("⚠️  Sensitive files in git staging:")
            for file in risk_analysis['staged_files']:
                print(f"    - {file}")
            print()

        if risk_analysis['recent_sensitive_reads']:
            print("⚠️  Recently read sensitive files:")
            for file in risk_analysis['recent_sensitive_reads']:
                print(f"    - {file}")
            print()

        # Request approval
        print("Allow this network operation? (yes/no): ", end="", flush=True)

        # Get user input
        try:
            response = input().strip().lower()
            return response in ["yes", "y"]
        except (EOFError, KeyboardInterrupt):
            # Treat input errors as denial
            print("\nInput error - operation denied")
            return False

    def _audit_log_append(self, entry: dict[str, Any]) -> None:
        """
        Append entry to immutable audit log.

        Log Format:
        timestamp | command | risk_level | user_decision | staged_files | sensitive_reads

        Immutability:
        - Append-only (never modify existing entries)
        - Write-only (no deletion)
        - For forensics after security incidents

        Args:
            entry: Log entry dict with timestamp, command, risk_level, user_decision, etc.
        """
        if not self.audit_log_path:
            return

        # Format entry as pipe-delimited line
        line = " | ".join([
            entry["timestamp"],
            entry["command"],
            entry["risk_level"],
            entry["user_decision"],
            ",".join(entry.get("staged_files", [])) or "none",
            ",".join(entry.get("sensitive_reads", [])) or "none"
        ])

        # Append to log (create if doesn't exist)
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            # Log to stderr but don't crash
            print(f"[network_audit] Warning: Failed to write audit log: {e}")
