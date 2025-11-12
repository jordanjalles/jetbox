"""
Security context for Rule of Two enforcement.

Tracks runtime security context that affects which Rule of Two properties
apply to behaviors. This enables context-aware, dynamic property resolution.

Key Concepts:
- workspace_trust_level: User-controlled trust level of workspace files
- sensitive_data_detected: Whether sensitive files (.env, keys) have been seen
- Agent monitors and can suggest trust level changes to user
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SecurityContext:
    """
    Runtime security context for Rule of Two property resolution.

    This context is passed to behaviors when computing their Rule of Two
    properties, enabling dynamic, context-aware security decisions.

    User-Controlled Settings:
    -------------------------
    workspace_trust_level: Set via IS_SANDBOX env var (aligns with Claude Code)
        - "isolated": IS_SANDBOX=1 → Agent-created sandbox, files are trusted (no prompt injection risk)
        - "user": IS_SANDBOX=0 or not set → User workspace with pre-existing files (untrusted, injection risk)
        - "mixed": Programmatically set → Combination, track per-file using WorkspaceManager.created_files

    Reactively Detected:
    -------------------
    sensitive_data_detected: Set to True when .env, *.key, credentials files accessed
    sensitive_files_seen: Accumulates list of sensitive file paths accessed
    prompt_injection_detected: Set when InputValidationBehavior detects injection
    injection_sources: Files that contained suspected prompt injection

    Future Expansion:
    ----------------
    network_policy: Network access restrictions (none/restricted/full)
    enforcement_level: How strictly to enforce violations (off/warn/block)
    """

    # User-controlled via IS_SANDBOX env var (IS_SANDBOX=1 → isolated, else → user)
    workspace_trust_level: Literal["isolated", "user", "mixed"] = "user"

    # Detected reactively during execution
    sensitive_data_detected: bool = False
    sensitive_files_seen: set[str] = field(default_factory=set)

    # Prompt injection tracking (Phase 4A)
    prompt_injection_detected: bool = False
    injection_sources: list[str] = field(default_factory=list)

    # Future expansion (placeholders for now)
    network_policy: Literal["none", "restricted", "full"] = "full"
    enforcement_level: Literal["off", "warn", "block"] = "warn"

    def mark_sensitive_file_accessed(self, file_path: str) -> None:
        """
        Mark that a sensitive file has been accessed.

        Updates sensitive_data_detected flag and adds to seen list.
        Called by ReadFileToolsBehavior when reading .env, *.key, etc.

        Args:
            file_path: Path to sensitive file that was accessed
        """
        self.sensitive_data_detected = True
        self.sensitive_files_seen.add(file_path)

    def mark_prompt_injection_detected(self, file_path: str) -> None:
        """
        Mark that prompt injection was detected in a file.

        Updates prompt_injection_detected flag and records source.
        Called by InputValidationBehavior (Phase 4A).

        Args:
            file_path: Path to file containing suspected injection
        """
        self.prompt_injection_detected = True
        if file_path not in self.injection_sources:
            self.injection_sources.append(file_path)

    def is_sensitive_file(self, file_path: str) -> bool:
        """
        Check if a file path matches sensitive file patterns.

        Patterns:
        - .env* (e.g., .env, .env.local, .env.production)
        - *.key, *.pem (SSH keys, certificates)
        - credentials*, *credentials* (credential files)
        - .aws/, .ssh/ (AWS/SSH config directories)
        - *secret*, *token*, *password* (secret-containing files)

        Args:
            file_path: Path to check

        Returns:
            True if file matches sensitive patterns, False otherwise
        """
        path_lower = file_path.lower()

        # Environment files
        if ".env" in path_lower:
            return True

        # Key/certificate files
        if path_lower.endswith((".key", ".pem", ".p12", ".pfx")):
            return True

        # Credential files
        if "credential" in path_lower:
            return True

        # AWS/SSH directories
        if "/.aws/" in path_lower or "/.ssh/" in path_lower:
            return True

        # Secret-containing files
        if any(keyword in path_lower for keyword in ["secret", "token", "password", "apikey"]):
            return True

        return False

    def update_trust_level(self, new_level: Literal["isolated", "user", "mixed"]) -> None:
        """
        Update workspace trust level (typically user-initiated).

        Called when user approves trust level change suggested by agent.

        Args:
            new_level: New trust level to set
        """
        self.workspace_trust_level = new_level

    def __repr__(self) -> str:
        """Readable representation for debugging."""
        return (
            f"SecurityContext("
            f"trust={self.workspace_trust_level}, "
            f"sensitive={self.sensitive_data_detected}, "
            f"injection={self.prompt_injection_detected})"
        )
