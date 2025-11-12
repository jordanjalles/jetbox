"""
Type definitions for the Rule of Two security model.

The Rule of Two principle states that an autonomous agent should satisfy
no more than two of the following properties:
    [A] Process untrustworthy inputs
    [B] Access sensitive systems/data
    [C] Change state or communicate externally

Agents satisfying all three [ABC] must acknowledge risks and receive
defense-in-depth protections (3 security layers).
"""

from enum import Enum


class RuleOfTwoProperty(Enum):
    """
    Properties that contribute to the Rule of Two security model.

    An agent should have at most 2 of these 3 properties to be considered
    safe without additional security measures.
    """

    UNTRUSTED_INPUT = "A"
    """
    [A] Process untrustworthy inputs

    The agent reads files, data, or content that could originate from
    external/untrusted sources (user files, downloaded content, etc.).
    This creates risk of prompt injection attacks.

    Examples:
    - Reading user-created files
    - Processing external data sources
    - Reading repository contents that may contain malicious READMEs
    """

    SENSITIVE_ACCESS = "B"
    """
    [B] Access sensitive systems/data

    The agent can read credentials, environment variables, system files,
    or other sensitive information that should not be exposed.

    Examples:
    - Reading .env files
    - Accessing environment variables
    - Reading SSH keys or API tokens
    - System file access
    """

    EXTERNAL_ACTION = "C"
    """
    [C] Change state or communicate externally

    The agent can modify the system state or communicate with external
    services, enabling potential exfiltration or unauthorized actions.

    Examples:
    - Writing files
    - Executing bash commands
    - Network operations (git push, HTTP requests)
    - Starting/stopping servers
    """

    def __str__(self) -> str:
        """Return the single-letter code (A, B, or C)."""
        return self.value

    def __repr__(self) -> str:
        """Return readable representation."""
        return f"RuleOfTwoProperty.{self.name}"
