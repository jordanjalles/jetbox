"""
Prompt injection detection behavior for Rule of Two defense-in-depth.

This is Defense Layer 1 of the Rule of Two security model.

The PromptInjectionDetectorBehavior intercepts file read operations and scans
content for known prompt injection patterns. It uses both pattern matching
and heuristic analysis to detect malicious instructions.

Detection Strategy:
- Pattern matching: 20+ regex patterns for known injection techniques
- Heuristic analysis: Statistical analysis of text characteristics
- Confidence scoring: 0.0 (benign) to 1.0 (obvious injection)

Response Actions:
- >0.8 confidence: Block (raise SecurityViolationError)
- 0.6-0.8 confidence: Warn (print warning, mark in context)
- <0.6 confidence: Log only (mark in context for tracking)

Key Design Principles:
- Prefer false negatives over false positives (usability matters)
- Clear, actionable warning messages
- Document known limitations (cannot detect all injections)
- Fail open (don't break agent on detection errors)
"""

import re
from typing import Any, TYPE_CHECKING
from behaviors.base import AgentBehavior
from behaviors.rule_of_two_types import RuleOfTwoProperty

if TYPE_CHECKING:
    from behaviors.base_agent import BaseAgent


class SecurityViolationError(Exception):
    """Raised when a high-confidence prompt injection is detected."""
    pass


class PromptInjectionDetectorBehavior(AgentBehavior):
    """
    Defense Layer 1: Prompt injection detection for untrusted inputs.

    This behavior intercepts read_file results and scans for prompt injection
    patterns. It's auto-injected by RuleOfTwoValidator for [ABC] agents.

    Rule of Two Properties:
    - This is a defense layer, not a capability behavior
    - Properties: [] (empty set - mitigates [A] but doesn't add properties)

    Detection Methods:
    1. Pattern matching: Known injection phrases/techniques
    2. Heuristic analysis: Statistical text analysis
    3. Confidence scoring: Combines both methods

    Response Actions:
    - >0.8 confidence: Block with SecurityViolationError
    - 0.6-0.8 confidence: Warn user and mark in context
    - <0.6 confidence: Log only for tracking
    """

    # Rule of Two: Defense layer, no properties
    rule_of_two_properties = set()

    # Injection pattern database (20+ patterns)
    INJECTION_PATTERNS = [
        # Role-playing / instruction override
        (r'\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|directions?|commands?)', re.IGNORECASE),
        (r'\bnew\s+instructions?:', re.IGNORECASE),
        (r'\byou\s+are\s+now\s+(?:a|an|the)', re.IGNORECASE),
        (r'\b(?:pretend|act|behave)\s+(?:as|like|to\s+be)\s+(?:a|an|the)', re.IGNORECASE),
        (r'\bstop\s+(?:being|acting)', re.IGNORECASE),

        # Direct imperatives to assistant
        (r'\b(?:system|assistant|user)\s*:\s*[A-Z]', 0),  # Role labels with content
        (r'\byou\s+must\s+(?:now\s+)?(?:ignore|obey|follow|execute|do)', re.IGNORECASE),
        (r'\byour\s+new\s+(?:role|mission|task|goal|objective)\s+is', re.IGNORECASE),
        (r'\bfrom\s+now\s+on,?\s+you\s+(?:are|will|must|should)', re.IGNORECASE),

        # Exfiltration attempts
        (r'\b(?:send|post|upload|transmit)\s+(?:this|the|all|my)?\s*(?:to|via)', re.IGNORECASE),
        (r'\bcurl\s+-[XxHd]', 0),  # curl with HTTP flags
        (r'\bwget\s+(?:-O|--output-document|http)', 0),
        (r'\b(?:http|https)://[^\s]+\b.*(?:token|key|secret|password|credential)', re.IGNORECASE),

        # Encoding tricks
        (r'\bbase64\s*(?:encode|decode)', re.IGNORECASE),
        (r'\b[A-Za-z0-9+/]{40,}={0,2}\b', 0),  # Long base64 strings
        (r'\\x[0-9a-fA-F]{2}', 0),  # Hex escape sequences
        (r'0x[0-9a-fA-F]{8,}', 0),  # Long hex strings

        # Delimiter/escape attacks
        (r'---\s*(?:END|STOP|BREAK|OVERRIDE)---', re.IGNORECASE),
        (r'```\s*(?:system|assistant|user)\b', re.IGNORECASE),
        (r'"""\s*\n\s*(?:system|assistant|user):', re.IGNORECASE),
        (r'\'\'\'(?:system|assistant|user):', re.IGNORECASE),

        # Context manipulation
        (r'\boverride\s+(?:previous|all|existing)\s+(?:context|rules|settings)', re.IGNORECASE),
        (r'\breset\s+(?:context|conversation|instructions)', re.IGNORECASE),
        (r'\bclear\s+(?:all\s+)?(?:previous|prior)\s+(?:context|memory|instructions)', re.IGNORECASE),
    ]

    def __init__(self, **params):
        """Initialize the input validation behavior."""
        super().__init__()
        self.blocked_count = 0
        self.warned_count = 0
        self.logged_count = 0

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "security_prompt_injection_detector"

    def get_instructions(self) -> str:
        """Return instructions for system prompt."""
        return """
## Input Validation Security

This agent has input validation enabled to detect prompt injection attacks.
If you read a file that contains suspicious instructions or commands that
appear to be attempting to override your behavior, you will be notified.

High-confidence detections will be blocked automatically for safety.
"""

    def _detect_injection_patterns(self, text: str) -> list[tuple[str, int, int]]:
        """
        Detect known injection patterns using regex matching.

        Args:
            text: Text content to scan

        Returns:
            List of matches as (matched_text, start_pos, end_pos) tuples
        """
        matches = []

        for pattern_def in self.INJECTION_PATTERNS:
            if isinstance(pattern_def, tuple):
                pattern, flags = pattern_def
            else:
                pattern, flags = pattern_def, 0

            try:
                regex = re.compile(pattern, flags)
                for match in regex.finditer(text):
                    matches.append((match.group(0), match.start(), match.end()))
            except re.error:
                # Skip invalid patterns
                continue

        return matches

    def _detect_injection_heuristics(self, text: str) -> float:
        """
        Detect injection attempts using heuristic analysis.

        Analyzes text characteristics that are common in prompt injections
        but rare in normal content:
        - Excessive imperative verbs per sentence
        - Excessive capitalization
        - Role-playing language patterns
        - Command-like syntax in prose

        Args:
            text: Text content to scan

        Returns:
            Confidence score 0.0-1.0 (0 = benign, 1 = obvious injection)
        """
        if not text or len(text.strip()) < 20:
            return 0.0

        score = 0.0

        # Count imperative verbs per sentence
        imperative_verbs = [
            'ignore', 'disregard', 'forget', 'pretend', 'act', 'behave',
            'stop', 'override', 'reset', 'clear', 'send', 'post', 'upload',
            'execute', 'run', 'do', 'must', 'obey', 'follow'
        ]

        # Split into sentences (rough approximation)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]

        if sentences:
            avg_imperatives_per_sentence = sum(
                sum(1 for verb in imperative_verbs if verb in sentence.lower())
                for sentence in sentences
            ) / len(sentences)

            if avg_imperatives_per_sentence > 2.5:
                score += 0.35
            elif avg_imperatives_per_sentence > 1.5:
                score += 0.2

        # Excessive capitalization (>30% uppercase letters)
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if caps_ratio > 0.4:
                score += 0.25
            elif caps_ratio > 0.3:
                score += 0.15

        # Role-playing language patterns
        role_playing_phrases = [
            'you are now', 'pretend to be', 'act as', 'behave like',
            'your new role', 'from now on', 'ignore previous'
        ]
        role_count = sum(1 for phrase in role_playing_phrases if phrase in text.lower())
        if role_count >= 3:
            score += 0.3
        elif role_count >= 2:
            score += 0.2

        # Command-like syntax (many lines starting with imperative verbs)
        lines = text.split('\n')
        lines = [l for l in lines if l.strip()]  # Filter empty lines
        if lines:
            command_lines = sum(
                1 for line in lines
                if any(
                    line.strip().lower().startswith(verb)
                    for verb in imperative_verbs
                )
            )
            if (command_lines / len(lines)) > 0.3:
                score += 0.2

        # Excessive special characters (encoding tricks)
        special_chars = sum(1 for c in text if c in '\\{}[]<>$`')
        if len(text) > 0 and (special_chars / len(text)) > 0.05:
            score += 0.1

        # Clamp to 0.0-1.0 range
        return min(1.0, score)

    def _compute_confidence(
        self,
        pattern_matches: list[tuple[str, int, int]],
        heuristic_score: float
    ) -> float:
        """
        Compute overall confidence score from pattern matches and heuristics.

        Combines both detection methods into a single confidence score.
        Pattern matches are weighted more heavily than heuristics.

        Args:
            pattern_matches: List of pattern matches
            heuristic_score: Heuristic analysis score (0.0-1.0)

        Returns:
            Overall confidence score 0.0-1.0
        """
        # Pattern matching is PRIMARY signal
        # Each match adds 0.35 (maxing at 1.0 for 3+ matches)
        pattern_score = min(1.0, len(pattern_matches) * 0.35)

        # Heuristic score adds additional weight
        heuristic_weight = heuristic_score * 0.5

        # Combined score
        confidence = pattern_score + heuristic_weight

        # Boost: If we have multiple patterns, high confidence
        if len(pattern_matches) >= 2:
            confidence += 0.2
        # Single pattern but strong heuristic
        elif len(pattern_matches) == 1 and heuristic_score > 0.3:
            confidence += 0.1

        return min(1.0, confidence)

    def _handle_detection(
        self,
        agent: "BaseAgent",
        file_path: str,
        confidence: float,
        pattern_matches: list[tuple[str, int, int]]
    ) -> None:
        """
        Handle detected injection based on confidence level.

        Actions:
        - >0.8: Block (raise SecurityViolationError)
        - 0.6-0.8: Warn (print warning, mark in context)
        - <0.6: Log only (mark in context)

        Args:
            agent: Agent instance
            file_path: Path to file with suspected injection
            confidence: Confidence score 0.0-1.0
            pattern_matches: List of pattern matches for context

        Raises:
            SecurityViolationError: If confidence > 0.8
        """
        context = agent.security_context

        if confidence >= 0.75:
            # Block: High confidence injection
            self.blocked_count += 1
            context.mark_prompt_injection_detected(file_path)

            # Build detailed error message
            patterns_str = "\n".join(
                f"  - '{match[0][:80]}...' at position {match[1]}"
                if len(match[0]) > 80 else f"  - '{match[0]}' at position {match[1]}"
                for match in pattern_matches[:5]  # Show first 5
            )

            error_msg = f"""
╔═══════════════════════════════════════════════════════════════╗
║  SECURITY ALERT: Prompt Injection Detected                   ║
╚═══════════════════════════════════════════════════════════════╝

File: {file_path}
Confidence: {confidence * 100:.1f}%

Detected patterns:
{patterns_str}

This file contains content that appears to be a prompt injection attack.
Reading this file has been BLOCKED for safety.

If this is a false positive, you can:
1. Review the file manually to verify it's safe
2. Adjust trust level: agent.security_context.update_trust_level("isolated")
3. Disable input validation (not recommended)

Learn more: docs/SECURITY_INPUT_VALIDATION.md
"""
            raise SecurityViolationError(error_msg)

        elif confidence >= 0.45:
            # Warn: Medium confidence
            self.warned_count += 1
            context.mark_prompt_injection_detected(file_path)

            patterns_str = ", ".join(
                f"'{match[0][:40]}...'"
                if len(match[0]) > 40 else f"'{match[0]}'"
                for match in pattern_matches[:3]
            )

            print(f"""
⚠️  SECURITY WARNING: Possible prompt injection detected
    File: {file_path}
    Confidence: {confidence * 100:.2f}%
    Patterns: {patterns_str}

    This file contains suspicious content. Proceed with caution.
    The file has been read, but marked in security context.
""")

        else:
            # Log only: Low confidence (tracked for analysis)
            self.logged_count += 1
            # Don't mark in context for low confidence (reduces noise)

    def on_tool_call(
        self,
        agent: "BaseAgent",
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any]
    ) -> None:
        """
        Intercept read_file results and scan for prompt injection.

        This hook is called after every tool execution. We scan read_file
        results for prompt injection patterns.

        Args:
            agent: Agent instance
            tool_name: Name of tool that was called
            args: Tool arguments
            result: Tool result
        """
        # Only scan read_file results
        if tool_name != "read_file":
            return

        # Check if read was successful and has content
        content = result.get("content", "")
        if not content or not isinstance(content, str):
            return

        file_path = args.get("path", "<unknown>")

        try:
            # Detect injection patterns
            pattern_matches = self._detect_injection_patterns(content)
            heuristic_score = self._detect_injection_heuristics(content)
            confidence = self._compute_confidence(pattern_matches, heuristic_score)

            # Handle based on confidence
            if confidence >= 0.45:
                self._handle_detection(agent, file_path, confidence, pattern_matches)

        except SecurityViolationError:
            # Re-raise security violations
            raise

        except Exception as e:
            # Fail open: Don't break agent on detection errors
            print(f"⚠️  Input validation error: {e}")
            print("    Continuing without validation (fail-open)")

    def on_goal_complete(
        self,
        agent: "BaseAgent",
        success: bool,
        summary: str
    ) -> None:
        """
        Log detection statistics at goal completion.

        Args:
            agent: Agent instance
            success: Whether goal completed successfully
            summary: Completion summary
        """
        if self.blocked_count > 0 or self.warned_count > 0 or self.logged_count > 0:
            print(f"""
Input Validation Statistics:
  Blocked: {self.blocked_count} high-confidence injections
  Warned: {self.warned_count} medium-confidence detections
  Logged: {self.logged_count} low-confidence detections
""")
