"""
RuleOfTwoValidator - Meta-behavior for Rule of Two enforcement.

This behavior validates that an agent's behavior configuration complies
with the Rule of Two security principle. If the agent has all three
properties [ABC], it either:
1. Raises SecurityError (if risk not acknowledged)
2. Auto-injects defense-in-depth layers (if risk acknowledged)

Rule of Two Principle:
An autonomous agent should satisfy NO MORE than two of:
- [A] Process untrustworthy inputs
- [B] Access sensitive systems/data
- [C] Change state or communicate externally

This is a meta-behavior that analyzes other behaviors, not a tool provider.
"""

from typing import Any, TYPE_CHECKING
from behaviors.base import AgentBehavior
from behaviors.rule_of_two_types import RuleOfTwoProperty

if TYPE_CHECKING:
    from behaviors.security_context import SecurityContext


class SecurityViolationError(Exception):
    """Raised when Rule of Two is violated without risk acknowledgment."""
    pass


class RuleOfTwoValidator(AgentBehavior):
    """
    Meta-behavior that enforces Rule of Two security principle.

    This behavior:
    1. Collects Rule of Two properties from all agent behaviors (context-aware)
    2. Detects [ABC] trifecta (all three properties present)
    3. Checks security config for risk acknowledgment
    4. Auto-injects defense layers if [ABC] + acknowledged
    5. Raises SecurityError if [ABC] + not acknowledged

    Usage:
        This behavior is automatically added to agents when security is enabled.
        It runs during agent initialization to validate configuration.
    """

    # Rule of Two: No properties - this is a validator, not a capability provider
    rule_of_two_properties = set()

    def __init__(self, security_config: dict | None = None):
        """
        Initialize Rule of Two validator.

        Args:
            security_config: Security configuration dict (from security_defaults.yaml)
                            If None, loads from agent_config
        """
        self.security_config = security_config or {}
        self.enforcement_level = self._get_enforcement_level()
        self.collected_properties: set[RuleOfTwoProperty] = set()
        self.has_abc_trifecta = False

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "rule_of_two_validator"

    def _get_enforcement_level(self) -> str:
        """
        Get enforcement level from config.

        Returns:
            "off" | "warn" | "block"
        """
        rule_of_two_config = self.security_config.get("rule_of_two", {})
        return rule_of_two_config.get("enforcement", "block")

    def _collect_properties(
        self,
        agent: "BaseAgent",
        context: "SecurityContext"
    ) -> set[RuleOfTwoProperty]:
        """
        Collect Rule of Two properties from all agent behaviors.

        Calls get_rule_of_two_properties() on each behavior with context,
        enabling context-aware dynamic property resolution.

        Args:
            agent: Agent instance with behaviors loaded
            context: SecurityContext for dynamic property resolution

        Returns:
            Set of all Rule of Two properties across all behaviors
        """
        all_properties = set()

        for behavior in agent.behaviors:
            # Skip self to avoid recursion
            if behavior is self:
                continue

            # Call context-aware method (falls back to static attribute)
            behavior_props = behavior.get_rule_of_two_properties(agent, context)
            all_properties.update(behavior_props)

        return all_properties

    def _detect_abc_trifecta(self, properties: set[RuleOfTwoProperty]) -> bool:
        """
        Detect if all three Rule of Two properties are present.

        Args:
            properties: Set of collected properties

        Returns:
            True if [ABC] trifecta detected (all three properties present)
        """
        all_three = {
            RuleOfTwoProperty.UNTRUSTED_INPUT,
            RuleOfTwoProperty.SENSITIVE_ACCESS,
            RuleOfTwoProperty.EXTERNAL_ACTION
        }
        return properties == all_three

    def _format_properties_display(self, properties: set[RuleOfTwoProperty]) -> str:
        """
        Format properties as [ABC] string for display.

        Args:
            properties: Set of properties

        Returns:
            Formatted string like "[AB]" or "[ABC]"
        """
        if not properties:
            return "[]"
        letters = sorted([str(prop) for prop in properties])
        return f"[{''.join(letters)}]"

    def _get_rich_error_message(
        self,
        properties: set[RuleOfTwoProperty],
        agent: "BaseAgent"
    ) -> str:
        """
        Generate detailed error message for Rule of Two violation.

        Args:
            properties: Collected properties
            agent: Agent instance

        Returns:
            Formatted error message with solutions
        """
        props_display = self._format_properties_display(properties)

        message = f"""
╔════════════════════════════════════════════════════════════════════════╗
║                    RULE OF TWO SECURITY VIOLATION                       ║
╚════════════════════════════════════════════════════════════════════════╝

Agent Configuration: {props_display} (UNTRUSTED_INPUT + SENSITIVE_ACCESS + EXTERNAL_ACTION)

⚠️  This agent has ALL THREE Rule of Two properties, creating high security risk:
   [A] UNTRUSTED_INPUT    - Reads files that may contain prompt injection
   [B] SENSITIVE_ACCESS   - Can access credentials, API keys, environment vars
   [C] EXTERNAL_ACTION    - Can write files, run commands, communicate externally

Risk: Malicious input → credential harvesting → exfiltration in one agent

═══════════════════════════════════════════════════════════════════════════

SOLUTION OPTIONS:

1. ACKNOWLEDGE RISK + Enable Defense-in-Depth (Recommended)
   Set in agent config YAML:
   ```yaml
   security:
     rule_of_two:
       acknowledge_abc_risk: true
       skip_defense_in_depth: false  # Enable 3 protection layers
   ```

   This automatically enables:
   - Layer 1: Input validation (prompt injection detection)
   - Layer 2: Access auditing (credential harvesting detection)
   - Layer 3: Network audit (exfiltration prevention)

2. SPLIT AGENT CAPABILITIES (Most Secure)
   Separate into multiple agents:
   - Agent A: Read files [AB] + Delegate
   - Agent B: Write files + Run commands [BC]
   - Orchestrator coordinates between them

3. REDUCE CAPABILITIES (Quick Fix)
   Remove one property:
   - Remove read_file → [BC] (no untrusted input)
   - Remove credential access → [AC] (no sensitive data)
   - Remove write/command → [AB] (read-only agent)

═══════════════════════════════════════════════════════════════════════════

Current Environment:
- Workspace Trust: {agent.security_context.workspace_trust_level}
- IS_SANDBOX: {agent.security_context.workspace_trust_level == 'isolated'}
- Enforcement: {self.enforcement_level}

Behaviors with properties:
"""
        # List each behavior and its properties
        for behavior in agent.behaviors:
            if behavior is self:
                continue
            props = behavior.get_rule_of_two_properties(agent, agent.security_context)
            if props:
                props_str = self._format_properties_display(props)
                message += f"  - {behavior.get_name()}: {props_str}\n"

        message += "\n"
        message += "See: docs/SECURITY_RULE_OF_TWO.md for complete documentation\n"
        message += "═" * 75 + "\n"

        return message

    def _check_risk_acknowledged(self, agent: "BaseAgent") -> bool:
        """
        Check if [ABC] risk has been acknowledged in config.

        Args:
            agent: Agent instance with config

        Returns:
            True if risk acknowledged in config
        """
        # Check agent-specific config first
        if hasattr(agent, 'config') and agent.config:
            security_config = getattr(agent.config, 'security', None)
            if security_config:
                rule_of_two = getattr(security_config, 'rule_of_two', None)
                if rule_of_two:
                    acknowledge = getattr(rule_of_two, 'acknowledge_abc_risk', False)
                    if acknowledge:
                        return True

        # Check global security config
        rule_of_two_config = self.security_config.get("rule_of_two", {})
        return rule_of_two_config.get("acknowledge_abc_risk", False)

    def _should_skip_defense_layers(self, agent: "BaseAgent") -> bool:
        """
        Check if defense-in-depth layers should be skipped.

        Even with acknowledged risk, user can opt out of auto-injection.

        Args:
            agent: Agent instance

        Returns:
            True if user explicitly skipped defense layers
        """
        # Check agent-specific config
        if hasattr(agent, 'config') and agent.config:
            security_config = getattr(agent.config, 'security', None)
            if security_config:
                rule_of_two = getattr(security_config, 'rule_of_two', None)
                if rule_of_two:
                    skip = getattr(rule_of_two, 'skip_defense_in_depth', False)
                    if skip:
                        return True

        # Check global config
        rule_of_two_config = self.security_config.get("rule_of_two", {})
        return rule_of_two_config.get("skip_defense_in_depth", False)

    def _inject_defense_layers(self, agent: "BaseAgent") -> None:
        """
        Auto-inject defense-in-depth behaviors for [ABC] agent.

        Injects (if enabled in config):
        - InputValidationBehavior (Layer 1)
        - SensitiveAccessAuditorBehavior (Layer 2)
        - NetworkAuditBehavior (Layer 3)

        Args:
            agent: Agent instance

        Note:
            This is called AFTER behaviors are loaded but BEFORE agent starts.
            Defense layers are added to end of behavior list.
        """
        defense_config = self.security_config.get("rule_of_two", {}).get("defense_in_depth", {})

        injected_count = 0

        # Layer 1: Input Validation (Phase 4A)
        if defense_config.get("input_validation", {}).get("enabled", True):
            try:
                from behaviors.security_input_validation import InputValidationBehavior
                agent.behaviors.append(InputValidationBehavior())
                injected_count += 1
                print(f"[{agent.name}]   → Layer 1: Input Validation (prompt injection detection)")
            except ImportError as e:
                print(f"[{agent.name}]   ⚠️  Layer 1: Failed to import InputValidationBehavior: {e}")

        # Layer 2: Access Auditing (Phase 4B)
        if defense_config.get("access_auditing", {}).get("enabled", True):
            try:
                from behaviors.security_access_auditor import SensitiveAccessAuditorBehavior
                agent.behaviors.append(SensitiveAccessAuditorBehavior())
                injected_count += 1
                print(f"[{agent.name}]   → Layer 2: Access Auditing (credential harvesting detection)")
            except ImportError as e:
                print(f"[{agent.name}]   ⚠️  Layer 2: Failed to import SensitiveAccessAuditorBehavior: {e}")

        # Layer 3: Network Audit (Phase 4C)
        if defense_config.get("network_audit", {}).get("enabled", True):
            try:
                from behaviors.security_network_audit import NetworkAuditBehavior
                agent.behaviors.append(NetworkAuditBehavior())
                injected_count += 1
                print(f"[{agent.name}]   → Layer 3: Network Audit (exfiltration prevention)")
            except ImportError as e:
                print(f"[{agent.name}]   ⚠️  Layer 3: Failed to import NetworkAuditBehavior: {e}")

        if injected_count > 0:
            print(f"[{agent.name}] Defense-in-depth: {injected_count} layer(s) auto-injected")
        else:
            print(f"[{agent.name}] Defense-in-depth: No layers enabled")

    def validate_agent_configuration(self, agent: "BaseAgent") -> None:
        """
        Validate agent's Rule of Two configuration.

        This is the main entry point called during agent initialization.

        Workflow:
        1. Collect properties from all behaviors (context-aware)
        2. Detect [ABC] trifecta
        3. If [ABC]:
           a. Check if risk acknowledged
           b. If not acknowledged → Raise SecurityError
           c. If acknowledged → Auto-inject defense layers (unless skipped)
        4. Log validation result

        Args:
            agent: Agent instance with behaviors loaded

        Raises:
            SecurityViolationError: If [ABC] detected without risk acknowledgment
        """
        # Collect properties (context-aware)
        self.collected_properties = self._collect_properties(
            agent,
            agent.security_context
        )

        # Detect [ABC] trifecta
        self.has_abc_trifecta = self._detect_abc_trifecta(self.collected_properties)

        props_display = self._format_properties_display(self.collected_properties)

        # Log collected properties
        print(f"[{agent.name}] Rule of Two validation: {props_display}")

        # If not [ABC], validation passes
        if not self.has_abc_trifecta:
            print(f"[{agent.name}]   ✓ Compliant (≤2 properties)")
            return

        # [ABC] detected - check enforcement level
        if self.enforcement_level == "off":
            print(f"[{agent.name}]   ⚠️  [ABC] detected but enforcement OFF")
            return

        # Check if risk acknowledged
        risk_acknowledged = self._check_risk_acknowledged(agent)

        if not risk_acknowledged:
            # Enforcement: warn or block
            error_message = self._get_rich_error_message(self.collected_properties, agent)

            if self.enforcement_level == "block":
                raise SecurityViolationError(error_message)
            else:  # warn
                print(f"[{agent.name}]   ⚠️  [ABC] WARNING:")
                print(error_message)
                return

        # Risk acknowledged - check if defense layers should be injected
        skip_defense = self._should_skip_defense_layers(agent)

        if skip_defense:
            print(f"[{agent.name}]   ⚠️  [ABC] risk acknowledged, defense layers SKIPPED")
        else:
            print(f"[{agent.name}]   ⚠️  [ABC] risk acknowledged, injecting defense layers...")
            self._inject_defense_layers(agent)

    def on_goal_start(self, agent: "BaseAgent", goal: str) -> None:
        """
        Validate configuration when goal starts.

        This ensures validation happens after all behaviors are loaded
        but before the agent starts processing.

        Args:
            agent: Agent instance
            goal: Goal string
        """
        # Only validate if security is enabled
        security_enabled = self.security_config.get("enabled", False)

        if not security_enabled:
            disabled_msg = self.security_config.get("rule_of_two", {}).get(
                "disabled_message",
                "Security features disabled"
            )
            print(f"[{agent.name}] Security: DISABLED")
            return

        # Run validation
        self.validate_agent_configuration(agent)
