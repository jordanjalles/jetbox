"""
Test script to verify global behavior defaults are loaded correctly.

This script verifies:
1. Global defaults are loaded from agent_config.yaml
2. Behaviors receive the correct parameter values from global defaults
3. Agent-specific overrides work correctly (if present)
"""

from pathlib import Path
from task_executor_agent import TaskExecutorAgent
from orchestrator_agent import OrchestratorAgent
from architect_agent import ArchitectAgent
import yaml


def test_global_defaults_loaded():
    """Verify that global defaults are loaded from agent_config.yaml."""
    print("\n=== Test 1: Global defaults loaded from agent_config.yaml ===")

    config_path = Path("agent_config.yaml")
    if not config_path.exists():
        print("❌ agent_config.yaml not found")
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "behavior_defaults" not in config:
        print("❌ behavior_defaults section not found in agent_config.yaml")
        return False

    defaults = config["behavior_defaults"]
    print(f"✓ Found behavior_defaults section with {len(defaults)} behavior types")

    # Verify expected behaviors are present
    expected_behaviors = [
        "CompactWhenNearFullBehavior",
        "LoopDetectionBehavior",
        "CommandToolsBehavior",
    ]

    for behavior_type in expected_behaviors:
        if behavior_type in defaults:
            print(f"✓ {behavior_type}: {defaults[behavior_type]}")
        else:
            print(f"❌ {behavior_type} not found in defaults")
            return False

    print("✓ All expected behavior defaults found\n")
    return True


def test_task_executor_behavior_params():
    """Verify TaskExecutor behaviors get correct parameters from global defaults."""
    print("\n=== Test 2: TaskExecutor behavior parameters ===")

    # Create agent in test workspace
    test_workspace = Path("/tmp/test_behavior_defaults")
    test_workspace.mkdir(parents=True, exist_ok=True)

    agent = TaskExecutorAgent(
        workspace=test_workspace,
        goal="test",
        use_behaviors=True
    )

    # Find CompactWhenNearFullBehavior
    compact_behavior = None
    for b in agent.behaviors:
        if b.get_name() == "compact_when_near_full":
            compact_behavior = b
            break

    if not compact_behavior:
        print("❌ CompactWhenNearFullBehavior not found")
        return False

    # Verify it has correct params from global defaults
    expected_max_tokens = 8000
    expected_compact_threshold = 0.75
    expected_keep_recent = 20

    if compact_behavior.max_tokens != expected_max_tokens:
        print(f"❌ max_tokens: expected {expected_max_tokens}, got {compact_behavior.max_tokens}")
        return False
    print(f"✓ CompactWhenNearFullBehavior.max_tokens = {compact_behavior.max_tokens}")

    if compact_behavior.compact_threshold != expected_compact_threshold:
        print(f"❌ compact_threshold: expected {expected_compact_threshold}, got {compact_behavior.compact_threshold}")
        return False
    print(f"✓ CompactWhenNearFullBehavior.compact_threshold = {compact_behavior.compact_threshold}")

    if compact_behavior.keep_recent != expected_keep_recent:
        print(f"❌ keep_recent: expected {expected_keep_recent}, got {compact_behavior.keep_recent}")
        return False
    print(f"✓ CompactWhenNearFullBehavior.keep_recent = {compact_behavior.keep_recent}")

    # Find LoopDetectionBehavior
    loop_behavior = None
    for b in agent.behaviors:
        if b.get_name() == "loop_detection":
            loop_behavior = b
            break

    if not loop_behavior:
        print("❌ LoopDetectionBehavior not found")
        return False

    expected_max_repeats = 5
    if loop_behavior.max_repeats != expected_max_repeats:
        print(f"❌ max_repeats: expected {expected_max_repeats}, got {loop_behavior.max_repeats}")
        return False
    print(f"✓ LoopDetectionBehavior.max_repeats = {loop_behavior.max_repeats}")

    # Find CommandToolsBehavior
    command_behavior = None
    for b in agent.behaviors:
        if b.get_name() == "command_tools":
            command_behavior = b
            break

    if not command_behavior:
        print("❌ CommandToolsBehavior not found")
        return False

    # CommandToolsBehavior doesn't store whitelist as an attribute, so we can't verify it directly
    # But we can verify the behavior was loaded without errors
    print("✓ CommandToolsBehavior loaded successfully")

    print("✓ All TaskExecutor behavior parameters correct\n")
    return True


def test_orchestrator_behavior_params():
    """Verify Orchestrator behaviors get correct parameters from global defaults."""
    print("\n=== Test 3: Orchestrator behavior parameters ===")

    # Create agent in test workspace
    test_workspace = Path("/tmp/test_behavior_defaults_orch")
    test_workspace.mkdir(parents=True, exist_ok=True)

    agent = OrchestratorAgent(
        workspace=test_workspace,
        use_behaviors=True
    )

    # Find CompactWhenNearFullBehavior
    compact_behavior = None
    for b in agent.behaviors:
        if b.get_name() == "compact_when_near_full":
            compact_behavior = b
            break

    if not compact_behavior:
        print("❌ CompactWhenNearFullBehavior not found")
        return False

    # Verify it has correct params from global defaults
    expected_max_tokens = 8000
    expected_compact_threshold = 0.75

    if compact_behavior.max_tokens != expected_max_tokens:
        print(f"❌ max_tokens: expected {expected_max_tokens}, got {compact_behavior.max_tokens}")
        return False
    print(f"✓ CompactWhenNearFullBehavior.max_tokens = {compact_behavior.max_tokens}")

    if compact_behavior.compact_threshold != expected_compact_threshold:
        print(f"❌ compact_threshold: expected {expected_compact_threshold}, got {compact_behavior.compact_threshold}")
        return False
    print(f"✓ CompactWhenNearFullBehavior.compact_threshold = {compact_behavior.compact_threshold}")

    # Find LoopDetectionBehavior
    loop_behavior = None
    for b in agent.behaviors:
        if b.get_name() == "loop_detection":
            loop_behavior = b
            break

    if not loop_behavior:
        print("❌ LoopDetectionBehavior not found")
        return False

    expected_max_repeats = 5
    if loop_behavior.max_repeats != expected_max_repeats:
        print(f"❌ max_repeats: expected {expected_max_repeats}, got {loop_behavior.max_repeats}")
        return False
    print(f"✓ LoopDetectionBehavior.max_repeats = {loop_behavior.max_repeats}")

    print("✓ All Orchestrator behavior parameters correct\n")
    return True


def test_architect_behavior_params():
    """Verify Architect behaviors get correct parameters from global defaults."""
    print("\n=== Test 4: Architect behavior parameters ===")

    # Create agent in test workspace
    test_workspace = Path("/tmp/test_behavior_defaults_arch")
    test_workspace.mkdir(parents=True, exist_ok=True)

    agent = ArchitectAgent(
        workspace=test_workspace,
        use_behaviors=True
    )

    # Find CompactWhenNearFullBehavior
    compact_behavior = None
    for b in agent.behaviors:
        if b.get_name() == "compact_when_near_full":
            compact_behavior = b
            break

    if not compact_behavior:
        print("❌ CompactWhenNearFullBehavior not found")
        return False

    # Verify it has correct params from global defaults
    expected_max_tokens = 8000
    expected_compact_threshold = 0.75

    if compact_behavior.max_tokens != expected_max_tokens:
        print(f"❌ max_tokens: expected {expected_max_tokens}, got {compact_behavior.max_tokens}")
        return False
    print(f"✓ CompactWhenNearFullBehavior.max_tokens = {compact_behavior.max_tokens}")

    if compact_behavior.compact_threshold != expected_compact_threshold:
        print(f"❌ compact_threshold: expected {expected_compact_threshold}, got {compact_behavior.compact_threshold}")
        return False
    print(f"✓ CompactWhenNearFullBehavior.compact_threshold = {compact_behavior.compact_threshold}")

    # Find LoopDetectionBehavior
    loop_behavior = None
    for b in agent.behaviors:
        if b.get_name() == "loop_detection":
            loop_behavior = b
            break

    if not loop_behavior:
        print("❌ LoopDetectionBehavior not found")
        return False

    expected_max_repeats = 5
    if loop_behavior.max_repeats != expected_max_repeats:
        print(f"❌ max_repeats: expected {expected_max_repeats}, got {loop_behavior.max_repeats}")
        return False
    print(f"✓ LoopDetectionBehavior.max_repeats = {loop_behavior.max_repeats}")

    print("✓ All Architect behavior parameters correct\n")
    return True


def test_override_mechanism():
    """Verify that agent-specific overrides work correctly."""
    print("\n=== Test 5: Override mechanism (optional) ===")

    # Create a test config with an override
    test_config = Path("/tmp/test_override_config.yaml")
    test_config.write_text("""
behaviors:
  - type: LoopDetectionBehavior
    params:
      max_repeats: 10  # Override global default (5)

  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 16000  # Override global default (8000)
""")

    # Create a minimal agent to test loading
    test_workspace = Path("/tmp/test_override")
    test_workspace.mkdir(parents=True, exist_ok=True)

    from base_agent import BaseAgent

    class TestAgent(BaseAgent):
        def get_tools(self):
            return []

        def get_system_prompt(self):
            return "test"

        def get_context_strategy(self):
            return "hierarchical"

        def build_context(self):
            return [{"role": "system", "content": "test"}]

    agent = TestAgent(
        name="test_agent",
        role="test",
        workspace=test_workspace,
        config=None
    )

    # Load behaviors from override config
    agent.load_behaviors_from_config(str(test_config))

    # Find LoopDetectionBehavior
    loop_behavior = None
    for b in agent.behaviors:
        if b.get_name() == "loop_detection":
            loop_behavior = b
            break

    if not loop_behavior:
        print("❌ LoopDetectionBehavior not found")
        return False

    # Verify override worked
    expected_max_repeats = 10  # Override value
    if loop_behavior.max_repeats != expected_max_repeats:
        print(f"❌ max_repeats override failed: expected {expected_max_repeats}, got {loop_behavior.max_repeats}")
        return False
    print(f"✓ LoopDetectionBehavior.max_repeats = {loop_behavior.max_repeats} (overridden)")

    # Find CompactWhenNearFullBehavior
    compact_behavior = None
    for b in agent.behaviors:
        if b.get_name() == "compact_when_near_full":
            compact_behavior = b
            break

    if not compact_behavior:
        print("❌ CompactWhenNearFullBehavior not found")
        return False

    # Verify override worked
    expected_max_tokens = 16000  # Override value
    if compact_behavior.max_tokens != expected_max_tokens:
        print(f"❌ max_tokens override failed: expected {expected_max_tokens}, got {compact_behavior.max_tokens}")
        return False
    print(f"✓ CompactWhenNearFullBehavior.max_tokens = {compact_behavior.max_tokens} (overridden)")

    # Clean up test config
    test_config.unlink()

    print("✓ Override mechanism works correctly\n")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Global Behavior Defaults System")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Global defaults loaded", test_global_defaults_loaded()))
    results.append(("TaskExecutor parameters", test_task_executor_behavior_params()))
    results.append(("Orchestrator parameters", test_orchestrator_behavior_params()))
    results.append(("Architect parameters", test_architect_behavior_params()))
    results.append(("Override mechanism", test_override_mechanism()))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Global behavior defaults system working correctly.")
        exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. See output above for details.")
        exit(1)
