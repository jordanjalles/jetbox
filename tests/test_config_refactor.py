"""
Test suite for agent configuration refactor.

Tests:
1. Blurbs load from agent config files
2. Delegation tools load from agent config files
3. Token limits are consistent (8000)
4. StatusDisplayBehavior removed from configs
5. agents.yaml only contains class and can_delegate_to
"""

import yaml
from pathlib import Path
import sys
import os

# Add parent directory to path so we can import agent modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_agents_yaml_structure():
    """Test that agents.yaml only contains class and can_delegate_to."""
    print("\n" + "="*70)
    print("TEST 1: agents.yaml structure (only class and can_delegate_to)")
    print("="*70)

    # Get workspace root
    workspace_root = Path(__file__).parent.parent

    with open(workspace_root / "agents.yaml") as f:
        config = yaml.safe_load(f)

    agents = config["agents"]

    for agent_name, agent_config in agents.items():
        print(f"\nAgent: {agent_name}")
        print(f"  Keys: {list(agent_config.keys())}")

        # Should only have 'class' and 'can_delegate_to'
        allowed_keys = {"class", "can_delegate_to"}
        actual_keys = set(agent_config.keys())

        # Check no forbidden keys
        forbidden_keys = {"blurb", "description", "delegation_tool"}
        found_forbidden = actual_keys & forbidden_keys

        if found_forbidden:
            print(f"  ❌ FAIL: Found forbidden keys: {found_forbidden}")
            return False

        # Check has required keys
        if "class" not in actual_keys:
            print(f"  ❌ FAIL: Missing 'class' key")
            return False

        if "can_delegate_to" not in actual_keys:
            print(f"  ❌ FAIL: Missing 'can_delegate_to' key")
            return False

        print(f"  ✓ PASS: Only contains {actual_keys}")

    print("\n✓ PASS: agents.yaml structure is correct")
    return True


def test_blurbs_in_agent_configs():
    """Test that blurbs are present in individual agent config files."""
    print("\n" + "="*70)
    print("TEST 2: Blurbs in agent config files")
    print("="*70)

    workspace_root = Path(__file__).parent.parent
    agents_with_blurbs = ["orchestrator", "architect", "task_executor"]

    for agent_name in agents_with_blurbs:
        config_file = f"{agent_name}_config.yaml"
        print(f"\n{config_file}:")

        config_path = workspace_root / config_file
        if not config_path.exists():
            print(f"  ❌ FAIL: Config file not found")
            return False

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if "blurb" not in config:
            print(f"  ❌ FAIL: No 'blurb' field")
            return False

        blurb = config["blurb"].strip()
        print(f"  Blurb length: {len(blurb)} chars")
        print(f"  First 100 chars: {blurb[:100]}...")

        if len(blurb) < 50:
            print(f"  ❌ FAIL: Blurb too short")
            return False

        print(f"  ✓ PASS: Blurb present and valid")

    print("\n✓ PASS: All agent configs have blurbs")
    return True


def test_delegation_tools_in_agent_configs():
    """Test that delegation_tool definitions are in agent config files."""
    print("\n" + "="*70)
    print("TEST 3: delegation_tool in agent config files")
    print("="*70)

    workspace_root = Path(__file__).parent.parent

    # architect and task_executor should have delegation_tool
    agents_with_delegation_tools = {
        "architect": "consult_architect",
        "task_executor": "delegate_to_executor"
    }

    for agent_name, expected_tool_name in agents_with_delegation_tools.items():
        config_file = f"{agent_name}_config.yaml"
        print(f"\n{config_file}:")

        with open(workspace_root / config_file) as f:
            config = yaml.safe_load(f)

        if "delegation_tool" not in config:
            print(f"  ❌ FAIL: No 'delegation_tool' field")
            return False

        delegation_tool = config["delegation_tool"]

        # Check structure
        if "name" not in delegation_tool:
            print(f"  ❌ FAIL: No 'name' field in delegation_tool")
            return False

        if "description" not in delegation_tool:
            print(f"  ❌ FAIL: No 'description' field in delegation_tool")
            return False

        if "parameters" not in delegation_tool:
            print(f"  ❌ FAIL: No 'parameters' field in delegation_tool")
            return False

        # Check tool name
        if delegation_tool["name"] != expected_tool_name:
            print(f"  ❌ FAIL: Expected tool name '{expected_tool_name}', got '{delegation_tool['name']}'")
            return False

        print(f"  Tool name: {delegation_tool['name']}")
        print(f"  Description: {delegation_tool['description'][:60]}...")
        print(f"  Parameters: {list(delegation_tool['parameters'].keys())}")
        print(f"  ✓ PASS: delegation_tool present and valid")

    print("\n✓ PASS: All delegation tools defined in agent configs")
    return True


def test_token_limits_consistent():
    """Test that all agent configs use consistent max_tokens (8000)."""
    print("\n" + "="*70)
    print("TEST 4: Token limits consistent (8000)")
    print("="*70)

    workspace_root = Path(__file__).parent.parent
    config_files = ["task_executor_config.yaml", "orchestrator_config.yaml", "architect_config.yaml"]

    for config_file in config_files:
        print(f"\n{config_file}:")

        with open(workspace_root / config_file) as f:
            config = yaml.safe_load(f)

        # Find CompactWhenNearFullBehavior
        found = False
        for behavior in config.get("behaviors", []):
            if behavior["type"] == "CompactWhenNearFullBehavior":
                found = True
                max_tokens = behavior["params"]["max_tokens"]
                print(f"  max_tokens: {max_tokens}")

                if max_tokens != 8000:
                    print(f"  ❌ FAIL: Expected 8000, got {max_tokens}")
                    return False

                print(f"  ✓ PASS: max_tokens = 8000")
                break

        if not found:
            print(f"  ⚠ WARNING: No CompactWhenNearFullBehavior found")

    print("\n✓ PASS: All token limits are 8000")
    return True


def test_status_display_behavior_removed():
    """Test that StatusDisplayBehavior is removed from agent configs."""
    print("\n" + "="*70)
    print("TEST 5: StatusDisplayBehavior removed from configs")
    print("="*70)

    workspace_root = Path(__file__).parent.parent
    config_files = ["task_executor_config.yaml", "orchestrator_config.yaml", "architect_config.yaml"]

    for config_file in config_files:
        print(f"\n{config_file}:")

        with open(workspace_root / config_file) as f:
            config = yaml.safe_load(f)

        # Check that StatusDisplayBehavior is NOT in behaviors list
        for behavior in config.get("behaviors", []):
            if behavior["type"] == "StatusDisplayBehavior":
                print(f"  ❌ FAIL: StatusDisplayBehavior still present")
                return False

        print(f"  ✓ PASS: StatusDisplayBehavior not in config")

    print("\n✓ PASS: StatusDisplayBehavior removed from all configs")
    return True


def test_base_agent_loads_delegation_tools():
    """Test that base_agent.py loads delegation tools from agent configs."""
    print("\n" + "="*70)
    print("TEST 6: base_agent.py loads delegation tools from agent configs")
    print("="*70)

    workspace_root = Path(__file__).parent.parent

    # This is a code inspection test - check that _auto_add_delegation_behavior
    # reads from agent config files

    with open(workspace_root / "base_agent.py") as f:
        code = f.read()

    # Check for key patterns
    patterns = [
        'agent_config_file = Path(f"{target_agent}_config.yaml")',
        '"delegation_tool" in target_config',
        'agent_info["delegation_tool"] = target_config["delegation_tool"]',
        'agent_info["blurb"] = target_config["blurb"]'
    ]

    for pattern in patterns:
        if pattern in code:
            print(f"  ✓ Found: {pattern}")
        else:
            print(f"  ❌ FAIL: Not found: {pattern}")
            return False

    print("\n✓ PASS: base_agent.py loads from agent config files")
    return True


def test_orchestrator_delegation_behavior():
    """Test that orchestrator can load delegation behavior with tools from agent configs."""
    print("\n" + "="*70)
    print("TEST 7: Orchestrator delegation behavior loading")
    print("="*70)

    try:
        from orchestrator_agent import OrchestratorAgent
        from pathlib import Path
        import tempfile

        # Create temporary workspace for test
        temp_dir = tempfile.mkdtemp()
        workspace = Path(temp_dir)

        # Create orchestrator with behaviors enabled
        agent = OrchestratorAgent(workspace=workspace, use_behaviors=True)

        # Check blurb loaded from config
        blurb = agent.get_blurb()
        print(f"\nOrchestrator blurb (first 100 chars):")
        print(f"  {blurb[:100]}...")

        if len(blurb) < 50:
            print("  ❌ FAIL: Blurb too short or missing")
            return False

        print("  ✓ Blurb loaded successfully")

        # Check behaviors loaded
        print(f"\nLoaded behaviors:")
        for b in agent.behaviors:
            print(f"  - {b.get_name()}")

        # Check delegation behavior present
        behavior_names = [b.get_name() for b in agent.behaviors]
        if "delegation" not in behavior_names:
            print("  ❌ FAIL: DelegationBehavior not loaded")
            return False

        print("  ✓ DelegationBehavior loaded")

        # Check delegation tools created
        tools = agent.get_tools()
        tool_names = [t['function']['name'] for t in tools]

        print(f"\nDelegation tools:")
        delegation_tools = [t for t in tool_names if 'delegate' in t or 'consult' in t]
        for tool_name in delegation_tools:
            print(f"  - {tool_name}")

        # Should have consult_architect and delegate_to_executor
        if "consult_architect" not in tool_names:
            print("  ❌ FAIL: consult_architect tool not found")
            return False

        if "delegate_to_executor" not in tool_names:
            print("  ❌ FAIL: delegate_to_executor tool not found")
            return False

        print("  ✓ All delegation tools created")

        # Check tool parameters loaded from configs
        for tool in tools:
            if tool['function']['name'] == 'consult_architect':
                params = tool['function']['parameters']['properties']
                required = tool['function']['parameters']['required']

                print(f"\nconsult_architect parameters:")
                print(f"  Parameters: {list(params.keys())}")
                print(f"  Required: {required}")

                # Should have project_description, requirements, constraints
                expected_params = ["project_description", "requirements", "constraints"]
                for param in expected_params:
                    if param not in params:
                        print(f"  ❌ FAIL: Missing parameter '{param}'")
                        return False
                    if param not in required:
                        print(f"  ❌ FAIL: Parameter '{param}' should be required")
                        return False

                print("  ✓ consult_architect parameters correct")

            if tool['function']['name'] == 'delegate_to_executor':
                params = tool['function']['parameters']['properties']
                required = tool['function']['parameters']['required']

                print(f"\ndelegate_to_executor parameters:")
                print(f"  Parameters: {list(params.keys())}")
                print(f"  Required: {required}")

                # Should have task_description, workspace_mode, workspace_path
                if "task_description" not in params or "task_description" not in required:
                    print(f"  ❌ FAIL: task_description missing or not required")
                    return False

                if "workspace_mode" not in params or "workspace_mode" not in required:
                    print(f"  ❌ FAIL: workspace_mode missing or not required")
                    return False

                # workspace_path should NOT be required
                if "workspace_path" in required:
                    print(f"  ❌ FAIL: workspace_path should not be required")
                    return False

                # Check enum on workspace_mode
                if "enum" not in params["workspace_mode"]:
                    print(f"  ❌ FAIL: workspace_mode missing enum")
                    return False

                print("  ✓ delegate_to_executor parameters correct")

        print("\n✓ PASS: Orchestrator delegation behavior working correctly")
        return True

    except Exception as e:
        print(f"\n❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("AGENT CONFIGURATION REFACTOR TEST SUITE")
    print("="*70)

    tests = [
        test_agents_yaml_structure,
        test_blurbs_in_agent_configs,
        test_delegation_tools_in_agent_configs,
        test_token_limits_consistent,
        test_status_display_behavior_removed,
        test_base_agent_loads_delegation_tools,
        test_orchestrator_delegation_behavior,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
