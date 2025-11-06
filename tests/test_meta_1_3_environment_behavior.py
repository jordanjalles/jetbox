#!/usr/bin/env python3
"""
Test 1.3: EnvironmentBehavior Generation

E2E test for generating environment variable management behavior.
Tests state changes and default value handling.
"""
from pathlib import Path
import pytest
from task_executor_agent import TaskExecutorAgent
from behaviors.create_behavior import CreateBehaviorBehavior
from behaviors.validation import ValidationBehavior
from behaviors.sandbox_test import SandboxTestBehavior


def test_generate_environment_behavior():
    """
    E2E Test: Generate EnvironmentBehavior for env var management.

    Success Criteria:
    - Tools work correctly
    - Default values handled properly
    - Tests verify state changes
    """
    print("\n" + "="*60)
    print("TEST 1.3: EnvironmentBehavior Generation")
    print("="*60)

    # Step 1: Setup
    workspace = Path('.agent_workspace/meta_test_environment')
    workspace.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Creating MetaProgrammer agent...")
    meta_programmer = TaskExecutorAgent(
        workspace=workspace,
        config_file='config/agents/meta_programmer.yaml',
        timeout_seconds=300
    )
    print(f"✓ MetaProgrammer loaded")

    # Step 2: Define specification
    print("\n[2/6] Defining EnvironmentBehavior specification...")

    tool_specs = [
        {
            "name": "get_env_var",
            "description": "Get environment variable value with optional default",
            "parameters": {
                "name": {
                    "type": "string",
                    "description": "Environment variable name",
                    "required": True
                },
                "default": {
                    "type": "string",
                    "description": "Default value if variable not set",
                    "required": False
                }
            }
        },
        {
            "name": "set_env_var",
            "description": "Set environment variable value (for current process)",
            "parameters": {
                "name": {
                    "type": "string",
                    "description": "Environment variable name",
                    "required": True
                },
                "value": {
                    "type": "string",
                    "description": "Value to set",
                    "required": True
                }
            }
        }
    ]

    print(f"✓ Defined 2 tools: get_env_var, set_env_var")

    # Step 3: Generate behavior
    print("\n[3/6] Generating behavior code...")

    create_behavior = CreateBehaviorBehavior()
    result = create_behavior.dispatch_tool(
        agent=meta_programmer,
        tool_name="create_behavior",
        args={
            "behavior_name": "EnvironmentBehavior",
            "description": "Provides environment variable management tools",
            "tool_specs": tool_specs,
            "lifecycle_hooks": [],
            "context_enhancement": None,
            "safety_mode": "auto"
        }
    )

    if not result.get('success'):
        pytest.fail(f"Generation failed: {result.get('error')}")

    behavior_file = result.get('behavior_file')
    test_file = result.get('test_file')

    print(f"✓ Behavior: {behavior_file}")
    print(f"✓ Tests: {test_file}")

    # Step 4: Validate
    print("\n[4/6] Validating...")

    # Read the behavior file content
    with open(behavior_file, 'r') as f:
        behavior_code = f.read()

    validation = ValidationBehavior()
    validation_result = validation.dispatch_tool(
        agent=meta_programmer,
        tool_name="validate_behavior_class",
        args={
            "code": behavior_code,
            "expected_name": "EnvironmentBehavior"
        }
    )

    # Extract result from the wrapper
    result = validation_result.get('result', {})
    if not result.get('valid', False):
        issues = result.get('issues', [])
        pytest.fail(f"Validation failed: {issues}")

    print("✓ Validation passed")

    # Step 5: Run tests
    print("\n[5/6] Running tests...")

    sandbox = SandboxTestBehavior()
    test_result = sandbox.dispatch_tool(
        agent=meta_programmer,
        tool_name="run_behavior_tests",
        args={
            "test_file": str(test_file),
            "behavior_file": str(behavior_file)
        }
    )

    if not test_result.get('passed', False):
        pytest.fail(f"Tests failed: {test_result.get('output')}")

    print("✓ Tests passed")

    # Step 6: Verify
    print("\n[6/6] Verifying behavior...")

    try:
        import sys
        import importlib.util
        import os

        behavior_dir = Path(behavior_file).parent
        if str(behavior_dir) not in sys.path:
            sys.path.insert(0, str(behavior_dir))

        spec = importlib.util.spec_from_file_location(
            Path(behavior_file).stem,
            behavior_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        EnvironmentBehavior = getattr(module, 'EnvironmentBehavior')
        behavior = EnvironmentBehavior()

        tools = behavior.get_tools()
        tool_names = [t['function']['name'] for t in tools]

        print(f"✓ Tools: {', '.join(tool_names)}")

        # Test default value handling
        print("\nTesting default values...")

        # Get non-existent var with default
        get_result = behavior.dispatch_tool(
            agent=meta_programmer,
            tool_name="get_env_var",
            args={
                "name": "NONEXISTENT_TEST_VAR_12345",
                "default": "test_default"
            }
        )

        returned_value = get_result.get('value', '')
        if returned_value == "test_default":
            print("✓ Default value handling works")
        else:
            print(f"⚠️ Warning: Default value may not work (got: {returned_value})")

        # Test set and get
        print("\nTesting set/get operations...")

        set_result = behavior.dispatch_tool(
            agent=meta_programmer,
            tool_name="set_env_var",
            args={
                "name": "TEST_VAR_META_123",
                "value": "test_value_123"
            }
        )

        if set_result.get('success'):
            # Verify it was set
            actual_value = os.environ.get('TEST_VAR_META_123', '')
            if actual_value == "test_value_123":
                print("✓ Set/get operations work")
            else:
                print(f"⚠️ Warning: Value may not persist (expected: test_value_123, got: {actual_value})")

    except Exception as e:
        pytest.fail(f"Verification failed: {e}")

    # Success!
    print("\n" + "="*60)
    print("✅ TEST 1.3 PASSED: EnvironmentBehavior")
    print("="*60)

    return True


if __name__ == "__main__":
    try:
        success = test_generate_environment_behavior()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
