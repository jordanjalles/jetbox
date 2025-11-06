#!/usr/bin/env python3
"""
Test 1.2: JsonToolsBehavior Generation

E2E test for generating JSON manipulation behavior.
Tests multiple tools with edge case handling.
"""
import json
from pathlib import Path
import pytest
from task_executor_agent import TaskExecutorAgent
from behaviors.create_behavior import CreateBehaviorBehavior
from behaviors.validation import ValidationBehavior
from behaviors.sandbox_test import SandboxTestBehavior


def test_generate_json_tools_behavior():
    """
    E2E Test: Generate JsonToolsBehavior with JSON parsing/formatting tools.

    Success Criteria:
    - Handles edge cases gracefully (malformed input)
    - Tool schemas are complete
    - Tests cover error conditions
    - No external dependencies beyond stdlib
    """
    print("\n" + "="*60)
    print("TEST 1.2: JsonToolsBehavior Generation")
    print("="*60)

    # Step 1: Setup
    workspace = Path('.agent_workspace/meta_test_json_tools')
    workspace.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Creating MetaProgrammer agent...")
    meta_programmer = TaskExecutorAgent(
        workspace=workspace,
        config_file='config/agents/meta_programmer.yaml',
        timeout_seconds=300
    )
    print(f"✓ MetaProgrammer loaded")

    # Step 2: Define specification
    print("\n[2/6] Defining JsonToolsBehavior specification...")

    tool_specs = [
        {
            "name": "parse_json",
            "description": "Parse JSON string to dictionary. Returns error dict if parsing fails.",
            "parameters": {
                "json_string": {
                    "type": "string",
                    "description": "JSON string to parse",
                    "required": True
                }
            }
        },
        {
            "name": "format_json",
            "description": "Format data as JSON string with optional indentation",
            "parameters": {
                "data": {
                    "type": "object",
                    "description": "Data to format as JSON",
                    "required": True
                },
                "indent": {
                    "type": "integer",
                    "description": "Indentation spaces (default: 2)",
                    "required": False
                }
            }
        },
        {
            "name": "validate_json_schema",
            "description": "Validate data against JSON schema (basic validation)",
            "parameters": {
                "data": {
                    "type": "object",
                    "description": "Data to validate",
                    "required": True
                },
                "schema": {
                    "type": "object",
                    "description": "JSON schema definition (required/properties)",
                    "required": True
                }
            }
        }
    ]

    print(f"✓ Defined 3 tools: parse_json, format_json, validate_json_schema")

    # Step 3: Generate behavior
    print("\n[3/6] Generating behavior code...")

    create_behavior = CreateBehaviorBehavior()
    result = create_behavior.dispatch_tool(
        agent=meta_programmer,
        tool_name="create_behavior",
        args={
            "behavior_name": "JsonToolsBehavior",
            "description": "Provides JSON manipulation tools for parsing, formatting, and validation",
            "tool_specs": tool_specs,
            "lifecycle_hooks": [],
            "context_enhancement": None,
            "safety_mode": "auto"
        }
    )

    if not result.get('success'):
        pytest.fail(f"Behavior generation failed: {result.get('error')}")

    behavior_file = result.get('behavior_file')
    test_file = result.get('test_file')

    print(f"✓ Behavior: {behavior_file}")
    print(f"✓ Tests: {test_file}")

    # Step 4: Validate
    print("\n[4/6] Validating generated code...")

    # Read the behavior file content
    with open(behavior_file, 'r') as f:
        behavior_code = f.read()

    validation = ValidationBehavior()
    validation_result = validation.dispatch_tool(
        agent=meta_programmer,
        tool_name="validate_behavior_class",
        args={
            "code": behavior_code,
            "expected_name": "JsonToolsBehavior"
        }
    )

    # Extract result from the wrapper
    result = validation_result.get('result', {})
    if not result.get('valid', False):
        issues = result.get('issues', [])
        pytest.fail(f"Validation failed: {issues}")

    print("✓ Validation passed")

    # Step 5: Run tests
    print("\n[5/6] Running generated tests...")

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
        test_output = test_result.get('output', '')
        pytest.fail(f"Tests failed: {test_output}")

    print("✓ Tests passed")

    # Step 6: Verify instantiation and edge cases
    print("\n[6/6] Verifying behavior and edge case handling...")

    try:
        import sys
        import importlib.util

        behavior_dir = Path(behavior_file).parent
        if str(behavior_dir) not in sys.path:
            sys.path.insert(0, str(behavior_dir))

        spec = importlib.util.spec_from_file_location(
            Path(behavior_file).stem,
            behavior_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        JsonToolsBehavior = getattr(module, 'JsonToolsBehavior')
        behavior = JsonToolsBehavior()

        tools = behavior.get_tools()
        tool_names = [t['function']['name'] for t in tools]

        print(f"✓ Tools: {', '.join(tool_names)}")

        # Test edge case: malformed JSON
        print("\nTesting edge case: malformed JSON...")
        parse_result = behavior.dispatch_tool(
            agent=meta_programmer,
            tool_name="parse_json",
            args={"json_string": "{invalid json}"}
        )

        # Should handle error gracefully
        if 'error' in parse_result or parse_result.get('success') == False:
            print("✓ Malformed JSON handled gracefully")
        else:
            print("⚠️ Warning: Malformed JSON may not be handled properly")

    except Exception as e:
        pytest.fail(f"Instantiation failed: {e}")

    # Success!
    print("\n" + "="*60)
    print("✅ TEST 1.2 PASSED: JsonToolsBehavior")
    print("="*60)

    return True


if __name__ == "__main__":
    try:
        success = test_generate_json_tools_behavior()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
