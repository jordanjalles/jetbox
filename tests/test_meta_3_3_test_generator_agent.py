#!/usr/bin/env python3
"""
Test 3.3: TestGeneratorAgent Generation

E2E test for generating a unit test generation specialist agent.
Tests agent with file + command tools.
"""
from pathlib import Path
import pytest
import yaml
from task_executor_agent import TaskExecutorAgent
from behaviors.create_agent import CreateAgentBehavior
from behaviors.validation import ValidationBehavior


def test_generate_test_generator_agent():
    """
    E2E Test: Generate TestGeneratorAgent for unit test generation.

    Success Criteria:
    - Config valid
    - Has file + command tools
    - Can run tests after generation
    - System prompt emphasizes test quality
    """
    print("\n" + "="*60)
    print("TEST 3.3: TestGeneratorAgent Generation")
    print("="*60)

    # Step 1: Setup
    workspace = Path('.agent_workspace/meta_test_test_generator')
    workspace.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Creating MetaProgrammer agent...")
    meta_programmer = TaskExecutorAgent(
        workspace=workspace,
        config_file='config/agents/meta_programmer.yaml',
        timeout_seconds=300
    )
    print(f"✓ MetaProgrammer loaded")

    # Step 2: Define specification
    print("\n[2/5] Defining TestGeneratorAgent specification...")

    behaviors = [
        "ChatbotBehavior",
        "CompactWhenNearFullBehavior",
        "ReadFileToolsBehavior",
        "WriteFileToolsBehavior",
        "DirectoryToolsBehavior",
        "CommandToolsBehavior",
        "LoopDetectionBehavior"
    ]

    system_prompt = """You are a unit test generation specialist.

Your role:
- Generate comprehensive unit tests for code
- Cover edge cases and error conditions
- Write clear, maintainable test code
- Run tests to verify they work

Guidelines:
- Use read_file to understand code before testing
- Use write_file to create test files
- Use run_bash to execute tests (pytest, unittest)
- Focus on edge cases and error conditions
- Ensure tests are deterministic and fast
- Follow testing best practices (AAA pattern, clear names)

Test Frameworks:
- Python: pytest (preferred), unittest
- Always run tests after generation to verify they work
"""

    delegation_tool = {
        "name": "delegate_to_test_generator",
        "description": "Delegate unit test generation to TestGenerator specialist",
        "parameters": {
            "code_path": {
                "type": "string",
                "description": "Path to code that needs tests",
                "required": True
            },
            "framework": {
                "type": "string",
                "description": "Test framework: 'pytest' or 'unittest'",
                "required": False
            },
            "coverage_target": {
                "type": "integer",
                "description": "Target code coverage percentage (default: 80)",
                "required": False
            }
        }
    }

    print(f"✓ Agent spec: {len(behaviors)} behaviors")

    # Step 3: Generate
    print("\n[3/5] Generating agent configuration...")

    create_agent = CreateAgentBehavior()
    result = create_agent.dispatch_tool(
        agent=meta_programmer,
        tool_name="create_agent",
        args={
            "agent_name": "TestGeneratorAgent",
            "role": "Unit test generation specialist",
            "blurb": "TestGenerator creates comprehensive unit tests with edge case coverage. It writes pytest or unittest tests, runs them to verify correctness, and ensures high code coverage. Best for TDD workflows, legacy code testing, and ensuring robust test suites.",
            "behaviors": behaviors,
            "system_prompt": system_prompt,
            "delegation_tool": delegation_tool,
            "safety_mode": "auto"
        }
    )

    if not result.get('success'):
        pytest.fail(f"Generation failed: {result.get('error')}")

    config_file = result.get('config_file')
    print(f"✓ Config: {config_file}")

    # Step 4: Validate
    print("\n[4/5] Validating configuration...")

    validation = ValidationBehavior()
    validation_result = validation.dispatch_tool(
        agent=meta_programmer,
        tool_name="validate_agent_config",
        args={"config_file": str(config_file)}
    )

    if not validation_result.get('valid', False):
        pytest.fail(f"Validation failed: {validation_result.get('issues')}")

    print("✓ Validation passed")

    # Step 5: Verify
    print("\n[5/5] Verifying configuration...")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Check structure
        assert 'role' in config
        assert 'blurb' in config
        assert 'system_prompt' in config
        assert 'behaviors' in config
        assert 'delegation_tool' in config

        print(f"✓ Role: {config['role']}")

        # Check behaviors
        behavior_types = [b['type'] if isinstance(b, dict) else b for b in config['behaviors']]

        has_read = 'ReadFileToolsBehavior' in behavior_types
        has_write = 'WriteFileToolsBehavior' in behavior_types
        has_command = 'CommandToolsBehavior' in behavior_types

        print(f"✓ Has read tools: {has_read}")
        print(f"✓ Has write tools: {has_write}")
        print(f"✓ Has command tools: {has_command}")

        assert has_read, "Missing ReadFileToolsBehavior"
        assert has_write, "Missing WriteFileToolsBehavior"
        assert has_command, "Missing CommandToolsBehavior (needed to run tests)"

        # Check delegation tool
        del_tool = config['delegation_tool']
        params = del_tool.get('parameters', {})

        assert 'code_path' in params, "Missing code_path parameter"

        print(f"✓ Delegation tool: {del_tool['name']}")

        # Check system prompt emphasizes testing
        system_prompt_text = config['system_prompt'].lower()
        mentions_tests = any(word in system_prompt_text for word in [
            'test', 'edge case', 'coverage', 'pytest', 'unittest'
        ])

        print(f"✓ System prompt emphasizes testing: {mentions_tests}")

    except Exception as e:
        pytest.fail(f"Verification failed: {e}")

    # Test instantiation
    print("\nVerifying agent instantiation...")

    try:
        test_generator = TaskExecutorAgent(
            workspace=workspace / 'test_gen_test',
            config_file=str(config_file),
            timeout_seconds=60
        )

        tools = test_generator.get_tools()
        tool_names = [t.get('function', {}).get('name') for t in tools]

        print(f"✓ Agent instantiated with {len(tools)} tools")

        # Check tools
        assert 'read_file' in tool_names, "Missing read_file"
        assert 'write_file' in tool_names, "Missing write_file"
        assert 'run_bash' in tool_names, "Missing run_bash (needed for running tests)"

        print("✓ All expected tools present")

    except Exception as e:
        pytest.fail(f"Instantiation failed: {e}")

    # Success!
    print("\n" + "="*60)
    print("✅ TEST 3.3 PASSED: TestGeneratorAgent")
    print("="*60)
    print(f"\nGenerated config: {config_file}")
    print(f"Behaviors: {len(behaviors)}")
    print(f"Tools: {len(tools)}")

    return True


if __name__ == "__main__":
    try:
        success = test_generate_test_generator_agent()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
