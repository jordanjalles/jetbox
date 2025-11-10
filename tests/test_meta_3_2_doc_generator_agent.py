#!/usr/bin/env python3
"""
Test 3.2: DocGeneratorAgent Generation

E2E test for generating a documentation generation specialist agent.
Tests agent config generation with standard behaviors.
"""
from pathlib import Path
import pytest
import yaml
from agents.task_executor_agent import TaskExecutorAgent
from behaviors.create_agent import CreateAgentBehavior
from behaviors.validation import ValidationBehavior


def test_generate_doc_generator_agent():
    """
    E2E Test: Generate DocGeneratorAgent for documentation tasks.

    Success Criteria:
    - Config valid
    - Has read + write tools
    - System prompt emphasizes documentation quality
    - Delegation tool schema complete
    """
    print("\n" + "="*60)
    print("TEST 3.2: DocGeneratorAgent Generation")
    print("="*60)

    # Step 1: Setup
    workspace = Path('.agent_workspace/meta_test_doc_generator')
    workspace.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Creating MetaProgrammer agent...")
    meta_programmer = TaskExecutorAgent(
        workspace=workspace,
        config_file='config/agents/meta_programmer.yaml',
        timeout_seconds=300
    )
    print(f"✓ MetaProgrammer loaded")

    # Step 2: Define agent specification
    print("\n[2/5] Defining DocGeneratorAgent specification...")

    behaviors = [
        "ChatbotBehavior",
        "CompactWhenNearFullBehavior",
        "ReadFileToolsBehavior",
        "WriteFileToolsBehavior",
        "DirectoryToolsBehavior",
        "LoopDetectionBehavior"
    ]

    system_prompt = """You are a documentation generation specialist.

Your role:
- Generate clear, comprehensive documentation from code
- Write README files, API docs, and user guides
- Create examples and usage instructions
- Maintain consistent documentation style

Guidelines:
- Use write_file to create documentation
- Use read_file to understand code before documenting
- Use list_dir to explore project structure
- Focus on clarity and completeness
- Include code examples where helpful
"""

    delegation_tool = {
        "name": "delegate_to_doc_generator",
        "description": "Delegate documentation generation tasks to DocGenerator specialist",
        "parameters": {
            "code_path": {
                "type": "string",
                "description": "Path to code that needs documentation",
                "required": True
            },
            "doc_type": {
                "type": "string",
                "description": "Type of documentation: 'readme', 'api', 'tutorial', 'reference'",
                "required": True
            },
            "output_path": {
                "type": "string",
                "description": "Where to write the documentation",
                "required": False
            }
        }
    }

    print(f"✓ Agent spec: {len(behaviors)} behaviors, delegation tool defined")

    # Step 3: Generate agent config
    print("\n[3/5] Generating agent configuration...")

    create_agent = CreateAgentBehavior()
    result = create_agent.dispatch_tool(
        agent=meta_programmer,
        tool_name="create_agent",
        args={
            "agent_name": "DocGeneratorAgent",
            "role": "Documentation generation specialist",
            "blurb": "DocGenerator creates clear, comprehensive documentation from code. It writes README files, API docs, user guides, and tutorials. Best for documenting codebases, explaining APIs, and creating usage examples.",
            "behaviors": behaviors,
            "system_prompt": system_prompt,
            "delegation_tool": delegation_tool,
            "safety_mode": "auto"
        }
    )

    if not result.get('success'):
        pytest.fail(f"Agent generation failed: {result.get('error')}")

    config_file = result.get('config_file')
    print(f"✓ Config generated: {config_file}")

    # Step 4: Validate config
    print("\n[4/5] Validating agent configuration...")

    validation = ValidationBehavior()
    validation_result = validation.dispatch_tool(
        agent=meta_programmer,
        tool_name="validate_agent_config",
        args={
            "config_file": str(config_file)
        }
    )

    if not validation_result.get('valid', False):
        issues = validation_result.get('issues', [])
        pytest.fail(f"Validation failed: {issues}")

    print("✓ Validation passed")

    # Step 5: Verify config structure
    print("\n[5/5] Verifying configuration structure...")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Check required fields
        assert 'role' in config, "Missing role"
        assert 'blurb' in config, "Missing blurb"
        assert 'system_prompt' in config, "Missing system_prompt"
        assert 'behaviors' in config, "Missing behaviors"
        assert 'delegation_tool' in config, "Missing delegation_tool"

        print(f"✓ Role: {config['role']}")
        print(f"✓ Behaviors: {len(config['behaviors'])} configured")

        # Check for read/write behaviors
        behavior_types = [b['type'] if isinstance(b, dict) else b for b in config['behaviors']]

        has_read = 'ReadFileToolsBehavior' in behavior_types
        has_write = 'WriteFileToolsBehavior' in behavior_types

        print(f"✓ Has read tools: {has_read}")
        print(f"✓ Has write tools: {has_write}")

        assert has_read, "Missing ReadFileToolsBehavior"
        assert has_write, "Missing WriteFileToolsBehavior"

        # Check delegation tool
        del_tool = config['delegation_tool']
        assert 'name' in del_tool, "Missing delegation tool name"
        assert 'description' in del_tool, "Missing delegation tool description"
        assert 'parameters' in del_tool, "Missing delegation tool parameters"

        params = del_tool['parameters']
        assert 'code_path' in params, "Missing code_path parameter"
        assert 'doc_type' in params, "Missing doc_type parameter"

        print(f"✓ Delegation tool: {del_tool['name']}")
        print(f"✓ Parameters: {', '.join(params.keys())}")

        # Check system prompt mentions documentation
        system_prompt_text = config['system_prompt'].lower()
        mentions_docs = any(word in system_prompt_text for word in [
            'documentation', 'readme', 'api', 'guide'
        ])

        print(f"✓ System prompt emphasizes documentation: {mentions_docs}")

        if not mentions_docs:
            print("⚠️ Warning: System prompt may not emphasize documentation")

    except Exception as e:
        pytest.fail(f"Config verification failed: {e}")

    # Test instantiation
    print("\nVerifying agent can be instantiated...")

    try:
        doc_generator = TaskExecutorAgent(
            workspace=workspace / 'doc_gen_test',
            config_file=str(config_file),
            timeout_seconds=60
        )

        tools = doc_generator.get_tools()
        tool_names = [t.get('function', {}).get('name') for t in tools]

        print(f"✓ Agent instantiated with {len(tools)} tools")
        print(f"  Tools: {', '.join(tool_names[:5])}...")

        # Check for expected tools
        assert 'read_file' in tool_names, "Missing read_file tool"
        assert 'write_file' in tool_names, "Missing write_file tool"
        assert 'list_dir' in tool_names, "Missing list_dir tool"

        print("✓ All expected tools present")

    except Exception as e:
        pytest.fail(f"Instantiation failed: {e}")

    # Success!
    print("\n" + "="*60)
    print("✅ TEST 3.2 PASSED: DocGeneratorAgent")
    print("="*60)
    print(f"\nGenerated config: {config_file}")
    print(f"Behaviors: {len(behaviors)}")
    print(f"Tools: {len(tools)}")
    print(f"Validation: PASS")

    return True


if __name__ == "__main__":
    try:
        success = test_generate_doc_generator_agent()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
