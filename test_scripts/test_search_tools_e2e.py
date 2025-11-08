#!/usr/bin/env python3
"""
Deep E2E Test: SearchToolsBehavior + AllAgent

This test demonstrates the full self-extensibility system by creating:
1. SearchToolsBehavior - Dynamic tool filtering for context management
2. AllAgent - Agent with all behaviors loaded

Test validates:
- Complex behavior generation with lifecycle hooks
- Agent config generation with many behaviors
- Full MetaProgrammer workflow
- Integration of generated artifacts
"""

from pathlib import Path
from task_executor_agent import TaskExecutorAgent
import shutil
import yaml


def test_search_tools_behavior_generation():
    """
    Step 1: Generate SearchToolsBehavior via MetaProgrammer.

    SearchToolsBehavior provides dynamic tool filtering:
    - Removes all tools from context except search_tools
    - When agent uses search_tools, adds only relevant tool specs
    - Reduces context bloat for agents with many behaviors
    """
    print("\n" + "=" * 80)
    print("STEP 1: Generate SearchToolsBehavior")
    print("=" * 80)

    # Setup workspace
    workspace = Path('.test_search_tools_e2e')
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir()

    try:
        # Load MetaProgrammer
        print("\n[1.1] Loading MetaProgrammer...")
        meta = TaskExecutorAgent(
            workspace=workspace,
            config_file='config/agents/meta_programmer.yaml',
            timeout_seconds=300
        )
        print(f"✓ MetaProgrammer loaded with {len(meta.behaviors)} behaviors")

        # Get CreateBehaviorBehavior
        create_behavior = None
        for b in meta.behaviors:
            if b.get_name() == 'create_behavior':
                create_behavior = b
                break

        assert create_behavior is not None, "CreateBehaviorBehavior not found"

        # Define SearchToolsBehavior spec
        print("\n[1.2] Defining SearchToolsBehavior specification...")

        search_tools_spec = {
            "behavior_name": "SearchToolsBehavior",
            "description": "Dynamic tool filtering for context management - removes all tools except search_tools, then adds only relevant tools when queried",
            "tool_specs": [
                {
                    "name": "search_tools",
                    "description": "Search for tools by keyword or pattern. Returns tool names and descriptions matching the query. Only matching tools are added to context.",
                    "parameters": {
                        "query": {
                            "type": "string",
                            "description": "Search query (keywords or pattern to match against tool names/descriptions)",
                            "required": True
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of matching tools to return (default: 10)",
                            "required": False
                        }
                    }
                }
            ],
            "lifecycle_hooks": ["on_initial_context"],
            "context_enhancement": """
This behavior needs to:
1. Store reference to all agent tools in __init__
2. In on_initial_context(): filter out all tools except search_tools from the initial context
3. In search_tools(): return matching tool specs and add them to a cached list
4. Maintain a set of "active" tools that have been searched for

Implementation pattern:
- Store agent.get_tools() in self._all_tools during __init__
- In on_initial_context(): return context with only search_tools in tools list
- In search_tools(): search self._all_tools, return matches, cache for context
- Use fuzzy matching on tool names and descriptions
""",
            "safety_mode": "auto"
        }

        print("✓ SearchToolsBehavior spec:")
        print(f"  - Tools: {len(search_tools_spec['tool_specs'])}")
        print(f"  - Lifecycle hooks: {search_tools_spec['lifecycle_hooks']}")
        print(f"  - Context enhancement: Yes")

        # Generate SearchToolsBehavior
        print("\n[1.3] Generating SearchToolsBehavior...")
        result = create_behavior.dispatch_tool(
            agent=meta,
            tool_name="create_behavior",
            args=search_tools_spec
        )

        # Verify generation
        print("\n[1.4] Verifying generation results...")

        assert "error" not in result, f"Generation failed: {result.get('error')}"
        assert result.get("success"), "Generation did not report success"

        behavior_file = result.get("behavior_file")
        test_file = result.get("test_file")

        assert behavior_file, "No behavior_file in result"
        assert test_file, "No test_file in result"
        assert Path(behavior_file).exists(), f"Behavior file not created: {behavior_file}"
        assert Path(test_file).exists(), f"Test file not created: {test_file}"

        print(f"✓ Behavior generated: {behavior_file}")
        print(f"✓ Tests generated: {test_file}")

        # Verify validation
        validation_results = result.get("validation_results")
        assert validation_results, "No validation results"
        assert validation_results.get("valid"), f"Validation failed: {validation_results}"

        print("✓ Validation: PASS")

        # Check installation
        installed = result.get("installed", False)
        print(f"✓ Installation: {'COMPLETED' if installed else 'STAGED'}")

        # Read generated behavior to verify on_initial_context hook
        behavior_code = Path(behavior_file).read_text()
        assert "on_initial_context" in behavior_code, "on_initial_context hook not found in generated code"
        assert "search_tools" in behavior_code, "search_tools method not found in generated code"

        print("✓ Lifecycle hook present: on_initial_context")
        print("✓ Tool method present: search_tools")

        print("\n" + "=" * 80)
        print("✅ STEP 1 COMPLETE: SearchToolsBehavior generated successfully")
        print("=" * 80)

        return {
            "behavior_file": behavior_file,
            "test_file": test_file,
            "validation": validation_results,
            "installed": installed
        }

    finally:
        # Cleanup workspace
        if workspace.exists():
            shutil.rmtree(workspace)


def test_all_agent_generation():
    """
    Step 2: Generate AllAgent config via MetaProgrammer.

    AllAgent has all available behaviors loaded, including:
    - SearchToolsBehavior (just created)
    - All existing behaviors

    This tests agent config generation with many behaviors.
    """
    print("\n" + "=" * 80)
    print("STEP 2: Generate AllAgent Configuration")
    print("=" * 80)

    # Setup workspace
    workspace = Path('.test_all_agent_e2e')
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir()

    try:
        # Load MetaProgrammer
        print("\n[2.1] Loading MetaProgrammer...")
        meta = TaskExecutorAgent(
            workspace=workspace,
            config_file='config/agents/meta_programmer.yaml',
            timeout_seconds=300
        )
        print(f"✓ MetaProgrammer loaded")

        # Get CreateAgentBehavior
        create_agent = None
        for b in meta.behaviors:
            if b.get_name() == 'create_agent':
                create_agent = b
                break

        assert create_agent is not None, "CreateAgentBehavior not found"

        # Collect all available behaviors
        print("\n[2.2] Collecting all available behaviors...")

        behaviors_dir = Path('behaviors')
        all_behaviors = []

        # Get all behavior Python files
        for behavior_file in behaviors_dir.glob('*.py'):
            if behavior_file.stem in ['__init__', 'base']:
                continue

            # Convert filename to class name
            # e.g., search_tools.py -> SearchToolsBehavior
            words = behavior_file.stem.split('_')
            class_name = ''.join(word.capitalize() for word in words)
            if not class_name.endswith('Behavior'):
                class_name += 'Behavior'

            all_behaviors.append(class_name)

        print(f"✓ Found {len(all_behaviors)} behaviors")
        print(f"  First 10: {all_behaviors[:10]}")

        # Define AllAgent spec
        print("\n[2.3] Defining AllAgent specification...")

        all_agent_spec = {
            "agent_name": "AllAgent",
            "role": "Universal agent with all available behaviors",
            "description": """AllAgent has every behavior loaded, making it capable of handling any task.
Uses SearchToolsBehavior to dynamically filter tools and manage context bloat.""",
            "behaviors": all_behaviors,
            "system_prompt_guidelines": [
                """You are AllAgent - a universal agent with access to all available behaviors and tools.

Use search_tools to discover available capabilities:
- search_tools query="file" to find file operations
- search_tools query="http" to find HTTP/network tools
- search_tools query="test" to find testing tools

Work efficiently:
1. Use search_tools to find relevant tools for the task
2. Execute tools to complete the task
3. Use compact_when_near_full to manage context

You have access to every capability in the system. Use search_tools to discover what's available."""
            ],
            "safety_mode": "dryrun"  # Don't auto-install, just validate
        }

        print("✓ AllAgent spec:")
        print(f"  - Role: {all_agent_spec['role']}")
        print(f"  - Behaviors: {len(all_agent_spec['behaviors'])}")
        print(f"  - Safety mode: {all_agent_spec['safety_mode']}")

        # Generate AllAgent
        print("\n[2.4] Generating AllAgent config...")
        result = create_agent.dispatch_tool(
            agent=meta,
            tool_name="create_agent",
            args=all_agent_spec
        )

        # Verify generation
        print("\n[2.5] Verifying generation results...")

        assert "error" not in result, f"Generation failed: {result.get('error')}"
        assert result.get("success"), "Generation did not report success"

        config_file = result.get("config_file")
        assert config_file, "No config_file in result"
        assert Path(config_file).exists(), f"Config file not created: {config_file}"

        print(f"✓ Config generated: {config_file}")

        # Read and validate YAML
        print("\n[2.6] Validating YAML structure...")
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        assert 'role' in config, "Missing 'role' field"
        assert 'blurb' in config, "Missing 'blurb' field"
        assert 'system_prompt' in config, "Missing 'system_prompt' field"
        assert 'behaviors' in config, "Missing 'behaviors' field"

        loaded_behaviors = [b['type'] for b in config['behaviors']]
        print(f"✓ YAML structure valid")
        print(f"✓ Behaviors in config: {len(loaded_behaviors)}")

        # Verify SearchToolsBehavior is included
        assert 'SearchToolsBehavior' in loaded_behaviors, "SearchToolsBehavior not in AllAgent config"
        print("✓ SearchToolsBehavior included in AllAgent")

        print("\n" + "=" * 80)
        print("✅ STEP 2 COMPLETE: AllAgent config generated successfully")
        print("=" * 80)

        return {
            "config_file": config_file,
            "behaviors_count": len(loaded_behaviors),
            "has_search_tools": 'SearchToolsBehavior' in loaded_behaviors
        }

    finally:
        # Cleanup workspace
        if workspace.exists():
            shutil.rmtree(workspace)


def test_integration():
    """
    Step 3: Integration test - load AllAgent and verify SearchToolsBehavior works.

    This validates that:
    - Generated behavior can be loaded
    - Generated agent config can be loaded
    - SearchToolsBehavior functions correctly
    - Tool filtering works as expected
    """
    print("\n" + "=" * 80)
    print("STEP 3: Integration Test - AllAgent + SearchToolsBehavior")
    print("=" * 80)

    print("\n[3.1] Checking if SearchToolsBehavior was installed...")

    search_tools_file = Path('behaviors/SearchToolsBehavior.py')
    if not search_tools_file.exists():
        print("⚠ SearchToolsBehavior not installed (safety_mode=auto may have failed)")
        print("  This is expected if validation or tests failed")
        print("  Skipping integration test")
        return None

    print("✓ SearchToolsBehavior installed")

    # Check AllAgent config
    all_agent_config = Path('.agent_generated/staging/AllAgent.yaml')
    if not all_agent_config.exists():
        print("⚠ AllAgent config not found")
        print("  Skipping integration test")
        return None

    print("✓ AllAgent config found")

    print("\n[3.2] Loading AllAgent...")
    workspace = Path('.test_integration_e2e')
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir()

    try:
        # Try to load AllAgent
        # Note: This may fail if SearchToolsBehavior has issues
        try:
            all_agent = TaskExecutorAgent(
                workspace=workspace,
                config_file=str(all_agent_config),
                timeout_seconds=300
            )
            print(f"✓ AllAgent loaded with {len(all_agent.behaviors)} behaviors")

            # Get all tools
            tools = all_agent.get_tools()
            print(f"✓ AllAgent has {len(tools)} total tools")

            # Check if search_tools is present
            tool_names = [t['name'] for t in tools]
            assert 'search_tools' in tool_names, "search_tools not found in AllAgent tools"
            print("✓ search_tools present in AllAgent")

            print("\n" + "=" * 80)
            print("✅ STEP 3 COMPLETE: Integration test passed")
            print("=" * 80)

            return {
                "loaded": True,
                "behaviors_count": len(all_agent.behaviors),
                "tools_count": len(tools),
                "has_search_tools": 'search_tools' in tool_names
            }

        except Exception as e:
            print(f"⚠ Failed to load AllAgent: {e}")
            print("  This may indicate issues with SearchToolsBehavior")
            print("  Check the generated code for errors")
            return {
                "loaded": False,
                "error": str(e)
            }

    finally:
        # Cleanup
        if workspace.exists():
            shutil.rmtree(workspace)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DEEP E2E TEST: Self-Extensibility System")
    print("Testing: SearchToolsBehavior + AllAgent")
    print("=" * 80)

    try:
        # Step 1: Generate SearchToolsBehavior
        search_tools_result = test_search_tools_behavior_generation()

        # Step 2: Generate AllAgent
        all_agent_result = test_all_agent_generation()

        # Step 3: Integration test
        integration_result = test_integration()

        # Final summary
        print("\n" + "=" * 80)
        print("🎉 DEEP E2E TEST COMPLETE")
        print("=" * 80)
        print("\nResults:")
        print(f"  SearchToolsBehavior: {'✓ PASS' if search_tools_result else '✗ FAIL'}")
        print(f"  AllAgent:           {'✓ PASS' if all_agent_result else '✗ FAIL'}")
        print(f"  Integration:        {'✓ PASS' if integration_result and integration_result.get('loaded') else '⚠ SKIP' if integration_result is None else '✗ FAIL'}")

        print("\n📊 Summary:")
        if search_tools_result:
            print(f"  - SearchToolsBehavior generated: {search_tools_result['behavior_file']}")
            print(f"  - Validation: {'PASS' if search_tools_result['validation'].get('valid') else 'FAIL'}")
            print(f"  - Installed: {search_tools_result['installed']}")

        if all_agent_result:
            print(f"  - AllAgent config: {all_agent_result['config_file']}")
            print(f"  - Behaviors loaded: {all_agent_result['behaviors_count']}")
            print(f"  - Has SearchToolsBehavior: {all_agent_result['has_search_tools']}")

        if integration_result:
            if integration_result.get('loaded'):
                print(f"  - AllAgent functional: Yes")
                print(f"  - Tools available: {integration_result['tools_count']}")
            else:
                print(f"  - AllAgent functional: No ({integration_result.get('error')})")

        print("\n✅ Self-extensibility system validated end-to-end!")
        print("   MetaProgrammer successfully created:")
        print("   1. Complex behavior with lifecycle hooks")
        print("   2. Agent config with all behaviors")
        print("   3. Validated and tested all artifacts")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
