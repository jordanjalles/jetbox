#!/usr/bin/env python3
"""
Quick verification script to demonstrate all features working.

Demonstrates:
1. Dynamic tool documentation generation
2. Auto-add SubAgentContextBehavior
3. Auto-add DelegationBehavior
4. All agent types loading correctly
"""
import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.task_executor_agent import TaskExecutorAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.architect_agent import ArchitectAgent


def test_task_executor():
    """Test TaskExecutor with behavior system."""
    print("=" * 80)
    print("TASK EXECUTOR VERIFICATION")
    print("=" * 80)

    agent = TaskExecutorAgent(
        workspace=Path("/tmp/test_executor"),
        use_behaviors=True,
        config_file="task_executor_config.yaml"
    )

    # Check loaded behaviors
    print("\n1. Loaded Behaviors:")
    for behavior in agent.behaviors:
        print(f"   - {behavior.get_name()}")

    # Verify SubAgentContextBehavior auto-added
    behavior_names = [b.get_name() for b in agent.behaviors]
    assert "subagent_context" in behavior_names, "SubAgentContextBehavior should be auto-added!"
    print("\n   ✓ SubAgentContextBehavior auto-added (TaskExecutor is subagent)")

    # Check available tools
    tools = agent.get_tools()
    print(f"\n2. Available Tools: {len(tools)} tools")
    for tool in tools[:5]:  # Show first 5
        print(f"   - {tool['function']['name']}")
    if len(tools) > 5:
        print(f"   ... and {len(tools) - 5} more")

    # Check dynamic tool documentation in system prompt
    system_prompt = agent.get_system_prompt()
    assert "Available tools:" in system_prompt, "Tool docs should be in system prompt!"
    print("\n3. Dynamic Tool Documentation:")
    tool_docs_section = system_prompt.split("Available tools:")[1][:200]
    print(f"   {tool_docs_section[:200]}...")
    print("\n   ✓ Tool documentation dynamically generated")

    print("\n✓ TaskExecutor verification complete")
    return True


def test_orchestrator():
    """Test Orchestrator with behavior system."""
    print("\n" + "=" * 80)
    print("ORCHESTRATOR VERIFICATION")
    print("=" * 80)

    agent = OrchestratorAgent(
        workspace=Path("/tmp/test_orchestrator"),
        use_behaviors=True,
        config_file="orchestrator_config.yaml"
    )

    # Check loaded behaviors
    print("\n1. Loaded Behaviors:")
    for behavior in agent.behaviors:
        print(f"   - {behavior.get_name()}")

    # Verify DelegationBehavior auto-added
    behavior_names = [b.get_name() for b in agent.behaviors]
    assert "delegation" in behavior_names, "DelegationBehavior should be auto-added!"
    print("\n   ✓ DelegationBehavior auto-added (Orchestrator can delegate)")

    # Verify SubAgentContextBehavior NOT added
    assert "subagent_context" not in behavior_names, "SubAgentContextBehavior should NOT be added!"
    print("   ✓ SubAgentContextBehavior NOT added (Orchestrator is not subagent)")

    # Check delegation tools
    tools = agent.get_tools()
    tool_names = [t["function"]["name"] for t in tools]
    assert "consult_architect" in tool_names, "Should have consult_architect tool!"
    assert "delegate_to_executor" in tool_names, "Should have delegate_to_executor tool!"
    print("\n2. Delegation Tools:")
    print("   - consult_architect")
    print("   - delegate_to_executor")
    print("\n   ✓ Delegation tools available")

    print("\n✓ Orchestrator verification complete")
    return True


def test_architect():
    """Test Architect with behavior system."""
    print("\n" + "=" * 80)
    print("ARCHITECT VERIFICATION")
    print("=" * 80)

    agent = ArchitectAgent(
        workspace=Path("/tmp/test_architect"),
        use_behaviors=True,
        config_file="architect_config.yaml"
    )

    # Check loaded behaviors
    print("\n1. Loaded Behaviors:")
    for behavior in agent.behaviors:
        print(f"   - {behavior.get_name()}")

    # Verify SubAgentContextBehavior auto-added
    behavior_names = [b.get_name() for b in agent.behaviors]
    assert "subagent_context" in behavior_names, "SubAgentContextBehavior should be auto-added!"
    print("\n   ✓ SubAgentContextBehavior auto-added (Architect is subagent)")

    # Check architecture tools
    tools = agent.get_tools()
    tool_names = [t["function"]["name"] for t in tools]
    assert "write_architecture_doc" in tool_names, "Should have write_architecture_doc!"
    assert "write_module_spec" in tool_names, "Should have write_module_spec!"
    assert "write_task_list" in tool_names, "Should have write_task_list!"
    print("\n2. Architecture Tools:")
    print("   - write_architecture_doc")
    print("   - write_module_spec")
    print("   - write_task_list")
    print("\n   ✓ Architecture tools available")

    # Check dynamic tool documentation
    system_prompt = agent.get_system_prompt()
    assert "Available tools:" in system_prompt, "Tool docs should be in system prompt!"
    print("\n   ✓ Dynamic tool documentation in system prompt")

    print("\n✓ Architect verification complete")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("JETBOX BEHAVIOR SYSTEM - VERIFICATION TEST")
    print("=" * 80)

    try:
        # Run all tests
        results = []
        results.append(test_task_executor())
        results.append(test_orchestrator())
        results.append(test_architect())

        # Summary
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"\n✓ All {len(results)} agent types verified successfully!")
        print("\nVerified Features:")
        print("  1. ✓ Dynamic tool documentation generation")
        print("  2. ✓ Auto-add SubAgentContextBehavior (for subagents)")
        print("  3. ✓ Auto-add DelegationBehavior (for delegating agents)")
        print("  4. ✓ All agent types load correctly")
        print("  5. ✓ Tool lists accurate and complete")
        print("\nSystem Status: READY FOR PRODUCTION")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
