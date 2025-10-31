"""
Test that context strategies correctly inject instructions and tools.
"""
from pathlib import Path
from task_executor_agent import TaskExecutorAgent
from context_strategies import HierarchicalStrategy, AppendUntilFullStrategy


def test_hierarchical_strategy_injection():
    """Test that HierarchicalStrategy injects instructions and tools."""
    print("\n" + "="*70)
    print("TEST: Hierarchical Strategy Injection")
    print("="*70)

    # Create agent with hierarchical strategy
    agent = TaskExecutorAgent(
        workspace=Path("."),
        context_strategy=HierarchicalStrategy(),
    )

    # Check system prompt includes hierarchical instructions
    system_prompt = agent.get_system_prompt()
    print("\n--- System Prompt (first 500 chars) ---")
    print(system_prompt[:500])
    print("...")

    assert "HIERARCHICAL WORKFLOW" in system_prompt, "Missing hierarchical workflow instructions"
    assert "decompose_task" in system_prompt, "Missing decompose_task mention"
    assert "mark_subtask_complete" in system_prompt, "Missing mark_subtask_complete mention"
    print("✅ System prompt includes hierarchical instructions")

    # Check tools include hierarchical tools
    tools = agent.get_tools()
    tool_names = {tool["function"]["name"] for tool in tools}

    print(f"\n--- Available Tools ({len(tools)} total) ---")
    for name in sorted(tool_names):
        print(f"  - {name}")

    assert "decompose_task" in tool_names, "Missing decompose_task tool"
    assert "mark_subtask_complete" in tool_names, "Missing mark_subtask_complete tool"
    assert "write_file" in tool_names, "Missing base tool write_file"
    assert "run_bash" in tool_names, "Missing base tool run_bash"
    print("✅ Tools include both base and hierarchical tools")

    print("\n✅ Hierarchical strategy injection PASSED")


def test_append_strategy_no_injection():
    """Test that AppendUntilFullStrategy doesn't inject hierarchical stuff."""
    print("\n" + "="*70)
    print("TEST: Append-Until-Full Strategy (No Hierarchical Injection)")
    print("="*70)

    # Create agent with append-until-full strategy
    agent = TaskExecutorAgent(
        workspace=Path("."),
        context_strategy=AppendUntilFullStrategy(),
    )

    # Check system prompt does NOT include hierarchical instructions
    system_prompt = agent.get_system_prompt()
    print("\n--- System Prompt (first 500 chars) ---")
    print(system_prompt[:500])
    print("...")

    assert "HIERARCHICAL WORKFLOW" not in system_prompt, "Should not have hierarchical workflow instructions"
    print("✅ System prompt does NOT include hierarchical instructions")

    # Check tools do NOT include hierarchical tools
    tools = agent.get_tools()
    tool_names = {tool["function"]["name"] for tool in tools}

    print(f"\n--- Available Tools ({len(tools)} total) ---")
    for name in sorted(tool_names):
        print(f"  - {name}")

    assert "decompose_task" not in tool_names, "Should not have decompose_task tool"
    assert "mark_subtask_complete" not in tool_names, "Should not have mark_subtask_complete tool"
    assert "write_file" in tool_names, "Missing base tool write_file"
    assert "run_bash" in tool_names, "Missing base tool run_bash"
    print("✅ Tools include only base tools (no hierarchical tools)")

    print("\n✅ Append strategy (no injection) PASSED")


def test_default_strategy():
    """Test that default strategy is append-until-full."""
    print("\n" + "="*70)
    print("TEST: Default Strategy (should be append-until-full)")
    print("="*70)

    # Create agent with default strategy
    agent = TaskExecutorAgent(workspace=Path("."))

    # Check which strategy is active
    strategy_name = agent.context_strategy.get_name()
    print(f"\n--- Default Strategy: {strategy_name} ---")

    # As per task_executor_agent.py:74, default is append-until-full
    assert strategy_name == "append_until_full", f"Expected 'append_until_full', got '{strategy_name}'"
    print("✅ Default strategy is append-until-full")

    # Tools should NOT include hierarchical tools by default
    tools = agent.get_tools()
    tool_names = {tool["function"]["name"] for tool in tools}

    assert "decompose_task" not in tool_names, "Default strategy should not have hierarchical tools"
    assert "mark_subtask_complete" not in tool_names, "Default strategy should not have hierarchical tools"
    print("✅ Default strategy does not inject hierarchical tools")

    print("\n✅ Default strategy test PASSED")


if __name__ == "__main__":
    test_hierarchical_strategy_injection()
    test_append_strategy_no_injection()
    test_default_strategy()

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✅")
    print("="*70)
    print("\nSummary:")
    print("- Hierarchical strategy correctly injects workflow instructions and tools")
    print("- Append-until-full strategy does NOT inject hierarchical stuff")
    print("- Base tools (write_file, run_bash, etc.) are available in all strategies")
    print("- Strategy-specific tools are added only when strategy is active")
