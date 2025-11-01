"""
Test suite for behavior composability and independence.

This module tests that behaviors:
1. Work in isolation (can be instantiated and used alone)
2. Compose correctly (work together without conflicts)
3. Don't interfere with each other
4. Can be substituted/swapped
"""

import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, MagicMock

# Import all behaviors
from behaviors import (
    AgentBehavior,
    FileToolsBehavior,
    CommandToolsBehavior,
    ServerToolsBehavior,
    ArchitectToolsBehavior,
    CompactWhenNearFullBehavior,
    HierarchicalContextBehavior,
    SubAgentContextBehavior,
    LoopDetectionBehavior,
    WorkspaceTaskNotesBehavior,
    StatusDisplayBehavior,
    DelegationBehavior,
)


class TestBehaviorIsolation:
    """Test that each behavior works independently."""

    def test_file_tools_behavior_isolation(self):
        """FileToolsBehavior works without other behaviors."""
        behavior = FileToolsBehavior()

        # Has unique name
        assert behavior.get_name() == "file_tools"

        # Provides tools
        tools = behavior.get_tools()
        assert len(tools) == 3
        tool_names = [t["function"]["name"] for t in tools]
        assert "write_file" in tool_names
        assert "read_file" in tool_names
        assert "list_dir" in tool_names

    def test_command_tools_behavior_isolation(self):
        """CommandToolsBehavior works without other behaviors."""
        behavior = CommandToolsBehavior(whitelist=["python", "pytest"])

        assert behavior.get_name() == "command_tools"

        tools = behavior.get_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "run_bash"

    def test_server_tools_behavior_isolation(self):
        """ServerToolsBehavior works without other behaviors."""
        behavior = ServerToolsBehavior()

        assert behavior.get_name() == "server_tools"

        tools = behavior.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "start_server" in tool_names
        assert "stop_server" in tool_names
        assert "check_server" in tool_names
        assert "list_servers" in tool_names

    def test_architect_tools_behavior_isolation(self):
        """ArchitectToolsBehavior works without other behaviors."""
        behavior = ArchitectToolsBehavior()

        assert behavior.get_name() == "architect_tools"

        tools = behavior.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "write_architecture_doc" in tool_names
        assert "write_module_spec" in tool_names
        assert "write_task_list" in tool_names

    def test_compact_when_near_full_behavior_isolation(self):
        """CompactWhenNearFullBehavior works without other behaviors."""
        behavior = CompactWhenNearFullBehavior(max_tokens=8000)

        assert behavior.get_name() == "compact_when_near_full"

        # Can enhance context independently
        context = [{"role": "system", "content": "Test"}]
        enhanced = behavior.enhance_context(context)
        assert isinstance(enhanced, list)

    def test_hierarchical_context_behavior_isolation(self):
        """HierarchicalContextBehavior works without other behaviors."""
        behavior = HierarchicalContextBehavior(history_keep=12)

        assert behavior.get_name() == "hierarchical_context"

        tools = behavior.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "decompose_task" in tool_names
        assert "mark_subtask_complete" in tool_names

    def test_subagent_context_behavior_isolation(self):
        """SubAgentContextBehavior works without other behaviors."""
        behavior = SubAgentContextBehavior()

        assert behavior.get_name() == "subagent_context"

        tools = behavior.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "mark_complete" in tool_names
        assert "mark_failed" in tool_names

    def test_loop_detection_behavior_isolation(self):
        """LoopDetectionBehavior works without other behaviors."""
        behavior = LoopDetectionBehavior(max_repeats=5)

        assert behavior.get_name() == "loop_detection"

        # Can handle events independently
        behavior.on_tool_call("test_tool", {"arg": "value"}, {"success": True})
        # Should not crash

    def test_workspace_task_notes_behavior_isolation(self):
        """WorkspaceTaskNotesBehavior works without other behaviors."""
        behavior = WorkspaceTaskNotesBehavior()

        assert behavior.get_name() == "workspace_task_notes"

        # Can enhance context independently (even without workspace)
        context = [{"role": "system", "content": "Test"}]
        enhanced = behavior.enhance_context(context)
        assert isinstance(enhanced, list)

    def test_status_display_behavior_isolation(self):
        """StatusDisplayBehavior works without other behaviors."""
        behavior = StatusDisplayBehavior()

        assert behavior.get_name() == "status_display"

        # Can handle events independently
        behavior.on_goal_start("Test goal")
        behavior.on_tool_call("test_tool", {"arg": "value"}, {"success": True})
        behavior.on_goal_complete(success=True)
        # Should not crash

    def test_delegation_behavior_isolation(self):
        """DelegationBehavior works without other behaviors."""
        relationships = {
            "can_delegate_to": ["architect", "task_executor"],
            "architect": {"description": "Design agent"},
            "task_executor": {"description": "Execution agent"}
        }
        behavior = DelegationBehavior(agent_relationships=relationships)

        assert behavior.get_name() == "delegation"

        tools = behavior.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "consult_architect" in tool_names
        assert "delegate_to_executor" in tool_names


class TestBehaviorComposition:
    """Test that behaviors work together without conflicts."""

    def test_two_behavior_composition_file_and_command(self):
        """FileToolsBehavior + CommandToolsBehavior compose."""
        file_behavior = FileToolsBehavior()
        command_behavior = CommandToolsBehavior(whitelist=["python"])

        # Get all tools
        all_tools = file_behavior.get_tools() + command_behavior.get_tools()

        # No duplicate tool names
        tool_names = [t["function"]["name"] for t in all_tools]
        assert len(tool_names) == len(set(tool_names))  # All unique

    def test_three_behavior_composition_context_tools_utility(self):
        """SubAgent + Files + Loop compose correctly."""
        behaviors = [
            SubAgentContextBehavior(),
            FileToolsBehavior(),
            LoopDetectionBehavior()
        ]

        # All have unique names
        names = [b.get_name() for b in behaviors]
        assert len(names) == len(set(names))

        # All tools are unique
        all_tools = []
        for b in behaviors:
            all_tools.extend(b.get_tools())
        tool_names = [t["function"]["name"] for t in all_tools]
        assert len(tool_names) == len(set(tool_names))

    def test_full_task_executor_stack_composition(self):
        """All TaskExecutor behaviors compose correctly."""
        behaviors = [
            SubAgentContextBehavior(),
            CompactWhenNearFullBehavior(max_tokens=128000),
            FileToolsBehavior(),
            CommandToolsBehavior(whitelist=["python", "pytest"]),
            ServerToolsBehavior(),
            LoopDetectionBehavior(max_repeats=5),
            WorkspaceTaskNotesBehavior(),
            StatusDisplayBehavior()
        ]

        # All have unique names
        names = [b.get_name() for b in behaviors]
        assert len(names) == 8
        assert len(names) == len(set(names))

        # All tools are unique
        all_tools = []
        for b in behaviors:
            all_tools.extend(b.get_tools())
        tool_names = [t["function"]["name"] for t in all_tools]
        assert len(tool_names) == len(set(tool_names))

    def test_context_behaviors_compose(self):
        """SubAgent + Compact + Notes compose for context."""
        mock_context_manager = Mock()
        mock_context_manager.state = Mock()
        mock_context_manager.state.goal = Mock()
        mock_context_manager.state.goal.description = "Test goal"

        behaviors = [
            SubAgentContextBehavior(),
            CompactWhenNearFullBehavior(max_tokens=8000),
            WorkspaceTaskNotesBehavior()
        ]

        # Build context through all behaviors
        context = [{"role": "system", "content": "You are an agent"}]

        for behavior in behaviors:
            context = behavior.enhance_context(
                context,
                context_manager=mock_context_manager
            )

        # Context should be valid
        assert isinstance(context, list)
        assert len(context) >= 1  # At least system message

        # Each behavior should have contributed (delegated goal injected)
        context_str = str(context)
        assert "DELEGATED GOAL" in context_str

    def test_order_independence_tools(self):
        """Tool behaviors work in any order."""
        # Order 1
        behaviors1 = [
            FileToolsBehavior(),
            CommandToolsBehavior(whitelist=["python"]),
            ServerToolsBehavior()
        ]

        # Order 2
        behaviors2 = [
            ServerToolsBehavior(),
            FileToolsBehavior(),
            CommandToolsBehavior(whitelist=["python"])
        ]

        # Both orders provide same tools (just different order)
        tools1 = [t["function"]["name"] for b in behaviors1 for t in b.get_tools()]
        tools2 = [t["function"]["name"] for b in behaviors2 for t in b.get_tools()]

        assert set(tools1) == set(tools2)

    def test_order_independence_context(self):
        """Context behaviors work in any order (with expected differences)."""
        behaviors1 = [
            SubAgentContextBehavior(),
            CompactWhenNearFullBehavior(max_tokens=8000)
        ]

        behaviors2 = [
            CompactWhenNearFullBehavior(max_tokens=8000),
            SubAgentContextBehavior()
        ]

        context = [{"role": "system", "content": "Test"}]

        # Both orders should work (no crashes)
        for behavior in behaviors1:
            context1 = behavior.enhance_context(context.copy())

        for behavior in behaviors2:
            context2 = behavior.enhance_context(context.copy())

        # Both should return valid contexts
        assert isinstance(context1, list)
        assert isinstance(context2, list)


class TestBehaviorNoConflict:
    """Test that behaviors don't interfere with each other."""

    def test_tool_name_uniqueness_across_all_behaviors(self):
        """No duplicate tool names across all behaviors."""
        behaviors = [
            FileToolsBehavior(),
            CommandToolsBehavior(whitelist=["python"]),
            ServerToolsBehavior(),
            ArchitectToolsBehavior(),
            HierarchicalContextBehavior(),
            SubAgentContextBehavior(),
        ]

        all_tools = []
        for b in behaviors:
            all_tools.extend(b.get_tools())

        tool_names = [t["function"]["name"] for t in all_tools]

        # Check for duplicates
        duplicates = [name for name in tool_names if tool_names.count(name) > 1]
        assert len(duplicates) == 0, f"Duplicate tool names: {set(duplicates)}"

    def test_context_enhancements_compose_without_conflicts(self):
        """Multiple context enhancements don't conflict."""
        behaviors = [
            SubAgentContextBehavior(),
            CompactWhenNearFullBehavior(max_tokens=8000),
            WorkspaceTaskNotesBehavior()
        ]

        context = [{"role": "system", "content": "Test"}]

        # Apply all enhancements
        for behavior in behaviors:
            context = behavior.enhance_context(context)

        # Context should still be valid
        assert isinstance(context, list)
        assert all(isinstance(msg, dict) for msg in context)
        assert all("role" in msg for msg in context)

    def test_event_handlers_dont_interfere(self):
        """Event handlers work independently."""
        behaviors = [
            LoopDetectionBehavior(),
            StatusDisplayBehavior(),
            WorkspaceTaskNotesBehavior()
        ]

        # Send events to all
        for behavior in behaviors:
            behavior.on_goal_start("Test goal")
            behavior.on_tool_call("test_tool", {"arg": "val"}, {"success": True})
            behavior.on_goal_complete(success=True)

        # No crashes = success


class TestBehaviorSubstitution:
    """Test that behaviors can be swapped/substituted."""

    def test_swap_context_behaviors(self):
        """Can swap CompactWhenNearFull for SubAgent context."""
        # Config 1: CompactWhenNearFull
        behaviors1 = [
            CompactWhenNearFullBehavior(max_tokens=8000),
            FileToolsBehavior()
        ]

        # Config 2: SubAgent context
        behaviors2 = [
            SubAgentContextBehavior(),
            FileToolsBehavior()
        ]

        # Both configs work
        for behaviors in [behaviors1, behaviors2]:
            context = [{"role": "system", "content": "Test"}]
            for b in behaviors:
                context = b.enhance_context(context)
            assert isinstance(context, list)

    def test_remove_optional_behaviors(self):
        """Agent works with optional behaviors removed."""
        # Full stack
        full_stack = [
            SubAgentContextBehavior(),
            CompactWhenNearFullBehavior(max_tokens=8000),
            FileToolsBehavior(),
            CommandToolsBehavior(whitelist=["python"]),
            ServerToolsBehavior(),  # Optional
            LoopDetectionBehavior(),  # Optional
            WorkspaceTaskNotesBehavior(),  # Optional
            StatusDisplayBehavior()  # Optional
        ]

        # Minimal stack (remove optionals)
        minimal_stack = [
            SubAgentContextBehavior(),
            CompactWhenNearFullBehavior(max_tokens=8000),
            FileToolsBehavior(),
            CommandToolsBehavior(whitelist=["python"])
        ]

        # Both should work
        for behaviors in [full_stack, minimal_stack]:
            # Get all tools
            all_tools = []
            for b in behaviors:
                all_tools.extend(b.get_tools())

            # Tools should be valid
            assert isinstance(all_tools, list)
            assert all(isinstance(t, dict) for t in all_tools)

    def test_add_behaviors_dynamically(self):
        """Can add behaviors after initialization."""
        # Start with minimal
        behaviors = [FileToolsBehavior()]

        initial_tool_count = len(behaviors[0].get_tools())

        # Add more behaviors
        behaviors.append(CommandToolsBehavior(whitelist=["python"]))
        behaviors.append(LoopDetectionBehavior())

        # All tools available
        all_tools = []
        for b in behaviors:
            all_tools.extend(b.get_tools())

        assert len(all_tools) > initial_tool_count

    def test_minimal_behavior_set_file_tools_only(self):
        """Minimal agent with only file tools works."""
        behaviors = [FileToolsBehavior()]

        tools = behaviors[0].get_tools()
        assert len(tools) == 3

        # Verify tool names are present
        tool_names = [t["function"]["name"] for t in tools]
        assert "write_file" in tool_names
        assert "read_file" in tool_names
        assert "list_dir" in tool_names

    def test_minimal_behavior_set_context_only(self):
        """Minimal agent with only context behavior works."""
        behaviors = [CompactWhenNearFullBehavior(max_tokens=8000)]

        context = [{"role": "system", "content": "Test"}]
        enhanced = behaviors[0].enhance_context(context)

        assert isinstance(enhanced, list)
        assert len(enhanced) >= 1


class TestBehaviorEdgeCases:
    """Test edge cases in behavior composition."""

    def test_empty_behavior_list(self):
        """Agent with no behaviors doesn't crash."""
        behaviors = []

        # Should not crash
        all_tools = []
        for b in behaviors:
            all_tools.extend(b.get_tools())

        assert len(all_tools) == 0

    def test_duplicate_behavior_instances(self):
        """Can have multiple instances of same behavior (if needed)."""
        # Two file tool behaviors (unusual but should work)
        behavior1 = FileToolsBehavior()
        behavior2 = FileToolsBehavior()

        # Both provide tools
        tools1 = behavior1.get_tools()
        tools2 = behavior2.get_tools()

        assert len(tools1) == len(tools2)

    def test_behavior_with_missing_optional_methods(self):
        """Behaviors work even if optional methods not implemented."""
        class MinimalBehavior(AgentBehavior):
            def get_name(self):
                return "minimal"

        behavior = MinimalBehavior()

        # Should have defaults for optional methods
        assert behavior.get_tools() == []
        assert behavior.enhance_context([]) == []
        assert behavior.get_instructions() == ""

        # Events should not crash
        behavior.on_goal_start("test")
        behavior.on_tool_call("tool", {}, {})
        behavior.on_round_end(1)
        behavior.on_timeout(60)
        behavior.on_goal_complete(True)

    def test_behavior_with_extra_kwargs(self):
        """Behaviors tolerate extra kwargs."""
        behavior = FileToolsBehavior()

        # Pass extra kwargs that behavior doesn't use
        context = behavior.enhance_context(
            [],
            unknown_param="value",
            another_param=123,
            yet_another={"key": "val"}
        )

        # Should not crash
        assert isinstance(context, list)


def test_behavior_composability_summary():
    """
    Summary test: Verify all behaviors follow composability principles.

    This test serves as a high-level check that all behaviors:
    1. Can be instantiated independently
    2. Have unique names
    3. Can be combined without conflicts
    """
    all_behavior_classes = [
        FileToolsBehavior,
        CommandToolsBehavior,
        ServerToolsBehavior,
        ArchitectToolsBehavior,
        CompactWhenNearFullBehavior,
        HierarchicalContextBehavior,
        SubAgentContextBehavior,
        LoopDetectionBehavior,
        WorkspaceTaskNotesBehavior,
        StatusDisplayBehavior,
    ]

    # 1. All can be instantiated
    behaviors = []
    for BehaviorClass in all_behavior_classes:
        if BehaviorClass == CommandToolsBehavior:
            behavior = BehaviorClass(whitelist=["python"])
        elif BehaviorClass == CompactWhenNearFullBehavior:
            behavior = BehaviorClass(max_tokens=8000)
        elif BehaviorClass == HierarchicalContextBehavior:
            behavior = BehaviorClass(history_keep=12)
        elif BehaviorClass == LoopDetectionBehavior:
            behavior = BehaviorClass(max_repeats=5)
        else:
            behavior = BehaviorClass()
        behaviors.append(behavior)

    # 2. All have unique names
    names = [b.get_name() for b in behaviors]
    assert len(names) == len(set(names)), f"Duplicate names: {names}"

    # 3. All tools are unique
    all_tools = []
    for b in behaviors:
        all_tools.extend(b.get_tools())
    tool_names = [t["function"]["name"] for t in all_tools]
    duplicates = [name for name in tool_names if tool_names.count(name) > 1]
    assert len(duplicates) == 0, f"Duplicate tools: {set(duplicates)}"

    print(f"\n✓ Composability verified for {len(behaviors)} behaviors")
    print(f"✓ {len(tool_names)} unique tools registered")
