"""
Tests for HierarchicalContextBehavior.

Tests:
- Hierarchy injection
- Message pruning (keep last N)
- decompose_task tool
- mark_subtask_complete tool
- Clear on transitions
"""

import pytest
from unittest.mock import Mock, MagicMock
from behaviors.hierarchical_context import HierarchicalContextBehavior


class TestHierarchicalContextBehavior:
    """Test suite for HierarchicalContextBehavior."""

    def test_get_name(self):
        """Test behavior returns correct name."""
        behavior = HierarchicalContextBehavior()
        assert behavior.get_name() == "hierarchical_context"

    def test_initialization_with_defaults(self):
        """Test behavior initializes with default parameters."""
        behavior = HierarchicalContextBehavior()
        assert behavior.history_keep == 12
        assert behavior.clear_on_transition is True

    def test_initialization_with_custom_params(self):
        """Test behavior initializes with custom parameters."""
        behavior = HierarchicalContextBehavior(
            history_keep=20,
            clear_on_transition=False
        )
        assert behavior.history_keep == 20
        assert behavior.clear_on_transition is False

    def test_enhance_context_injects_hierarchy(self):
        """Test hierarchy information is injected into context."""
        behavior = HierarchicalContextBehavior()

        # Mock context manager with goal/task/subtask
        mock_cm = Mock()
        mock_goal = Mock()
        mock_goal.description = "Create a calculator"
        mock_cm.state.goal = mock_goal

        mock_task = Mock()
        mock_task.description = "Implement addition"
        mock_subtask = Mock()
        mock_subtask.description = "Write add function"
        mock_subtask.depth = 1
        mock_subtask.rounds_used = 2
        mock_task.active_subtask.return_value = mock_subtask
        mock_cm._get_current_task.return_value = mock_task
        mock_cm.state.loop_counts = {}

        context = [
            {"role": "system", "content": "You are an assistant"}
        ]

        result = behavior.enhance_context(context, context_manager=mock_cm)

        # Should have system + hierarchy info
        assert len(result) >= 2
        hierarchy_msg = result[1]
        assert "GOAL: Create a calculator" in hierarchy_msg["content"]
        assert "CURRENT TASK: Implement addition" in hierarchy_msg["content"]
        assert "ACTIVE SUBTASK: Write add function" in hierarchy_msg["content"]

    def test_enhance_context_no_tasks_yet_warning(self):
        """Test warning is shown when no tasks exist yet."""
        behavior = HierarchicalContextBehavior()

        # Mock context manager with goal but no tasks
        mock_cm = Mock()
        mock_goal = Mock()
        mock_goal.description = "Create a project"
        mock_cm.state.goal = mock_goal
        mock_cm._get_current_task.return_value = None
        mock_cm.state.loop_counts = {}

        context = [{"role": "system", "content": "System"}]

        result = behavior.enhance_context(context, context_manager=mock_cm)

        hierarchy_msg = result[1]
        assert "NO TASKS YET" in hierarchy_msg["content"]
        assert "decompose_task" in hierarchy_msg["content"]

    def test_enhance_context_includes_loop_warnings(self):
        """Test loop warnings are included when loops detected."""
        behavior = HierarchicalContextBehavior()

        # Mock context manager with loop counts
        mock_cm = Mock()
        mock_goal = Mock()
        mock_goal.description = "Test goal"
        mock_cm.state.goal = mock_goal
        mock_cm._get_current_task.return_value = None
        mock_cm.state.loop_counts = {
            "write_file::test.py": 3,
            "run_bash::pytest": 5
        }

        context = [{"role": "system", "content": "System"}]

        result = behavior.enhance_context(context, context_manager=mock_cm)

        hierarchy_msg = result[1]
        assert "LOOP DETECTION WARNING" in hierarchy_msg["content"]
        assert "repeated" in hierarchy_msg["content"]

    def test_enhance_context_includes_probe_state(self):
        """Test filesystem probe state is included."""
        behavior = HierarchicalContextBehavior()

        # Mock context manager and probe function
        mock_cm = Mock()
        mock_goal = Mock()
        mock_goal.description = "Test"
        mock_cm.state.goal = mock_goal
        mock_cm._get_current_task.return_value = None
        mock_cm.state.loop_counts = {}

        def mock_probe():
            return {
                "files_exist": ["test.py", "main.py"],
                "recent_errors": ["ImportError: module not found"]
            }

        context = [{"role": "system", "content": "System"}]

        result = behavior.enhance_context(
            context,
            context_manager=mock_cm,
            probe_state_func=mock_probe
        )

        hierarchy_msg = result[1]
        assert "FILES CREATED" in hierarchy_msg["content"]
        assert "test.py" in hierarchy_msg["content"]
        assert "RECENT ERRORS" in hierarchy_msg["content"]

    def test_enhance_context_prunes_old_messages(self):
        """Test old messages are pruned, keeping only recent N exchanges."""
        behavior = HierarchicalContextBehavior(history_keep=2)  # Keep 2 exchanges = 4 messages

        mock_cm = Mock()
        mock_goal = Mock()
        mock_goal.description = "Test"
        mock_cm.state.goal = mock_goal
        mock_cm._get_current_task.return_value = None
        mock_cm.state.loop_counts = {}

        # Create context with many messages
        context = [
            {"role": "system", "content": "System"},
        ]
        # Add 10 message exchanges (20 messages)
        for i in range(10):
            context.append({"role": "user", "content": f"Message {i}"})
            context.append({"role": "assistant", "content": f"Response {i}"})

        result = behavior.enhance_context(context, context_manager=mock_cm)

        # Should have: system + hierarchy + 4 recent messages (2 exchanges)
        assert len(result) <= 6  # system + hierarchy + 4 messages

        # Check that recent messages are preserved
        content_str = str(result)
        assert "Message 9" in content_str  # Most recent
        assert "Message 8" in content_str  # Second most recent

        # Old messages should be pruned
        assert "Message 0" not in content_str
        assert "Message 1" not in content_str

    def test_get_tools_returns_hierarchical_tools(self):
        """Test get_tools returns decompose_task and mark_subtask_complete."""
        behavior = HierarchicalContextBehavior()
        tools = behavior.get_tools()

        assert len(tools) == 2
        tool_names = [t["function"]["name"] for t in tools]
        assert "mark_subtask_complete" in tool_names
        assert "decompose_task" in tool_names

    def test_dispatch_tool_mark_subtask_complete_success(self):
        """Test dispatch_tool handles mark_subtask_complete successfully."""
        behavior = HierarchicalContextBehavior()

        mock_cm = Mock()
        mock_cm.mark_subtask_complete.return_value = "Subtask marked complete"

        result = behavior.dispatch_tool(
            "mark_subtask_complete",
            {"success": True, "reason": "Tests pass"},
            context_manager=mock_cm
        )

        assert result["success"] is True
        mock_cm.mark_subtask_complete.assert_called_once_with(
            success=True,
            reason="Tests pass"
        )

    def test_dispatch_tool_mark_subtask_complete_failure(self):
        """Test dispatch_tool handles mark_subtask_complete with failure."""
        behavior = HierarchicalContextBehavior()

        mock_cm = Mock()
        mock_cm.mark_subtask_complete.return_value = "Subtask marked failed"

        result = behavior.dispatch_tool(
            "mark_subtask_complete",
            {"success": False, "reason": "Tests fail"},
            context_manager=mock_cm
        )

        assert result["success"] is True
        mock_cm.mark_subtask_complete.assert_called_once_with(
            success=False,
            reason="Tests fail"
        )

    def test_dispatch_tool_decompose_task(self):
        """Test dispatch_tool handles decompose_task."""
        behavior = HierarchicalContextBehavior()

        mock_cm = Mock()

        subtasks = ["Write code", "Write tests", "Run linter"]
        result = behavior.dispatch_tool(
            "decompose_task",
            {"subtasks": subtasks},
            context_manager=mock_cm
        )

        assert result["success"] is True
        assert "3 subtasks" in result["result"]
        mock_cm.create_task_with_subtasks.assert_called_once()

    def test_dispatch_tool_decompose_task_no_subtasks(self):
        """Test dispatch_tool handles decompose_task with empty subtasks."""
        behavior = HierarchicalContextBehavior()

        mock_cm = Mock()

        result = behavior.dispatch_tool(
            "decompose_task",
            {"subtasks": []},
            context_manager=mock_cm
        )

        assert "error" in result
        assert "No subtasks" in result["error"]

    def test_dispatch_tool_no_context_manager(self):
        """Test dispatch_tool returns error when no context manager provided."""
        behavior = HierarchicalContextBehavior()

        result = behavior.dispatch_tool(
            "mark_subtask_complete",
            {"success": True}
        )

        assert "error" in result
        assert "No context manager" in result["error"]

    def test_dispatch_tool_unknown_tool(self):
        """Test dispatch_tool raises NotImplementedError for unknown tools."""
        behavior = HierarchicalContextBehavior()

        mock_cm = Mock()

        with pytest.raises(NotImplementedError):
            behavior.dispatch_tool("unknown_tool", {}, context_manager=mock_cm)

    def test_get_instructions_returns_workflow(self):
        """Test get_instructions returns hierarchical workflow."""
        behavior = HierarchicalContextBehavior()
        instructions = behavior.get_instructions()

        assert "HIERARCHICAL WORKFLOW" in instructions
        assert "decompose_task" in instructions
        assert "mark_subtask_complete" in instructions

    def test_enhance_context_no_context_manager(self):
        """Test enhance_context returns unchanged context if no context manager."""
        behavior = HierarchicalContextBehavior()

        context = [{"role": "system", "content": "System"}]

        result = behavior.enhance_context(context)

        assert result == context

    def test_enhance_context_with_config_shows_depth_and_rounds(self):
        """Test hierarchy info includes depth and rounds when config provided."""
        behavior = HierarchicalContextBehavior()

        # Mock context manager with config
        mock_cm = Mock()
        mock_goal = Mock()
        mock_goal.description = "Test"
        mock_cm.state.goal = mock_goal

        mock_task = Mock()
        mock_task.description = "Task 1"
        mock_subtask = Mock()
        mock_subtask.description = "Subtask 1"
        mock_subtask.depth = 2
        mock_subtask.rounds_used = 3
        mock_task.active_subtask.return_value = mock_subtask
        mock_cm._get_current_task.return_value = mock_task
        mock_cm.state.loop_counts = {}

        # Mock config
        mock_config = Mock()
        mock_config.hierarchy.max_depth = 5
        mock_config.rounds.max_per_subtask = 6

        context = [{"role": "system", "content": "System"}]

        result = behavior.enhance_context(
            context,
            context_manager=mock_cm,
            config=mock_config
        )

        hierarchy_msg = result[1]
        assert "Subtask Depth: 2/5" in hierarchy_msg["content"]
        assert "Rounds Used: 3/6" in hierarchy_msg["content"]

    def test_enhance_context_includes_jetbox_notes(self):
        """Test jetbox notes are included when available."""
        behavior = HierarchicalContextBehavior()

        mock_cm = Mock()
        mock_goal = Mock()
        mock_goal.description = "Test"
        mock_cm.state.goal = mock_goal
        mock_cm._get_current_task.return_value = None
        mock_cm.state.loop_counts = {}

        context = [{"role": "system", "content": "System"}]

        # Mock jetbox_notes module
        with pytest.mock.patch('behaviors.hierarchical_context.jetbox_notes') as mock_notes:
            mock_notes.load_jetbox_notes.return_value = "Previous work: Created calculator"

            result = behavior.enhance_context(
                context,
                context_manager=mock_cm,
                workspace="/workspace"
            )

            hierarchy_msg = result[1]
            assert "JETBOX NOTES" in hierarchy_msg["content"]
            assert "Previous work: Created calculator" in hierarchy_msg["content"]
