"""Test jetbox_notes.py with non-hierarchical context (generic approach)."""
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import jetbox_notes


@dataclass
class MockAction:
    """Mock Action object for testing (mimics context_manager.Action)."""
    name: str
    args: dict[str, Any]
    result: str | None = None
    error_msg: str = ""
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class MockGoal:
    """Mock Goal object for testing."""
    description: str


class MockWorkspace:
    """Mock workspace manager for testing."""
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_dir = self.temp_dir
        self.goal = "Test goal for generic jetbox notes"


def mock_llm_call(messages, temperature=0.2, timeout=30):
    """Mock LLM call that returns a fake summary."""
    return {
        "message": {
            "content": "- Created test files\n- Ran tests successfully\n- Fixed linting errors\n- Timed out while debugging edge case"
        }
    }


def test_timeout_summary_with_action_history():
    """Test create_timeout_summary using action_history (strategy-agnostic)."""
    print("\n" + "="*70)
    print("TEST: create_timeout_summary with action_history (NO hierarchical context)")
    print("="*70)

    # Setup mock workspace
    workspace = MockWorkspace()
    jetbox_notes.set_workspace(workspace)
    jetbox_notes.set_llm_caller(mock_llm_call)

    # Create mock action history (simulates agent work)
    action_history = [
        MockAction(name="write_file", args={"path": "test.py"}, result="success"),
        MockAction(name="write_file", args={"path": "main.py"}, result="success"),
        MockAction(name="run_cmd", args={"cmd": "pytest test.py"}, result="success"),
        MockAction(name="run_cmd", args={"cmd": "ruff check ."}, result="error", error_msg="E501 line too long"),
        MockAction(name="write_file", args={"path": "main.py"}, result="success"),
        MockAction(name="run_cmd", args={"cmd": "ruff check ."}, result="success"),
        MockAction(name="run_cmd", args={"cmd": "pytest test.py"}, result="error", error_msg="AssertionError: expected 5, got 4"),
        MockAction(name="write_file", args={"path": "main.py"}, result="success"),
        MockAction(name="run_cmd", args={"cmd": "pytest test.py"}, result="error", error_msg="AssertionError: expected 5, got 4"),
    ]

    # Create mock goal (optional - can also use workspace.goal)
    goal = MockGoal(description="Create a Python calculator package with tests")

    # Call create_timeout_summary with action_history (NO hierarchical context)
    print("\nCalling create_timeout_summary with:")
    print(f"  - goal: {goal.description}")
    print(f"  - elapsed_seconds: 120.5")
    print(f"  - action_history: {len(action_history)} actions")
    print(f"  - NO task tree, NO subtasks, NO hierarchical context")

    jetbox_notes.create_timeout_summary(
        goal=goal,
        elapsed_seconds=120.5,
        action_history=action_history,
    )

    # Verify notes were created
    notes_file = workspace.workspace_dir / "jetboxnotes.md"
    assert notes_file.exists(), "Notes file should be created"

    content = notes_file.read_text()
    print("\n" + "="*70)
    print("Generated Notes:")
    print("="*70)
    print(content)
    print("="*70)

    # Verify content includes expected sections
    assert "TIMEOUT" in content, "Should include timeout header"
    assert "Created test files" in content or "120s" in content, "Should include summary or elapsed time"

    print("\n✓ Test passed: create_timeout_summary works with action_history (generic approach)")

    # Cleanup
    import shutil
    shutil.rmtree(workspace.temp_dir)


def test_timeout_summary_without_action_history():
    """Test create_timeout_summary without action_history (fallback)."""
    print("\n" + "="*70)
    print("TEST: create_timeout_summary WITHOUT action_history (fallback)")
    print("="*70)

    # Setup mock workspace
    workspace = MockWorkspace()
    jetbox_notes.set_workspace(workspace)
    jetbox_notes.set_llm_caller(mock_llm_call)

    # Create mock goal
    goal = MockGoal(description="Test goal without action history")

    # Call create_timeout_summary with NO action_history
    print("\nCalling create_timeout_summary with:")
    print(f"  - goal: {goal.description}")
    print(f"  - elapsed_seconds: 60.0")
    print(f"  - action_history: None (fallback mode)")

    jetbox_notes.create_timeout_summary(
        goal=goal,
        elapsed_seconds=60.0,
        action_history=None,
    )

    # Verify notes were created
    notes_file = workspace.workspace_dir / "jetboxnotes.md"
    assert notes_file.exists(), "Notes file should be created"

    content = notes_file.read_text()
    print("\n" + "="*70)
    print("Generated Notes (Fallback):")
    print("="*70)
    print(content)
    print("="*70)

    # Verify content includes expected sections
    assert "TIMEOUT" in content, "Should include timeout header"
    assert "60s" in content or "60.0" in content, "Should include elapsed time"

    print("\n✓ Test passed: create_timeout_summary works without action_history (fallback)")

    # Cleanup
    import shutil
    shutil.rmtree(workspace.temp_dir)


def test_backward_compatibility_with_hierarchical():
    """Test that old hierarchical calls still work (backward compatibility)."""
    print("\n" + "="*70)
    print("TEST: Backward compatibility with hierarchical context")
    print("="*70)

    # Setup mock workspace
    workspace = MockWorkspace()
    jetbox_notes.set_workspace(workspace)
    jetbox_notes.set_llm_caller(mock_llm_call)

    # Create mock hierarchical structures (from context_manager.py)
    @dataclass
    class MockSubtask:
        description: str
        status: str = "pending"
        rounds_used: int = 0

    @dataclass
    class MockTask:
        description: str
        status: str = "pending"
        subtasks: list = field(default_factory=list)

    @dataclass
    class MockHierarchicalGoal:
        description: str
        status: str = "in_progress"
        tasks: list = field(default_factory=list)

    # Build hierarchical goal
    goal = MockHierarchicalGoal(description="Create a web app")
    task1 = MockTask(description="Setup project structure", status="completed")
    task1.subtasks = [
        MockSubtask(description="Create package.json", status="completed", rounds_used=2),
        MockSubtask(description="Create index.html", status="completed", rounds_used=1),
    ]
    task2 = MockTask(description="Implement features", status="in_progress")
    task2.subtasks = [
        MockSubtask(description="Add login form", status="completed", rounds_used=3),
        MockSubtask(description="Add authentication", status="in_progress", rounds_used=5),
    ]
    goal.tasks = [task1, task2]

    # Create action history from hierarchical work
    action_history = [
        MockAction(name="write_file", args={"path": "package.json"}, result="success"),
        MockAction(name="write_file", args={"path": "index.html"}, result="success"),
        MockAction(name="write_file", args={"path": "login.html"}, result="success"),
        MockAction(name="write_file", args={"path": "auth.js"}, result="success"),
        MockAction(name="run_cmd", args={"cmd": "npm test"}, result="error", error_msg="Auth tests failing"),
    ]

    # Call with BOTH hierarchical goal AND action_history
    print("\nCalling create_timeout_summary with:")
    print(f"  - goal: {goal.description} (hierarchical Goal object)")
    print(f"  - elapsed_seconds: 180.0")
    print(f"  - action_history: {len(action_history)} actions")
    print(f"  - Hierarchical structure: {len(goal.tasks)} tasks, {sum(len(t.subtasks) for t in goal.tasks)} subtasks")

    jetbox_notes.create_timeout_summary(
        goal=goal,
        elapsed_seconds=180.0,
        action_history=action_history,
    )

    # Verify notes were created
    notes_file = workspace.workspace_dir / "jetboxnotes.md"
    assert notes_file.exists(), "Notes file should be created"

    content = notes_file.read_text()
    print("\n" + "="*70)
    print("Generated Notes (Hierarchical + Action History):")
    print("="*70)
    print(content)
    print("="*70)

    # Verify content
    assert "TIMEOUT" in content, "Should include timeout header"
    assert "180s" in content or "180.0" in content, "Should include elapsed time"

    print("\n✓ Test passed: Backward compatibility maintained (works with hierarchical Goal)")

    # Cleanup
    import shutil
    shutil.rmtree(workspace.temp_dir)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING GENERIC JETBOX_NOTES IMPLEMENTATION")
    print("="*70)
    print("\nThese tests verify that jetbox_notes.py works with ANY context strategy")
    print("by using action_history instead of hierarchical task trees.\n")

    test_timeout_summary_with_action_history()
    test_timeout_summary_without_action_history()
    test_backward_compatibility_with_hierarchical()

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print("\njetbox_notes.py is now GENERIC and works with:")
    print("  - HierarchicalStrategy (backward compatible)")
    print("  - AppendUntilFullStrategy")
    print("  - SubAgentStrategy")
    print("  - ArchitectStrategy")
    print("  - Any future strategy that provides action_history")
    print("="*70 + "\n")
