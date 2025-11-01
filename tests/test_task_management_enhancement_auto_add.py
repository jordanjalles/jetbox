"""
Test that TaskManagementEnhancement is auto-added to OrchestratorAgent and ArchitectAgent.

This test verifies that:
1. OrchestratorAgent auto-detects task breakdown and adds enhancement during build_context()
2. ArchitectAgent auto-detects task breakdown and adds enhancement during build_context()
3. Enhancement provides task status in context
4. Enhancement provides task management tools
"""
import tempfile
from pathlib import Path
import json

from orchestrator_agent import OrchestratorAgent
from architect_agent import ArchitectAgent
from context_strategies import TaskManagementEnhancement


def test_orchestrator_auto_adds_task_management():
    """Test that OrchestratorAgent auto-adds TaskManagementEnhancement when task breakdown exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create orchestrator
        orchestrator = OrchestratorAgent(workspace=workspace)

        # Initially no enhancements
        assert len(orchestrator.enhancements) == 0

        # Create task breakdown file
        arch_dir = workspace / "architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        task_breakdown = {
            "tasks": [
                {
                    "id": "T1",
                    "description": "Implement feature A",
                    "module": "core",
                    "status": "pending",
                    "dependencies": [],
                    "priority": 1,
                    "estimated_complexity": "medium"
                },
                {
                    "id": "T2",
                    "description": "Implement feature B",
                    "module": "api",
                    "status": "pending",
                    "dependencies": ["T1"],
                    "priority": 2,
                    "estimated_complexity": "low"
                }
            ]
        }
        task_file = arch_dir / "task-breakdown.json"
        task_file.write_text(json.dumps(task_breakdown, indent=2))

        # Build context - should auto-add enhancement
        context = orchestrator.build_context()

        # Verify enhancement was added
        assert len(orchestrator.enhancements) == 1
        assert isinstance(orchestrator.enhancements[0], TaskManagementEnhancement)

        # Verify context includes task status
        context_str = "\n".join([msg.get("content", "") for msg in context])
        assert "TASK BREAKDOWN STATUS" in context_str
        assert "Total Tasks: 2" in context_str
        assert "T1" in context_str or "Implement feature A" in context_str

        # Verify tools include task management tools
        tools = orchestrator.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "read_task_breakdown" in tool_names
        assert "get_next_task" in tool_names
        assert "mark_task_status" in tool_names

        print("✓ OrchestratorAgent auto-adds TaskManagementEnhancement")


def test_orchestrator_only_adds_once():
    """Test that OrchestratorAgent doesn't add enhancement twice."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create task breakdown
        arch_dir = workspace / "architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        task_breakdown = {"tasks": [{"id": "T1", "description": "Test", "module": "core", "status": "pending", "dependencies": [], "priority": 1, "estimated_complexity": "low"}]}
        (arch_dir / "task-breakdown.json").write_text(json.dumps(task_breakdown))

        # Create orchestrator
        orchestrator = OrchestratorAgent(workspace=workspace)

        # Build context twice
        orchestrator.build_context()
        orchestrator.build_context()

        # Should only have one enhancement
        assert len(orchestrator.enhancements) == 1

        print("✓ OrchestratorAgent only adds enhancement once")


def test_architect_auto_adds_task_management():
    """Test that ArchitectAgent auto-adds TaskManagementEnhancement when task breakdown exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create architect
        architect = ArchitectAgent(workspace=workspace, project_description="Test Project")

        # Initially no enhancements (configure_workspace runs during __init__ but no task breakdown yet)
        assert len(architect.enhancements) == 0

        # Create task breakdown file
        arch_dir = workspace / "architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        task_breakdown = {
            "tasks": [
                {
                    "id": "T1",
                    "description": "Design module A",
                    "module": "core",
                    "status": "pending",
                    "dependencies": [],
                    "priority": 1,
                    "estimated_complexity": "high"
                }
            ]
        }
        task_file = arch_dir / "task-breakdown.json"
        task_file.write_text(json.dumps(task_breakdown, indent=2))

        # Build context - should auto-add enhancement
        context = architect.build_context()

        # Verify enhancement was added
        assert len(architect.enhancements) == 1
        assert isinstance(architect.enhancements[0], TaskManagementEnhancement)

        # Verify context includes task status
        context_str = "\n".join([msg.get("content", "") for msg in context])
        assert "TASK BREAKDOWN STATUS" in context_str
        assert "Total Tasks: 1" in context_str

        # Verify tools include task management tools
        tools = architect.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "read_task_breakdown" in tool_names
        assert "get_next_task" in tool_names
        assert "mark_task_status" in tool_names

        print("✓ ArchitectAgent auto-adds TaskManagementEnhancement")


def test_architect_only_adds_once():
    """Test that ArchitectAgent doesn't add enhancement twice."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create task breakdown
        arch_dir = workspace / "architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        task_breakdown = {"tasks": [{"id": "T1", "description": "Test", "module": "core", "status": "pending", "dependencies": [], "priority": 1, "estimated_complexity": "low"}]}
        (arch_dir / "task-breakdown.json").write_text(json.dumps(task_breakdown))

        # Create architect (configure_workspace runs and adds enhancement)
        architect = ArchitectAgent(workspace=workspace, project_description="Test")

        # Build context multiple times
        architect.build_context()
        architect.build_context()

        # Should only have one enhancement
        assert len(architect.enhancements) == 1

        print("✓ ArchitectAgent only adds enhancement once")


def test_enhancement_provides_task_context():
    """Test that enhancement injects task status into context properly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create task breakdown with various statuses
        arch_dir = workspace / "architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        task_breakdown = {
            "tasks": [
                {"id": "T1", "description": "Task 1", "module": "core", "status": "completed", "dependencies": [], "priority": 1, "estimated_complexity": "low"},
                {"id": "T2", "description": "Task 2", "module": "api", "status": "in_progress", "dependencies": ["T1"], "priority": 2, "estimated_complexity": "medium"},
                {"id": "T3", "description": "Task 3", "module": "ui", "status": "pending", "dependencies": ["T2"], "priority": 3, "estimated_complexity": "high"},
                {"id": "T4", "description": "Task 4", "module": "core", "status": "pending", "dependencies": [], "priority": 4, "estimated_complexity": "low"},
            ]
        }
        (arch_dir / "task-breakdown.json").write_text(json.dumps(task_breakdown))

        # Create orchestrator
        orchestrator = OrchestratorAgent(workspace=workspace)
        context = orchestrator.build_context()

        # Find task status in context
        task_status_msg = None
        for msg in context:
            if msg.get("role") == "user" and "TASK BREAKDOWN STATUS" in msg.get("content", ""):
                task_status_msg = msg["content"]
                break

        assert task_status_msg is not None

        # Verify status counts
        assert "Total Tasks: 4" in task_status_msg
        assert "Pending:     2" in task_status_msg
        assert "In Progress: 1" in task_status_msg
        assert "Completed:   1" in task_status_msg

        # Verify task list shows status icons
        assert "✓" in task_status_msg  # Completed
        assert "⟳" in task_status_msg  # In progress
        assert "○" in task_status_msg  # Pending

        print("✓ Enhancement provides task context with status counts and icons")


def test_enhancement_provides_next_task():
    """Test that enhancement shows next pending task respecting dependencies."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create task breakdown with dependencies
        arch_dir = workspace / "architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        task_breakdown = {
            "tasks": [
                {"id": "T1", "description": "Foundation", "module": "core", "status": "completed", "dependencies": [], "priority": 1, "estimated_complexity": "medium"},
                {"id": "T2", "description": "Build on foundation", "module": "api", "status": "pending", "dependencies": ["T1"], "priority": 2, "estimated_complexity": "low"},
                {"id": "T3", "description": "Depends on T2", "module": "ui", "status": "pending", "dependencies": ["T2"], "priority": 3, "estimated_complexity": "high"},
            ]
        }
        (arch_dir / "task-breakdown.json").write_text(json.dumps(task_breakdown))

        # Create orchestrator
        orchestrator = OrchestratorAgent(workspace=workspace)
        context = orchestrator.build_context()

        # Find task status in context
        context_str = "\n".join([msg.get("content", "") for msg in context])

        # Should show T2 as next pending task (T1 is completed, T2 dependencies satisfied)
        assert "NEXT PENDING TASK:" in context_str
        assert "T2" in context_str
        assert "Build on foundation" in context_str

        print("✓ Enhancement shows next pending task respecting dependencies")


if __name__ == "__main__":
    print("Testing TaskManagementEnhancement auto-add functionality...\n")

    test_orchestrator_auto_adds_task_management()
    test_orchestrator_only_adds_once()
    test_architect_auto_adds_task_management()
    test_architect_only_adds_once()
    test_enhancement_provides_task_context()
    test_enhancement_provides_next_task()

    print("\n✓ All tests passed!")
