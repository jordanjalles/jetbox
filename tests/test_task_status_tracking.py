"""
Test task status tracking functionality.

Tests the enhanced task-breakdown.json schema with status fields
and the task management tools for updating task status.
"""
import json
import tempfile
from pathlib import Path
from datetime import datetime

import architect_tools
import task_management_tools
from workspace_manager import WorkspaceManager


class TestTaskStatusTracking:
    """Test suite for task status tracking system."""

    def setup_method(self):
        """Set up test workspace for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)

        # Create workspace manager
        self.workspace_manager = WorkspaceManager(
            goal="test-task-status",
            workspace_path=self.workspace_path
        )

        # Configure tools with workspace
        class SimpleWorkspace:
            def __init__(self, workspace_dir):
                self.workspace_dir = workspace_dir

        architect_tools.set_workspace(SimpleWorkspace(self.workspace_path))
        task_management_tools.set_workspace(self.workspace_manager)

    def teardown_method(self):
        """Clean up test workspace."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_write_task_list_initializes_status_fields(self):
        """Test that write_task_list initializes all status tracking fields."""
        print("\n" + "="*70)
        print("TEST: Write task list initializes status fields")
        print("="*70)

        # Create task list without status fields
        tasks = [
            {
                "id": "T1",
                "description": "Implement auth module",
                "module": "auth-service",
                "priority": 1,
                "dependencies": [],
                "estimated_complexity": "medium"
            },
            {
                "id": "T2",
                "description": "Implement API gateway",
                "module": "api-gateway",
                "priority": 2,
                "dependencies": ["T1"],
                "estimated_complexity": "high"
            }
        ]

        # Write task list
        result = architect_tools.write_task_list(tasks)
        print(f"\n✓ Task list written: {result['file_path']}")

        assert result["status"] == "success"
        assert result["task_count"] == 2

        # Read the file and verify status fields were added
        task_file = self.workspace_path / "architecture" / "task-breakdown.json"
        assert task_file.exists()

        with open(task_file) as f:
            data = json.load(f)

        print(f"\n✓ Loaded task breakdown with {len(data['tasks'])} tasks")

        # Verify each task has all status fields
        for task in data["tasks"]:
            print(f"\nTask {task['id']}:")
            print(f"  Status: {task['status']}")
            print(f"  Started: {task['started_at']}")
            print(f"  Completed: {task['completed_at']}")
            print(f"  Result: {task['result']}")
            print(f"  Attempts: {task['attempts']}")
            print(f"  Notes: {task['notes']}")

            assert task["status"] == "pending"
            assert task["started_at"] is None
            assert task["completed_at"] is None
            assert task["result"] is None
            assert task["attempts"] == 0
            assert task["notes"] == []

        print("\n✅ All status fields initialized correctly")

    def test_mark_task_in_progress_sets_timestamps(self):
        """Test that marking task as in_progress sets started_at and increments attempts."""
        print("\n" + "="*70)
        print("TEST: Mark task in_progress sets timestamps")
        print("="*70)

        # Create and write task list
        tasks = [
            {
                "id": "T1",
                "description": "Implement auth module",
                "module": "auth-service",
                "priority": 1,
                "dependencies": [],
                "estimated_complexity": "medium"
            }
        ]
        architect_tools.write_task_list(tasks)

        # Mark task as in_progress
        result = task_management_tools.mark_task_status(
            task_id="T1",
            status="in_progress",
            notes="Starting work on auth module"
        )

        print(f"\n✓ Task status updated: {result['message']}")
        assert result["status"] == "success"
        assert result["new_status"] == "in_progress"

        # Read task and verify timestamps
        task_file = self.workspace_path / "architecture" / "task-breakdown.json"
        with open(task_file) as f:
            data = json.load(f)

        task = data["tasks"][0]
        print("\nTask T1 after marking in_progress:")
        print(f"  Status: {task['status']}")
        print(f"  Started: {task['started_at']}")
        print(f"  Attempts: {task['attempts']}")
        print(f"  Notes: {len(task['notes'])} note(s)")

        assert task["status"] == "in_progress"
        assert task["started_at"] is not None
        assert task["attempts"] == 1
        assert len(task["notes"]) == 1
        assert task["notes"][0]["note"] == "Starting work on auth module"

        # Verify started_at is a valid ISO timestamp
        datetime.fromisoformat(task["started_at"])

        print("\n✅ Timestamps and attempts set correctly")

    def test_mark_task_completed_sets_timestamps_and_result(self):
        """Test that marking task as completed sets completed_at and result."""
        print("\n" + "="*70)
        print("TEST: Mark task completed sets timestamps and result")
        print("="*70)

        # Create task list
        tasks = [
            {
                "id": "T1",
                "description": "Implement auth module",
                "module": "auth-service",
                "priority": 1,
                "dependencies": [],
                "estimated_complexity": "medium"
            }
        ]
        architect_tools.write_task_list(tasks)

        # Mark as in_progress first
        task_management_tools.mark_task_status(
            task_id="T1",
            status="in_progress",
            notes="Started implementation"
        )

        # Mark as completed
        result = task_management_tools.mark_task_status(
            task_id="T1",
            status="completed",
            result="Created auth module with JWT support and user registration",
            notes="All tests passing"
        )

        print(f"\n✓ Task marked completed: {result['message']}")
        assert result["status"] == "success"
        assert result["new_status"] == "completed"

        # Read task and verify
        task_file = self.workspace_path / "architecture" / "task-breakdown.json"
        with open(task_file) as f:
            data = json.load(f)

        task = data["tasks"][0]
        print("\nTask T1 after completion:")
        print(f"  Status: {task['status']}")
        print(f"  Started: {task['started_at']}")
        print(f"  Completed: {task['completed_at']}")
        print(f"  Result: {task['result']}")
        print(f"  Attempts: {task['attempts']}")
        print(f"  Notes: {len(task['notes'])} note(s)")

        assert task["status"] == "completed"
        assert task["completed_at"] is not None
        assert task["result"] == "Created auth module with JWT support and user registration"
        assert len(task["notes"]) == 2  # in_progress note + completed note

        # Verify completed_at is a valid ISO timestamp
        datetime.fromisoformat(task["completed_at"])

        print("\n✅ Completion timestamps and result stored correctly")

    def test_get_next_task_respects_dependencies(self):
        """Test that get_next_task returns tasks in dependency order."""
        print("\n" + "="*70)
        print("TEST: Get next task respects dependencies")
        print("="*70)

        # Create task list with dependencies
        tasks = [
            {
                "id": "T1",
                "description": "Set up database",
                "module": "infrastructure",
                "priority": 1,
                "dependencies": [],
                "estimated_complexity": "low"
            },
            {
                "id": "T2",
                "description": "Create user model",
                "module": "auth-service",
                "priority": 2,
                "dependencies": ["T1"],
                "estimated_complexity": "medium"
            },
            {
                "id": "T3",
                "description": "Create admin dashboard",
                "module": "admin-ui",
                "priority": 3,
                "dependencies": ["T1", "T2"],
                "estimated_complexity": "high"
            }
        ]
        architect_tools.write_task_list(tasks)

        # Get next task (should be T1 - no dependencies)
        result = task_management_tools.get_next_task()
        print(f"\n✓ First next task: {result['task']['id']} - {result['task']['description']}")
        assert result["status"] == "success"
        assert result["task"]["id"] == "T1"

        # Mark T1 as completed
        task_management_tools.mark_task_status("T1", "completed", result="Database configured")

        # Get next task (should be T2 - T1 is completed)
        result = task_management_tools.get_next_task()
        print(f"✓ Second next task: {result['task']['id']} - {result['task']['description']}")
        assert result["status"] == "success"
        assert result["task"]["id"] == "T2"

        # Mark T2 as in_progress (not completed)
        task_management_tools.mark_task_status("T2", "in_progress")

        # Get next task (should be None - T3 depends on T2 which is not completed)
        result = task_management_tools.get_next_task()
        print(f"✓ Third next task: {result['task']} (should be None)")
        assert result["status"] == "success"
        assert result["task"] is None

        # Mark T2 as completed
        task_management_tools.mark_task_status("T2", "completed", result="User model created")

        # Get next task (should be T3 - all dependencies completed)
        result = task_management_tools.get_next_task()
        print(f"✓ Fourth next task: {result['task']['id']} - {result['task']['description']}")
        assert result["status"] == "success"
        assert result["task"]["id"] == "T3"

        print("\n✅ Dependency ordering works correctly")

    def test_read_task_breakdown_counts_statuses(self):
        """Test that read_task_breakdown returns accurate status counts."""
        print("\n" + "="*70)
        print("TEST: Read task breakdown counts statuses")
        print("="*70)

        # Create task list
        tasks = [
            {"id": "T1", "description": "Task 1", "module": "mod1", "priority": 1, "dependencies": []},
            {"id": "T2", "description": "Task 2", "module": "mod2", "priority": 2, "dependencies": []},
            {"id": "T3", "description": "Task 3", "module": "mod3", "priority": 3, "dependencies": []},
            {"id": "T4", "description": "Task 4", "module": "mod4", "priority": 4, "dependencies": []},
        ]
        architect_tools.write_task_list(tasks)

        # Initial state - all pending
        result = task_management_tools.read_task_breakdown()
        print("\n✓ Initial breakdown:")
        print(f"  Total: {result['total_tasks']}")
        print(f"  Pending: {result['pending_count']}")
        print(f"  In progress: {result['in_progress_count']}")
        print(f"  Completed: {result['completed_count']}")
        print(f"  Failed: {result['failed_count']}")

        assert result["total_tasks"] == 4
        assert result["pending_count"] == 4
        assert result["in_progress_count"] == 0
        assert result["completed_count"] == 0
        assert result["failed_count"] == 0

        # Update various tasks
        task_management_tools.mark_task_status("T1", "completed")
        task_management_tools.mark_task_status("T2", "in_progress")
        task_management_tools.mark_task_status("T3", "failed", result="Missing dependencies")

        # Check updated counts
        result = task_management_tools.read_task_breakdown()
        print("\n✓ Updated breakdown:")
        print(f"  Total: {result['total_tasks']}")
        print(f"  Pending: {result['pending_count']}")
        print(f"  In progress: {result['in_progress_count']}")
        print(f"  Completed: {result['completed_count']}")
        print(f"  Failed: {result['failed_count']}")

        assert result["total_tasks"] == 4
        assert result["pending_count"] == 1  # T4
        assert result["in_progress_count"] == 1  # T2
        assert result["completed_count"] == 1  # T1
        assert result["failed_count"] == 1  # T3

        print("\n✅ Status counts are accurate")

    def test_multiple_attempts_increment_correctly(self):
        """Test that marking a task as in_progress multiple times increments attempts."""
        print("\n" + "="*70)
        print("TEST: Multiple attempts increment correctly")
        print("="*70)

        # Create task
        tasks = [
            {"id": "T1", "description": "Complex task", "module": "mod1", "priority": 1, "dependencies": []}
        ]
        architect_tools.write_task_list(tasks)

        # First attempt
        task_management_tools.mark_task_status("T1", "in_progress", notes="First attempt")

        # Read and verify
        task_file = self.workspace_path / "architecture" / "task-breakdown.json"
        with open(task_file) as f:
            data = json.load(f)
        task = data["tasks"][0]
        print(f"\n✓ After first attempt: attempts = {task['attempts']}")
        assert task["attempts"] == 1

        # Reset to pending, then try again
        task_management_tools.mark_task_status("T1", "pending")
        task_management_tools.mark_task_status("T1", "in_progress", notes="Second attempt")

        with open(task_file) as f:
            data = json.load(f)
        task = data["tasks"][0]
        print(f"✓ After second attempt: attempts = {task['attempts']}")
        assert task["attempts"] == 2

        # Third attempt
        task_management_tools.mark_task_status("T1", "pending")
        task_management_tools.mark_task_status("T1", "in_progress", notes="Third attempt")

        with open(task_file) as f:
            data = json.load(f)
        task = data["tasks"][0]
        print(f"✓ After third attempt: attempts = {task['attempts']}")
        assert task["attempts"] == 3

        print("\n✅ Attempts increment correctly across retries")

    def test_notes_accumulate_over_time(self):
        """Test that notes accumulate as task progresses."""
        print("\n" + "="*70)
        print("TEST: Notes accumulate over time")
        print("="*70)

        # Create task
        tasks = [
            {"id": "T1", "description": "Task with notes", "module": "mod1", "priority": 1, "dependencies": []}
        ]
        architect_tools.write_task_list(tasks)

        # Add multiple notes
        task_management_tools.mark_task_status("T1", "in_progress", notes="Started implementation")
        task_management_tools.mark_task_status("T1", "in_progress", notes="Encountered issue with database")
        task_management_tools.mark_task_status("T1", "in_progress", notes="Fixed database connection")
        task_management_tools.mark_task_status("T1", "completed", result="Task complete", notes="All tests passing")

        # Read and verify notes
        task_file = self.workspace_path / "architecture" / "task-breakdown.json"
        with open(task_file) as f:
            data = json.load(f)
        task = data["tasks"][0]

        print(f"\n✓ Task has {len(task['notes'])} notes:")
        for i, note in enumerate(task["notes"], 1):
            print(f"  {i}. [{note['timestamp']}] {note['note']}")

        assert len(task["notes"]) == 4
        assert task["notes"][0]["note"] == "Started implementation"
        assert task["notes"][1]["note"] == "Encountered issue with database"
        assert task["notes"][2]["note"] == "Fixed database connection"
        assert task["notes"][3]["note"] == "All tests passing"

        # Verify all notes have timestamps
        for note in task["notes"]:
            datetime.fromisoformat(note["timestamp"])

        print("\n✅ Notes accumulate correctly with timestamps")


def test_task_status_tracking_basic():
    """Basic smoke test for task status tracking."""
    suite = TestTaskStatusTracking()
    suite.setup_method()

    try:
        print("\n" + "="*70)
        print("TASK STATUS TRACKING TEST SUITE")
        print("="*70)

        # Run all tests
        suite.test_write_task_list_initializes_status_fields()
        suite.test_mark_task_in_progress_sets_timestamps()
        suite.test_mark_task_completed_sets_timestamps_and_result()
        suite.test_get_next_task_respects_dependencies()
        suite.test_read_task_breakdown_counts_statuses()
        suite.test_multiple_attempts_increment_correctly()
        suite.test_notes_accumulate_over_time()

        print("\n" + "="*70)
        print("✅ ALL TASK STATUS TRACKING TESTS PASSED")
        print("="*70)

    finally:
        suite.teardown_method()


if __name__ == "__main__":
    test_task_status_tracking_basic()
