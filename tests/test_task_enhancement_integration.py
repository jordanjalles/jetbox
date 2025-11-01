"""
Integration test demonstrating TaskManagementEnhancement auto-add in realistic workflow.

This test simulates:
1. Orchestrator consulting architect
2. Architect creating task breakdown
3. Orchestrator seeing task status automatically
4. Orchestrator delegating tasks and tracking progress
"""
import tempfile
from pathlib import Path
import json

from orchestrator_agent import OrchestratorAgent
from architect_agent import ArchitectAgent
from context_strategies import TaskManagementEnhancement


def test_orchestrator_architect_task_tracking_workflow():
    """
    Realistic workflow: architect creates tasks, orchestrator tracks them.

    This demonstrates the auto-add feature in action:
    1. Architect creates task breakdown
    2. Orchestrator auto-detects and adds TaskManagementEnhancement
    3. Orchestrator sees task status in context
    4. Orchestrator can update task status
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Step 1: Create architect in workspace
        print("\n[Step 1] Creating architect agent")
        architect = ArchitectAgent(
            workspace=workspace,
            project_description="Real-time analytics platform"
        )

        # Verify no enhancement yet
        assert len(architect.enhancements) == 0

        # Step 2: Architect creates task breakdown (simulated)
        print("[Step 2] Architect creates task breakdown")
        arch_dir = workspace / "architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)

        task_breakdown = {
            "tasks": [
                {
                    "id": "T1",
                    "description": "Implement ingestion module",
                    "module": "ingestion",
                    "status": "pending",
                    "dependencies": [],
                    "priority": 1,
                    "estimated_complexity": "high"
                },
                {
                    "id": "T2",
                    "description": "Implement processing module",
                    "module": "processing",
                    "status": "pending",
                    "dependencies": ["T1"],
                    "priority": 2,
                    "estimated_complexity": "high"
                },
                {
                    "id": "T3",
                    "description": "Implement storage module",
                    "module": "storage",
                    "status": "pending",
                    "dependencies": ["T2"],
                    "priority": 3,
                    "estimated_complexity": "medium"
                },
                {
                    "id": "T4",
                    "description": "Implement API module",
                    "module": "api",
                    "status": "pending",
                    "dependencies": ["T3"],
                    "priority": 4,
                    "estimated_complexity": "medium"
                },
                {
                    "id": "T5",
                    "description": "Add monitoring and metrics",
                    "module": "observability",
                    "status": "pending",
                    "dependencies": ["T1", "T2", "T3", "T4"],
                    "priority": 5,
                    "estimated_complexity": "low"
                }
            ]
        }

        task_file = arch_dir / "task-breakdown.json"
        task_file.write_text(json.dumps(task_breakdown, indent=2))
        print(f"[Step 2] Created {len(task_breakdown['tasks'])} tasks")

        # Step 3: Architect builds context - should auto-add enhancement
        print("[Step 3] Architect builds context (auto-adds TaskManagementEnhancement)")
        architect_context = architect.build_context()

        # Verify enhancement was added
        assert len(architect.enhancements) == 1
        assert isinstance(architect.enhancements[0], TaskManagementEnhancement)
        print("[Step 3] ✓ Architect auto-added TaskManagementEnhancement")

        # Verify architect sees task status in context
        context_str = "\n".join([msg.get("content", "") for msg in architect_context])
        assert "TASK BREAKDOWN STATUS" in context_str
        assert "Total Tasks: 5" in context_str
        assert "Pending:     5" in context_str
        print("[Step 3] ✓ Architect sees task status in context")

        # Step 4: Create orchestrator in same workspace
        print("[Step 4] Creating orchestrator agent in same workspace")
        orchestrator = OrchestratorAgent(workspace=workspace)

        # Verify no enhancement initially
        assert len(orchestrator.enhancements) == 0

        # Step 5: Orchestrator builds context - should auto-add enhancement
        print("[Step 5] Orchestrator builds context (auto-adds TaskManagementEnhancement)")
        orchestrator_context = orchestrator.build_context()

        # Verify enhancement was added
        assert len(orchestrator.enhancements) == 1
        assert isinstance(orchestrator.enhancements[0], TaskManagementEnhancement)
        print("[Step 5] ✓ Orchestrator auto-added TaskManagementEnhancement")

        # Verify orchestrator sees task status
        orch_context_str = "\n".join([msg.get("content", "") for msg in orchestrator_context])
        assert "TASK BREAKDOWN STATUS" in orch_context_str
        assert "Total Tasks: 5" in orch_context_str
        assert "NEXT PENDING TASK:" in orch_context_str
        assert "T1" in orch_context_str  # First task (no dependencies)
        print("[Step 5] ✓ Orchestrator sees task status and next task (T1)")

        # Step 6: Verify orchestrator has task management tools
        print("[Step 6] Verifying orchestrator has task management tools")
        tools = orchestrator.get_tools()
        tool_names = [t["function"]["name"] for t in tools]

        assert "read_task_breakdown" in tool_names
        assert "get_next_task" in tool_names
        assert "mark_task_status" in tool_names
        assert "update_task" in tool_names
        print("[Step 6] ✓ Orchestrator has all task management tools")

        # Step 7: Simulate marking a task complete
        print("[Step 7] Simulating task completion (mark T1 as completed)")
        # Update task breakdown file (simulating mark_task_status call result)
        task_breakdown["tasks"][0]["status"] = "completed"
        task_file.write_text(json.dumps(task_breakdown, indent=2))

        # Build context again - should show updated status
        orchestrator_context2 = orchestrator.build_context()
        context_str2 = "\n".join([msg.get("content", "") for msg in orchestrator_context2])

        assert "Completed:   1" in context_str2
        assert "Pending:     4" in context_str2
        assert "T2" in context_str2  # Next task (T1 dependency satisfied)
        print("[Step 7] ✓ Orchestrator sees updated task status (T1 completed, T2 next)")

        # Step 8: Mark another task in progress
        print("[Step 8] Simulating task in progress (mark T2 as in_progress)")
        task_breakdown["tasks"][1]["status"] = "in_progress"
        task_file.write_text(json.dumps(task_breakdown, indent=2))

        orchestrator_context3 = orchestrator.build_context()
        context_str3 = "\n".join([msg.get("content", "") for msg in orchestrator_context3])

        assert "In Progress: 1" in context_str3
        assert "Completed:   1" in context_str3
        assert "⟳" in context_str3  # In progress icon
        print("[Step 8] ✓ Orchestrator sees in-progress task with ⟳ icon")

        print("\n" + "="*70)
        print("WORKFLOW COMPLETE - TaskManagementEnhancement auto-add works!")
        print("="*70)
        print("\nSummary:")
        print("- Architect created task breakdown → auto-added enhancement")
        print("- Orchestrator detected task breakdown → auto-added enhancement")
        print("- Both agents see task status in context")
        print("- Both agents have task management tools")
        print("- Context updates as tasks progress")
        print("="*70)


def test_enhancement_persists_across_context_builds():
    """Test that enhancement persists and doesn't get re-added on each build."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create task breakdown
        arch_dir = workspace / "architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        task_breakdown = {
            "tasks": [
                {"id": "T1", "description": "Test", "module": "core",
                 "status": "pending", "dependencies": [], "priority": 1,
                 "estimated_complexity": "low"}
            ]
        }
        (arch_dir / "task-breakdown.json").write_text(json.dumps(task_breakdown))

        # Create orchestrator
        orchestrator = OrchestratorAgent(workspace=workspace)

        # Build context 5 times
        for i in range(5):
            orchestrator.build_context()

        # Should only have one enhancement
        assert len(orchestrator.enhancements) == 1
        print("✓ Enhancement persists across multiple context builds (only added once)")


def test_no_enhancement_without_task_breakdown():
    """Test that enhancement is NOT added when no task breakdown exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create orchestrator (no task breakdown)
        orchestrator = OrchestratorAgent(workspace=workspace)

        # Build context
        orchestrator.build_context()

        # Should have no enhancements
        assert len(orchestrator.enhancements) == 0
        print("✓ No enhancement added when task breakdown doesn't exist")


if __name__ == "__main__":
    print("Testing TaskManagementEnhancement integration in realistic workflow...\n")

    test_orchestrator_architect_task_tracking_workflow()
    print()
    test_enhancement_persists_across_context_builds()
    print()
    test_no_enhancement_without_task_breakdown()

    print("\n✓ All integration tests passed!")
