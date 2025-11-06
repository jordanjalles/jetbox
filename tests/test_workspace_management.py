"""Test WorkspaceManagementBehavior functionality."""
import tempfile
from pathlib import Path
from behaviors.workspace_management import WorkspaceManagementBehavior


def test_workspace_management_behavior():
    """Test workspace management tools."""
    # Create temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        workspaces_root = Path(tmpdir) / ".agent_workspaces"
        workspaces_root.mkdir()

        # Create some test workspaces
        ws1 = workspaces_root / "test-workspace-1"
        ws1.mkdir()
        (ws1 / "test.py").write_text("print('hello')")

        ws2 = workspaces_root / "test-workspace-2"
        ws2.mkdir()
        (ws2 / "main.py").write_text("def main(): pass")
        (ws2 / "utils.py").write_text("def util(): pass")

        # Create behavior
        behavior = WorkspaceManagementBehavior(workspaces_root=str(workspaces_root))

        # Test 1: list_workspaces
        print("Test 1: list_workspaces")
        result = behavior._list_workspaces(limit=10, sort_by="name")
        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["workspaces"]) == 2
        assert result["workspaces"][0]["name"] == "test-workspace-1"
        assert result["workspaces"][1]["name"] == "test-workspace-2"
        print(f"✓ Found {result['count']} workspaces")

        # Test 2: search_workspaces
        print("\nTest 2: search_workspaces")
        result = behavior._search_workspaces("workspace-2")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["workspaces"][0]["name"] == "test-workspace-2"
        print("✓ Search for 'workspace-2' found 1 match")

        # Test 3: get_workspace_info
        print("\nTest 3: get_workspace_info")
        result = behavior._get_workspace_info("test-workspace-2")
        assert result["success"] is True
        assert result["workspace"]["name"] == "test-workspace-2"
        assert result["workspace"]["file_count"] == 2
        assert len(result["workspace"]["files"]) == 2
        print(f"✓ Workspace info shows {result['workspace']['file_count']} files")

        # Test 4: get_workspace_info with notes
        print("\nTest 4: get_workspace_info with notes")
        notes_file = ws2 / "workspace_task_notes.md"
        notes_file.write_text("# Task Notes\n\n- Task 1 completed\n- Task 2 in progress")
        result = behavior._get_workspace_info("test-workspace-2")
        assert result["success"] is True
        assert result["workspace"]["notes"] is not None
        assert "Task 1 completed" in result["workspace"]["notes"]
        print("✓ Notes loaded successfully")

        # Test 5: Nonexistent workspace
        print("\nTest 5: Nonexistent workspace")
        result = behavior._get_workspace_info("nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]
        print("✓ Nonexistent workspace returns error")

        # Test 6: Empty workspaces directory
        print("\nTest 6: Empty workspaces directory")
        empty_behavior = WorkspaceManagementBehavior(
            workspaces_root=str(Path(tmpdir) / "nonexistent")
        )
        result = empty_behavior._list_workspaces()
        assert result["success"] is True
        assert len(result["workspaces"]) == 0
        print("✓ Empty directory returns empty list")

    print("\n✅ All workspace management tests passed!")


if __name__ == "__main__":
    test_workspace_management_behavior()
