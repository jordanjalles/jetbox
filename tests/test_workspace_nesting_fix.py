"""
Test for workspace nesting fix.

This test verifies that when Orchestrator delegates to Architect and then to TaskExecutor,
the workspace is reused correctly without creating nested workspaces.
"""
import pytest
from pathlib import Path
import shutil
import tempfile
from task_executor_agent import TaskExecutorAgent
from architect_agent import ArchitectAgent
from workspace_manager import WorkspaceManager


def test_workspace_reuse_no_nesting():
    """
    Test that TaskExecutor reuses workspace without creating nested directories.

    Simulates: Orchestrator → Architect → TaskExecutor flow

    Expected: Files created by Architect should be accessible by TaskExecutor
    Actual (before fix): TaskExecutor creates nested workspace and can't find files
    """
    # Create temporary test workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir) / ".agent_workspace" / "test-project"
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Simulate Architect creating architecture files
        print("\n[TEST] Step 1: Architect creates architecture docs")
        arch_workspace = WorkspaceManager(
            goal="test-architecture",
            workspace_path=workspace_path
        )

        # Create architecture directory structure
        arch_dir = workspace_path / "architecture" / "modules"
        arch_dir.mkdir(parents=True, exist_ok=True)

        # Write architecture file
        arch_file = arch_dir / "api-client.md"
        arch_file.write_text("# API Client Module\n\nImplement the ApiClient class...")

        print(f"[TEST] Architect created: {arch_file}")
        assert arch_file.exists(), "Architecture file should exist"

        # Step 2: Simulate TaskExecutor being delegated with workspace parameter
        print("\n[TEST] Step 2: TaskExecutor receives workspace parameter")
        executor = TaskExecutorAgent(
            workspace=workspace_path,  # Pass existing workspace
            goal="Implement ApiClient class per architecture/modules/api-client.md",
            max_rounds=1,
        )

        # Verify TaskExecutor's workspace manager
        print(f"[TEST] TaskExecutor workspace_dir: {executor.workspace_manager.workspace_dir}")

        # Check that workspace is NOT nested
        workspace_str = str(executor.workspace_manager.workspace_dir)
        nesting_count = workspace_str.count('.agent_workspace')

        print(f"[TEST] Workspace path: {workspace_str}")
        print(f"[TEST] Nesting count (.agent_workspace appears): {nesting_count}")

        assert nesting_count == 1, f"Workspace should not be nested (found {nesting_count} occurrences of '.agent_workspace')"

        # Step 3: Verify TaskExecutor can read architecture files
        print("\n[TEST] Step 3: TaskExecutor reads architecture file")
        arch_file_relative = Path("architecture/modules/api-client.md")
        resolved_path = executor.workspace_manager.resolve_path(arch_file_relative)

        print(f"[TEST] Resolved path: {resolved_path}")
        print(f"[TEST] File exists: {resolved_path.exists()}")

        assert resolved_path.exists(), f"TaskExecutor should be able to read architecture file at {resolved_path}"

        # Step 4: Verify no nested workspace directories
        print("\n[TEST] Step 4: Check for nested workspaces")
        nested_workspaces = list(workspace_path.glob("**/.agent_workspace"))
        print(f"[TEST] Nested .agent_workspace directories found: {len(nested_workspaces)}")

        assert len(nested_workspaces) == 0, "Should not have nested .agent_workspace directories"

        print("\n[TEST] ✅ All checks passed - workspace reuse works correctly!")


def test_workspace_create_new():
    """
    Test that TaskExecutor creates NEW workspace when no workspace parameter provided.

    This ensures the fix doesn't break the default isolated workspace behavior.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to temp directory
        import os
        original_cwd = os.getcwd()

        try:
            os.chdir(tmpdir)

            # Step 1: Create TaskExecutor without workspace parameter
            print("\n[TEST] Creating TaskExecutor without workspace parameter")
            executor = TaskExecutorAgent(
                workspace=None,  # No workspace = create new
                goal="Create a simple calculator",
                max_rounds=1,
            )

            # Step 2: Verify workspace was created under .agent_workspace
            print(f"[TEST] TaskExecutor workspace_dir: {executor.workspace_manager.workspace_dir}")

            workspace_str = str(executor.workspace_manager.workspace_dir)

            # Should contain .agent_workspace
            assert '.agent_workspace' in workspace_str, "Workspace should be under .agent_workspace"

            # Should exist
            assert executor.workspace_manager.workspace_dir.exists(), "Workspace directory should exist"

            print("\n[TEST] ✅ New workspace creation works correctly!")

        finally:
            os.chdir(original_cwd)


def test_workspace_parameter_semantics():
    """
    Test workspace parameter semantics are clear and correct.

    - workspace=None → create new isolated workspace
    - workspace=Path → reuse existing workspace
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Test 1: workspace=None creates new
        print("\n[TEST] Test 1: workspace=None creates new workspace")
        existing_workspace_path = tmppath / ".agent_workspace" / "test-workspace"
        existing_workspace_path.mkdir(parents=True, exist_ok=True)

        import os
        original_cwd = os.getcwd()

        try:
            os.chdir(tmppath)

            executor1 = TaskExecutorAgent(
                workspace=None,
                goal="test goal 1",
                max_rounds=1,
            )

            # Should create a NEW workspace (not reuse existing)
            assert executor1.workspace_manager.workspace_dir != existing_workspace_path
            print(f"[TEST] Created new workspace: {executor1.workspace_manager.workspace_dir}")

            # Test 2: workspace=Path reuses existing
            print("\n[TEST] Test 2: workspace=Path reuses existing workspace")
            executor2 = TaskExecutorAgent(
                workspace=existing_workspace_path,
                goal="test goal 2",
                max_rounds=1,
            )

            # Should REUSE the existing workspace
            assert executor2.workspace_manager.workspace_dir == existing_workspace_path.resolve()
            print(f"[TEST] Reused workspace: {executor2.workspace_manager.workspace_dir}")

            print("\n[TEST] ✅ Workspace parameter semantics are correct!")

        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    print("="*70)
    print("WORKSPACE NESTING FIX - TEST SUITE")
    print("="*70)

    # Run tests
    test_workspace_reuse_no_nesting()
    test_workspace_create_new()
    test_workspace_parameter_semantics()

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✅")
    print("="*70)
