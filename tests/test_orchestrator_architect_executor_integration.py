"""
Integration test for Orchestrator → Architect → TaskExecutor workflow.

This test verifies that the workspace nesting fix works in the full delegation chain.
"""
import pytest
from pathlib import Path
import tempfile
import shutil
from workspace_manager import WorkspaceManager
from architect_agent import ArchitectAgent
from task_executor_agent import TaskExecutorAgent


def test_orchestrator_architect_executor_integration():
    """
    Integration test simulating the full Orchestrator → Architect → TaskExecutor flow.

    Steps:
    1. Orchestrator creates a workspace for the project
    2. Architect creates architecture docs in that workspace
    3. Orchestrator delegates task to TaskExecutor with workspace parameter
    4. TaskExecutor should REUSE workspace (no nesting)
    5. TaskExecutor should be able to read Architect's files
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_base = Path(tmpdir) / ".agent_workspace"
        workspace_base.mkdir(parents=True, exist_ok=True)

        # Step 1: Orchestrator creates workspace
        print("\n[INTEGRATION] Step 1: Orchestrator creates workspace")
        project_workspace = workspace_base / "rest-api-client-library"
        project_workspace.mkdir(parents=True, exist_ok=True)

        # Step 2: Architect consultation
        print("\n[INTEGRATION] Step 2: Architect creates architecture")
        architect = ArchitectAgent(
            workspace=project_workspace,
            project_description="REST API client library with authentication"
        )

        # Simulate architect creating files
        arch_dir = project_workspace / "architecture" / "modules"
        arch_dir.mkdir(parents=True, exist_ok=True)

        api_client_spec = arch_dir / "api-client.md"
        api_client_spec.write_text("""# API Client Module

## Responsibility
Core HTTP client for making REST API requests

## Interfaces
- `get(url, params)` - GET request
- `post(url, data)` - POST request
- `put(url, data)` - PUT request
- `delete(url)` - DELETE request

## Dependencies
- requests library
- auth_handler module

## Technologies
- Python 3.11+
- requests 2.31+
""")

        auth_handler_spec = arch_dir / "auth-handler.md"
        auth_handler_spec.write_text("""# Authentication Handler Module

## Responsibility
Handle API authentication (API key, OAuth)

## Interfaces
- `add_auth_header(request)` - Add auth to request

## Dependencies
None

## Technologies
- Python 3.11+
""")

        print(f"[INTEGRATION] Architect created:")
        print(f"  - {api_client_spec.relative_to(workspace_base)}")
        print(f"  - {auth_handler_spec.relative_to(workspace_base)}")

        # Verify files exist
        assert api_client_spec.exists()
        assert auth_handler_spec.exists()

        # Step 3: Orchestrator delegates to TaskExecutor
        print("\n[INTEGRATION] Step 3: Orchestrator delegates task to TaskExecutor")
        print(f"[INTEGRATION] Delegating with workspace: {project_workspace}")

        # Create TaskExecutor with existing workspace (simulate subprocess delegation)
        executor = TaskExecutorAgent(
            workspace=project_workspace,  # Reuse workspace
            goal="Implement ApiClient class per architecture/modules/api-client.md",
            max_rounds=1,
        )

        # Step 4: Verify no nesting
        print("\n[INTEGRATION] Step 4: Verify no workspace nesting")
        executor_workspace = executor.workspace_manager.workspace_dir
        print(f"[INTEGRATION] TaskExecutor workspace: {executor_workspace}")

        # Check nesting count
        workspace_str = str(executor_workspace)
        nesting_count = workspace_str.count('.agent_workspace')

        print(f"[INTEGRATION] Nesting count: {nesting_count}")
        assert nesting_count == 1, f"Should have exactly 1 occurrence of .agent_workspace, found {nesting_count}"

        # Workspace should match project workspace
        assert executor_workspace == project_workspace.resolve()

        # Step 5: Verify TaskExecutor can read architecture files
        print("\n[INTEGRATION] Step 5: Verify TaskExecutor can access Architect's files")

        # Read api-client spec
        api_client_relative = Path("architecture/modules/api-client.md")
        resolved_api_client = executor.workspace_manager.resolve_path(api_client_relative)

        print(f"[INTEGRATION] Reading: {api_client_relative}")
        print(f"[INTEGRATION] Resolved to: {resolved_api_client}")
        print(f"[INTEGRATION] File exists: {resolved_api_client.exists()}")

        assert resolved_api_client.exists(), "TaskExecutor should be able to read api-client.md"

        # Read auth-handler spec
        auth_handler_relative = Path("architecture/modules/auth-handler.md")
        resolved_auth_handler = executor.workspace_manager.resolve_path(auth_handler_relative)

        print(f"[INTEGRATION] Reading: {auth_handler_relative}")
        print(f"[INTEGRATION] Resolved to: {resolved_auth_handler}")
        print(f"[INTEGRATION] File exists: {resolved_auth_handler.exists()}")

        assert resolved_auth_handler.exists(), "TaskExecutor should be able to read auth-handler.md"

        # Step 6: Verify no nested workspaces
        print("\n[INTEGRATION] Step 6: Check for nested workspace directories")
        nested_workspaces = list(project_workspace.glob("**/.agent_workspace"))
        print(f"[INTEGRATION] Nested .agent_workspace directories: {len(nested_workspaces)}")

        assert len(nested_workspaces) == 0, "Should not have any nested .agent_workspace directories"

        print("\n[INTEGRATION] ✅ Full integration test passed!")
        print("[INTEGRATION] Orchestrator → Architect → TaskExecutor flow works correctly!")


if __name__ == "__main__":
    print("="*70)
    print("ORCHESTRATOR → ARCHITECT → EXECUTOR INTEGRATION TEST")
    print("="*70)

    test_orchestrator_architect_executor_integration()

    print("\n" + "="*70)
    print("INTEGRATION TEST PASSED ✅")
    print("="*70)
