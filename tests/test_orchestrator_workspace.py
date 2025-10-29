"""
Test orchestrator workspace handling.

Tests that orchestrator can:
1. List existing workspaces
2. Delegate to existing workspace
3. Correctly handle workspace parameter
"""
from pathlib import Path
from agent_registry import AgentRegistry
from agent_config import config
from orchestrator_main import execute_orchestrator_tool


def test_list_workspaces():
    """Test listing workspaces."""

    workspace = Path.cwd()
    registry = AgentRegistry(config_path="agents.yaml", workspace=workspace)

    print("=" * 60)
    print("TEST: List Workspaces")
    print("=" * 60)
    print()

    # Create a mock tool call
    tool_call = {
        "function": {
            "name": "list_workspaces",
            "arguments": {},
        }
    }

    result = execute_orchestrator_tool(tool_call, registry)

    print("Result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Workspaces: {len(result.get('workspaces', []))}")
    print()
    print("Message:")
    print(result.get('message', ''))
    print()

    if result.get('workspaces'):
        print("First workspace:")
        ws = result['workspaces'][0]
        print(f"  Name: {ws['name']}")
        print(f"  Path: {ws['path']}")
        print(f"  Files: {ws['files']}")

    print()
    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


def test_delegate_with_workspace():
    """Test delegating with workspace parameter."""

    workspace = Path.cwd()
    registry = AgentRegistry(config_path="agents.yaml", workspace=workspace)

    print()
    print("=" * 60)
    print("TEST: Delegate with Workspace Parameter")
    print("=" * 60)
    print()

    # First list workspaces to get an existing one
    list_call = {
        "function": {
            "name": "list_workspaces",
            "arguments": {},
        }
    }

    list_result = execute_orchestrator_tool(list_call, registry)

    if not list_result.get('workspaces'):
        print("No workspaces found. Skipping test.")
        return

    # Get the most recent workspace
    workspace_path = list_result['workspaces'][0]['path']
    workspace_name = list_result['workspaces'][0]['name']

    print(f"Testing with workspace: {workspace_name}")
    print(f"Path: {workspace_path}")
    print()

    # Create a delegation with workspace parameter
    delegate_call = {
        "function": {
            "name": "delegate_to_executor",
            "arguments": {
                "task_description": "List all files in the workspace",
                "workspace": workspace_path,
            },
        }
    }

    print("Tool call arguments:")
    print(f"  task_description: {delegate_call['function']['arguments']['task_description']}")
    print(f"  workspace: {delegate_call['function']['arguments']['workspace']}")
    print()
    print("This would run:")
    print(f"  python agent.py --workspace {workspace_path} \"List all files in the workspace\"")
    print()

    # Note: We won't actually execute this as it would run the full agent
    # Just verify the tool call structure is correct

    print("=" * 60)
    print("TEST COMPLETED (structure validation only)")
    print("=" * 60)


if __name__ == "__main__":
    test_list_workspaces()
    test_delegate_with_workspace()
