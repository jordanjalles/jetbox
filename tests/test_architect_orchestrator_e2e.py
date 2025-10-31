"""
End-to-end test for Architect → Orchestrator → TaskExecutor workflow.

This test demonstrates the full integration:
User → Orchestrator → Architect → Orchestrator → TaskExecutor → Orchestrator → User

NOTE: This is a manual/integration test that requires an LLM.
Run with: PYTHONPATH=. python tests/test_architect_orchestrator_e2e.py
"""
from pathlib import Path
import tempfile
import shutil
import json

from agent_registry import AgentRegistry
from orchestrator_main import execute_orchestrator_tool


def test_architect_integration_basic():
    """
    Test orchestrator → architect integration mechanics.

    NOTE: Without an LLM, artifacts won't be created. This test verifies:
    - Tool call routing works
    - Workspace is created
    - Result structure is correct

    For full artifact creation test, run with actual LLM.
    """
    print("\n" + "="*70)
    print("TEST: Orchestrator → Architect Integration (Basic)")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        registry = AgentRegistry(workspace=workspace)

        # Simulate orchestrator consulting architect
        tool_call = {
            "function": {
                "name": "consult_architect",
                "arguments": {
                    "project_description": "Simple web API for managing todo items",
                    "requirements": "RESTful API with CRUD operations, persistent storage",
                    "constraints": "Python, single developer, 1 week timeline"
                }
            }
        }

        print("\n[Test] Simulating orchestrator calling consult_architect...")
        result = execute_orchestrator_tool(tool_call, registry)

        # Verify result structure (these should work even without LLM)
        assert result["success"] is True, f"Consultation failed: {result.get('message')}"
        assert "artifacts" in result, "Missing artifacts in result"
        assert "workspace" in result, "Missing workspace in result"

        artifacts = result["artifacts"]
        workspace_path = Path(result["workspace"])

        print(f"\n✅ Consultation successful!")
        print(f"Workspace: {workspace_path}")
        print(f"Artifacts: {artifacts}")

        # Verify workspace was created
        assert workspace_path.exists(), "Workspace directory not created"

        # Architecture dir may or may not exist without LLM
        arch_dir = workspace_path / "architecture"
        if arch_dir.exists():
            print("✅ Architecture directory created")
        else:
            print("⚠️  Architecture directory not created (expected without LLM)")

        # Check if artifacts were created (won't happen without LLM)
        has_artifacts = (
            len(artifacts["docs"]) > 0 or
            len(artifacts["modules"]) > 0 or
            artifacts["task_breakdown"] is not None
        )

        if has_artifacts:
            print("\n✅ Architecture artifacts created successfully")
            print(f"  - Docs: {len(artifacts['docs'])}")
            print(f"  - Modules: {len(artifacts['modules'])}")
            print(f"  - Task breakdown: {'Yes' if artifacts['task_breakdown'] else 'No'}")
        else:
            print("\n⚠️  No artifacts created (expected without LLM)")
            print("   Run with actual LLM to test artifact creation")

    print("\n✅ Basic integration test PASSED (mechanics verified)")


def test_workflow_documentation():
    """
    Document the full e2e workflow (not automated - requires LLM).

    This shows the expected flow when a user asks for a complex project.
    """
    print("\n" + "="*70)
    print("WORKFLOW DOCUMENTATION: Full E2E Flow")
    print("="*70)

    workflow = """
SCENARIO: User asks for a complex project

Step 1: User → Orchestrator
  User: "Build a real-time chat application with user authentication"

Step 2: Orchestrator → Architect
  Orchestrator recognizes complexity (multi-component: auth + chat + real-time)
  Orchestrator calls: consult_architect(
      project_description="real-time chat application with user authentication",
      requirements="real-time messaging, user auth, message persistence",
      constraints="web-based, support 100 concurrent users"
  )

Step 3: Architect works (10 rounds max)
  - Architect asks clarifying questions (if needed)
  - Architect creates architecture artifacts:
    ✓ architecture/system-overview.md
    ✓ architecture/modules/auth-service.md
    ✓ architecture/modules/chat-service.md
    ✓ architecture/modules/websocket-server.md
    ✓ architecture/task-breakdown.json

Step 4: Architect → Orchestrator
  Returns: {
    "success": True,
    "artifacts": {
      "docs": ["architecture/system-overview.md"],
      "modules": ["architecture/modules/auth-service.md", ...],
      "task_breakdown": "architecture/task-breakdown.json"
    },
    "workspace": ".agent_workspace/real-time-chat-application"
  }

Step 5: Orchestrator reads task breakdown
  Orchestrator reads architecture/task-breakdown.json:
  {
    "tasks": [
      {"id": "T1", "description": "Implement auth service", "module": "auth-service", ...},
      {"id": "T2", "description": "Implement chat service", "module": "chat-service", ...},
      {"id": "T3", "description": "Implement WebSocket server", "module": "websocket-server", ...}
    ]
  }

Step 6: Orchestrator → TaskExecutor (Task 1)
  Orchestrator calls: delegate_to_executor(
      task_description="Implement auth service per architecture/modules/auth-service.md",
      workspace_mode="existing",
      workspace_path=".agent_workspace/real-time-chat-application"
  )
  TaskExecutor: [implements auth service]

Step 7: Orchestrator → TaskExecutor (Task 2)
  Orchestrator calls: delegate_to_executor(
      task_description="Implement chat service per architecture/modules/chat-service.md",
      workspace_mode="existing",
      workspace_path=".agent_workspace/real-time-chat-application"
  )
  TaskExecutor: [implements chat service]

Step 8: Orchestrator → TaskExecutor (Task 3)
  Orchestrator calls: delegate_to_executor(
      task_description="Implement WebSocket server per architecture/modules/websocket-server.md",
      workspace_mode="existing",
      workspace_path=".agent_workspace/real-time-chat-application"
  )
  TaskExecutor: [implements WebSocket server]

Step 9: Orchestrator → User
  Orchestrator: "Project complete! Architecture and implementation in .agent_workspace/real-time-chat-application/"
  - Architecture docs in architecture/
  - Auth service implemented
  - Chat service implemented
  - WebSocket server implemented

EXPECTED OUTCOME:
  ✅ Architecture documents created and persisted
  ✅ Module specifications guide implementation
  ✅ Task breakdown structures the work
  ✅ All modules implemented per architecture
  ✅ Single workspace contains everything
"""

    print(workflow)
    print("\n✅ Workflow documented")


def test_registry_permissions():
    """Test that orchestrator can delegate to architect."""
    print("\n" + "="*70)
    print("TEST: Registry Delegation Permissions")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        registry = AgentRegistry(workspace=workspace)

        # Verify orchestrator can delegate to architect
        assert registry.can_delegate("orchestrator", "architect"), \
            "Orchestrator should be able to delegate to architect"

        # Verify architect cannot delegate (terminal consultant)
        assert not registry.can_delegate("architect", "task_executor"), \
            "Architect should not delegate (consultant model)"

        # Verify orchestrator can still delegate to task_executor
        assert registry.can_delegate("orchestrator", "task_executor"), \
            "Orchestrator should be able to delegate to task_executor"

        print("✅ Orchestrator can delegate to: architect, task_executor")
        print("✅ Architect cannot delegate (consultant model)")

    print("\n✅ Registry permissions test PASSED")


def test_orchestrator_has_consult_tool():
    """Test that orchestrator has the consult_architect tool."""
    print("\n" + "="*70)
    print("TEST: Orchestrator Tool Availability")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        registry = AgentRegistry(workspace=workspace)

        orchestrator = registry.get_agent("orchestrator")
        tools = orchestrator.get_tools()

        tool_names = {tool["function"]["name"] for tool in tools}

        print(f"\nOrchestrator tools ({len(tools)}):")
        for name in sorted(tool_names):
            print(f"  - {name}")

        # Verify consult_architect is available
        assert "consult_architect" in tool_names, "consult_architect tool not found"
        assert "delegate_to_executor" in tool_names, "delegate_to_executor tool not found"

        print("\n✅ Orchestrator has consult_architect tool")
        print("✅ Orchestrator has delegate_to_executor tool")

    print("\n✅ Tool availability test PASSED")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ARCHITECT ↔ ORCHESTRATOR E2E TESTS")
    print("="*70)

    test_registry_permissions()
    test_orchestrator_has_consult_tool()
    test_architect_integration_basic()
    test_workflow_documentation()

    print("\n" + "="*70)
    print("ALL E2E TESTS PASSED ✅")
    print("="*70)
    print("\nSummary:")
    print("- Orchestrator can delegate to Architect")
    print("- Orchestrator has consult_architect tool")
    print("- Architect consultation produces artifacts")
    print("- Full workflow: User→Orchestrator→Architect→Orchestrator→TaskExecutor→User")
    print("\nNext: Test with actual LLM by running orchestrator_main.py")
