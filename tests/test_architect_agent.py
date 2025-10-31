"""
Test the Architect agent basic functionality.
"""
from pathlib import Path
import tempfile
import shutil
import json

from architect_agent import ArchitectAgent
from agent_registry import AgentRegistry
import architect_tools


def test_architect_agent_creation():
    """Test that ArchitectAgent can be created."""
    print("\n" + "="*70)
    print("TEST: Architect Agent Creation")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        agent = ArchitectAgent(
            workspace=workspace,
            project_description="Test project"
        )

        assert agent.name == "architect"
        assert agent.role == "Software architecture consultant"
        assert agent.workspace == workspace
        print("✅ ArchitectAgent created successfully")

    print("\n✅ Architect agent creation test PASSED")


def test_architect_tools():
    """Test that architect tools can create artifacts."""
    print("\n" + "="*70)
    print("TEST: Architect Tools")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Configure tools with workspace
        class SimpleWorkspace:
            def __init__(self, workspace_dir):
                self.workspace_dir = workspace_dir

        architect_tools.set_workspace(SimpleWorkspace(workspace))

        # Test write_architecture_doc
        result = architect_tools.write_architecture_doc(
            title="System Overview",
            content="This is a test architecture document."
        )

        assert result["status"] == "success"
        assert "architecture/" in result["file_path"]
        print(f"✅ Created architecture doc: {result['file_path']}")

        # Verify file exists
        doc_path = workspace / result["file_path"]
        assert doc_path.exists()
        content = doc_path.read_text()
        assert "System Overview" in content
        assert "test architecture document" in content

        # Test write_module_spec
        result = architect_tools.write_module_spec(
            module_name="auth-service",
            responsibility="Handle user authentication",
            interfaces={
                "inputs": ["username: string", "password: string"],
                "outputs": ["token: JWT"],
                "apis": ["POST /auth/login - Authenticate user"]
            },
            dependencies=["database", "redis"],
            technologies={"language": "Python", "framework": "FastAPI"},
            implementation_notes="Use bcrypt for password hashing"
        )

        assert result["status"] == "success"
        assert "architecture/modules/" in result["file_path"]
        print(f"✅ Created module spec: {result['file_path']}")

        # Verify module spec file
        module_path = workspace / result["file_path"]
        assert module_path.exists()
        content = module_path.read_text()
        assert "auth-service" in content
        assert "bcrypt" in content

        # Test write_task_list
        result = architect_tools.write_task_list(
            tasks=[
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
                    "description": "Write tests for auth",
                    "module": "auth-service",
                    "priority": 2,
                    "dependencies": ["T1"],
                    "estimated_complexity": "low"
                }
            ]
        )

        assert result["status"] == "success"
        assert result["task_count"] == 2
        print(f"✅ Created task list: {result['file_path']}")

        # Verify task list file
        task_path = workspace / result["file_path"]
        assert task_path.exists()
        with open(task_path) as f:
            task_data = json.load(f)
        assert task_data["total_tasks"] == 2
        assert len(task_data["tasks"]) == 2
        assert task_data["tasks"][0]["id"] == "T1"

        # Test list_architecture_docs
        result = architect_tools.list_architecture_docs()
        assert result["status"] == "success"
        assert len(result["docs"]) == 1  # system-overview.md
        assert len(result["modules"]) == 1  # auth-service.md
        assert result["task_breakdown"] is not None  # task-breakdown.json
        print(f"✅ Listed artifacts: {len(result['docs'])} docs, {len(result['modules'])} modules")

    print("\n✅ Architect tools test PASSED")


def test_architect_in_registry():
    """Test that ArchitectAgent can be retrieved from registry."""
    print("\n" + "="*70)
    print("TEST: Architect in Agent Registry")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        registry = AgentRegistry(workspace=workspace)

        # Get architect agent
        architect = registry.get_agent("architect")

        assert architect is not None
        assert architect.name == "architect"
        assert isinstance(architect, ArchitectAgent)
        print("✅ ArchitectAgent retrieved from registry")

        # Check delegation permissions
        assert registry.can_delegate("orchestrator", "architect")
        assert not registry.can_delegate("architect", "task_executor")
        print("✅ Delegation permissions correct")

    print("\n✅ Registry test PASSED")


def test_architect_strategy():
    """Test that ArchitectStrategy works correctly."""
    print("\n" + "="*70)
    print("TEST: Architect Strategy")
    print("="*70)

    from context_strategies import ArchitectStrategy
    from context_manager import ContextManager

    strategy = ArchitectStrategy()

    # Test basic properties
    assert strategy.get_name() == "architect"
    assert strategy.should_use_jetbox_notes() is False
    assert strategy.should_clear_on_transition() is False
    print("✅ ArchitectStrategy properties correct")

    # Test context building
    cm = ContextManager()
    cm.load_or_init("Test project")

    class DummyConfig:
        pass

    context = strategy.build_context(
        context_manager=cm,
        messages=[],
        system_prompt="Test prompt",
        config=DummyConfig(),
    )

    assert len(context) >= 1  # At least system prompt
    assert context[0]["role"] == "system"
    assert context[0]["content"] == "Test prompt"
    print("✅ ArchitectStrategy builds context correctly")

    print("\n✅ Strategy test PASSED")


if __name__ == "__main__":
    test_architect_agent_creation()
    test_architect_tools()
    test_architect_in_registry()
    test_architect_strategy()

    print("\n" + "="*70)
    print("ALL ARCHITECT TESTS PASSED ✅")
    print("="*70)
    print("\nSummary:")
    print("- ArchitectAgent can be created and configured")
    print("- Architect tools create artifacts in correct workspace locations")
    print("- ArchitectAgent integrates with agent registry")
    print("- ArchitectStrategy manages context correctly")
    print("- Jetbox notes disabled by default for architect")
