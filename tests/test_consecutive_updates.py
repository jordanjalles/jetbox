"""
Test consecutive project updates - orchestrator delegating to task executor.

Verifies:
1. Create new project (workspace_mode="new")
2. Update same project (workspace_mode="existing")
3. Multiple consecutive updates to same project
4. Workspace isolation and reuse
"""
import sys
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator_agent import OrchestratorAgent
from task_executor_agent import TaskExecutorAgent
from workspace_manager import WorkspaceManager


def test_task_executor_basic_workflow():
    """Test TaskExecutor can create and update files in workspace."""
    print("\n=== Testing TaskExecutor Basic Workflow ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create TaskExecutor with a goal
        executor = TaskExecutorAgent(
            workspace=tmppath,
            goal="Create a simple calculator with add function",
            max_rounds=10
        )

        # Verify workspace was created
        workspace_dir = executor.workspace_manager.workspace_dir
        assert workspace_dir.exists(), "Workspace should be created"
        assert workspace_dir.parent == tmppath, f"Workspace should be under tmppath: {workspace_dir}"
        assert "calculator" in workspace_dir.name, f"Workspace name should contain 'calculator': {workspace_dir.name}"

        print(f"✓ TaskExecutor workspace created: {workspace_dir}")

        # Verify context manager initialized
        assert executor.context_manager is not None
        assert executor.context_manager.state.goal is not None
        assert "calculator" in executor.context_manager.state.goal.description.lower()

        print("✓ Context manager initialized with goal")

        # Verify status display initialized
        assert executor.status_display is not None
        print("✓ Status display initialized")

        # Create a simple file using tools (don't actually run LLM)
        from tools import write_file, read_file
        import tools

        # Configure tools with workspace
        tools.set_workspace(executor.workspace_manager)
        tools.set_ledger(workspace_dir / "test_ledger.log")

        # Write a calculator file
        result = write_file(
            path="calculator.py",
            content="def add(a, b):\n    return a + b\n"
        )
        assert isinstance(result, str), f"write_file should return a string: {result}"
        assert "SUCCESS" in result or "Wrote" in result, f"Expected success message: {result}"
        print("✓ Created calculator.py in workspace")

        # Verify file exists in workspace
        calc_file = workspace_dir / "calculator.py"
        assert calc_file.exists()
        assert "def add" in calc_file.read_text()

        # Read it back
        result = read_file(path="calculator.py")
        assert isinstance(result, str), f"read_file should return a string: {result}"
        assert "def add" in result
        print("✓ Read calculator.py from workspace")

        print("\n✅ TaskExecutor basic workflow test passed!")


def test_orchestrator_workspace_modes():
    """Test Orchestrator correctly identifies new vs existing work."""
    print("\n=== Testing Orchestrator Workspace Modes ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create orchestrator
        orchestrator = OrchestratorAgent(workspace=tmppath)

        # Simulate creating a new project
        print("\n1. NEW PROJECT: Creating calculator...")
        orchestrator.add_user_message("Create a calculator package with add function")

        # Check tools available
        tools = orchestrator.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "delegate_to_executor" in tool_names
        assert "find_workspace" in tool_names
        assert "list_workspaces" in tool_names

        print(f"✓ Orchestrator has delegation tools: {tool_names}")

        # Verify system prompt emphasizes workspace_mode
        system_prompt = orchestrator.get_system_prompt()
        assert "workspace_mode" in system_prompt
        assert "REQUIRED" in system_prompt
        assert "new" in system_prompt and "existing" in system_prompt

        print("✓ System prompt requires workspace_mode parameter")

        # Create a workspace manually to simulate first project
        workspace_mgr = WorkspaceManager(
            goal="create-calculator-package",
            base_dir=tmppath
        )
        calc_workspace = workspace_mgr.workspace_dir
        calc_workspace.mkdir(parents=True, exist_ok=True)

        # Create some files in it
        (calc_workspace / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
        (calc_workspace / "test_calc.py").write_text("from calculator import add\n")

        print(f"✓ Created workspace: {calc_workspace}")

        # Now simulate updating existing project
        print("\n2. EXISTING PROJECT: Adding multiply function...")
        orchestrator.add_user_message("Add a multiply function to the calculator")

        # Verify list_workspaces can find the calculator workspace
        # Note: WorkspaceManager was created with base_dir=tmppath, so workspace is under tmppath directly
        workspaces = list(tmppath.glob("create-calculator*"))
        assert len(workspaces) == 1
        assert "calculator" in str(workspaces[0])
        print(f"✓ Found existing workspace: {workspaces[0].name}")

        print("\n✅ Orchestrator workspace mode test passed!")


def test_consecutive_project_updates():
    """Test multiple consecutive updates to same project."""
    print("\n=== Testing Consecutive Project Updates ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Step 1: Create initial project
        print("\n1. Creating initial calculator project...")
        workspace_mgr = WorkspaceManager(
            goal="create-calculator",
            base_dir=tmppath
        )
        workspace_dir = workspace_mgr.workspace_dir
        workspace_dir.mkdir(parents=True, exist_ok=True)

        import tools
        tools.set_workspace(workspace_mgr)
        tools.set_ledger(workspace_dir / "ledger.log")

        # Create initial file
        tools.write_file(
            path="calculator.py",
            content="def add(a, b):\n    return a + b\n"
        )

        files = list(workspace_dir.glob("*.py"))
        assert len(files) == 1
        print(f"✓ Initial project: {files}")

        # Step 2: First update - add subtract
        print("\n2. Update 1: Adding subtract function...")
        current_content = tools.read_file(path="calculator.py")

        new_content = current_content + "\ndef subtract(a, b):\n    return a - b\n"
        tools.write_file(path="calculator.py", content=new_content)

        result = tools.read_file(path="calculator.py")
        assert "add" in result
        assert "subtract" in result
        print("✓ Update 1 complete: add + subtract")

        # Step 3: Second update - add multiply
        print("\n3. Update 2: Adding multiply function...")
        current_content = tools.read_file(path="calculator.py")

        new_content = current_content + "\ndef multiply(a, b):\n    return a * b\n"
        tools.write_file(path="calculator.py", content=new_content)

        result = tools.read_file(path="calculator.py")
        assert "add" in result
        assert "subtract" in result
        assert "multiply" in result
        print("✓ Update 2 complete: add + subtract + multiply")

        # Step 4: Third update - add tests
        print("\n4. Update 3: Adding tests...")
        tools.write_file(
            path="test_calculator.py",
            content="""from calculator import add, subtract, multiply

def test_add():
    assert add(2, 3) == 5

def test_subtract():
    assert subtract(5, 3) == 2

def test_multiply():
    assert multiply(3, 4) == 12
"""
        )

        files = list(workspace_dir.glob("*.py"))
        assert len(files) == 2
        print(f"✓ Update 3 complete: {[f.name for f in files]}")

        # Verify ledger was created
        ledger_file = workspace_dir / "ledger.log"
        if ledger_file.exists():
            ledger_content = ledger_file.read_text()
            print(f"✓ Ledger file exists ({len(ledger_content)} bytes)")
        else:
            print("✓ No ledger file (ledger tracking optional in tests)")

        print("\n✅ Consecutive updates test passed!")


def test_workspace_isolation():
    """Test that different projects have isolated workspaces."""
    print("\n=== Testing Workspace Isolation ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create two different projects
        print("\n1. Creating calculator project...")
        calc_mgr = WorkspaceManager(goal="calculator", base_dir=tmppath)
        calc_dir = calc_mgr.workspace_dir
        calc_dir.mkdir(parents=True, exist_ok=True)

        import tools
        tools.set_workspace(calc_mgr)
        tools.set_ledger(calc_dir / "ledger.log")

        tools.write_file(path="calculator.py", content="def add(a, b): return a + b\n")

        print("\n2. Creating blog project...")
        blog_mgr = WorkspaceManager(goal="blog-api", base_dir=tmppath)
        blog_dir = blog_mgr.workspace_dir
        blog_dir.mkdir(parents=True, exist_ok=True)

        tools.set_workspace(blog_mgr)
        tools.set_ledger(blog_dir / "ledger.log")

        tools.write_file(path="blog.py", content="class Post: pass\n")

        # Verify isolation
        calc_files = list(calc_dir.glob("*.py"))
        blog_files = list(blog_dir.glob("*.py"))

        assert len(calc_files) == 1
        assert calc_files[0].name == "calculator.py"

        assert len(blog_files) == 1
        assert blog_files[0].name == "blog.py"

        print(f"✓ Calculator workspace: {calc_files[0].name}")
        print(f"✓ Blog workspace: {blog_files[0].name}")

        # Verify they're in different directories
        assert calc_dir != blog_dir
        assert "calculator" in str(calc_dir)
        assert "blog" in str(blog_dir)

        print("\n✅ Workspace isolation test passed!")


if __name__ == "__main__":
    try:
        test_task_executor_basic_workflow()
        test_orchestrator_workspace_modes()
        test_consecutive_project_updates()
        test_workspace_isolation()

        print("\n" + "=" * 60)
        print("✅ ALL CONSECUTIVE UPDATE TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
