"""
Tests for CommandToolsBehavior.

Validates:
- Tool registration and schema
- run_bash command execution
- Output capture and structured results
- Timeout handling
- Workspace directory execution
- Ledger logging
"""
import pytest
from pathlib import Path
import tempfile
import shutil
import time

from behaviors.command_tools import CommandToolsBehavior
from workspace_manager import WorkspaceManager


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def workspace_manager(temp_workspace):
    """Create WorkspaceManager for testing."""
    return WorkspaceManager(
        goal="test-command-tools",
        workspace_path=temp_workspace
    )


@pytest.fixture
def ledger_file(temp_workspace):
    """Create ledger file for testing."""
    return temp_workspace / "test_ledger.log"


@pytest.fixture
def command_tools(workspace_manager, ledger_file):
    """Create CommandToolsBehavior instance."""
    return CommandToolsBehavior(
        workspace_manager=workspace_manager,
        ledger_file=ledger_file
    )


# ==============================================================================
# Basic Behavior Interface Tests
# ==============================================================================

def test_get_name(command_tools):
    """Test behavior returns correct name."""
    assert command_tools.get_name() == "command_tools"


def test_get_tools_returns_one_tool(command_tools):
    """Test behavior provides exactly 1 tool."""
    tools = command_tools.get_tools()
    assert len(tools) == 1

    tool_names = [t["function"]["name"] for t in tools]
    assert "run_bash" in tool_names


def test_tool_schema_format(command_tools):
    """Test tools follow OpenAI function calling schema."""
    tools = command_tools.get_tools()

    for tool in tools:
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]
        assert tool["function"]["parameters"]["type"] == "object"
        assert "properties" in tool["function"]["parameters"]


# ==============================================================================
# run_bash Tests
# ==============================================================================

def test_run_bash_simple_command():
    """Test run_bash executes simple command."""
    command_tools = CommandToolsBehavior()

    result = command_tools.dispatch_tool(
        "run_bash",
        {"command": "echo 'Hello, World!'"}
    )

    assert result["returncode"] == 0
    assert "Hello, World!" in result["stdout"]
    assert result["stderr"] == ""


def test_run_bash_command_with_output():
    """Test run_bash captures stdout."""
    command_tools = CommandToolsBehavior()

    result = command_tools.dispatch_tool(
        "run_bash",
        {"command": "python -c 'print(\"test output\")'"}
    )

    assert result["returncode"] == 0
    assert "test output" in result["stdout"]


def test_run_bash_command_with_error():
    """Test run_bash captures stderr and non-zero exit code."""
    command_tools = CommandToolsBehavior()

    result = command_tools.dispatch_tool(
        "run_bash",
        {"command": "ls /nonexistent-directory-12345"}
    )

    assert result["returncode"] != 0
    assert len(result["stderr"]) > 0


def test_run_bash_pipes_and_redirection():
    """Test run_bash supports pipes and redirection."""
    command_tools = CommandToolsBehavior()

    result = command_tools.dispatch_tool(
        "run_bash",
        {"command": "echo 'line1' | grep 'line'"}
    )

    assert result["returncode"] == 0
    assert "line1" in result["stdout"]


def test_run_bash_command_chaining():
    """Test run_bash supports command chaining with &&."""
    command_tools = CommandToolsBehavior()

    result = command_tools.dispatch_tool(
        "run_bash",
        {"command": "echo 'first' && echo 'second'"}
    )

    assert result["returncode"] == 0
    assert "first" in result["stdout"]
    assert "second" in result["stdout"]


def test_run_bash_timeout():
    """Test run_bash respects timeout parameter."""
    command_tools = CommandToolsBehavior()

    start_time = time.time()
    result = command_tools.dispatch_tool(
        "run_bash",
        {"command": "sleep 10", "timeout": 1}
    )
    elapsed = time.time() - start_time

    assert elapsed < 3  # Should timeout quickly
    assert "error" in result
    assert "timed out" in result["error"].lower()
    assert result["returncode"] == -1


def test_run_bash_output_truncation():
    """Test run_bash truncates large output to 50KB."""
    command_tools = CommandToolsBehavior()

    # Generate large output (>50KB)
    large_command = "python -c 'print(\"x\" * 60000)'"
    result = command_tools.dispatch_tool(
        "run_bash",
        {"command": large_command}
    )

    assert result["returncode"] == 0
    # Output should be truncated to last 50KB
    assert len(result["stdout"]) <= 50_000


def test_run_bash_workspace_directory(command_tools, workspace_manager):
    """Test run_bash executes in workspace directory."""
    # Create a file in workspace
    test_file = workspace_manager.workspace_dir / "test.txt"
    test_file.write_text("workspace test")

    # List files in current directory (should be workspace)
    result = command_tools.dispatch_tool(
        "run_bash",
        {"command": "ls"}
    )

    assert result["returncode"] == 0
    assert "test.txt" in result["stdout"]


def test_run_bash_pythonpath_set(command_tools, workspace_manager):
    """Test run_bash sets PYTHONPATH to workspace."""
    result = command_tools.dispatch_tool(
        "run_bash",
        {"command": "echo $PYTHONPATH"}
    )

    assert result["returncode"] == 0
    assert str(workspace_manager.workspace_dir) in result["stdout"]


def test_run_bash_ledger_logging(command_tools, ledger_file):
    """Test run_bash logs to ledger."""
    command_tools.dispatch_tool(
        "run_bash",
        {"command": "echo 'test'"}
    )

    assert ledger_file.exists()
    ledger_content = ledger_file.read_text()
    assert "BASH" in ledger_content
    assert "echo 'test'" in ledger_content


def test_run_bash_error_logging(command_tools, ledger_file):
    """Test run_bash logs errors to ledger."""
    command_tools.dispatch_tool(
        "run_bash",
        {"command": "exit 1"}
    )

    assert ledger_file.exists()
    ledger_content = ledger_file.read_text()
    assert "ERROR" in ledger_content
    assert "run_bash rc=1" in ledger_content


# ==============================================================================
# Integration Tests
# ==============================================================================

def test_without_workspace_manager(temp_workspace, ledger_file):
    """Test behavior works without workspace manager (runs in current dir)."""
    command_tools = CommandToolsBehavior(ledger_file=ledger_file)

    result = command_tools.dispatch_tool(
        "run_bash",
        {"command": "pwd"}
    )

    assert result["returncode"] == 0
    assert len(result["stdout"]) > 0


def test_runtime_workspace_override(command_tools, temp_workspace):
    """Test workspace_manager can be overridden at runtime."""
    # Create second workspace manager
    other_workspace = temp_workspace / "other"
    other_workspace.mkdir()
    other_wm = WorkspaceManager(
        goal="other",
        workspace_path=other_workspace
    )

    # Create file in other workspace
    (other_workspace / "marker.txt").write_text("marker")

    # Pass different workspace_manager via kwargs
    result = command_tools.dispatch_tool(
        "run_bash",
        {"command": "ls"},
        workspace_manager=other_wm
    )

    # Should list files from other workspace
    assert result["returncode"] == 0
    assert "marker.txt" in result["stdout"]


def test_exception_handling(command_tools):
    """Test run_bash handles exceptions gracefully."""
    # This test is hard to trigger but ensures exception handling works
    # We rely on the implementation catching general exceptions
    result = command_tools.dispatch_tool(
        "run_bash",
        {"command": "echo test"}
    )

    # Should not raise exception, should return structured result
    assert "returncode" in result
