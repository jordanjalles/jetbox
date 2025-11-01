"""
Tests for FileToolsBehavior.

Validates:
- Tool registration and schema
- write_file, read_file, list_dir operations
- Workspace-aware path resolution
- Safety checks (forbidden files in edit mode)
- Ledger logging
- Parameter invention tolerance
"""
import pytest
from pathlib import Path
import tempfile
import shutil

from behaviors.file_tools import FileToolsBehavior
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
        goal="test-file-tools",
        workspace_path=temp_workspace
    )


@pytest.fixture
def ledger_file(temp_workspace):
    """Create ledger file for testing."""
    return temp_workspace / "test_ledger.log"


@pytest.fixture
def file_tools(workspace_manager, ledger_file):
    """Create FileToolsBehavior instance."""
    return FileToolsBehavior(
        workspace_manager=workspace_manager,
        ledger_file=ledger_file
    )


# ==============================================================================
# Basic Behavior Interface Tests
# ==============================================================================

def test_get_name(file_tools):
    """Test behavior returns correct name."""
    assert file_tools.get_name() == "file_tools"


def test_get_tools_returns_three_tools(file_tools):
    """Test behavior provides exactly 3 tools."""
    tools = file_tools.get_tools()
    assert len(tools) == 3

    tool_names = [t["function"]["name"] for t in tools]
    assert "write_file" in tool_names
    assert "read_file" in tool_names
    assert "list_dir" in tool_names


def test_tool_schema_format(file_tools):
    """Test tools follow OpenAI function calling schema."""
    tools = file_tools.get_tools()

    for tool in tools:
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]
        assert tool["function"]["parameters"]["type"] == "object"
        assert "properties" in tool["function"]["parameters"]


# ==============================================================================
# write_file Tests
# ==============================================================================

def test_write_file_creates_file(file_tools, workspace_manager):
    """Test write_file creates a file successfully."""
    result = file_tools.dispatch_tool(
        "write_file",
        {"path": "test.txt", "content": "Hello, World!"}
    )

    assert "Wrote 13 chars" in result
    assert "test.txt" in result

    # Verify file exists
    file_path = workspace_manager.resolve_path("test.txt")
    assert file_path.exists()
    assert file_path.read_text() == "Hello, World!"


def test_write_file_append_mode(file_tools, workspace_manager):
    """Test write_file append mode."""
    # Write initial content
    file_tools.dispatch_tool(
        "write_file",
        {"path": "test.txt", "content": "Line 1\n"}
    )

    # Append more content
    result = file_tools.dispatch_tool(
        "write_file",
        {"path": "test.txt", "content": "Line 2\n", "append": True}
    )

    assert "Appended" in result

    file_path = workspace_manager.resolve_path("test.txt")
    assert file_path.read_text() == "Line 1\nLine 2\n"


def test_write_file_overwrite_false(file_tools, workspace_manager):
    """Test write_file with overwrite=False prevents overwriting."""
    # Create initial file
    file_tools.dispatch_tool(
        "write_file",
        {"path": "test.txt", "content": "Original"}
    )

    # Try to overwrite with overwrite=False
    result = file_tools.dispatch_tool(
        "write_file",
        {"path": "test.txt", "content": "New", "overwrite": False}
    )

    assert "[ERROR]" in result
    assert "overwrite=False" in result

    # Original content should remain
    file_path = workspace_manager.resolve_path("test.txt")
    assert file_path.read_text() == "Original"


def test_write_file_creates_parent_directories(file_tools, workspace_manager):
    """Test write_file creates parent directories."""
    result = file_tools.dispatch_tool(
        "write_file",
        {"path": "subdir/nested/file.txt", "content": "Nested"}
    )

    assert "Wrote" in result

    file_path = workspace_manager.resolve_path("subdir/nested/file.txt")
    assert file_path.exists()
    assert file_path.read_text() == "Nested"


def test_write_file_line_endings(file_tools, workspace_manager):
    """Test write_file with custom line endings."""
    content = "Line 1\nLine 2\nLine 3"

    # Windows line endings
    file_tools.dispatch_tool(
        "write_file",
        {"path": "windows.txt", "content": content, "line_end": "\r\n"}
    )

    file_path = workspace_manager.resolve_path("windows.txt")
    raw_content = file_path.read_bytes()
    assert b"\r\n" in raw_content


def test_write_file_parameter_invention_tolerance(file_tools, workspace_manager, capsys):
    """Test write_file ignores unsupported parameters gracefully."""
    result = file_tools.dispatch_tool(
        "write_file",
        {
            "path": "test.txt",
            "content": "Content",
            "timeout": 30,  # Unsupported parameter
            "mode": "create"  # Unsupported parameter
        }
    )

    # Should still succeed
    assert "Wrote" in result

    # Should print warning about ignored parameters
    captured = capsys.readouterr()
    assert "ignoring unsupported parameters" in captured.out


def test_write_file_ledger_logging(file_tools, ledger_file):
    """Test write_file logs to ledger."""
    file_tools.dispatch_tool(
        "write_file",
        {"path": "test.txt", "content": "Content"}
    )

    assert ledger_file.exists()
    ledger_content = ledger_file.read_text()
    assert "WRITE" in ledger_content
    assert "test.txt" in ledger_content


def test_write_file_edit_mode_safety(temp_workspace, ledger_file):
    """Test write_file prevents modifying agent code in edit mode."""
    # Create workspace manager in edit mode (workspace_path triggers edit mode)
    wm = WorkspaceManager(
        goal="test",
        workspace_path=temp_workspace
    )

    file_tools = FileToolsBehavior(
        workspace_manager=wm,
        ledger_file=ledger_file
    )

    # Try to modify forbidden file
    result = file_tools.dispatch_tool(
        "write_file",
        {"path": "agent.py", "content": "Malicious code"}
    )

    assert "[SAFETY]" in result
    assert "agent.py" in result

    # File should not exist
    file_path = wm.resolve_path("agent.py")
    assert not file_path.exists()


# ==============================================================================
# read_file Tests
# ==============================================================================

def test_read_file_reads_content(file_tools, workspace_manager):
    """Test read_file reads file content."""
    # Create file
    file_path = workspace_manager.resolve_path("test.txt")
    file_path.write_text("Test content")

    result = file_tools.dispatch_tool(
        "read_file",
        {"path": "test.txt"}
    )

    assert result == "Test content"


def test_read_file_max_size_truncation(file_tools, workspace_manager):
    """Test read_file truncates large files."""
    # Create large file
    large_content = "x" * 10000
    file_path = workspace_manager.resolve_path("large.txt")
    file_path.write_text(large_content)

    result = file_tools.dispatch_tool(
        "read_file",
        {"path": "large.txt", "max_size": 1000}
    )

    assert len(result) > 1000  # Includes truncation message
    assert "[TRUNCATED:" in result
    assert result.startswith("x" * 1000)


def test_read_file_encoding_handling(file_tools, workspace_manager):
    """Test read_file with custom encoding."""
    file_path = workspace_manager.resolve_path("utf16.txt")
    file_path.write_text("Unicode: é", encoding="utf-16")

    result = file_tools.dispatch_tool(
        "read_file",
        {"path": "utf16.txt", "encoding": "utf-16"}
    )

    assert "Unicode: é" in result


def test_read_file_parameter_invention_tolerance(file_tools, workspace_manager, capsys):
    """Test read_file ignores unsupported parameters."""
    file_path = workspace_manager.resolve_path("test.txt")
    file_path.write_text("Content")

    result = file_tools.dispatch_tool(
        "read_file",
        {"path": "test.txt", "binary": True}  # Unsupported parameter
    )

    assert result == "Content"

    captured = capsys.readouterr()
    assert "ignoring unsupported parameters" in captured.out


# ==============================================================================
# list_dir Tests
# ==============================================================================

def test_list_dir_lists_files(file_tools, workspace_manager):
    """Test list_dir returns sorted file list."""
    # Create some files
    (workspace_manager.resolve_path("file1.txt")).touch()
    (workspace_manager.resolve_path("file2.txt")).touch()
    (workspace_manager.resolve_path("file3.txt")).touch()

    result = file_tools.dispatch_tool(
        "list_dir",
        {"path": "."}
    )

    assert isinstance(result, list)
    assert "file1.txt" in result
    assert "file2.txt" in result
    assert "file3.txt" in result
    assert result == sorted(result)  # Should be sorted


def test_list_dir_subdirectory(file_tools, workspace_manager):
    """Test list_dir with subdirectory."""
    # Create subdirectory with files
    subdir = workspace_manager.resolve_path("subdir")
    subdir.mkdir()
    (subdir / "a.txt").touch()
    (subdir / "b.txt").touch()

    result = file_tools.dispatch_tool(
        "list_dir",
        {"path": "subdir"}
    )

    assert "a.txt" in result
    assert "b.txt" in result


def test_list_dir_missing_directory(file_tools):
    """Test list_dir with missing directory returns error."""
    result = file_tools.dispatch_tool(
        "list_dir",
        {"path": "nonexistent"}
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert "__error__" in result[0]


def test_list_dir_default_path(file_tools, workspace_manager):
    """Test list_dir with no path defaults to current directory."""
    (workspace_manager.resolve_path("file.txt")).touch()

    result = file_tools.dispatch_tool(
        "list_dir",
        {}
    )

    assert "file.txt" in result


# ==============================================================================
# Workspace Integration Tests
# ==============================================================================

def test_workspace_path_resolution(file_tools, workspace_manager):
    """Test path resolution works through workspace manager."""
    result = file_tools.dispatch_tool(
        "write_file",
        {"path": "test.txt", "content": "Test"}
    )

    # File should be in workspace directory
    file_path = workspace_manager.workspace_dir / "test.txt"
    assert file_path.exists()


def test_without_workspace_manager(temp_workspace, ledger_file):
    """Test behavior works without workspace manager (absolute paths)."""
    file_tools = FileToolsBehavior(ledger_file=ledger_file)

    # Use absolute path
    test_file = temp_workspace / "test.txt"

    result = file_tools.dispatch_tool(
        "write_file",
        {"path": str(test_file), "content": "Test"}
    )

    assert "Wrote" in result
    assert test_file.exists()


def test_runtime_workspace_override(file_tools, temp_workspace):
    """Test workspace_manager can be overridden at runtime."""
    # Create second workspace manager
    other_workspace = temp_workspace / "other"
    other_workspace.mkdir()
    other_wm = WorkspaceManager(
        goal="other",
        workspace_path=other_workspace
    )

    # Pass different workspace_manager via kwargs
    result = file_tools.dispatch_tool(
        "write_file",
        {"path": "test.txt", "content": "Test"},
        workspace_manager=other_wm
    )

    # File should be in other workspace
    assert (other_workspace / "test.txt").exists()
    assert not (file_tools.workspace_manager.workspace_dir / "test.txt").exists()
