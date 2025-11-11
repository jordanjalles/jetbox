"""
Test suite for workspace path resolution safety.

These tests detect unsafe path patterns that could break during refactors:
- Path() calls without proper anchors (PROJECT_ROOT or workspace_manager)
- Absolute paths leaking into workspace operations
- Path escapes with ../
- CWD-dependent resolution

Run this test before merging any workspace-related changes.
"""
import pytest
from pathlib import Path
import tempfile
import ast
import re


class TestWorkspacePathValidation:
    """Test workspace path resolution validation in WorkspaceManager."""

    def test_resolve_path_rejects_absolute_paths(self):
        """Absolute paths should be rejected with clear error."""
        from src.workspace_manager import WorkspaceManager

        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorkspaceManager("test", workspace_path=tmpdir)

            with pytest.raises(ValueError, match="must use relative paths"):
                wm.resolve_path("/etc/passwd")

    def test_resolve_path_blocks_escape_attempts(self):
        """Path escapes with ../ should be blocked."""
        from src.workspace_manager import WorkspaceManager

        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorkspaceManager("test", workspace_path=tmpdir)

            with pytest.raises(ValueError, match="escapes workspace boundary"):
                wm.resolve_path("../../../etc/passwd")

    def test_resolve_path_allows_normal_relative(self):
        """Normal relative paths should work."""
        from src.workspace_manager import WorkspaceManager

        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorkspaceManager("test", workspace_path=tmpdir)

            result = wm.resolve_path("test.txt")
            assert result.is_absolute()
            assert result.parent == Path(tmpdir).resolve()

    def test_resolve_path_error_messages_include_docs_link(self):
        """Error messages should guide user to documentation."""
        from src.workspace_manager import WorkspaceManager

        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorkspaceManager("test", workspace_path=tmpdir)

            with pytest.raises(ValueError, match="WORKSPACE_PATH_ARCHITECTURE.md"):
                wm.resolve_path("/absolute/path")


class TestAgentInitializationValidation:
    """Test agent workspace validation during initialization."""

    def test_base_agent_normalizes_relative_workspace(self):
        """BaseAgent should convert relative workspace to absolute."""
        from base_agent import BaseAgent
        from agent_config import PROJECT_ROOT

        config_file = PROJECT_ROOT / "config/agents/task_executor.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use relative path
            relative_workspace = Path(tmpdir).name

            # Change to parent so relative path works
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(Path(tmpdir).parent)

                agent = BaseAgent(
                    name="test",
                    workspace=relative_workspace,
                    config_file=str(config_file)
                )

                # Should be converted to absolute
                assert agent.workspace.is_absolute()
            finally:
                os.chdir(original_cwd)

    def test_base_agent_logs_workspace_path(self, capsys):
        """BaseAgent should log workspace being used."""
        from base_agent import BaseAgent
        from agent_config import PROJECT_ROOT

        config_file = PROJECT_ROOT / "config/agents/task_executor.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = BaseAgent(
                name="test",
                workspace=tmpdir,
                config_file=str(config_file)
            )

            captured = capsys.readouterr()
            assert "Using custom workspace" in captured.out
            assert tmpdir in captured.out


class TestDelegationWorkspaceHandoff:
    """Test workspace handoff during agent delegation."""

    def test_delegation_validates_workspace_parent_exists(self):
        """Delegation should fail fast if workspace parent doesn't exist."""
        # This test verifies the validation logic exists
        # Full delegation test would require complete agent setup

        # Verify workspace validation catches nonexistent parents
        nonexistent_workspace = Path("/nonexistent/parent/workspace")
        assert not nonexistent_workspace.parent.exists(), \
            "Test precondition: parent directory should not exist"

        # The actual validation happens in delegation.py:939-948
        # where it checks workspace.parent.exists() before delegation
        from behaviors.delegation import DelegationBehavior
        from agent_config import PROJECT_ROOT

        delegation_file = PROJECT_ROOT / "behaviors/delegation.py"
        content = delegation_file.read_text()

        # Verify validation code exists
        assert "workspace.parent.exists()" in content, \
            "delegation.py should validate workspace parent exists"


class TestPathResolutionPatterns:
    """Static analysis tests for unsafe path patterns in code."""

    def test_no_bare_path_calls_in_behaviors(self):
        """
        Scan behavior files for Path() calls without proper anchors.

        This catches patterns like:
        - Path("relative/path")  # BAD - CWD dependent
        - Path("agents/foo.py")  # BAD - should use PROJECT_ROOT

        NOTE: This test currently detects 13 known unsafe patterns to be fixed.
        It's enabled to catch NEW unsafe patterns introduced during development.
        """
        from agent_config import PROJECT_ROOT

        behaviors_dir = PROJECT_ROOT / "behaviors"
        if not behaviors_dir.exists():
            pytest.skip("behaviors directory not found")

        unsafe_patterns = []

        # Known unsafe patterns (to be fixed in follow-up)
        known_issues = {
            "create_agent.py:20",
            "create_agent.py:204",
            "create_behavior.py:40",
            "create_behavior.py:275",
            "create_behavior.py:757",
            "delegation.py:571",
            "sandbox_test.py:205",
            "sandbox_test.py:214",
            "server_tools.py:257",
            "server_tools.py:288",
            "server_tools.py:307",
            "server_tools.py:318",
            "server_tools.py:337",
        }

        for py_file in behaviors_dir.glob("*.py"):
            content = py_file.read_text()

            # Skip __init__.py
            if py_file.name == "__init__.py":
                continue

            # Look for Path("...") patterns
            # Negative lookahead: skip if preceded by PROJECT_ROOT / or workspace_manager.
            pattern = r'(?<!PROJECT_ROOT / )(?<!workspace_manager\.)Path\(["\'](?!/)([^"\']+)["\']\)'

            for match in re.finditer(pattern, content):
                path_arg = match.group(1)

                # Exceptions - these are OK:
                # - Path(".") for current directory listing
                # - Path that's actually a variable name followed by path construction
                if path_arg == ".":
                    continue

                # Found unsafe pattern
                line_num = content[:match.start()].count('\n') + 1
                location = f"{py_file.name}:{line_num}"

                # Skip known issues
                if location in known_issues:
                    continue

                # Found NEW unsafe pattern
                unsafe_patterns.append({
                    "file": py_file.name,
                    "pattern": match.group(0),
                    "line": line_num
                })

        if unsafe_patterns:
            msg = "Found NEW unsafe Path() calls without PROJECT_ROOT or workspace_manager:\n"
            for item in unsafe_patterns:
                msg += f"  {item['file']}:{item['line']} - {item['pattern']}\n"
            msg += "\nUse PROJECT_ROOT for code/config, workspace_manager.resolve_path() for user files."
            msg += "\nSee docs/WORKSPACE_PATH_ARCHITECTURE.md"
            pytest.fail(msg)

    def test_delegation_uses_absolute_paths(self):
        """
        Verify delegation.py uses PROJECT_ROOT for agent/config paths.
        """
        from agent_config import PROJECT_ROOT

        delegation_file = PROJECT_ROOT / "behaviors/delegation.py"
        if not delegation_file.exists():
            pytest.skip("delegation.py not found")

        content = delegation_file.read_text()

        # Check for PROJECT_ROOT usage
        assert "from agent_config import PROJECT_ROOT" in content, \
            "delegation.py should import PROJECT_ROOT for absolute paths"

        # Check it's used for config files
        assert 'PROJECT_ROOT / f"config/agents/' in content, \
            "delegation.py should use PROJECT_ROOT for config file resolution"


class TestCommandToolsCWDIsolation:
    """Test that command_tools.py uses workspace CWD, not system CWD."""

    def test_run_bash_uses_workspace_cwd(self):
        """
        Verify run_bash executes in workspace directory, not calling CWD.
        """
        from agent_config import PROJECT_ROOT

        command_tools_file = PROJECT_ROOT / "behaviors/command_tools.py"
        if not command_tools_file.exists():
            pytest.skip("command_tools.py not found")

        content = command_tools_file.read_text()

        # Should set cwd to workspace_manager.workspace_dir
        assert "cwd=cwd" in content or "cwd=workspace" in content, \
            "run_bash should set cwd parameter for subprocess.run()"

        # Should NOT use os.chdir
        assert "os.chdir" not in content, \
            "command_tools.py should not use os.chdir (affects all code)"


class TestCrossCuttingPathSafety:
    """Integration tests for path safety across components."""

    def test_workspace_paths_stay_within_boundary(self):
        """
        End-to-end test: file operations stay within workspace.
        """
        from src.workspace_manager import WorkspaceManager

        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorkspaceManager("test", workspace_path=tmpdir)

            # These should all stay within tmpdir
            safe_paths = [
                "file.txt",
                "subdir/file.txt",
                "deep/nested/path/file.txt"
            ]

            for path in safe_paths:
                resolved = wm.resolve_path(path)
                assert str(resolved).startswith(tmpdir), \
                    f"Path {path} resolved outside workspace: {resolved}"

    def test_project_root_paths_absolute(self):
        """
        Verify PROJECT_ROOT resolves to absolute path.
        """
        from agent_config import PROJECT_ROOT

        assert PROJECT_ROOT.is_absolute(), \
            "PROJECT_ROOT must be absolute for CWD-independent resolution"

        # Should point to workspace root (where agent_config.py lives)
        assert (PROJECT_ROOT / "agent_config.py").exists(), \
            "PROJECT_ROOT should point to repository root"


# pytest markers for selective runs
pytestmark = [
    pytest.mark.workspace_safety,  # All tests in this file
]
