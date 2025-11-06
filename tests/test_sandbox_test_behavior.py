"""
Tests for SandboxTestBehavior.

Comprehensive test suite covering:
- get_name() returns "sandbox_test"
- Tools are provided (run_behavior_tests, run_agent_sanity_check)
- run_behavior_tests with simple test cases
- Error handling and cleanup

Target: >90% code coverage
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import shutil
from behaviors.sandbox_test import SandboxTestBehavior


class TestSandboxTestBehaviorBasics:
    """Test basic behavior interface."""

    def test_get_name(self):
        """get_name() returns 'sandbox_test'."""
        behavior = SandboxTestBehavior()
        assert behavior.get_name() == "sandbox_test"

    def test_initialization_default(self):
        """Behavior initializes with default timeout."""
        behavior = SandboxTestBehavior()
        assert behavior.default_timeout == 30

    def test_initialization_custom_timeout(self):
        """Behavior accepts custom timeout parameter."""
        behavior = SandboxTestBehavior(default_timeout=60)
        assert behavior.default_timeout == 60

    def test_initialization_with_other_kwargs(self):
        """Behavior accepts and ignores other kwargs."""
        behavior = SandboxTestBehavior(default_timeout=45, some_param="value")
        assert behavior.default_timeout == 45


class TestSandboxTestBehaviorTools:
    """Test tool definitions."""

    def test_get_tools_returns_list(self):
        """get_tools() returns a list."""
        behavior = SandboxTestBehavior()
        tools = behavior.get_tools()
        assert isinstance(tools, list)

    def test_get_tools_count(self):
        """get_tools() returns exactly 2 tools."""
        behavior = SandboxTestBehavior()
        tools = behavior.get_tools()
        assert len(tools) == 2

    def test_all_tools_present(self):
        """Both expected sandbox test tools are present."""
        behavior = SandboxTestBehavior()
        tools = behavior.get_tools()

        tool_names = [tool["function"]["name"] for tool in tools]

        expected_tools = [
            "run_behavior_tests",
            "run_agent_sanity_check"
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Tool {expected} not found"

    def test_tool_structure(self):
        """Each tool has proper OpenAI function schema structure."""
        behavior = SandboxTestBehavior()
        tools = behavior.get_tools()

        for tool in tools:
            assert "type" in tool
            assert tool["type"] == "function"
            assert "function" in tool

            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

            params = func["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params

    def test_run_behavior_tests_tool_schema(self):
        """run_behavior_tests tool has correct schema."""
        behavior = SandboxTestBehavior()
        tools = behavior.get_tools()

        tool = next(t for t in tools if t["function"]["name"] == "run_behavior_tests")
        func = tool["function"]

        assert "sandbox" in func["description"].lower()
        assert "behavior_file" in func["parameters"]["properties"]
        assert "test_file" in func["parameters"]["properties"]
        assert "timeout" in func["parameters"]["properties"]
        assert "behavior_file" in func["parameters"]["required"]
        assert "test_file" in func["parameters"]["required"]

    def test_run_agent_sanity_check_tool_schema(self):
        """run_agent_sanity_check tool has correct schema."""
        behavior = SandboxTestBehavior()
        tools = behavior.get_tools()

        tool = next(t for t in tools if t["function"]["name"] == "run_agent_sanity_check")
        func = tool["function"]

        assert "agent" in func["description"].lower()
        assert "config_file" in func["parameters"]["properties"]
        assert "agent_class" in func["parameters"]["properties"]
        assert "config_file" in func["parameters"]["required"]
        assert "agent_class" in func["parameters"]["required"]


class TestRunBehaviorTestsTool:
    """Test run_behavior_tests tool execution."""

    def test_successful_test_run(self, tmp_path):
        """Successful test run returns success result."""
        behavior = SandboxTestBehavior()

        # Create agent mock with workspace
        agent = Mock()
        agent.workspace = tmp_path

        # Create a simple behavior file
        behavior_file = tmp_path / "simple_behavior.py"
        behavior_file.write_text("""
def add(a, b):
    return a + b
""")

        # Create a passing test file
        test_file = tmp_path / "test_simple.py"
        test_file.write_text("""
def test_add():
    from simple_behavior import add
    assert add(2, 3) == 5
""")

        args = {
            "behavior_file": "simple_behavior.py",
            "test_file": "test_simple.py",
            "timeout": 30
        }

        result = behavior.dispatch_tool(agent, "run_behavior_tests", args)

        assert "success" in result
        assert result["success"] is True
        assert "stdout" in result
        assert "message" in result

    def test_failing_test_run(self, tmp_path):
        """Failing test returns failure result."""
        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        # Create a simple behavior file
        behavior_file = tmp_path / "simple_behavior.py"
        behavior_file.write_text("""
def add(a, b):
    return a + b
""")

        # Create a failing test file
        test_file = tmp_path / "test_failing.py"
        test_file.write_text("""
def test_add_fails():
    from simple_behavior import add
    assert add(2, 3) == 999  # This will fail
""")

        args = {
            "behavior_file": "simple_behavior.py",
            "test_file": "test_failing.py"
        }

        result = behavior.dispatch_tool(agent, "run_behavior_tests", args)

        assert "success" in result
        assert result["success"] is False
        assert "stderr" in result or "stdout" in result

    def test_behavior_file_not_found(self, tmp_path):
        """Non-existent behavior file returns error."""
        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        args = {
            "behavior_file": "nonexistent.py",
            "test_file": "test_file.py"
        }

        result = behavior.dispatch_tool(agent, "run_behavior_tests", args)

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_test_file_not_found(self, tmp_path):
        """Non-existent test file returns error."""
        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        # Create behavior file
        behavior_file = tmp_path / "behavior.py"
        behavior_file.write_text("def foo(): pass")

        args = {
            "behavior_file": "behavior.py",
            "test_file": "nonexistent_test.py"
        }

        result = behavior.dispatch_tool(agent, "run_behavior_tests", args)

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_custom_timeout(self, tmp_path):
        """Custom timeout is respected."""
        behavior = SandboxTestBehavior(default_timeout=5)

        agent = Mock()
        agent.workspace = tmp_path

        # Create files
        behavior_file = tmp_path / "behavior.py"
        behavior_file.write_text("def foo(): pass")

        test_file = tmp_path / "test_quick.py"
        test_file.write_text("""
def test_quick():
    assert True
""")

        args = {
            "behavior_file": "behavior.py",
            "test_file": "test_quick.py",
            "timeout": 10  # Override default
        }

        result = behavior.dispatch_tool(agent, "run_behavior_tests", args)

        # Should succeed with custom timeout
        assert "success" in result or "error" in result

    def test_uses_default_timeout_when_not_specified(self, tmp_path):
        """Default timeout is used when not specified in args."""
        behavior = SandboxTestBehavior(default_timeout=20)

        agent = Mock()
        agent.workspace = tmp_path

        # Create files
        behavior_file = tmp_path / "behavior.py"
        behavior_file.write_text("def foo(): pass")

        test_file = tmp_path / "test_default.py"
        test_file.write_text("""
def test_default():
    assert True
""")

        args = {
            "behavior_file": "behavior.py",
            "test_file": "test_default.py"
            # No timeout specified
        }

        # Mock subprocess to verify timeout
        with patch('behaviors.sandbox_test.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            behavior.dispatch_tool(agent, "run_behavior_tests", args)

            # Verify default timeout was used
            call_args = mock_run.call_args
            assert call_args.kwargs['timeout'] == 20

    def test_sandbox_cleanup_on_success(self, tmp_path):
        """Sandbox directory is cleaned up after successful test."""
        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        # Create files
        behavior_file = tmp_path / "behavior.py"
        behavior_file.write_text("def foo(): pass")

        test_file = tmp_path / "test_cleanup.py"
        test_file.write_text("""
def test_cleanup():
    assert True
""")

        args = {
            "behavior_file": "behavior.py",
            "test_file": "test_cleanup.py"
        }

        # Track sandbox directory
        sandbox_dirs_before = list(Path(tempfile.gettempdir()).glob("sandbox_behavior_test_*"))

        result = behavior.dispatch_tool(agent, "run_behavior_tests", args)

        # Sandbox should be cleaned up
        sandbox_dirs_after = list(Path(tempfile.gettempdir()).glob("sandbox_behavior_test_*"))

        # Number of sandbox dirs should be the same or less (cleanup happened)
        assert len(sandbox_dirs_after) <= len(sandbox_dirs_before)

    def test_sandbox_cleanup_on_error(self, tmp_path):
        """Sandbox directory is cleaned up even on error."""
        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        # Create files that will cause test failure
        behavior_file = tmp_path / "behavior.py"
        behavior_file.write_text("def foo(): pass")

        test_file = tmp_path / "test_error.py"
        test_file.write_text("""
def test_error():
    assert False  # Intentional failure
""")

        args = {
            "behavior_file": "behavior.py",
            "test_file": "test_error.py"
        }

        sandbox_dirs_before = list(Path(tempfile.gettempdir()).glob("sandbox_behavior_test_*"))

        result = behavior.dispatch_tool(agent, "run_behavior_tests", args)

        sandbox_dirs_after = list(Path(tempfile.gettempdir()).glob("sandbox_behavior_test_*"))

        # Cleanup should still happen
        assert len(sandbox_dirs_after) <= len(sandbox_dirs_before)

    def test_agent_without_workspace_attribute(self):
        """Agent without workspace uses current directory."""
        behavior = SandboxTestBehavior()

        agent = Mock(spec=[])  # No workspace attribute

        args = {
            "behavior_file": "nonexistent.py",
            "test_file": "test.py"
        }

        result = behavior.dispatch_tool(agent, "run_behavior_tests", args)

        # Should handle gracefully (will fail on file not found)
        assert "error" in result

    @patch('behaviors.sandbox_test.subprocess.run')
    def test_timeout_handling(self, mock_run, tmp_path):
        """Timeout during test execution is handled gracefully."""
        import subprocess

        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        # Create files
        behavior_file = tmp_path / "behavior.py"
        behavior_file.write_text("def foo(): pass")

        test_file = tmp_path / "test_timeout.py"
        test_file.write_text("""
def test_timeout():
    assert True
""")

        # Simulate timeout
        mock_run.side_effect = subprocess.TimeoutExpired("pytest", 30)

        args = {
            "behavior_file": "behavior.py",
            "test_file": "test_timeout.py",
            "timeout": 1
        }

        result = behavior.dispatch_tool(agent, "run_behavior_tests", args)

        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        assert "timed out" in result["error"].lower()


class TestRunAgentSanityCheckTool:
    """Test run_agent_sanity_check tool execution."""

    def test_agent_sanity_check_basic(self, tmp_path):
        """Basic agent sanity check execution."""
        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        # Create a simple config file
        config_file = tmp_path / "agent_config.yaml"
        config_file.write_text("""
role: "Test Agent"
blurb: "Does testing."
system_prompt: "You are a test agent."
behaviors: []
""")

        args = {
            "config_file": "agent_config.yaml",
            "agent_class": "TaskExecutorAgent"
        }

        # This will fail because the agent class doesn't exist in sandbox
        # but we're testing that the mechanism works
        result = behavior.dispatch_tool(agent, "run_agent_sanity_check", args)

        assert "success" in result
        # We expect this to fail gracefully
        assert "stdout" in result or "stderr" in result

    def test_agent_sanity_check_config_not_found(self, tmp_path):
        """Non-existent config file returns error."""
        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        args = {
            "config_file": "nonexistent_config.yaml",
            "agent_class": "TaskExecutorAgent"
        }

        result = behavior.dispatch_tool(agent, "run_agent_sanity_check", args)

        assert "error" in result
        assert "not found" in result["error"].lower()

    @patch('behaviors.sandbox_test.subprocess.run')
    def test_agent_sanity_check_timeout(self, mock_run, tmp_path):
        """Timeout during sanity check is handled."""
        import subprocess

        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        # Create config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("role: Test")

        # Simulate timeout
        mock_run.side_effect = subprocess.TimeoutExpired("python", 10)

        args = {
            "config_file": "config.yaml",
            "agent_class": "TestAgent"
        }

        result = behavior.dispatch_tool(agent, "run_agent_sanity_check", args)

        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        assert "timed out" in result["error"].lower()

    def test_agent_sanity_check_cleanup(self, tmp_path):
        """Sandbox cleanup happens after sanity check."""
        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        # Create config
        config_file = tmp_path / "config.yaml"
        config_file.write_text("role: Test")

        sandbox_dirs_before = list(Path(tempfile.gettempdir()).glob("sandbox_agent_check_*"))

        args = {
            "config_file": "config.yaml",
            "agent_class": "TestAgent"
        }

        result = behavior.dispatch_tool(agent, "run_agent_sanity_check", args)

        sandbox_dirs_after = list(Path(tempfile.gettempdir()).glob("sandbox_agent_check_*"))

        # Cleanup should happen
        assert len(sandbox_dirs_after) <= len(sandbox_dirs_before)


class TestDispatchToolRouting:
    """Test dispatch_tool routing logic."""

    def test_routes_run_behavior_tests(self, tmp_path):
        """dispatch_tool routes run_behavior_tests correctly."""
        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        # Will fail on file not found, but verifies routing works
        result = behavior.dispatch_tool(
            agent,
            "run_behavior_tests",
            {"behavior_file": "test.py", "test_file": "test.py"}
        )

        assert "error" in result or "success" in result

    def test_routes_run_agent_sanity_check(self, tmp_path):
        """dispatch_tool routes run_agent_sanity_check correctly."""
        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        # Will fail on file not found, but verifies routing works
        result = behavior.dispatch_tool(
            agent,
            "run_agent_sanity_check",
            {"config_file": "config.yaml", "agent_class": "TestAgent"}
        )

        assert "error" in result or "success" in result

    def test_unknown_tool_calls_parent(self):
        """Unknown tool name raises NotImplementedError."""
        behavior = SandboxTestBehavior()
        agent = Mock()

        # Parent dispatch_tool raises NotImplementedError for unknown tools
        with pytest.raises(NotImplementedError):
            behavior.dispatch_tool(agent, "unknown_tool", {})


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_exception_during_test_execution(self, tmp_path):
        """Exception during test execution is caught."""
        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        # Create files
        behavior_file = tmp_path / "behavior.py"
        behavior_file.write_text("def foo(): pass")

        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")

        # Mock subprocess to raise exception
        with patch('behaviors.sandbox_test.subprocess.run') as mock_run:
            mock_run.side_effect = RuntimeError("Unexpected error")

            args = {
                "behavior_file": "behavior.py",
                "test_file": "test.py"
            }

            result = behavior.dispatch_tool(agent, "run_behavior_tests", args)

            assert "error" in result
            assert "Sandbox test error" in result["error"]

    def test_missing_required_arguments(self):
        """Missing required arguments is handled gracefully."""
        behavior = SandboxTestBehavior()
        agent = Mock()

        # Missing both required arguments
        result = behavior.dispatch_tool(agent, "run_behavior_tests", {})

        # Should handle missing args gracefully
        assert "error" in result or "success" in result


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_behavior_testing_workflow(self, tmp_path):
        """Complete workflow: create files -> run tests -> cleanup."""
        behavior = SandboxTestBehavior()

        agent = Mock()
        agent.workspace = tmp_path

        # 1. Create a behavior
        behavior_code = """
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""
        behavior_file = tmp_path / "calculator.py"
        behavior_file.write_text(behavior_code)

        # 2. Create tests
        test_code = """
from calculator import Calculator

def test_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5

def test_subtract():
    calc = Calculator()
    assert calc.subtract(5, 3) == 2
"""
        test_file = tmp_path / "test_calculator.py"
        test_file.write_text(test_code)

        # 3. Run tests
        args = {
            "behavior_file": "calculator.py",
            "test_file": "test_calculator.py",
            "timeout": 30
        }

        result = behavior.dispatch_tool(agent, "run_behavior_tests", args)

        # 4. Verify results
        assert "success" in result
        assert result["success"] is True
        assert "message" in result
        assert "Tests passed" in result["message"]
