"""
Tests for CreateBehaviorBehavior.

Tests:
- Behavior identifier
- Tool schemas
- Tool dispatch
- Code generation
- Validation integration
- Sandbox testing integration
- Safety mode handling
- No cross-behavior dependencies
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
from behaviors.create_behavior import CreateBehaviorBehavior


class TestCreateBehaviorBehavior:
    """Test suite for CreateBehaviorBehavior."""

    def test_get_name(self):
        """Behavior returns correct identifier."""
        behavior = CreateBehaviorBehavior()
        assert behavior.get_name() == "create_behavior"

    def test_initialization(self):
        """Behavior initializes without errors."""
        behavior = CreateBehaviorBehavior()
        assert behavior is not None
        assert behavior.staging_dir == ".agent_generated/staging"
        assert behavior.default_safety_mode == "review"

    def test_initialization_with_params(self):
        """Behavior accepts custom parameters."""
        behavior = CreateBehaviorBehavior(
            staging_dir="/tmp/staging",
            default_safety_mode="auto"
        )
        assert behavior.staging_dir == "/tmp/staging"
        assert behavior.default_safety_mode == "auto"

    def test_tool_schema(self):
        """Tool schemas are well-formed."""
        behavior = CreateBehaviorBehavior()
        tools = behavior.get_tools()

        assert len(tools) == 1
        tool = tools[0]

        assert tool["type"] == "function"
        assert tool["function"]["name"] == "create_behavior"
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]

        # Check required parameters
        params = tool["function"]["parameters"]
        assert "behavior_name" in params["properties"]
        assert "description" in params["properties"]
        assert "tool_specs" in params["properties"]
        assert "lifecycle_hooks" in params["properties"]
        assert "safety_mode" in params["properties"]

        assert "behavior_name" in params["required"]
        assert "description" in params["required"]
        assert "tool_specs" in params["required"]

    def test_tool_dispatch_create_behavior(self):
        """Tool dispatch routes to create_behavior handler."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        # Mock the execution method to avoid actual LLM calls
        with patch.object(behavior, '_execute_create_behavior', return_value={"success": True}):
            result = behavior.dispatch_tool(
                agent=mock_agent,
                tool_name="create_behavior",
                args={
                    "behavior_name": "test_behavior",
                    "description": "Test behavior",
                    "tool_specs": [{"name": "test_tool", "description": "Test", "parameters": {}}]
                }
            )

            assert result == {"success": True}

    def test_tool_dispatch_unknown_tool(self):
        """Unknown tools raise NotImplementedError."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        with pytest.raises(NotImplementedError):
            behavior.dispatch_tool(
                agent=mock_agent,
                tool_name="unknown_tool",
                args={}
            )

    def test_missing_required_parameters(self):
        """Missing required parameters return error."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()
        mock_agent.workspace = Path.cwd()

        # Missing behavior_name
        result = behavior._execute_create_behavior(mock_agent, {
            "description": "Test",
            "tool_specs": []
        })
        assert "error" in result

        # Missing description
        result = behavior._execute_create_behavior(mock_agent, {
            "behavior_name": "test",
            "tool_specs": []
        })
        assert "error" in result

        # Missing tool_specs
        result = behavior._execute_create_behavior(mock_agent, {
            "behavior_name": "test",
            "description": "Test"
        })
        assert "error" in result

        # Empty tool_specs
        result = behavior._execute_create_behavior(mock_agent, {
            "behavior_name": "test",
            "description": "Test",
            "tool_specs": []
        })
        assert "error" in result

    @patch('behaviors.create_behavior.Path')
    def test_generate_behavior_code_template_not_found(self, mock_path):
        """Code generation fails gracefully if template missing."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        # Mock template not existing
        mock_template = MagicMock()
        mock_template.exists.return_value = False
        mock_path.return_value = mock_template

        result = behavior._generate_behavior_code(
            mock_agent,
            behavior_name="test_behavior",
            description="Test",
            tool_specs=[{"name": "test", "description": "Test", "parameters": {}}],
            lifecycle_hooks=[]
        )

        assert "error" in result
        assert "Template not found" in result["error"]

    @patch('llm_utils.chat_with_inactivity_timeout')
    @patch('behaviors.create_behavior.Path')
    def test_generate_behavior_code_success(self, mock_path, mock_chat):
        """Code generation succeeds with valid LLM response."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        # Mock template exists
        mock_template = MagicMock()
        mock_template.exists.return_value = True
        mock_template.read_text.return_value = "# Template code"
        mock_path.return_value = mock_template

        # Mock LLM response
        mock_chat.return_value = {
            "message": {
                "content": "# Generated behavior code\nclass TestBehavior:\n    pass"
            }
        }

        result = behavior._generate_behavior_code(
            mock_agent,
            behavior_name="test_behavior",
            description="Test",
            tool_specs=[{"name": "test", "description": "Test", "parameters": {}}],
            lifecycle_hooks=[]
        )

        assert "code" in result
        assert "class TestBehavior" in result["code"]

    @patch('llm_utils.chat_with_inactivity_timeout')
    @patch('behaviors.create_behavior.Path')
    def test_generate_behavior_code_extracts_from_markdown(self, mock_path, mock_chat):
        """Code generation extracts code from markdown blocks."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        # Mock template
        mock_template = MagicMock()
        mock_template.exists.return_value = True
        mock_template.read_text.return_value = "# Template"
        mock_path.return_value = mock_template

        # Mock LLM response with markdown
        mock_chat.return_value = {
            "message": {
                "content": "Here's the code:\n```python\nclass TestBehavior:\n    pass\n```"
            }
        }

        result = behavior._generate_behavior_code(
            mock_agent,
            behavior_name="test_behavior",
            description="Test",
            tool_specs=[],
            lifecycle_hooks=[]
        )

        assert "code" in result
        assert "class TestBehavior" in result["code"]
        assert "Here's the code:" not in result["code"]

    @patch('llm_utils.chat_with_inactivity_timeout')
    @patch('behaviors.create_behavior.Path')
    def test_generate_test_code_success(self, mock_path, mock_chat):
        """Test generation succeeds with valid LLM response."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        # Mock template
        mock_template = MagicMock()
        mock_template.exists.return_value = True
        mock_template.read_text.return_value = "# Test template"
        mock_path.return_value = mock_template

        # Mock LLM response
        mock_chat.return_value = {
            "message": {
                "content": "def test_something():\n    assert True"
            }
        }

        result = behavior._generate_test_code(
            mock_agent,
            behavior_name="test_behavior",
            description="Test",
            tool_specs=[]
        )

        assert "code" in result
        assert "test_something" in result["code"]

    def test_save_to_staging(self):
        """Save to staging creates files correctly."""
        behavior = CreateBehaviorBehavior()

        # Use temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_agent = Mock()
            mock_agent.workspace = Path(tmpdir)

            result = behavior._save_to_staging(
                mock_agent,
                behavior_name="test_behavior",
                behavior_code="# Behavior code",
                test_code="# Test code"
            )

            assert "error" not in result
            assert "behavior_file" in result
            assert "test_file" in result

            # Verify files exist
            behavior_file = Path(result["behavior_file"])
            test_file = Path(result["test_file"])

            assert behavior_file.exists()
            assert test_file.exists()
            assert behavior_file.read_text() == "# Behavior code"
            assert test_file.read_text() == "# Test code"

    def test_validate_generated_code_no_validation_behavior(self):
        """Validation fails if ValidationBehavior not available."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()
        mock_agent.behaviors = []

        result = behavior._validate_generated_code(
            mock_agent,
            behavior_name="test_behavior",
            behavior_code="# Code",
            behavior_file="/tmp/test.py"
        )

        assert "error" in result
        assert "ValidationBehavior not available" in result["error"]

    def test_validate_generated_code_success(self):
        """Validation succeeds with valid code."""
        behavior = CreateBehaviorBehavior()

        # Mock agent with validation behavior
        mock_validation = Mock()
        mock_validation.get_name.return_value = "validation"
        mock_validation.dispatch_tool.return_value = {
            "result": {"valid": True}
        }

        mock_agent = Mock()
        mock_agent.behaviors = [mock_validation]
        mock_agent.workspace = Path.cwd()

        # Use absolute path to avoid path resolution issues
        behavior_file = str(Path.cwd() / "test.py")

        result = behavior._validate_generated_code(
            mock_agent,
            behavior_name="test_behavior",
            behavior_code="class TestBehaviorBehavior: pass",
            behavior_file=behavior_file
        )

        assert "error" not in result
        assert result.get("valid") is True

    def test_validate_generated_code_failure(self):
        """Validation fails with invalid code."""
        behavior = CreateBehaviorBehavior()

        # Mock validation behavior returning errors
        mock_validation = Mock()
        mock_validation.get_name.return_value = "validation"

        def mock_dispatch(agent, tool_name, args):
            if tool_name == "validate_python_syntax":
                return {"result": {"valid": False, "error": "Syntax error"}}
            return {"result": {"valid": True}}

        mock_validation.dispatch_tool.side_effect = mock_dispatch

        mock_agent = Mock()
        mock_agent.behaviors = [mock_validation]
        mock_agent.workspace = Path.cwd()

        # Use absolute path
        behavior_file = str(Path.cwd() / "test.py")

        result = behavior._validate_generated_code(
            mock_agent,
            behavior_name="test_behavior",
            behavior_code="invalid code",
            behavior_file=behavior_file
        )

        assert result.get("valid") is False
        assert "error" in result

    def test_test_in_sandbox_no_sandbox_behavior(self):
        """Sandbox test fails if SandboxTestBehavior not available."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()
        mock_agent.behaviors = []

        result = behavior._test_in_sandbox(
            mock_agent,
            behavior_file="/tmp/behavior.py",
            test_file="/tmp/test.py"
        )

        assert "error" in result
        assert "SandboxTestBehavior not available" in result["error"]

    def test_test_in_sandbox_success(self):
        """Sandbox test succeeds with passing tests."""
        behavior = CreateBehaviorBehavior()

        # Mock sandbox behavior
        mock_sandbox = Mock()
        mock_sandbox.get_name.return_value = "sandbox_test"
        mock_sandbox.dispatch_tool.return_value = {
            "success": True,
            "stdout": "Tests passed"
        }

        mock_agent = Mock()
        mock_agent.behaviors = [mock_sandbox]
        mock_agent.workspace = Path.cwd()

        # Use absolute paths
        behavior_file = str(Path.cwd() / "behavior.py")
        test_file = str(Path.cwd() / "test.py")

        result = behavior._test_in_sandbox(
            mock_agent,
            behavior_file=behavior_file,
            test_file=test_file
        )

        assert result.get("success") is True

    def test_safety_mode_dryrun(self):
        """Dryrun mode saves to staging only."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        result = behavior._handle_safety_mode(
            mock_agent,
            safety_mode="dryrun",
            behavior_name="test_behavior",
            behavior_file="/tmp/behavior.py",
            test_file="/tmp/test.py",
            validation_result={"valid": True},
            test_result={"success": True}
        )

        assert result["success"] is True
        assert result["installed"] is False
        assert "dryrun" in result["message"]

    def test_safety_mode_review(self):
        """Review mode saves to staging and returns for approval."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        result = behavior._handle_safety_mode(
            mock_agent,
            safety_mode="review",
            behavior_name="test_behavior",
            behavior_file="/tmp/behavior.py",
            test_file="/tmp/test.py",
            validation_result={"valid": True},
            test_result={"success": True}
        )

        assert result["success"] is True
        assert result["installed"] is False
        assert "review" in result["message"]

    @patch.object(CreateBehaviorBehavior, '_install_behavior')
    def test_safety_mode_auto(self, mock_install):
        """Auto mode installs immediately if valid."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        mock_install.return_value = {
            "success": True,
            "behavior_path": "/behaviors/test.py",
            "test_path": "/tests/test_test.py"
        }

        result = behavior._handle_safety_mode(
            mock_agent,
            safety_mode="auto",
            behavior_name="test_behavior",
            behavior_file="/tmp/behavior.py",
            test_file="/tmp/test.py",
            validation_result={"valid": True},
            test_result={"success": True}
        )

        assert result["success"] is True
        assert result["installed"] is True
        assert "installed successfully" in result["message"]
        mock_install.assert_called_once()

    @patch.object(CreateBehaviorBehavior, '_install_behavior')
    def test_safety_mode_auto_installation_fails(self, mock_install):
        """Auto mode handles installation failure."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        mock_install.return_value = {
            "error": "Installation failed"
        }

        result = behavior._handle_safety_mode(
            mock_agent,
            safety_mode="auto",
            behavior_name="test_behavior",
            behavior_file="/tmp/behavior.py",
            test_file="/tmp/test.py",
            validation_result={"valid": True},
            test_result={"success": True}
        )

        assert result["success"] is False
        assert result["installed"] is False
        assert "error" in result

    def test_safety_mode_strict(self):
        """Strict mode requires manual review."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        result = behavior._handle_safety_mode(
            mock_agent,
            safety_mode="strict",
            behavior_name="test_behavior",
            behavior_file="/tmp/behavior.py",
            test_file="/tmp/test.py",
            validation_result={"valid": True},
            test_result={"success": True}
        )

        assert result["success"] is True
        assert result["installed"] is False
        assert "Strict mode" in result["message"]

    def test_safety_mode_unknown(self):
        """Unknown safety mode returns error."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        result = behavior._handle_safety_mode(
            mock_agent,
            safety_mode="invalid_mode",
            behavior_name="test_behavior",
            behavior_file="/tmp/behavior.py",
            test_file="/tmp/test.py",
            validation_result={"valid": True},
            test_result={"success": True}
        )

        assert result["success"] is False
        assert "Unknown safety mode" in result["error"]

    @patch('utils.installer.install_with_rollback')
    def test_install_behavior_success(self, mock_install):
        """Installation succeeds with valid behavior."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        mock_install.return_value = {
            "success": True,
            "backup_file": None
        }

        result = behavior._install_behavior(
            mock_agent,
            behavior_name="test_behavior",
            behavior_file="/tmp/behavior.py",
            test_file="/tmp/test.py"
        )

        assert result["success"] is True
        assert "behavior_path" in result
        assert "test_path" in result

    @patch('utils.installer.install_with_rollback')
    def test_install_behavior_behavior_install_fails(self, mock_install):
        """Installation fails if behavior file install fails."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        mock_install.return_value = {
            "success": False,
            "error": "Permission denied"
        }

        result = behavior._install_behavior(
            mock_agent,
            behavior_name="test_behavior",
            behavior_file="/tmp/behavior.py",
            test_file="/tmp/test.py"
        )

        assert "error" in result
        assert "Failed to install behavior" in result["error"]

    @patch('utils.installer.install_with_rollback')
    def test_install_behavior_test_install_fails(self, mock_install):
        """Installation fails if test file install fails."""
        behavior = CreateBehaviorBehavior()
        mock_agent = Mock()

        # First call (behavior) succeeds, second call (test) fails
        mock_install.side_effect = [
            {"success": True, "backup_file": None},
            {"success": False, "error": "Permission denied"}
        ]

        result = behavior._install_behavior(
            mock_agent,
            behavior_name="test_behavior",
            behavior_file="/tmp/behavior.py",
            test_file="/tmp/test.py"
        )

        assert "error" in result
        assert "Failed to install test" in result["error"]

    def test_full_workflow_integration(self):
        """Test full workflow with mocked dependencies."""
        behavior = CreateBehaviorBehavior()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock agent with behaviors
            mock_validation = Mock()
            mock_validation.get_name.return_value = "validation"
            mock_validation.dispatch_tool.return_value = {
                "result": {"valid": True}
            }

            mock_sandbox = Mock()
            mock_sandbox.get_name.return_value = "sandbox_test"
            mock_sandbox.dispatch_tool.return_value = {
                "success": True,
                "stdout": "Tests passed"
            }

            mock_agent = Mock()
            mock_agent.workspace = Path(tmpdir)
            mock_agent.behaviors = [mock_validation, mock_sandbox]

            # Mock LLM calls
            with patch('llm_utils.chat_with_inactivity_timeout') as mock_chat:
                mock_chat.return_value = {
                    "message": {
                        "content": "class TestBehaviorBehavior:\n    pass"
                    }
                }

                # Mock template files
                with patch('behaviors.create_behavior.Path') as mock_path_class:
                    mock_template = MagicMock()
                    mock_template.exists.return_value = True
                    mock_template.read_text.return_value = "# Template"
                    mock_path_class.return_value = mock_template

                    # Execute with review mode (no installation)
                    result = behavior._execute_create_behavior(
                        mock_agent,
                        {
                            "behavior_name": "test_behavior",
                            "description": "Test behavior",
                            "tool_specs": [{"name": "test_tool", "description": "Test", "parameters": {}}],
                            "safety_mode": "review"
                        }
                    )

                    assert result["success"] is True
                    assert result["installed"] is False
                    assert "behavior_file" in result
                    assert "test_file" in result

    def test_no_cross_behavior_dependencies(self):
        """Behavior has no cross-behavior dependencies."""
        # Read the source file directly
        from pathlib import Path
        source_file = Path("behaviors/create_behavior.py")
        source = source_file.read_text()

        # Should only import from behaviors.base
        assert "from behaviors.base import" in source

        # Should NOT import specific behaviors directly
        assert "from behaviors.validation import" not in source
        assert "from behaviors.sandbox_test import" not in source

        # Should access behaviors through agent.behaviors list
        assert "agent.behaviors" in source
