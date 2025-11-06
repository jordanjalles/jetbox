"""
Tests for CreateAgentBehavior.

Test coverage:
- Basic tests (3 tests)
- Parameter validation (5 tests)
- Config generation (3 tests)
- Validation integration (3 tests)
- Team integration (3 tests)
- Safety modes (4 tests)
- Installation (2 tests)
- Error handling (3 tests)
- Full workflow (2 tests)

Total: 28 tests
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import yaml
from behaviors.create_agent import CreateAgentBehavior


class TestCreateAgentBehavior:
    """Test suite for CreateAgentBehavior."""

    # ============================================================
    # Basic Tests (3 tests)
    # ============================================================

    def test_get_name(self):
        """Behavior returns correct identifier."""
        behavior = CreateAgentBehavior()
        assert behavior.get_name() == "create_agent"

    def test_initialization(self):
        """Behavior initializes without errors."""
        behavior = CreateAgentBehavior()
        assert behavior is not None
        assert behavior.staging_dir == ".agent_generated/staging"
        assert behavior.default_safety_mode == "review"

    def test_tool_schema(self):
        """Tool schemas are well-formed."""
        behavior = CreateAgentBehavior()
        tools = behavior.get_tools()

        assert len(tools) == 1
        tool = tools[0]

        assert tool["type"] == "function"
        assert tool["function"]["name"] == "create_agent"
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]

        # Check required parameters
        params = tool["function"]["parameters"]
        assert "agent_name" in params["properties"]
        assert "role" in params["properties"]
        assert "description" in params["properties"]
        assert "behaviors" in params["properties"]
        assert "safety_mode" in params["properties"]

        assert "agent_name" in params["required"]
        assert "role" in params["required"]
        assert "description" in params["required"]
        assert "behaviors" in params["required"]

    # ============================================================
    # Parameter Validation (5 tests)
    # ============================================================

    def test_missing_agent_name(self):
        """Missing agent_name parameter is rejected."""
        behavior = CreateAgentBehavior()
        result = behavior._validate_and_extract_params({
            "role": "Test Agent",
            "description": "A test agent",
            "behaviors": ["ChatbotBehavior"]
        })
        assert "error" in result
        assert "agent_name is required" in result["error"]

    def test_missing_role(self):
        """Missing role parameter is rejected."""
        behavior = CreateAgentBehavior()
        result = behavior._validate_and_extract_params({
            "agent_name": "test-agent",
            "description": "A test agent",
            "behaviors": ["ChatbotBehavior"]
        })
        assert "error" in result
        assert "role is required" in result["error"]

    def test_missing_description(self):
        """Missing description parameter is rejected."""
        behavior = CreateAgentBehavior()
        result = behavior._validate_and_extract_params({
            "agent_name": "test-agent",
            "role": "Test Agent",
            "behaviors": ["ChatbotBehavior"]
        })
        assert "error" in result
        assert "description is required" in result["error"]

    def test_missing_behaviors(self):
        """Missing behaviors parameter is rejected."""
        behavior = CreateAgentBehavior()
        result = behavior._validate_and_extract_params({
            "agent_name": "test-agent",
            "role": "Test Agent",
            "description": "A test agent"
        })
        assert "error" in result
        assert "behaviors must be a non-empty list" in result["error"]

    def test_valid_parameters(self):
        """Valid parameters are accepted."""
        behavior = CreateAgentBehavior()
        result = behavior._validate_and_extract_params({
            "agent_name": "test-agent",
            "role": "Test Agent",
            "description": "A test agent for testing",
            "behaviors": ["ChatbotBehavior", "FileToolsBehavior"]
        })
        assert "params" in result
        assert result["params"]["agent_name"] == "test-agent"
        assert result["params"]["role"] == "Test Agent"
        assert result["params"]["description"] == "A test agent for testing"
        assert result["params"]["behaviors"] == ["ChatbotBehavior", "FileToolsBehavior"]

    # ============================================================
    # Config Generation (3 tests)
    # ============================================================

    def test_config_generation_success(self):
        """Config generation succeeds with valid LLM response."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()
        mock_agent.workspace = Path("/tmp/test")

        template_content = "role: {ROLE}\nblurb: {BLURB}"
        generated_yaml = "role: Test Agent\nblurb: This is a test agent."

        with patch('llm_utils.chat_with_inactivity_timeout') as mock_chat:
            mock_chat.return_value = {
                "message": {"content": generated_yaml}
            }

            result = behavior._generate_agent_config(
                mock_agent,
                "test-agent",
                "Test Agent",
                "A test agent for testing purposes.",
                ["ChatbotBehavior"],
                [],
                {},
                template_content
            )

            assert "config" in result
            assert "Test Agent" in result["config"]

    def test_config_generation_extracts_yaml_from_markdown(self):
        """Config generation extracts YAML from markdown code blocks."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()
        mock_agent.workspace = Path("/tmp/test")

        template_content = "role: {ROLE}"
        generated_yaml_with_markdown = "```yaml\nrole: Test Agent\n```"

        with patch('llm_utils.chat_with_inactivity_timeout') as mock_chat:
            mock_chat.return_value = {
                "message": {"content": generated_yaml_with_markdown}
            }

            result = behavior._generate_agent_config(
                mock_agent,
                "test-agent",
                "Test Agent",
                "A test agent.",
                ["ChatbotBehavior"],
                [],
                {},
                template_content
            )

            assert "config" in result
            assert result["config"] == "role: Test Agent"

    def test_config_generation_llm_failure(self):
        """Config generation handles LLM failures gracefully."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()

        template_content = "role: {ROLE}"

        with patch('llm_utils.chat_with_inactivity_timeout') as mock_chat:
            mock_chat.side_effect = Exception("LLM timeout")

            result = behavior._generate_agent_config(
                mock_agent,
                "test-agent",
                "Test Agent",
                "A test agent.",
                ["ChatbotBehavior"],
                [],
                {},
                template_content
            )

            assert "error" in result
            assert "Config generation failed" in result["error"]

    # ============================================================
    # Validation Integration (3 tests)
    # ============================================================

    def test_yaml_syntax_validation(self):
        """YAML syntax validation integrates with ValidationBehavior."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()

        mock_validation_behavior = Mock()
        mock_validation_behavior.get_name.return_value = "validation"
        mock_validation_behavior.dispatch_tool.return_value = {
            "result": {"valid": True}
        }

        mock_agent.behaviors = [mock_validation_behavior]

        yaml_content = "role: Test Agent\nblurb: Test"

        result = behavior._validate_yaml_syntax(mock_agent, yaml_content)

        assert result.get("valid") == True
        mock_validation_behavior.dispatch_tool.assert_called_once()

    def test_agent_config_validation(self):
        """Agent config validation integrates with ValidationBehavior."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()

        mock_validation_behavior = Mock()
        mock_validation_behavior.get_name.return_value = "validation"
        mock_validation_behavior.dispatch_tool.return_value = {
            "result": {"valid": True}
        }

        mock_agent.behaviors = [mock_validation_behavior]

        yaml_content = "role: Test Agent\nbehaviors:\n  - type: ChatbotBehavior"

        result = behavior._validate_agent_config(mock_agent, yaml_content)

        assert result.get("valid") == True
        mock_validation_behavior.dispatch_tool.assert_called_once()

    def test_dag_validation(self):
        """DAG validation integrates with ValidationBehavior."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()

        mock_validation_behavior = Mock()
        mock_validation_behavior.get_name.return_value = "validation"
        mock_validation_behavior.dispatch_tool.return_value = {
            "result": {"valid": True}
        }

        mock_agent.behaviors = [mock_validation_behavior]

        team_config = {
            "agents": {
                "test-agent": {"can_delegate_to": []}
            }
        }

        result = behavior._validate_agent_dag(mock_agent, team_config)

        assert result.get("valid") == True
        mock_validation_behavior.dispatch_tool.assert_called_once()

    # ============================================================
    # Team Integration (3 tests)
    # ============================================================

    def test_team_integration_success(self):
        """Agent successfully integrates with existing team."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()
        mock_agent.workspace = Path("/tmp/test")

        team_yaml = """
name: Test Team
agents:
  orchestrator:
    class: OrchestratorAgent
    config: orchestrator
    can_delegate_to: []
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            team_file = Path(tmpdir) / "config" / "teams" / "test.yaml"
            team_file.parent.mkdir(parents=True, exist_ok=True)
            team_file.write_text(team_yaml)

            mock_agent.workspace = Path(tmpdir)

            # Mock validation behavior
            mock_validation_behavior = Mock()
            mock_validation_behavior.get_name.return_value = "validation"
            mock_validation_behavior.dispatch_tool.return_value = {
                "result": {"valid": True}
            }
            mock_agent.behaviors = [mock_validation_behavior]

            result = behavior._integrate_with_team(
                mock_agent,
                "test-agent",
                "test",
                []
            )

            assert result.get("success") == True
            assert "team_file" in result

    def test_team_integration_agent_already_exists(self):
        """Team integration fails if agent already exists."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()

        team_yaml = """
name: Test Team
agents:
  test-agent:
    class: TestAgentAgent
    config: test-agent
    can_delegate_to: []
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            team_file = Path(tmpdir) / "config" / "teams" / "test.yaml"
            team_file.parent.mkdir(parents=True, exist_ok=True)
            team_file.write_text(team_yaml)

            mock_agent.workspace = Path(tmpdir)

            result = behavior._integrate_with_team(
                mock_agent,
                "test-agent",
                "test",
                []
            )

            assert "error" in result
            assert "already exists" in result["error"]

    def test_team_integration_dag_cycle_detection(self):
        """Team integration detects DAG cycles."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()

        team_yaml = """
name: Test Team
agents:
  orchestrator:
    class: OrchestratorAgent
    config: orchestrator
    can_delegate_to: []
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            team_file = Path(tmpdir) / "config" / "teams" / "test.yaml"
            team_file.parent.mkdir(parents=True, exist_ok=True)
            team_file.write_text(team_yaml)

            mock_agent.workspace = Path(tmpdir)

            # Mock validation behavior that detects cycle
            mock_validation_behavior = Mock()
            mock_validation_behavior.get_name.return_value = "validation"
            mock_validation_behavior.dispatch_tool.return_value = {
                "error": "Cycle detected in delegation graph"
            }
            mock_agent.behaviors = [mock_validation_behavior]

            result = behavior._integrate_with_team(
                mock_agent,
                "test-agent",
                "test",
                ["orchestrator"]
            )

            assert "error" in result

    # ============================================================
    # Safety Modes (4 tests)
    # ============================================================

    def test_safety_mode_dryrun(self):
        """Dryrun mode saves to staging only."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()

        result = behavior._handle_safety_mode(
            mock_agent,
            "dryrun",
            "test-agent",
            "/tmp/test-agent.yaml",
            None,
            {"valid": True},
            {"valid": True},
            None
        )

        assert result["success"] == True
        assert result["installed"] == False
        assert "dryrun mode" in result["message"]

    def test_safety_mode_review(self):
        """Review mode validates and returns paths."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()

        result = behavior._handle_safety_mode(
            mock_agent,
            "review",
            "test-agent",
            "/tmp/test-agent.yaml",
            None,
            {"valid": True},
            {"valid": True},
            None
        )

        assert result["success"] == True
        assert result["installed"] == False
        assert "Ready for review" in result["message"]
        assert result["agent_config_file"] == "/tmp/test-agent.yaml"

    def test_safety_mode_auto(self):
        """Auto mode installs if validation passes."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()

        with patch.object(behavior, '_install_agent', return_value={"success": True}):
            result = behavior._handle_safety_mode(
                mock_agent,
                "auto",
                "test-agent",
                "/tmp/test-agent.yaml",
                None,
                {"valid": True},
                {"valid": True},
                None
            )

            assert result["success"] == True
            assert result["installed"] == True
            assert "installed successfully" in result["message"]

    def test_safety_mode_strict(self):
        """Strict mode requires manual review."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()

        result = behavior._handle_safety_mode(
            mock_agent,
            "strict",
            "test-agent",
            "/tmp/test-agent.yaml",
            None,
            {"valid": True},
            {"valid": True},
            None
        )

        assert result["success"] == True
        assert result["installed"] == False
        assert "manual review" in result["message"]

    # ============================================================
    # Installation (2 tests)
    # ============================================================

    def test_install_agent_success(self):
        """Agent config installs successfully."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            staging_file = Path(tmpdir) / "staging" / "test-agent.yaml"
            staging_file.parent.mkdir(parents=True, exist_ok=True)
            staging_file.write_text("role: Test")

            with patch('utils.installer.install_with_rollback') as mock_install:
                mock_install.return_value = {
                    "success": True,
                    "backup_file": None
                }

                result = behavior._install_agent(
                    mock_agent,
                    "test-agent",
                    str(staging_file),
                    None
                )

                assert result["success"] == True
                assert "agent_config_path" in result
                mock_install.assert_called_once()

    def test_install_agent_with_team(self):
        """Agent and team configs install together."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            staging_agent_file = Path(tmpdir) / "staging" / "test-agent.yaml"
            staging_agent_file.parent.mkdir(parents=True, exist_ok=True)
            staging_agent_file.write_text("role: Test")

            staging_team_file = Path(tmpdir) / "staging" / "test.yaml"
            staging_team_file.write_text("name: Test Team")

            with patch('utils.installer.install_with_rollback') as mock_install:
                mock_install.return_value = {
                    "success": True,
                    "backup_file": None
                }

                result = behavior._install_agent(
                    mock_agent,
                    "test-agent",
                    str(staging_agent_file),
                    str(staging_team_file)
                )

                assert result["success"] == True
                assert "agent_config_path" in result
                assert "team_config_path" in result
                assert mock_install.call_count == 2

    # ============================================================
    # Error Handling (3 tests)
    # ============================================================

    def test_missing_template_file(self):
        """Missing template file returns error."""
        behavior = CreateAgentBehavior()

        with patch.object(Path, 'exists', return_value=False):
            result = behavior._load_template()

            assert "error" in result
            assert "Template not found" in result["error"]

    def test_validation_behavior_not_available(self):
        """Missing ValidationBehavior returns error."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()
        mock_agent.behaviors = []

        result = behavior._validate_yaml_syntax(mock_agent, "role: Test")

        assert "error" in result
        assert "ValidationBehavior not available" in result["error"]

    def test_installation_failure(self):
        """Installation failures are handled gracefully."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()

        with patch('utils.installer.install_with_rollback') as mock_install:
            mock_install.return_value = {
                "success": False,
                "error": "Permission denied"
            }

            result = behavior._install_agent(
                mock_agent,
                "test-agent",
                "/tmp/test-agent.yaml",
                None
            )

            assert "error" in result
            assert "Permission denied" in result["error"]

    # ============================================================
    # Full Workflow (2 tests)
    # ============================================================

    def test_end_to_end_agent_generation(self):
        """End-to-end agent generation workflow."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()
        mock_agent.workspace = Path("/tmp/test")

        # Mock all dependencies
        with patch.object(behavior, '_load_template', return_value={"template": "role: {ROLE}"}), \
             patch.object(behavior, '_generate_agent_config', return_value={"config": "role: Test Agent"}), \
             patch.object(behavior, '_validate_yaml_syntax', return_value={"valid": True}), \
             patch.object(behavior, '_validate_agent_config', return_value={"valid": True}), \
             patch.object(behavior, '_save_to_staging', return_value={"agent_config_file": "/tmp/test-agent.yaml"}), \
             patch.object(behavior, '_handle_safety_mode', return_value={"success": True}):

            params = {
                "agent_name": "test-agent",
                "role": "Test Agent",
                "description": "A test agent",
                "behaviors": ["ChatbotBehavior"],
                "system_prompt_guidelines": [],
                "delegation_tool_params": {},
                "add_to_team": "",
                "can_delegate_to": [],
                "safety_mode": "review"
            }

            result = behavior._run_agent_generation_workflow(mock_agent, params)

            assert result["success"] == True

    def test_end_to_end_agent_with_team_integration(self):
        """End-to-end agent generation with team integration."""
        behavior = CreateAgentBehavior()
        mock_agent = Mock()
        mock_agent.workspace = Path("/tmp/test")

        # Mock all dependencies
        with patch.object(behavior, '_load_template', return_value={"template": "role: {ROLE}"}), \
             patch.object(behavior, '_generate_agent_config', return_value={"config": "role: Test Agent"}), \
             patch.object(behavior, '_validate_yaml_syntax', return_value={"valid": True}), \
             patch.object(behavior, '_validate_agent_config', return_value={"valid": True}), \
             patch.object(behavior, '_save_to_staging', return_value={"agent_config_file": "/tmp/test-agent.yaml"}), \
             patch.object(behavior, '_integrate_with_team', return_value={"success": True, "team_file": "/tmp/test.yaml"}), \
             patch.object(behavior, '_handle_safety_mode', return_value={"success": True}):

            params = {
                "agent_name": "test-agent",
                "role": "Test Agent",
                "description": "A test agent",
                "behaviors": ["ChatbotBehavior"],
                "system_prompt_guidelines": [],
                "delegation_tool_params": {},
                "add_to_team": "default",
                "can_delegate_to": ["task-executor"],
                "safety_mode": "review"
            }

            result = behavior._run_agent_generation_workflow(mock_agent, params)

            assert result["success"] == True
