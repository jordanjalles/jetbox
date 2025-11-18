"""Tests for specialized agents system components."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from behaviors.home_assistant import HomeAssistantBehavior
from behaviors.scheduled_mode import ScheduledModeBehavior
from src.process_manager import ProcessInfo, ProcessManager


class TestScheduledModeBehavior:
    """Test ScheduledModeBehavior."""

    def test_behavior_name(self):
        """Test behavior returns correct name."""
        behavior = ScheduledModeBehavior()
        assert behavior.get_name() == "scheduled_mode"

    def test_initial_context_injection(self):
        """Test scheduled mode instructions are injected."""
        behavior = ScheduledModeBehavior()
        context = [{"role": "system", "content": "Base system prompt"}]

        result = behavior.on_initial_context(None, context)

        assert len(result) == 2
        assert "SCHEDULED MODE" in result[1]["content"]
        assert "Exit cleanly when done" in result[1]["content"]

    def test_round_nudge_near_limit(self):
        """Test nudge is added when approaching max rounds."""
        behavior = ScheduledModeBehavior(max_rounds=10)
        context = []

        # Should not nudge at round 5
        result = behavior.on_round_start(None, 5, context)
        assert len(result) == 0

        # Should nudge at round 8 (max-2)
        result = behavior.on_round_start(None, 8, [])
        assert len(result) == 1
        assert "REMINDER" in result[0]["content"]
        assert "8/10 rounds" in result[0]["content"]

    def test_completion_logging(self):
        """Test completion events are logged."""
        behavior = ScheduledModeBehavior()

        # Should not raise errors
        behavior.on_goal_complete(None, success=True, summary="Task completed")
        behavior.on_goal_complete(None, success=False, summary="Task failed")


class TestHomeAssistantBehavior:
    """Test HomeAssistantBehavior."""

    def test_behavior_name(self):
        """Test behavior returns correct name."""
        behavior = HomeAssistantBehavior()
        assert behavior.get_name() == "home_assistant"

    def test_tools_registration(self):
        """Test all expected tools are registered."""
        behavior = HomeAssistantBehavior()
        tools = behavior.get_tools()

        tool_names = [t["function"]["name"] for t in tools]

        assert "ha_list_devices" in tool_names
        assert "ha_get_state" in tool_names
        assert "ha_call_service" in tool_names
        assert "ha_list_automations" in tool_names
        assert "ha_trigger_automation" in tool_names

    def test_unconfigured_behavior(self):
        """Test behavior handles missing configuration."""
        behavior = HomeAssistantBehavior(api_url=None, access_token=None)

        result = behavior.dispatch_tool(None, "ha_list_devices", {})

        assert "error" in result
        assert "not configured" in result["error"]

    @patch("behaviors.home_assistant.urlopen")
    def test_list_devices(self, mock_urlopen):
        """Test listing devices."""
        behavior = HomeAssistantBehavior(
            api_url="http://test:8123", access_token="test_token"
        )

        # Mock API response
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            [
                {
                    "entity_id": "light.living_room",
                    "state": "on",
                    "attributes": {"friendly_name": "Living Room Light"},
                },
                {
                    "entity_id": "switch.fan",
                    "state": "off",
                    "attributes": {"friendly_name": "Fan"},
                },
            ]
        ).encode("utf-8")
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        mock_urlopen.return_value = mock_response

        result = behavior.dispatch_tool(None, "ha_list_devices", {})

        assert "devices" in result
        assert len(result["devices"]) == 2
        assert result["devices"][0]["entity_id"] == "light.living_room"
        assert result["devices"][1]["entity_id"] == "switch.fan"

    @patch("behaviors.home_assistant.urlopen")
    def test_get_state(self, mock_urlopen):
        """Test getting device state."""
        behavior = HomeAssistantBehavior(
            api_url="http://test:8123", access_token="test_token"
        )

        # Mock API response
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {
                    "friendly_name": "Living Room Light",
                    "brightness": 255,
                },
                "last_changed": "2024-01-15T10:00:00",
                "last_updated": "2024-01-15T10:00:00",
            }
        ).encode("utf-8")
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        mock_urlopen.return_value = mock_response

        result = behavior.dispatch_tool(
            None, "ha_get_state", {"entity_id": "light.living_room"}
        )

        assert result["entity_id"] == "light.living_room"
        assert result["state"] == "on"
        assert result["name"] == "Living Room Light"

    @patch("behaviors.home_assistant.urlopen")
    def test_call_service(self, mock_urlopen):
        """Test calling a service."""
        behavior = HomeAssistantBehavior(
            api_url="http://test:8123", access_token="test_token"
        )

        # Mock API response
        mock_response = Mock()
        mock_response.read.return_value = json.dumps([]).encode("utf-8")
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        mock_urlopen.return_value = mock_response

        result = behavior.dispatch_tool(
            None,
            "ha_call_service",
            {
                "domain": "light",
                "service": "turn_on",
                "entity_id": "light.living_room",
                "data": {"brightness": 255},
            },
        )

        assert result["success"] is True
        assert result["service"] == "light.turn_on"
        assert result["entity_id"] == "light.living_room"


class TestProcessManager:
    """Test ProcessManager."""

    def test_initialization(self):
        """Test process manager initializes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProcessManager(log_dir=Path(tmpdir))

            assert manager.log_dir.exists()
            assert len(manager.processes) == 0

    def test_spawn_agent(self):
        """Test spawning an agent process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProcessManager(log_dir=Path(tmpdir))

            # Spawn a simple process (just sleep)
            with patch("subprocess.Popen") as mock_popen:
                mock_process = Mock()
                mock_process.pid = 12345
                mock_popen.return_value = mock_process

                proc_info = manager.spawn_agent(
                    agent_name="test_agent",
                    config_file="config/agents/test.yaml",
                    goal="Test goal",
                )

                assert proc_info.agent_name == "test_agent"
                assert proc_info.pid == 12345
                assert proc_info.status == "running"
                assert "test_agent" in manager.processes

    def test_check_process_status(self):
        """Test checking process status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProcessManager(log_dir=Path(tmpdir))

            # Create a mock process
            mock_process = Mock()
            mock_process.poll.return_value = None  # Still running

            from datetime import datetime

            proc_info = ProcessInfo(
                agent_name="test",
                process=mock_process,
                pid=123,
                started_at=datetime.now(),
                last_run=datetime.now(),
                next_run=None,
                status="running",
                exit_code=None,
                crash_count=0,
                log_path=Path(tmpdir) / "test.log",
            )

            manager.processes["test"] = proc_info

            status = manager.check_process_status("test")

            assert status == "running"

            # Simulate process completion
            mock_process.poll.return_value = 0

            status = manager.check_process_status("test")

            assert status == "completed"
            assert proc_info.exit_code == 0

    def test_stop_process(self):
        """Test stopping a process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProcessManager(log_dir=Path(tmpdir))

            # Create a mock process
            mock_process = Mock()
            mock_process.pid = 123

            from datetime import datetime

            proc_info = ProcessInfo(
                agent_name="test",
                process=mock_process,
                pid=123,
                started_at=datetime.now(),
                last_run=datetime.now(),
                next_run=None,
                status="running",
                exit_code=None,
                crash_count=0,
                log_path=Path(tmpdir) / "test.log",
            )

            manager.processes["test"] = proc_info

            result = manager.stop_process("test")

            assert result is True
            mock_process.terminate.assert_called_once()
            assert proc_info.status == "stopped"

    def test_tail_logs(self):
        """Test tailing log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProcessManager(log_dir=Path(tmpdir))

            # Create a log file
            log_path = Path(tmpdir) / "test.log"
            log_path.write_text("Line 1\nLine 2\nLine 3\n")

            from datetime import datetime

            proc_info = ProcessInfo(
                agent_name="test",
                process=None,
                pid=None,
                started_at=datetime.now(),
                last_run=datetime.now(),
                next_run=None,
                status="running",
                exit_code=None,
                crash_count=0,
                log_path=log_path,
            )

            manager.processes["test"] = proc_info

            lines = manager.tail_logs("test", lines=2)

            assert len(lines) == 2
            assert "Line 2" in lines[0]
            assert "Line 3" in lines[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
