"""
Unit tests for behavior config loading.

Tests the config file loading, behavior instantiation, and registration.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Any

from behaviors import AgentBehavior


# Mock behaviors for testing

class MockBehaviorA(AgentBehavior):
    """Mock behavior for testing."""

    def __init__(self, param1: str = "default", param2: int = 42) -> None:
        self.param1 = param1
        self.param2 = param2

    def get_name(self) -> str:
        return "mock_a"


class MockBehaviorB(AgentBehavior):
    """Mock behavior with tools."""

    def get_name(self) -> str:
        return "mock_b"

    def get_tools(self) -> list[dict[str, Any]]:
        return [{
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "A mock tool",
                "parameters": {"type": "object", "properties": {}}
            }
        }]

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any
    ) -> dict[str, Any]:
        if tool_name == "mock_tool":
            return {"result": "mock_result"}
        return super().dispatch_tool(tool_name, args, **kwargs)


class MockBehaviorC(AgentBehavior):
    """Another mock behavior with tools (for conflict testing)."""

    def get_name(self) -> str:
        return "mock_c"

    def get_tools(self) -> list[dict[str, Any]]:
        return [{
            "type": "function",
            "function": {
                "name": "mock_tool",  # Same name as MockBehaviorB!
                "description": "Conflicting tool",
                "parameters": {"type": "object", "properties": {}}
            }
        }]


# Helper function to simulate config loading

def load_behaviors_from_config_dict(
    config: dict[str, Any],
    behavior_classes: dict[str, type[AgentBehavior]]
) -> list[AgentBehavior]:
    """
    Simulate config loading for testing.

    This mimics what BaseAgent.load_behaviors_from_config() will do.

    Args:
        config: Config dict with 'behaviors' list
        behavior_classes: Map of behavior type name to class

    Returns:
        List of instantiated behaviors
    """
    behaviors = []
    for behavior_spec in config.get("behaviors", []):
        behavior_type = behavior_spec["type"]
        behavior_params = behavior_spec.get("params", {})

        # Get behavior class
        if behavior_type not in behavior_classes:
            raise ValueError(f"Unknown behavior type: {behavior_type}")

        behavior_class = behavior_classes[behavior_type]
        behavior = behavior_class(**behavior_params)
        behaviors.append(behavior)

    return behaviors


def check_tool_conflicts(behaviors: list[AgentBehavior]) -> None:
    """
    Check for tool name conflicts.

    Raises:
        ValueError: If two behaviors provide the same tool name
    """
    tool_registry: dict[str, str] = {}
    for behavior in behaviors:
        for tool in behavior.get_tools():
            tool_name = tool["function"]["name"]
            if tool_name in tool_registry:
                raise ValueError(
                    f"Tool '{tool_name}' already registered by "
                    f"{tool_registry[tool_name]}"
                )
            tool_registry[tool_name] = behavior.get_name()


class TestBehaviorConfigLoading:
    """Test config file loading and behavior instantiation."""

    def test_empty_config(self) -> None:
        """Empty config loads no behaviors."""
        config = {"behaviors": []}
        behaviors = load_behaviors_from_config_dict(config, {})
        assert behaviors == []

    def test_single_behavior_no_params(self) -> None:
        """Load single behavior with no parameters."""
        config = {
            "behaviors": [
                {"type": "MockBehaviorA", "params": {}}
            ]
        }
        behavior_classes = {"MockBehaviorA": MockBehaviorA}
        behaviors = load_behaviors_from_config_dict(config, behavior_classes)

        assert len(behaviors) == 1
        assert isinstance(behaviors[0], MockBehaviorA)
        assert behaviors[0].get_name() == "mock_a"
        assert behaviors[0].param1 == "default"
        assert behaviors[0].param2 == 42

    def test_single_behavior_with_params(self) -> None:
        """Load single behavior with custom parameters."""
        config = {
            "behaviors": [
                {
                    "type": "MockBehaviorA",
                    "params": {
                        "param1": "custom",
                        "param2": 99
                    }
                }
            ]
        }
        behavior_classes = {"MockBehaviorA": MockBehaviorA}
        behaviors = load_behaviors_from_config_dict(config, behavior_classes)

        assert len(behaviors) == 1
        assert behaviors[0].param1 == "custom"
        assert behaviors[0].param2 == 99

    def test_multiple_behaviors(self) -> None:
        """Load multiple behaviors in order."""
        config = {
            "behaviors": [
                {"type": "MockBehaviorA", "params": {"param1": "first"}},
                {"type": "MockBehaviorB", "params": {}},
                {"type": "MockBehaviorA", "params": {"param1": "second"}}
            ]
        }
        behavior_classes = {
            "MockBehaviorA": MockBehaviorA,
            "MockBehaviorB": MockBehaviorB
        }
        behaviors = load_behaviors_from_config_dict(config, behavior_classes)

        assert len(behaviors) == 3
        assert behaviors[0].get_name() == "mock_a"
        assert behaviors[0].param1 == "first"  # type: ignore[attr-defined]
        assert behaviors[1].get_name() == "mock_b"
        assert behaviors[2].get_name() == "mock_a"
        assert behaviors[2].param1 == "second"  # type: ignore[attr-defined]

    def test_unknown_behavior_type(self) -> None:
        """Unknown behavior type raises error."""
        config = {
            "behaviors": [
                {"type": "UnknownBehavior", "params": {}}
            ]
        }
        behavior_classes = {"MockBehaviorA": MockBehaviorA}

        with pytest.raises(ValueError) as exc_info:
            load_behaviors_from_config_dict(config, behavior_classes)
        assert "Unknown behavior type" in str(exc_info.value)
        assert "UnknownBehavior" in str(exc_info.value)

    def test_missing_required_param(self) -> None:
        """Missing required parameter raises error."""
        class RequiredParamBehavior(AgentBehavior):
            def __init__(self, required_param: str) -> None:
                self.required_param = required_param

            def get_name(self) -> str:
                return "required_param"

        config = {
            "behaviors": [
                {"type": "RequiredParamBehavior", "params": {}}
            ]
        }
        behavior_classes = {"RequiredParamBehavior": RequiredParamBehavior}

        with pytest.raises(TypeError):
            load_behaviors_from_config_dict(config, behavior_classes)

    def test_yaml_config_file_loading(self) -> None:
        """Load behaviors from actual YAML file."""
        config_yaml = """
behaviors:
  - type: MockBehaviorA
    params:
      param1: "from_yaml"
      param2: 123
  - type: MockBehaviorB
    params: {}
"""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.yaml',
            delete=False
        ) as f:
            f.write(config_yaml)
            config_path = f.name

        try:
            # Load and parse YAML
            with open(config_path) as f:
                config = yaml.safe_load(f)

            behavior_classes = {
                "MockBehaviorA": MockBehaviorA,
                "MockBehaviorB": MockBehaviorB
            }
            behaviors = load_behaviors_from_config_dict(config, behavior_classes)

            assert len(behaviors) == 2
            assert behaviors[0].param1 == "from_yaml"  # type: ignore[attr-defined]
            assert behaviors[0].param2 == 123  # type: ignore[attr-defined]
            assert behaviors[1].get_name() == "mock_b"
        finally:
            Path(config_path).unlink()

    def test_no_tool_conflicts(self) -> None:
        """No error when behaviors have different tool names."""
        behaviors = [MockBehaviorA(), MockBehaviorB()]
        # Should not raise
        check_tool_conflicts(behaviors)

    def test_tool_name_conflict_detection(self) -> None:
        """Detect when two behaviors provide the same tool name."""
        behaviors = [MockBehaviorB(), MockBehaviorC()]

        with pytest.raises(ValueError) as exc_info:
            check_tool_conflicts(behaviors)
        assert "mock_tool" in str(exc_info.value)
        assert "already registered" in str(exc_info.value)

    def test_behavior_order_preserved(self) -> None:
        """Behaviors are loaded in config order."""
        config = {
            "behaviors": [
                {"type": "MockBehaviorA", "params": {"param2": 1}},
                {"type": "MockBehaviorB", "params": {}},
                {"type": "MockBehaviorA", "params": {"param2": 2}},
                {"type": "MockBehaviorA", "params": {"param2": 3}}
            ]
        }
        behavior_classes = {
            "MockBehaviorA": MockBehaviorA,
            "MockBehaviorB": MockBehaviorB
        }
        behaviors = load_behaviors_from_config_dict(config, behavior_classes)

        assert [b.param2 if hasattr(b, 'param2') else 0 for b in behaviors] == [  # type: ignore[attr-defined]
            1, 0, 2, 3
        ]

    def test_params_field_optional(self) -> None:
        """'params' field is optional (defaults to empty dict)."""
        config = {
            "behaviors": [
                {"type": "MockBehaviorA"}  # No 'params' field
            ]
        }
        behavior_classes = {"MockBehaviorA": MockBehaviorA}
        behaviors = load_behaviors_from_config_dict(config, behavior_classes)

        assert len(behaviors) == 1
        assert behaviors[0].param1 == "default"

    def test_collect_tools_from_multiple_behaviors(self) -> None:
        """Can collect tools from multiple behaviors."""
        behaviors = [
            MockBehaviorA(),  # No tools
            MockBehaviorB(),  # Has 'mock_tool'
        ]

        all_tools = []
        for behavior in behaviors:
            all_tools.extend(behavior.get_tools())

        assert len(all_tools) == 1
        assert all_tools[0]["function"]["name"] == "mock_tool"

    def test_enhance_context_composition(self) -> None:
        """Multiple behaviors can enhance context in sequence."""
        class EnhancerA(AgentBehavior):
            def get_name(self) -> str:
                return "enhancer_a"

            def enhance_context(
                self,
                context: list[dict[str, Any]],
                **kwargs: Any
            ) -> list[dict[str, Any]]:
                if context:
                    context[0]["content"] += " [A]"
                return context

        class EnhancerB(AgentBehavior):
            def get_name(self) -> str:
                return "enhancer_b"

            def enhance_context(
                self,
                context: list[dict[str, Any]],
                **kwargs: Any
            ) -> list[dict[str, Any]]:
                if context:
                    context[0]["content"] += " [B]"
                return context

        behaviors = [EnhancerA(), EnhancerB()]
        context = [{"role": "system", "content": "Start"}]

        # Apply enhancements in order
        for behavior in behaviors:
            context = behavior.enhance_context(context)

        assert context[0]["content"] == "Start [A] [B]"

    def test_real_config_files_are_valid_yaml(self) -> None:
        """Verify actual config files are valid YAML."""
        config_files = [
            "/workspace/task_executor_config.yaml",
            "/workspace/orchestrator_config.yaml",
            "/workspace/architect_config.yaml"
        ]

        for config_file in config_files:
            config_path = Path(config_file)
            if not config_path.exists():
                pytest.skip(f"Config file not found: {config_file}")

            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Basic structure validation
            assert "behaviors" in config, f"{config_file} missing 'behaviors' key"
            assert isinstance(config["behaviors"], list)

            for i, behavior_spec in enumerate(config["behaviors"]):
                assert "type" in behavior_spec, (
                    f"{config_file} behavior {i} missing 'type'"
                )
                # 'params' is optional
                if "params" in behavior_spec:
                    assert isinstance(behavior_spec["params"], dict)
