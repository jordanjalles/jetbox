"""
Tests for ContextInspectorBehavior.

Verifies that the behavior:
- Captures snapshots without modifying context
- Saves valid JSON files
- Calculates metrics correctly
- Handles edge cases gracefully
"""

import json
import shutil
from pathlib import Path
import pytest
from behaviors.context_inspector import ContextInspectorBehavior


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name="test_agent"):
        self.name = name
        self._behaviors = []

    def get_tools(self):
        """Return mock tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "mock_tool",
                    "description": "A mock tool for testing",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arg": {"type": "string"}
                        },
                        "required": ["arg"]
                    }
                }
            }
        ]


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "test_context_inspection"
    output_dir.mkdir()
    yield output_dir
    # Cleanup
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def behavior(temp_output_dir):
    """Create ContextInspectorBehavior instance."""
    return ContextInspectorBehavior(
        output_dir=str(temp_output_dir),
        save_full_context=True,
        compress_large_contexts=False,
        capture_tools=True,
        capture_metrics=True
    )


@pytest.fixture
def mock_agent():
    """Create mock agent."""
    return MockAgent(name="test_agent")


@pytest.fixture
def sample_context():
    """Create sample context."""
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Follow instructions carefully."
        },
        {
            "role": "user",
            "content": "Please help me with a task."
        }
    ]


def test_behavior_name(behavior):
    """Test that behavior returns correct name."""
    assert behavior.get_name() == "context_inspector"


def test_on_initial_context_returns_unchanged(behavior, mock_agent, sample_context):
    """Test that on_initial_context returns context unchanged."""
    original_context = sample_context.copy()
    result = behavior.on_initial_context(mock_agent, sample_context)

    # Context should be unchanged (pure observer)
    assert result == original_context
    assert len(result) == len(original_context)


def test_on_initial_context_creates_snapshot(behavior, mock_agent, sample_context, temp_output_dir):
    """Test that on_initial_context creates a snapshot file."""
    behavior.on_initial_context(mock_agent, sample_context)

    # Check snapshot file was created
    snapshot_files = list(temp_output_dir.glob("*.json"))
    assert len(snapshot_files) == 1

    # Check filename format
    snapshot_file = snapshot_files[0]
    assert snapshot_file.name == "test_agent_round_000_initial.json"


def test_snapshot_structure(behavior, mock_agent, sample_context, temp_output_dir):
    """Test that snapshot has correct structure."""
    behavior.on_initial_context(mock_agent, sample_context)

    # Load snapshot
    snapshot_file = temp_output_dir / "test_agent_round_000_initial.json"
    with open(snapshot_file) as f:
        snapshot = json.load(f)

    # Verify structure
    assert snapshot["agent_name"] == "test_agent"
    assert snapshot["round"] == 0
    assert snapshot["phase"] == "initial"
    assert "timestamp" in snapshot
    assert isinstance(snapshot["timestamp"], (int, float))

    # Verify context is captured
    assert "context" in snapshot
    assert len(snapshot["context"]) == 2
    assert snapshot["context"][0]["role"] == "system"
    assert snapshot["context"][1]["role"] == "user"

    # Verify tools are captured
    assert "tools" in snapshot
    assert len(snapshot["tools"]) == 1
    assert snapshot["tools"][0]["function"]["name"] == "mock_tool"

    # Verify behaviors are captured
    assert "behaviors_loaded" in snapshot
    assert isinstance(snapshot["behaviors_loaded"], list)

    # Verify metrics are captured
    assert "metrics" in snapshot
    assert "total_messages" in snapshot["metrics"]
    assert "system_prompt_length" in snapshot["metrics"]
    assert "total_context_length" in snapshot["metrics"]
    assert "tool_count" in snapshot["metrics"]
    assert "tool_definition_length" in snapshot["metrics"]
    assert "estimated_tokens" in snapshot["metrics"]


def test_metrics_calculation(behavior, mock_agent, sample_context, temp_output_dir):
    """Test that metrics are calculated correctly."""
    behavior.on_initial_context(mock_agent, sample_context)

    # Load snapshot
    snapshot_file = temp_output_dir / "test_agent_round_000_initial.json"
    with open(snapshot_file) as f:
        snapshot = json.load(f)

    metrics = snapshot["metrics"]

    # Verify counts
    assert metrics["total_messages"] == 2
    assert metrics["tool_count"] == 1

    # Verify lengths are positive
    assert metrics["system_prompt_length"] > 0
    assert metrics["total_context_length"] > 0
    assert metrics["tool_definition_length"] > 0
    assert metrics["estimated_tokens"] > 0

    # Verify total >= system
    assert metrics["total_context_length"] >= metrics["system_prompt_length"]


def test_on_round_start_creates_snapshot(behavior, mock_agent, sample_context, temp_output_dir):
    """Test that on_round_start creates a snapshot file."""
    # First capture initial context to set agent name
    behavior.on_initial_context(mock_agent, sample_context)

    # Then capture round 1
    behavior.on_round_start(mock_agent, 1, sample_context)

    # Check snapshot files
    snapshot_files = sorted(temp_output_dir.glob("*.json"))
    assert len(snapshot_files) == 2

    # Check filenames
    assert snapshot_files[0].name == "test_agent_round_000_initial.json"
    assert snapshot_files[1].name == "test_agent_round_001_pre_llm.json"


def test_on_round_start_returns_unchanged(behavior, mock_agent, sample_context):
    """Test that on_round_start returns context unchanged."""
    # Set agent name first
    behavior.on_initial_context(mock_agent, sample_context)

    original_context = sample_context.copy()
    result = behavior.on_round_start(mock_agent, 1, sample_context)

    # Context should be unchanged (pure observer)
    assert result == original_context
    assert len(result) == len(original_context)


def test_multiple_rounds(behavior, mock_agent, sample_context, temp_output_dir):
    """Test capturing multiple rounds."""
    # Initial context
    behavior.on_initial_context(mock_agent, sample_context)

    # Multiple rounds
    for round_num in range(1, 6):
        behavior.on_round_start(mock_agent, round_num, sample_context)

    # Check all snapshots created
    snapshot_files = sorted(temp_output_dir.glob("*.json"))
    assert len(snapshot_files) == 6  # 1 initial + 5 rounds

    # Verify round numbers
    for i, snapshot_file in enumerate(snapshot_files):
        with open(snapshot_file) as f:
            snapshot = json.load(f)
        assert snapshot["round"] == i


def test_empty_context(behavior, mock_agent, temp_output_dir):
    """Test handling of empty context."""
    empty_context = []

    # Should not crash
    result = behavior.on_initial_context(mock_agent, empty_context)
    assert result == empty_context

    # Snapshot should still be created
    snapshot_files = list(temp_output_dir.glob("*.json"))
    assert len(snapshot_files) == 1


def test_large_context(behavior, mock_agent, temp_output_dir):
    """Test handling of large context."""
    large_context = [
        {"role": "system", "content": "System prompt " * 1000},
        {"role": "user", "content": "User message " * 1000},
    ]

    # Should not crash
    result = behavior.on_initial_context(mock_agent, large_context)
    assert result == large_context

    # Snapshot should be created
    snapshot_files = list(temp_output_dir.glob("*.json"))
    assert len(snapshot_files) == 1

    # Verify metrics reflect large size
    with open(snapshot_files[0]) as f:
        snapshot = json.load(f)

    assert snapshot["metrics"]["total_context_length"] > 10000


def test_compress_large_contexts(temp_output_dir, mock_agent):
    """Test that large contexts are compressed when configured."""
    behavior = ContextInspectorBehavior(
        output_dir=str(temp_output_dir),
        compress_large_contexts=True
    )

    large_context = [
        {"role": "system", "content": "x" * 10000},
    ]

    behavior.on_initial_context(mock_agent, large_context)

    # Load snapshot
    snapshot_files = list(temp_output_dir.glob("*.json"))
    with open(snapshot_files[0]) as f:
        snapshot = json.load(f)

    # Content should be truncated
    assert "_truncated" in snapshot["context"][0]
    assert len(snapshot["context"][0]["content"]) < 10000


def test_save_full_context_disabled(temp_output_dir, mock_agent, sample_context):
    """Test that context is not saved when save_full_context=False."""
    behavior = ContextInspectorBehavior(
        output_dir=str(temp_output_dir),
        save_full_context=False
    )

    behavior.on_initial_context(mock_agent, sample_context)

    # Load snapshot
    snapshot_files = list(temp_output_dir.glob("*.json"))
    with open(snapshot_files[0]) as f:
        snapshot = json.load(f)

    # Context should not be included
    assert "context" not in snapshot
    # But count should be
    assert "context_message_count" in snapshot
    assert snapshot["context_message_count"] == 2


def test_capture_tools_disabled(temp_output_dir, mock_agent, sample_context):
    """Test that tools are not captured when capture_tools=False."""
    behavior = ContextInspectorBehavior(
        output_dir=str(temp_output_dir),
        capture_tools=False
    )

    behavior.on_initial_context(mock_agent, sample_context)

    # Load snapshot
    snapshot_files = list(temp_output_dir.glob("*.json"))
    with open(snapshot_files[0]) as f:
        snapshot = json.load(f)

    # Tools should not be included
    assert "tools" not in snapshot


def test_capture_metrics_disabled(temp_output_dir, mock_agent, sample_context):
    """Test that metrics are not captured when capture_metrics=False."""
    behavior = ContextInspectorBehavior(
        output_dir=str(temp_output_dir),
        capture_metrics=False
    )

    behavior.on_initial_context(mock_agent, sample_context)

    # Load snapshot
    snapshot_files = list(temp_output_dir.glob("*.json"))
    with open(snapshot_files[0]) as f:
        snapshot = json.load(f)

    # Metrics should not be included
    assert "metrics" not in snapshot


def test_agent_without_behaviors(temp_output_dir, sample_context):
    """Test handling agent without _behaviors attribute."""
    behavior = ContextInspectorBehavior(output_dir=str(temp_output_dir))

    # Agent without _behaviors
    class MinimalAgent:
        name = "minimal"
        def get_tools(self):
            return []

    agent = MinimalAgent()

    # Should not crash
    result = behavior.on_initial_context(agent, sample_context)
    assert result == sample_context

    # Snapshot should be created
    snapshot_files = list(temp_output_dir.glob("*.json"))
    assert len(snapshot_files) == 1


def test_agent_without_tools(temp_output_dir, sample_context):
    """Test handling agent without get_tools method."""
    behavior = ContextInspectorBehavior(output_dir=str(temp_output_dir))

    # Agent without get_tools
    class MinimalAgent:
        name = "minimal"
        _behaviors = []

    agent = MinimalAgent()

    # Should not crash
    result = behavior.on_initial_context(agent, sample_context)
    assert result == sample_context

    # Snapshot should be created
    snapshot_files = list(temp_output_dir.glob("*.json"))
    assert len(snapshot_files) == 1

    # Tools should be empty list
    with open(snapshot_files[0]) as f:
        snapshot = json.load(f)
    assert snapshot["tools"] == []


def test_no_instructions(behavior):
    """Test that behavior returns empty instructions."""
    # This is an observer behavior, shouldn't add instructions
    assert behavior.get_instructions() == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
