"""
Integration test for ContextInspectorBehavior.

Tests the behavior in a more realistic scenario with a minimal agent.
"""

import json
import shutil
from pathlib import Path
import pytest


def test_context_inspector_with_base_agent(tmp_path):
    """Test ContextInspectorBehavior with BaseAgent."""
    from base_agent import BaseAgent
    from behaviors.context_inspector import ContextInspectorBehavior

    # Create temp workspace and output dir
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output_dir = tmp_path / "context_inspection"

    # Create a minimal config file
    config_content = """
behaviors:
  - type: ContextInspectorBehavior
    params:
      output_dir: "{output_dir}"
      save_full_context: true
      capture_tools: true
      capture_metrics: true
""".format(output_dir=str(output_dir))

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)

    # Create agent with ContextInspectorBehavior
    agent = BaseAgent(
        name="test_agent",
        workspace=workspace,
        config_file=str(config_file),
        timeout_seconds=10
    )

    # Verify behavior was loaded
    behavior_names = [b.get_name() for b in agent._behaviors]
    assert "context_inspector" in behavior_names

    # Get the context inspector behavior
    context_inspector = None
    for b in agent._behaviors:
        if b.get_name() == "context_inspector":
            context_inspector = b
            break

    assert context_inspector is not None
    assert context_inspector.output_dir == output_dir

    # Manually trigger snapshot to test functionality
    test_context = [
        {"role": "system", "content": "Test system prompt"},
        {"role": "user", "content": "Test user message"}
    ]

    context_inspector.on_initial_context(agent, test_context)

    # Verify snapshot was created
    snapshot_files = list(output_dir.glob("*.json"))
    assert len(snapshot_files) == 1

    # Verify snapshot structure
    snapshot_file = snapshot_files[0]
    assert snapshot_file.name.startswith("test_agent_round_000_initial")

    with open(snapshot_file) as f:
        snapshot = json.load(f)

    # Basic validation
    assert snapshot["agent_name"] == "test_agent"
    assert snapshot["round"] == 0
    assert snapshot["phase"] == "initial"
    assert "context" in snapshot
    assert "tools" in snapshot
    assert "behaviors_loaded" in snapshot
    assert "context_inspector" in snapshot["behaviors_loaded"]
    assert "metrics" in snapshot

    print(f"✅ Integration test passed!")
    print(f"   - Snapshot created: {snapshot_file.name}")
    print(f"   - Behaviors loaded: {snapshot['behaviors_loaded']}")
    print(f"   - Tools captured: {len(snapshot['tools'])}")
    print(f"   - Messages: {snapshot['metrics']['total_messages']}")
    print(f"   - Estimated tokens: {snapshot['metrics']['estimated_tokens']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
