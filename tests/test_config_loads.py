"""
Quick test to verify the task executor config loads with new behaviors.
"""
import sys
import yaml
from pathlib import Path

# Add parent directory to path so we can import behaviors
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import behavior classes
from behaviors import (
    DirectoryToolsBehavior,
    ReadFileToolsBehavior,
    WriteFileToolsBehavior,
)

def test_config_loads():
    """Test that task_executor_config.yaml loads successfully."""
    config_path = Path("/workspace/task_executor_config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Check behaviors section exists
    assert "behaviors" in config, "Config should have behaviors section"
    behaviors = config["behaviors"]

    # Find the file tool behaviors
    behavior_types = [b["type"] for b in behaviors]

    print(f"✓ Config loaded successfully")
    print(f"✓ Found {len(behavior_types)} behaviors")
    print(f"  Behavior types: {', '.join(behavior_types)}")

    # Check for our new behaviors
    assert "DirectoryToolsBehavior" in behavior_types, "Should have DirectoryToolsBehavior"
    assert "ReadFileToolsBehavior" in behavior_types, "Should have ReadFileToolsBehavior"
    assert "WriteFileToolsBehavior" in behavior_types, "Should have WriteFileToolsBehavior"

    print(f"✓ All three split behaviors found in config")

    # Verify old FileToolsBehavior is not present
    assert "FileToolsBehavior" not in behavior_types, "Old FileToolsBehavior should not be in config"
    print(f"✓ Old FileToolsBehavior not present (correctly removed)")

    # Test instantiation
    dir_behavior = DirectoryToolsBehavior()
    read_behavior = ReadFileToolsBehavior()
    write_behavior = WriteFileToolsBehavior()

    print(f"✓ All behaviors instantiate successfully")

    # Check tools
    dir_tools = dir_behavior.get_tools()
    read_tools = read_behavior.get_tools()
    write_tools = write_behavior.get_tools()

    assert len(dir_tools) == 1 and dir_tools[0]["function"]["name"] == "list_dir"
    assert len(read_tools) == 1 and read_tools[0]["function"]["name"] == "read_file"
    assert len(write_tools) == 1 and write_tools[0]["function"]["name"] == "write_file"

    print(f"✓ All tools defined correctly")
    print("\n✅ Configuration loads successfully with split behaviors!")

if __name__ == "__main__":
    test_config_loads()
