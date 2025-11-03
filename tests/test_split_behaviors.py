"""
Test script to verify split behaviors work independently.

Tests:
1. DirectoryToolsBehavior - list_dir functionality
2. ReadFileToolsBehavior - read_file functionality
3. WriteFileToolsBehavior - write_file functionality
"""
import sys
import tempfile
import os
from pathlib import Path

# Add parent directory to path so we can import behaviors
sys.path.insert(0, str(Path(__file__).parent.parent))

from behaviors.directory_tools import DirectoryToolsBehavior
from behaviors.read_file_tools import ReadFileToolsBehavior
from behaviors.write_file_tools import WriteFileToolsBehavior


def test_directory_tools():
    """Test DirectoryToolsBehavior independently."""
    print("\n=== Testing DirectoryToolsBehavior ===")

    # Create temp directory with some files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create test files
        (tmppath / "file1.txt").write_text("test1")
        (tmppath / "file2.txt").write_text("test2")
        (tmppath / "subdir").mkdir()

        # Initialize behavior
        behavior = DirectoryToolsBehavior()

        # Test get_name
        assert behavior.get_name() == "directory_tools", "Behavior name should be 'directory_tools'"
        print("✓ get_name() works")

        # Test get_tools
        tools = behavior.get_tools()
        assert len(tools) == 1, "Should provide 1 tool"
        assert tools[0]["function"]["name"] == "list_dir", "Tool should be list_dir"
        print("✓ get_tools() works")

        # Test dispatch_tool - list current directory
        result = behavior.dispatch_tool(
            "list_dir",
            {"path": str(tmppath)}
        )
        assert isinstance(result, list), "Result should be a list"
        assert "file1.txt" in result, "Should list file1.txt"
        assert "file2.txt" in result, "Should list file2.txt"
        assert "subdir" in result, "Should list subdir"
        print(f"✓ dispatch_tool() works - found {len(result)} items")

        # Test error handling - non-existent directory
        result = behavior.dispatch_tool(
            "list_dir",
            {"path": str(tmppath / "nonexistent")}
        )
        assert "__error__" in result[0], "Should return error for non-existent directory"
        print("✓ Error handling works")

    print("✅ DirectoryToolsBehavior test PASSED")


def test_read_file_tools():
    """Test ReadFileToolsBehavior independently."""
    print("\n=== Testing ReadFileToolsBehavior ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        test_file = tmppath / "test.txt"
        test_content = "Hello, World!\nThis is a test file."
        test_file.write_text(test_content)

        # Initialize behavior
        behavior = ReadFileToolsBehavior()

        # Test get_name
        assert behavior.get_name() == "read_file_tools", "Behavior name should be 'read_file_tools'"
        print("✓ get_name() works")

        # Test get_tools
        tools = behavior.get_tools()
        assert len(tools) == 1, "Should provide 1 tool"
        assert tools[0]["function"]["name"] == "read_file", "Tool should be read_file"
        print("✓ get_tools() works")

        # Test dispatch_tool - read file
        result = behavior.dispatch_tool(
            "read_file",
            {"path": str(test_file)}
        )
        assert result == test_content, "Should read correct content"
        print(f"✓ dispatch_tool() works - read {len(result)} chars")

        # Test max_size parameter
        result = behavior.dispatch_tool(
            "read_file",
            {"path": str(test_file), "max_size": 10}
        )
        assert len(result) > 10, "Should truncate and add warning"
        assert "TRUNCATED" in result, "Should include truncation warning"
        print("✓ max_size parameter works")

    print("✅ ReadFileToolsBehavior test PASSED")


def test_write_file_tools():
    """Test WriteFileToolsBehavior independently."""
    print("\n=== Testing WriteFileToolsBehavior ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        test_file = tmppath / "output.txt"
        test_content = "Test content\nLine 2"

        # Initialize behavior
        behavior = WriteFileToolsBehavior()

        # Test get_name
        assert behavior.get_name() == "write_file_tools", "Behavior name should be 'write_file_tools'"
        print("✓ get_name() works")

        # Test get_tools
        tools = behavior.get_tools()
        assert len(tools) == 1, "Should provide 1 tool"
        assert tools[0]["function"]["name"] == "write_file", "Tool should be write_file"
        print("✓ get_tools() works")

        # Test dispatch_tool - write file
        result = behavior.dispatch_tool(
            "write_file",
            {"path": str(test_file), "content": test_content}
        )
        assert "Wrote" in result, "Should indicate write success"
        assert test_file.exists(), "File should exist"
        assert test_file.read_text() == test_content, "Content should match"
        print("✓ dispatch_tool() works - file written")

        # Test append mode
        append_content = "\nLine 3"
        result = behavior.dispatch_tool(
            "write_file",
            {"path": str(test_file), "content": append_content, "append": True}
        )
        assert "Appended" in result, "Should indicate append success"
        final_content = test_file.read_text()
        assert final_content == test_content + append_content, "Content should be appended"
        print("✓ append mode works")

        # Test overwrite=False protection
        test_file2 = tmppath / "protected.txt"
        test_file2.write_text("existing")
        result = behavior.dispatch_tool(
            "write_file",
            {"path": str(test_file2), "content": "new", "overwrite": False}
        )
        assert "ERROR" in result, "Should return error when overwrite=False"
        assert test_file2.read_text() == "existing", "File should not be modified"
        print("✓ overwrite=False protection works")

    print("✅ WriteFileToolsBehavior test PASSED")


def test_all_behaviors_compose():
    """Test that all three behaviors can work together."""
    print("\n=== Testing Behavior Composition ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Initialize all three behaviors
        dir_behavior = DirectoryToolsBehavior()
        read_behavior = ReadFileToolsBehavior()
        write_behavior = WriteFileToolsBehavior()

        # Write a file
        test_file = tmppath / "composed.txt"
        test_content = "Composition test"
        write_result = write_behavior.dispatch_tool(
            "write_file",
            {"path": str(test_file), "content": test_content}
        )
        assert "Wrote" in write_result
        print("✓ WriteFileToolsBehavior wrote file")

        # List directory to verify file exists
        dir_result = dir_behavior.dispatch_tool(
            "list_dir",
            {"path": str(tmppath)}
        )
        assert "composed.txt" in dir_result
        print("✓ DirectoryToolsBehavior found file")

        # Read the file back
        read_result = read_behavior.dispatch_tool(
            "read_file",
            {"path": str(test_file)}
        )
        assert read_result == test_content
        print("✓ ReadFileToolsBehavior read file")

    print("✅ Behavior composition test PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Split File Tools Behaviors")
    print("=" * 60)

    try:
        test_directory_tools()
        test_read_file_tools()
        test_write_file_tools()
        test_all_behaviors_compose()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
