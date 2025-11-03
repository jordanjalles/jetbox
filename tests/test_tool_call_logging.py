"""Test improved tool call logging with argument previews."""
from base_agent import BaseAgent


def test_format_tool_call_preview():
    """Test the _format_tool_call_preview method."""
    agent = BaseAgent(workspace=".", name="test_agent", config_file="agent_config.yaml")

    # Test 1: No arguments
    preview = agent._format_tool_call_preview("mark_complete", {})
    assert preview == "mark_complete()"
    print(f"✓ No args: {preview}")

    # Test 2: Short argument
    preview = agent._format_tool_call_preview("write_file", {"path": "test.py"})
    assert preview == "write_file(path=test.py)"
    print(f"✓ Short arg: {preview}")

    # Test 3: Long argument (should truncate)
    long_content = "def hello():\n    print('Hello world')\n    return 42"
    preview = agent._format_tool_call_preview("write_file", {
        "path": "calculator.py",
        "content": long_content
    })
    assert "calculator.py" in preview
    assert "content=" in preview
    assert "..." in preview
    assert len(preview) < 100  # Should be reasonably short
    print(f"✓ Long arg (truncated): {preview}")

    # Test 4: Multiple arguments
    preview = agent._format_tool_call_preview("run_bash", {
        "command": "pytest -q",
        "timeout": 30
    })
    assert "command=pytest -q" in preview
    assert "timeout=30" in preview
    print(f"✓ Multiple args: {preview}")

    # Test 5: Newlines in content (should be escaped)
    content_with_newlines = "line1\nline2\nline3"
    preview = agent._format_tool_call_preview("write_file", {
        "path": "test.txt",
        "content": content_with_newlines
    })
    assert "\\n" in preview  # Newlines should be escaped
    assert "\n" not in preview  # No actual newlines
    print(f"✓ Newlines escaped: {preview}")

    # Test 6: Very long value (should truncate to max_length)
    very_long = "x" * 100
    preview = agent._format_tool_call_preview("test_tool", {"arg": very_long}, max_length=30)
    assert preview == "test_tool(arg=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...)"
    print(f"✓ Custom max_length: {preview}")

    print("\n✅ All tool call preview formatting tests passed!")


if __name__ == "__main__":
    test_format_tool_call_preview()
