#!/usr/bin/env python3
"""
Test command whitelist functionality.

Verifies that:
1. jetbox_commands_whitelist file is loaded correctly
2. Allowed commands execute successfully
3. Blocked commands are rejected with clear error messages
"""
from pathlib import Path
from behaviors.command_tools import CommandToolsBehavior

def test_whitelist_loading():
    """Test that whitelist loads from jetbox_commands_whitelist file."""
    behavior = CommandToolsBehavior()

    assert behavior.whitelist is not None, "Whitelist should be loaded from file"
    assert "python" in behavior.whitelist, "python should be in whitelist"
    assert "pytest" in behavior.whitelist, "pytest should be in whitelist"

    print("✓ Whitelist loaded successfully")
    print(f"  Allowed commands: {sorted(behavior.whitelist)}")

def test_allowed_command():
    """Test that allowed commands execute successfully."""
    behavior = CommandToolsBehavior()

    result = behavior._run_bash("python --version")

    assert "error" not in result, "Allowed command should not error"
    assert result["returncode"] == 0, "Python command should succeed"

    print("✓ Allowed command executed successfully")
    print(f"  Output: {result['stdout'][:50]}")

def test_blocked_command():
    """Test that blocked commands are rejected."""
    behavior = CommandToolsBehavior()

    result = behavior._run_bash("rm -rf /tmp/test")

    assert "error" in result, "Blocked command should error"
    assert "not allowed" in result["error"].lower(), "Error should indicate command not allowed"
    assert result["returncode"] == -1, "Blocked command should return -1"

    print("✓ Blocked command rejected successfully")
    print(f"  Error: {result['error'][:80]}")

def test_override_whitelist():
    """Test that whitelist parameter overrides file."""
    behavior = CommandToolsBehavior(whitelist=["echo", "ls"])

    # echo should work (in override whitelist)
    result = behavior._run_bash("echo 'test'")
    assert "error" not in result, "echo should be allowed"

    # python should not work (not in override whitelist)
    result = behavior._run_bash("python --version")
    assert "error" in result, "python should be blocked when using override whitelist"

    print("✓ Whitelist override works correctly")

if __name__ == "__main__":
    print("="*80)
    print("COMMAND WHITELIST TESTS")
    print("="*80)
    print()

    test_whitelist_loading()
    print()
    test_allowed_command()
    print()
    test_blocked_command()
    print()
    test_override_whitelist()

    print()
    print("="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
