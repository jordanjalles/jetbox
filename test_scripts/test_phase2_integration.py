#!/usr/bin/env python3
"""
Integration test for Phase 2 CLI flag system.

Tests that behaviors can be loaded via CLI flags and environment variables.
"""
import os
import sys
from pathlib import Path
from base_agent import BaseAgent


def test_cli_flag_loading():
    """Test that extra behaviors are loaded via CLI flag."""

    # Set environment variable (simulating what agent.py does)
    os.environ['JETBOX_EXTRA_BEHAVIORS'] = 'TestCliInjectorBehavior'

    print("Creating BaseAgent with environment variable set...")

    # Create a minimal agent to test behavior loading
    agent = BaseAgent(
        name="test_agent",
        workspace=Path("/tmp/test_phase2"),
        config_file="config/agents/task_executor.yaml",
    )

    # Check if behavior was loaded
    behavior_names = [b.get_name() for b in agent.behaviors]
    print(f"Loaded behaviors: {behavior_names}")

    if "test_cli_injector" in behavior_names:
        print("✅ TestCliInjectorBehavior loaded successfully via environment variable")
        return True
    else:
        print("❌ TestCliInjectorBehavior not found in loaded behaviors")
        return False


def test_direct_parameter():
    """Test that extra behaviors are loaded via parameter."""

    # Clear environment variable
    if 'JETBOX_EXTRA_BEHAVIORS' in os.environ:
        del os.environ['JETBOX_EXTRA_BEHAVIORS']

    print("\nCreating BaseAgent with extra_behaviors parameter...")

    # Create agent with extra_behaviors parameter
    agent = BaseAgent(
        name="test_agent2",
        workspace=Path("/tmp/test_phase2_2"),
        config_file="config/agents/task_executor.yaml",
        extra_behaviors=["TestCliInjectorBehavior"],
    )

    # Check if behavior was loaded
    behavior_names = [b.get_name() for b in agent.behaviors]
    print(f"Loaded behaviors: {behavior_names}")

    if "test_cli_injector" in behavior_names:
        print("✅ TestCliInjectorBehavior loaded successfully via parameter")
        return True
    else:
        print("❌ TestCliInjectorBehavior not found in loaded behaviors")
        return False


def test_duplicate_prevention():
    """Test that behaviors aren't loaded twice if already in config."""

    # Set environment variable with a behavior that's already in config
    os.environ['JETBOX_EXTRA_BEHAVIORS'] = 'ChatbotBehavior'

    print("\nCreating BaseAgent with duplicate behavior (ChatbotBehavior)...")

    # Create agent
    agent = BaseAgent(
        name="test_agent3",
        workspace=Path("/tmp/test_phase2_3"),
        config_file="config/agents/task_executor.yaml",
    )

    # Count ChatbotBehavior instances
    chatbot_count = sum(1 for b in agent.behaviors if b.get_name() == "chatbot")
    print(f"ChatbotBehavior instances: {chatbot_count}")

    if chatbot_count == 1:
        print("✅ Duplicate prevention works correctly")
        return True
    else:
        print(f"❌ Duplicate prevention failed (found {chatbot_count} instances)")
        return False


if __name__ == "__main__":
    print("Testing Phase 2 CLI flag integration...\n")
    print("=" * 60)

    results = []
    results.append(test_cli_flag_loading())
    results.append(test_direct_parameter())
    results.append(test_duplicate_prevention())

    print("=" * 60)

    if all(results):
        print("\n✅ All integration tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some integration tests failed")
        sys.exit(1)
