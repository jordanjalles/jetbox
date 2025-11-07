#!/usr/bin/env python3
"""Test that chatbot team loads correct config."""
from pathlib import Path
from agent_registry import AgentRegistry

# Load chatbot team
registry = AgentRegistry(team_name="chatbot", workspace=Path("."))

# Get simple_chatbot agent
agent = registry.get_agent("simple_chatbot")

# Check agent properties
print(f"Agent name: {agent.name}")
print(f"Agent class: {agent.__class__.__name__}")

# Get base system prompt (from config, no behavior instructions)
from agent_config import load_agent_config
config = load_agent_config("simple-chatbot")
base_system_prompt = config.get("system_prompt", "")

# Get full system prompt (with behavior instructions)
full_system_prompt = agent.get_system_prompt()

print(f"Base system prompt length: {len(base_system_prompt)} chars (from config)")
print(f"Full system prompt length: {len(full_system_prompt)} chars (with behavior instructions)")
print(f"Number of behaviors: {len(agent.behaviors)}")

# List loaded behaviors
print("\nLoaded behaviors:")
for behavior in agent.behaviors:
    print(f"  - {behavior.get_name()}")

# Check for file/command tools (should NOT be present)
has_file_tools = any(
    behavior.get_name() in ['read_file_tools', 'write_file_tools', 'directory_tools', 'command_tools']
    for behavior in agent.behaviors
)

print(f"\nHas file/command tools: {has_file_tools}")

# Expected results
print("\n" + "="*60)
print("EXPECTED RESULTS:")
print("="*60)
print("Agent name: simple_chatbot")
print("Agent class: BaseAgent")
print("Base system prompt: 404 chars (from config)")
print("Full system prompt: includes behavior instructions")
print("Number of behaviors: 2")
print("Loaded behaviors:")
print("  - chatbot")
print("  - compact_when_near_full")
print("Has file/command tools: False")

print("\n" + "="*60)
print("TEST RESULT:")
print("="*60)

# Validate
success = True
if agent.name != "simple_chatbot":
    print(f"❌ FAIL: Agent name is '{agent.name}', expected 'simple_chatbot'")
    success = False

if agent.__class__.__name__ != "BaseAgent":
    print(f"❌ FAIL: Agent class is '{agent.__class__.__name__}', expected 'BaseAgent'")
    success = False

if len(base_system_prompt) != 404:
    print(f"❌ FAIL: Base system prompt is {len(base_system_prompt)} chars, expected 404")
    success = False

if len(agent.behaviors) != 2:
    print(f"❌ FAIL: {len(agent.behaviors)} behaviors loaded, expected 2")
    success = False

if has_file_tools:
    print(f"❌ FAIL: File/command tools loaded, should NOT be present")
    success = False

if success:
    print("✅ ALL CHECKS PASSED - Chatbot loads correct config!")
else:
    print("❌ SOME CHECKS FAILED - See details above")
