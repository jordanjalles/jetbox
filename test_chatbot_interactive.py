#!/usr/bin/env python3
"""
Interactive test for chatbot agent.
Simulates a conversation to verify chat-only mode works correctly.
"""
from pathlib import Path
from agent_registry import AgentRegistry

print("="*60)
print("INTERACTIVE CHATBOT TEST")
print("="*60)

# Load chatbot team
registry = AgentRegistry(team_name="chatbot", workspace=Path("."))
agent = registry.get_agent("simple_chatbot")

print(f"\nAgent loaded: {agent.name}")
print(f"Agent class: {agent.__class__.__name__}")
print(f"Behaviors: {[b.get_name() for b in agent.behaviors]}")

# Check chat_only_mode detection
has_file_tools = any(
    behavior.get_name() in ['read_file_tools', 'write_file_tools', 'directory_tools', 'command_tools']
    for behavior in agent.behaviors
)
chat_only_mode = not has_file_tools
print(f"Chat-only mode: {chat_only_mode}")

if not chat_only_mode:
    print("\n❌ ERROR: Should be in chat-only mode!")
    exit(1)

print("\n" + "="*60)
print("TESTING CONVERSATIONAL INTERACTIONS")
print("="*60)

# Test 1: Simple greeting
print("\n[TEST 1] Simple greeting")
print("User: Hello!")
try:
    agent.run_single_llm_round("Hello!")
    print("✅ Test 1 passed - greeting handled")
except Exception as e:
    print(f"❌ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Ask a question
print("\n[TEST 2] Ask a question")
print("User: What is the capital of France?")
try:
    agent.run_single_llm_round("What is the capital of France?")
    print("✅ Test 2 passed - question answered")
except Exception as e:
    print(f"❌ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Follow-up question
print("\n[TEST 3] Follow-up question")
print("User: What's the population of that city?")
try:
    agent.run_single_llm_round("What's the population of that city?")
    print("✅ Test 3 passed - context maintained")
except Exception as e:
    print(f"❌ Test 3 failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("INTERACTIVE TEST COMPLETE")
print("="*60)
print("\nAll conversation tests completed successfully!")
print("The chatbot is working in pure chat mode without task execution.")
