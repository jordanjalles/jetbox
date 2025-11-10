#!/usr/bin/env python3
"""
Interactive test of simple-chatbot to verify the chatbot mode fix.

This simulates asking questions that the chatbot cannot answer,
to verify it doesn't call mark_failed and end the conversation.
"""
from pathlib import Path
from agents.task_executor_agent import TaskExecutorAgent


def test_chatbot_unanswerable_question():
    """Test that chatbot handles unanswerable questions gracefully."""
    print("=" * 60)
    print("Interactive Test: Simple Chatbot - Unanswerable Question")
    print("=" * 60)

    workspace = Path('.agent_workspace/chatbot_interactive_test')
    workspace.mkdir(parents=True, exist_ok=True)

    # Create simple-chatbot without a goal
    agent = TaskExecutorAgent(
        workspace=workspace,
        config_file='config/agents/simple-chatbot.yaml',
        goal=None,  # Pure conversation mode
        timeout_seconds=60
    )

    print(f"\n✓ Agent: {agent.name}")
    print(f"  Goal: {agent.goal}")
    print(f"  Behaviors: {len(agent.behaviors)}")

    # Check tools
    tools = agent.get_tools()
    tool_names = [t.get("function", {}).get("name") for t in tools]
    print(f"  Tools: {', '.join(tool_names)}")

    # Verify no completion tools
    has_mark_complete = "mark_complete" in tool_names
    has_mark_failed = "mark_failed" in tool_names

    if has_mark_complete or has_mark_failed:
        print("\n❌ ERROR: Agent has completion tools (should not)")
        return False

    print("\n✅ Agent configured correctly (no completion tools)")
    print("\n" + "-" * 60)
    print("Test Scenario: Ask weather question (unanswerable)")
    print("-" * 60)

    # Simulate asking a question the chatbot cannot answer
    # In the broken version, this would cause mark_failed to be called
    # In the fixed version, the agent should respond with text explaining limitation

    # Build context with the question
    context = agent.build_context()

    # Add user question
    context.append({
        "role": "user",
        "content": "What's the weather like today?"
    })

    print("\nUser: What's the weather like today?")
    print("\nAgent response:")
    print("-" * 60)

    # Get LLM response
    try:
        from ollama import chat

        response = chat(
            model=agent.model,
            messages=context,
            options={'temperature': agent.temperature},
            tools=tools if tools else None
        )

        message = response.get('message', {})
        content = message.get('content', '')
        tool_calls = message.get('tool_calls', [])

        if content:
            print(content)

        if tool_calls:
            print(f"\nTool calls made: {len(tool_calls)}")
            for tc in tool_calls:
                func = tc.get('function', {})
                name = func.get('name', 'unknown')
                args = func.get('arguments', {})
                print(f"  - {name}({args})")

                # Check if mark_failed was called
                if name == "mark_failed":
                    print("\n❌ FAILED: Agent called mark_failed (should not)")
                    print("   This means the conversation would end prematurely.")
                    return False

        if not tool_calls:
            print("\n✅ SUCCESS: Agent responded with text (no tool calls)")
            print("   Conversation can continue.")
            return True

        # If other tools were called (like clarify_with_user), that's OK
        print("\n✅ SUCCESS: Agent used conversational tools (not completion tools)")
        print("   Conversation can continue.")
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def main():
    """Run interactive test."""
    print("\n🔧 Testing Simple Chatbot - Unanswerable Question Handling\n")

    success = test_chatbot_unanswerable_question()

    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED: Chatbot handles limitations gracefully")
    else:
        print("❌ TEST FAILED: Chatbot does not handle limitations correctly")
    print("=" * 60)

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
