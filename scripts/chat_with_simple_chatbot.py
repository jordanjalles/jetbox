#!/usr/bin/env python3
"""
Simple launcher for the simple-chatbot agent.

Usage:
    python chat_with_simple_chatbot.py                  # Interactive mode
    python chat_with_simple_chatbot.py "Your message"   # Single message
"""
import sys
from pathlib import Path
from agents.task_executor_agent import TaskExecutorAgent


def main():
    # Get message from command line if provided
    goal = None
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])

    # Create workspace
    workspace = Path('.agent_workspace/simple_chatbot')
    workspace.mkdir(parents=True, exist_ok=True)

    # Create the simple chatbot agent
    print("Loading simple-chatbot...")
    agent = TaskExecutorAgent(
        workspace=workspace,
        config_file='config/agents/simple-chatbot.yaml',
        goal=goal,  # None = interactive chatbot mode
        timeout_seconds=300
    )

    print(f"✓ {agent.name} loaded")
    print(f"  Behaviors: {len(agent.behaviors)}")
    print(f"  Tools: {len(agent.get_tools())}")
    print()

    if goal:
        # Single message mode
        print(f"You: {goal}")
        print()
        result = agent.run()
        print()
        if isinstance(result, dict):
            success = result.get('success', False)
        else:
            success = getattr(result, 'success', False)

        if success:
            print("✓ Conversation complete")
        else:
            print("✗ Conversation ended")
    else:
        # Interactive mode
        print("="*60)
        print("Simple Chatbot - Interactive Mode")
        print("="*60)
        print("Type your messages and press Enter.")
        print("Type 'exit' or 'quit' to end the conversation.")
        print()

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nGoodbye! 👋")
                    break

                # Set the user's message as the goal and run one round
                agent.goal = user_input
                agent.run()
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except EOFError:
                break


if __name__ == "__main__":
    main()
