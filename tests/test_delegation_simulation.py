#!/usr/bin/env python3
"""
Simulate the actual orchestrator -> architect delegation to reproduce empty rounds.
"""

from pathlib import Path
from agents.orchestrator_agent import OrchestratorAgent

def main():
    """Test L7 task via orchestrator delegation."""

    print("="*80)
    print("DELEGATION SIMULATION - L7 Task")
    print("="*80)

    # Create orchestrator (like eval does)
    workspace = Path(".agent_workspaces/tests/delegation_test")
    orchestrator = OrchestratorAgent(
        workspace=workspace,
        exclude_behaviors=["ChatbotBehavior"]  # Autonomous mode
    )

    # Add L7 task (like eval does)
    task = "Create a full-stack Flask app with user auth, posts, and comments. Use SQLite. Include frontend templates."
    orchestrator.add_message({
        "role": "user",
        "content": task
    })

    print("\nRunning orchestrator (max 2 rounds to see delegation)...")
    print("="*80)

    # Run orchestrator - should delegate to architect
    result = orchestrator.run(max_rounds=2)

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Status: {result.get('status')}")
    print(f"Details: {result.get('reason', result.get('summary', 'N/A'))[:200]}")

if __name__ == "__main__":
    main()
