#!/usr/bin/env python3
"""
Simple integration test for orchestrator with architect and task executors.
Tests workspace coordination across multiple delegations.
"""

from orchestrator_agent import OrchestratorAgent
from pathlib import Path

# Create orchestrator with a clear, complex goal
workspace_dir = Path(".agent_workspaces/integration_test")
workspace_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("INTEGRATION TEST: Orchestrator + Architect + TaskExecutors")
print("="*60)
print()

goal = """Build a Flask web app with user authentication and blog posts with comments.

Requirements:
1. Database models for User, Post, Comment
2. Authentication (login/logout routes)
3. Blog post CRUD operations
4. Comment functionality
5. Basic HTML templates
6. All code should pass ruff checks
"""

print(f"Goal: {goal.strip()}")
print()
print("Starting orchestrator...")
print()

orchestrator = OrchestratorAgent(
    workspace=workspace_dir
)

# Add user message with the goal
orchestrator.add_message({"role": "user", "content": goal})

# Run orchestrator (it will delegate to architect and task_executor as needed)
result = orchestrator.run(max_rounds=50)

print()
print("="*60)
print("TEST COMPLETE")
print("="*60)
print()
print(f"Status: {result.get('status', 'unknown')}")
print(f"Workspace: {workspace_dir}")
print()

# List files created
if workspace_dir.exists():
    print("Files created:")
    for item in sorted(workspace_dir.rglob("*")):
        if item.is_file() and not item.name.startswith('.'):
            rel_path = item.relative_to(workspace_dir)
            print(f"  {rel_path}")
    print()

# Check for delegation
delegation_behavior = None
for behavior in orchestrator.behaviors:
    if behavior.get_name() == "delegation":
        delegation_behavior = behavior
        break

if delegation_behavior:
    print(f"Delegations performed: {len(delegation_behavior.delegated_tasks)}")
    for delegation in delegation_behavior.delegated_tasks:
        print(f"  → {delegation['agent']}: {delegation['task'][:60]}...")
    print()

print("Test finished.")
