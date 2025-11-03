#!/usr/bin/env python3
"""Check if architect has a goal set."""

from pathlib import Path
from architect_agent import ArchitectAgent

architect = ArchitectAgent(workspace=Path(".agent_workspaces/tests/test"))

# Add task (what orchestrator does when delegating)
architect.add_message({
    "role": "user",
    "content": "Create architecture for a simple blog API."
})

print("Goal from context_manager:", architect.context_manager.state.goal)
print("Has goal?", architect.context_manager.state.goal is not None)

# Check if subagent mode behavior set the goal
for behavior in architect._behaviors:
    if behavior.get_name() == "subagent_mode":
        print("SubAgentModeBehavior found")
        print("Goal set by behavior:", hasattr(behavior, 'goal'))
