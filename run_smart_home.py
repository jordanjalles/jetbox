#!/usr/bin/env python3
"""Quick runner for smart home controller agent."""
import os
import sys
from pathlib import Path
from base_agent import BaseAgent

# Set environment variables
os.environ["HOME_ASSISTANT_URL"] = "http://192.168.50.4:8123"
os.environ["HOME_ASSISTANT_TOKEN"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJlMDdkYzRkMTU2YTM0YzIxYjExNDJhMDIzNTMwMzdiNyIsImlhdCI6MTc2MzQyNTI1OSwiZXhwIjoyMDc4Nzg1MjU5fQ.-cpKRwjeIHBICXkIH4340lAoQXo0KlXg9UvDlI2esH8"

# Get goal from command line
goal = sys.argv[1] if len(sys.argv) > 1 else "List all devices"

print(f"\n🏠 Running Smart Home Controller")
print(f"📋 Goal: {goal}\n")

# Create agent with simple workspace
agent = BaseAgent(
    name="smart_home_controller",
    workspace=Path(".agent_workspaces/smart_home"),
    config_file="config/agents/smart_home_controller.yaml",
    timeout_seconds=120
)

# Set goal and run
agent.set_goal(goal)
agent.run()
