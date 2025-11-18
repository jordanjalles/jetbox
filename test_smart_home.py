#!/usr/bin/env python3
"""Direct test of smart home agent."""
import os
from pathlib import Path
from base_agent import BaseAgent

# Set environment variables
os.environ["HOME_ASSISTANT_URL"] = "http://192.168.50.4:8123"
os.environ["HOME_ASSISTANT_TOKEN"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJlMDdkYzRkMTU2YTM0YzIxYjExNDJhMDIzNTMwMzdiNyIsImlhdCI6MTc2MzQyNTI1OSwiZXhwIjoyMDc4Nzg1MjU5fQ.-cpKRwjeIHBICXkIH4340lAoQXo0KlXg9UvDlI2esH8"

# Create agent
agent = BaseAgent(
    name="smart_home_test",
    workspace=Path("/tmp/smart_home_test"),
    config_file="config/agents/smart_home_controller.yaml",
    timeout_seconds=120
)

# Set goal and run
agent.set_goal("Turn Jordan's office lights to 50% brightness")
agent.run()
