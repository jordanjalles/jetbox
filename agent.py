#!/usr/bin/env python3
"""
agent.py - Universal agent wrapper based on agents.yaml configuration

This file dynamically loads and runs the first agent defined in agents.yaml.
By default, this is the orchestrator agent, but it can be changed via config.

Usage:
    python agent.py [args]  # Runs first agent in agents.yaml with all args

The actual agent implementation and CLI logic lives in the agent classes
(orchestrator_agent.py, task_executor_agent.py, architect_agent.py).
"""
import sys
import yaml
from pathlib import Path


def get_first_agent_class():
    """
    Load agents.yaml and return the first agent's class.

    Returns:
        The agent class to instantiate
    """
    # Load agents.yaml
    config_path = Path("agents.yaml")
    if not config_path.exists():
        print("Error: agents.yaml not found")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get first agent
    agents = config.get("agents", {})
    if not agents:
        print("Error: No agents defined in agents.yaml")
        sys.exit(1)

    # Get first agent (dict maintains insertion order in Python 3.7+)
    first_agent_name = next(iter(agents.keys()))
    first_agent_config = agents[first_agent_name]

    # Get class name
    class_name = first_agent_config.get("class")
    if not class_name:
        print(f"Error: No class defined for agent '{first_agent_name}' in agents.yaml")
        sys.exit(1)

    print(f"[agent.py] Loading first agent from agents.yaml: {first_agent_name} ({class_name})")

    # Import the class dynamically
    # Map class names to modules
    module_map = {
        "OrchestratorAgent": "orchestrator_agent",
        "TaskExecutorAgent": "task_executor_agent",
        "ArchitectAgent": "architect_agent",
    }

    module_name = module_map.get(class_name)
    if not module_name:
        print(f"Error: Unknown agent class '{class_name}'. Add to module_map in agent.py")
        sys.exit(1)

    # Import and return the class
    try:
        module = __import__(module_name, fromlist=[class_name])
        agent_class = getattr(module, class_name)
        return agent_class
    except ImportError as e:
        print(f"Error: Could not import {class_name} from {module_name}: {e}")
        sys.exit(1)
    except AttributeError as e:
        print(f"Error: Class {class_name} not found in module {module_name}: {e}")
        sys.exit(1)


def main():
    """Main entry point - load first agent from agents.yaml and run its main()."""
    agent_class = get_first_agent_class()

    # Call the agent's main() method, which handles all CLI parsing and execution
    agent_class.main()


if __name__ == "__main__":
    main()
