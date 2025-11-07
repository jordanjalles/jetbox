#!/usr/bin/env python3
"""
agent.py - Universal agent wrapper with team selection support

This file dynamically loads and runs agents based on team configuration.
Supports multiple teams via config/teams/*.yaml files.

Usage:
    python agent.py [args]           # Auto-select team (interactive if multiple)
    python agent.py --team solo      # Use specific team
    python agent.py --list-teams     # List available teams
    python agent.py --help           # Show help

The actual agent implementation and CLI logic lives in the agent classes
(orchestrator_agent.py, task_executor_agent.py, architect_agent.py).
"""
import sys
import os
import yaml
from pathlib import Path

# IMPORTANT: Clear any OLLAMA_MODEL env var set by test scripts
# Test files often set os.environ["OLLAMA_MODEL"] = "gpt-oss:20b" for benchmarking
# This persists across shell sessions and overrides config files
# Users should use config/llm_config.yaml to set the model, not env vars
if "OLLAMA_MODEL" in os.environ:
    print(f"[agent.py] Clearing OLLAMA_MODEL env var (was: {os.environ['OLLAMA_MODEL']})")
    print(f"[agent.py] Will use model from config/llm_config.yaml")
    del os.environ["OLLAMA_MODEL"]

from agent_config import list_available_teams, load_team_config


def parse_extra_behaviors(argv: list[str]) -> tuple[list[str], list[str]]:
    """
    Parse --BehaviorName flags from argv for dynamic behavior injection.

    Supports both:
    - --ContextInspectorBehavior (full name)
    - --ContextInspector (short name, appends 'Behavior')

    Any flag starting with -- and a capital letter is treated as a behavior flag.
    This enables CLI injection of any behavior for debugging/inspection.

    Returns:
        (extra_behaviors, remaining_args)
    """
    extra_behaviors = []
    remaining_args = [argv[0]]  # Keep script name

    i = 1
    while i < len(argv):
        arg = argv[i]

        if arg.startswith('--'):
            flag_name = arg[2:]  # Strip '--'

            # Check if it's a behavior flag (starts with capital letter)
            if flag_name and flag_name[0].isupper():
                # Ensure it ends with 'Behavior'
                if not flag_name.endswith('Behavior'):
                    flag_name += 'Behavior'
                extra_behaviors.append(flag_name)
                i += 1
                continue

        # Not a behavior flag, keep in args
        remaining_args.append(arg)
        i += 1

    return extra_behaviors, remaining_args


def print_usage():
    """Print usage information."""
    print("Usage: python agent.py [OPTIONS] [goal]")
    print()
    print("Options:")
    print("  --team NAME         Use specific team (e.g., --team solo)")
    print("  --list-teams        List all available teams and exit")
    print("  --help              Show this help message")
    print()
    print("Other options:")
    print("  --workspace PATH    Use specific workspace directory")
    print("  --timeout SECONDS   Set subprocess timeout (default: 600)")
    print("  --once              Exit after initial response")
    print("  --chat              Force interactive chat mode")
    print()
    print("Examples:")
    print("  python agent.py 'Create a calculator'")
    print("  python agent.py --team solo 'Write a Flask app'")
    print("  python agent.py --list-teams")


def list_teams_and_exit():
    """List available teams and exit."""
    teams = list_available_teams()

    if not teams:
        print("No agent teams found.")
        print()
        print("Please create a team configuration in config/teams/")
        print("See CONFIG_REFACTOR_PLAN.md for details.")
        sys.exit(1)

    print("Available agent teams:")
    print()

    for team in teams:
        print(f"{team['file']} - {team['name']}")
        if team['description']:
            print(f"  {team['description']}")
        print(f"  Agents: {', '.join(team['agents'])}")
        print()

    sys.exit(0)


def select_team_interactively(teams: list[dict]) -> str:
    """
    Prompt user to select a team interactively.

    Args:
        teams: List of team configs from list_available_teams()

    Returns:
        Selected team file name (without .yaml)
    """
    print("Multiple agent teams available. Which team would you like to use?")
    print()

    for idx, team in enumerate(teams, 1):
        print(f"{idx}. {team['name']} ({team['file']}.yaml)")
        if team['description']:
            print(f"   {team['description']}")
        print()

    while True:
        try:
            choice = input(f"Select team [1-{len(teams)}] (default: 1): ").strip()
            if not choice:
                return teams[0]['file']

            choice_num = int(choice)
            if 1 <= choice_num <= len(teams):
                return teams[choice_num - 1]['file']
            else:
                print(f"Please enter a number between 1 and {len(teams)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            sys.exit(0)


def get_team_name() -> str:
    """
    Determine which team to use based on CLI args and available teams.

    Returns:
        Team file name (without .yaml extension)
    """
    # Check for --list-teams flag
    if "--list-teams" in sys.argv:
        list_teams_and_exit()

    # Check for --help flag
    if "--help" in sys.argv:
        print_usage()
        sys.exit(0)

    # Check for --team flag
    if "--team" in sys.argv:
        idx = sys.argv.index("--team")
        if idx + 1 < len(sys.argv):
            team_name = sys.argv[idx + 1]
            # Remove --team and its argument from sys.argv so BaseAgent.parse_cli_args doesn't see it
            sys.argv.pop(idx)
            sys.argv.pop(idx)
            return team_name
        else:
            print("Error: --team requires a team name")
            print()
            print_usage()
            sys.exit(1)

    # Auto-select team
    teams = list_available_teams()

    if not teams:
        print("Error: No agent teams found")
        print()
        print("Please create a team configuration in config/teams/")
        print("See CONFIG_REFACTOR_PLAN.md for details.")
        sys.exit(1)

    if len(teams) == 1:
        # Single team - use it automatically
        return teams[0]['file']

    # Multiple teams - prompt user
    return select_team_interactively(teams)


def get_first_agent_info(team_name: str) -> tuple:
    """
    Load team config and return the first agent's class and config file.

    Args:
        team_name: Team file name (without .yaml extension)

    Returns:
        Tuple of (agent_class, config_file_path or None, agent_name)
    """
    # Load team config
    team_config = load_team_config(team_name)

    if not team_config:
        print(f"Error: Could not load team config '{team_name}'")
        sys.exit(1)

    # Get first agent from team
    agents = team_config.get("agents", {})
    if not agents:
        print(f"Error: No agents defined in team '{team_name}'")
        sys.exit(1)

    # Get first agent (dict maintains insertion order in Python 3.7+)
    first_agent_name = next(iter(agents.keys()))
    first_agent_config = agents[first_agent_name]

    # Get class name
    class_name = first_agent_config.get("class")
    if not class_name:
        print(f"Error: No class defined for agent '{first_agent_name}' in team '{team_name}'")
        sys.exit(1)

    # Get config file if specified
    config_file = None
    if "config" in first_agent_config:
        config_file = f"config/agents/{first_agent_config['config']}.yaml"

    team_display_name = team_config.get("name", team_name)
    print(f"[agent.py] Using team: {team_display_name}")
    print(f"[agent.py] Starting agent: {first_agent_name} ({class_name})")
    if config_file:
        print(f"[agent.py] Using config: {config_file}")

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
        return (agent_class, config_file, first_agent_name)
    except ImportError as e:
        print(f"Error: Could not import {class_name} from {module_name}: {e}")
        sys.exit(1)
    except AttributeError as e:
        print(f"Error: Class {class_name} not found in module {module_name}: {e}")
        sys.exit(1)


def main():
    """Main entry point - select team and run first agent."""
    # Parse extra behaviors from CLI flags (e.g., --ContextInspector)
    extra_behaviors, remaining_args = parse_extra_behaviors(sys.argv)

    # Store in env var for session-wide propagation to sub-agents
    if extra_behaviors:
        os.environ['JETBOX_EXTRA_BEHAVIORS'] = ','.join(extra_behaviors)
        print(f"[agent.py] Extra behaviors enabled: {', '.join(extra_behaviors)}")

    # Update sys.argv to remove behavior flags
    sys.argv = remaining_args

    # Determine which team to use
    team_name = get_team_name()

    # Load first agent from team
    agent_class, config_file, agent_name = get_first_agent_info(team_name)

    # If team config specifies a custom config file, we need to override
    # the agent class's create_agent_instance to pass it
    if config_file:
        # Create a wrapper that overrides create_agent_instance to pass config_file
        original_create = agent_class.create_agent_instance

        @classmethod
        def create_with_config(cls, workspace, args):
            """Override to pass custom config file and agent name from team config."""
            initial_message = args.get("initial_message")
            force_chat_mode = args.get("force_chat_mode", False)
            timeout_seconds = args.get("timeout_seconds", 600)

            # Determine if ChatbotBehavior should be excluded
            exclude_behaviors = []
            if initial_message and not force_chat_mode:
                # Autonomous mode: exclude chatbot to run goal directly
                exclude_behaviors = ["ChatbotBehavior"]

            # Get extra behaviors from environment (set by parse_extra_behaviors in main)
            extra_behaviors_env = os.environ.get('JETBOX_EXTRA_BEHAVIORS', '')
            extra_behaviors = [b.strip() for b in extra_behaviors_env.split(',') if b.strip()] if extra_behaviors_env else None

            # Create agent with custom config file and agent name from team
            # NOTE: TaskExecutorAgent.__init__ has name="task_executor" hardcoded,
            # but BaseAgent.__init__ will override it if we instantiate BaseAgent directly
            from base_agent import BaseAgent
            return BaseAgent(
                name=agent_name,  # Use team config name, not hardcoded class name
                workspace=workspace,
                config_file=config_file,
                timeout_seconds=timeout_seconds,
                exclude_behaviors=exclude_behaviors,
                extra_behaviors=extra_behaviors,
            )

        # Temporarily replace the method
        agent_class.create_agent_instance = create_with_config

    # Call the agent's main() method, which handles all CLI parsing and execution
    agent_class.main()


if __name__ == "__main__":
    main()
