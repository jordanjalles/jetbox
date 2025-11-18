#!/usr/bin/env python3
"""Jetbox CLI for managing specialized agents."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from src.agent_manager import AgentManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_start(args):
    """Start the agent manager daemon."""
    manager = AgentManager()

    if args.foreground:
        # Run in foreground
        logger.info("Starting agent manager in foreground mode")
        manager.start(foreground=True)
    else:
        # Daemonize
        logger.info("Starting agent manager as daemon")

        # Fork to background
        if os.fork() > 0:
            # Parent process
            print("Agent manager started in background")
            sys.exit(0)

        # Child process continues
        # Redirect stdio
        sys.stdout.flush()
        sys.stderr.flush()

        # Start manager
        manager.start(foreground=False)

        # Keep running
        try:
            while manager.running:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop()


def cmd_stop(args):
    """Stop the agent manager daemon."""
    manager = AgentManager()

    # Check if running
    if not manager.pid_file.exists():
        print("Agent manager is not running")
        return

    try:
        with open(manager.pid_file, "r") as f:
            pid = int(f.read().strip())

        print(f"Stopping agent manager (PID {pid})...")

        # Send SIGTERM
        os.kill(pid, 15)  # SIGTERM

        # Wait for it to stop
        for i in range(10):
            time.sleep(0.5)
            if not manager.pid_file.exists():
                print("Agent manager stopped successfully")
                return

        print("Warning: Manager may not have stopped cleanly")

    except FileNotFoundError:
        print("PID file not found, manager may have already stopped")
    except ProcessLookupError:
        print("Process not found, removing stale PID file")
        manager.pid_file.unlink()
    except Exception as e:
        print(f"Error stopping manager: {e}")


def cmd_status(args):
    """Show status of all agents."""
    manager = AgentManager()

    # Check if running
    if not manager.pid_file.exists():
        print("AgentManager: NOT RUNNING")
        return

    try:
        # Load state file
        if manager.state_file.exists():
            import json

            with open(manager.state_file, "r") as f:
                state = json.load(f)

            print(
                f"AgentManager: RUNNING (PID {state.get('pid')}, uptime: {state.get('uptime', 'unknown')})"
            )
            print()

            if not state.get("agents"):
                print("No agents configured")
                return

            print("Agents:")
            for agent in state["agents"]:
                status_icon = {
                    "running": "●",
                    "completed": "✓",
                    "crashed": "✗",
                    "scheduled": "⏰",
                    "stopped": "○",
                    "not_started": "-",
                }.get(agent["status"], "?")

                print(
                    f"  {agent['name']:20s} [{status_icon} {agent['status'].upper():12s}]"
                )

                if agent["mode"] == "scheduled":
                    print(
                        f"    Schedule: {agent['schedule']} | Last: {agent.get('last_run', 'never')}"
                    )

                if agent.get("crash_count", 0) > 0:
                    print(
                        f"    Crashes: {agent['crash_count']}"
                    )

                print()

        else:
            print("No state file found")

    except Exception as e:
        print(f"Error reading status: {e}")


def cmd_logs(args):
    """View agent logs."""
    manager = AgentManager()

    agent_name = args.agent
    lines = args.tail or 50
    follow = args.follow

    if follow:
        # Follow mode (tail -f)
        print(f"Following logs for {agent_name} (Ctrl+C to stop)...")
        print()

        log_path = manager.log_dir / f"{agent_name}.log"

        if not log_path.exists():
            print(f"Log file not found: {log_path}")
            return

        # Get initial position
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                f.seek(0, 2)  # Seek to end
                position = f.tell()

            # Follow loop
            while True:
                with open(log_path, "r", encoding="utf-8") as f:
                    f.seek(position)
                    new_lines = f.readlines()
                    position = f.tell()

                if new_lines:
                    for line in new_lines:
                        print(line, end="")

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nStopped following logs")
        except Exception as e:
            print(f"Error following logs: {e}")

    else:
        # Tail mode
        log_lines = manager.get_agent_logs(agent_name, lines=lines)

        if log_lines is None:
            print(f"Agent {agent_name} not found")
            return

        if not log_lines:
            print(f"No logs found for {agent_name}")
            return

        for line in log_lines:
            print(line, end="")


def cmd_run(args):
    """Run a single agent for testing (not via manager)."""
    agent_name = args.agent
    goal = args.goal or "Execute test task"

    config_file = Path(f"config/agents/{agent_name}.yaml")

    if not config_file.exists():
        print(f"Config file not found: {config_file}")
        return

    print(f"Running agent {agent_name} with goal: {goal}")
    print()

    # Run agent directly (not through manager)
    import subprocess

    cmd = ["python", "agent.py", "--config", str(config_file), goal]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Agent exited with code {e.returncode}")
    except KeyboardInterrupt:
        print("\nInterrupted")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Jetbox - Manage specialized agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Start command
    start_parser = subparsers.add_parser(
        "start", help="Start the agent manager daemon"
    )
    start_parser.add_argument(
        "--foreground",
        "-f",
        action="store_true",
        help="Run in foreground (don't daemonize)",
    )
    start_parser.set_defaults(func=cmd_start)

    # Stop command
    stop_parser = subparsers.add_parser(
        "stop", help="Stop the agent manager daemon"
    )
    stop_parser.set_defaults(func=cmd_stop)

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Show status of all agents"
    )
    status_parser.set_defaults(func=cmd_status)

    # Logs command
    logs_parser = subparsers.add_parser(
        "logs", help="View agent logs"
    )
    logs_parser.add_argument("agent", help="Agent name")
    logs_parser.add_argument(
        "--tail", "-n", type=int, help="Number of lines to show"
    )
    logs_parser.add_argument(
        "--follow", "-f", action="store_true", help="Follow log output"
    )
    logs_parser.set_defaults(func=cmd_logs)

    # Run command
    run_parser = subparsers.add_parser(
        "run", help="Run a single agent for testing"
    )
    run_parser.add_argument("agent", help="Agent name")
    run_parser.add_argument(
        "--goal", "-g", help="Goal for the agent"
    )
    run_parser.set_defaults(func=cmd_run)

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
