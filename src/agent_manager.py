"""AgentManager daemon for orchestrating multiple specialized agents."""

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import schedule
import yaml

from agent_config import PROJECT_ROOT
from src.process_manager import ProcessManager

logger = logging.getLogger(__name__)


@dataclass
class AgentDeploymentConfig:
    """Deployment configuration for an agent."""

    enabled: bool = False
    mode: str = "scheduled"  # scheduled, service, on_demand
    schedule: str = "0 * * * *"  # cron syntax
    restart_on_crash: bool = True
    max_retries: int = 3
    goal: str = "Execute scheduled task"


@dataclass
class AgentConfig:
    """Complete agent configuration."""

    name: str
    config_file: Path
    deployment: AgentDeploymentConfig


class AgentManager:
    """Manages lifecycle of multiple specialized agents."""

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        state_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ):
        """Initialize agent manager.

        Args:
            config_dir: Directory containing agent configs (default: config/agents)
            state_dir: Directory for manager state (default: ~/.jetbox)
            log_dir: Directory for agent logs (default: ~/.jetbox/logs)
        """
        self.config_dir = config_dir or PROJECT_ROOT / "config" / "agents"
        self.state_dir = (
            state_dir or Path.home() / ".jetbox"
        )
        self.log_dir = log_dir or self.state_dir / "logs"

        # Create directories
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # State files
        self.pid_file = self.state_dir / "manager.pid"
        self.state_file = self.state_dir / "manager_state.json"

        # Process manager
        self.process_manager = ProcessManager(log_dir=self.log_dir)

        # Agent configurations
        self.agents: dict[str, AgentConfig] = {}

        # Scheduling
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running = False

        # Shutdown handling
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def load_agent_configs(self) -> None:
        """Discover and load all enabled agent configurations."""
        logger.info(f"Loading agent configs from {self.config_dir}")

        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
            return

        # Scan for YAML files
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                # Check if deployment is configured
                deployment_config = config.get("deployment", {})

                if not deployment_config.get("enabled", False):
                    logger.debug(
                        f"Skipping {config_file.stem} (deployment not enabled)"
                    )
                    continue

                # Parse deployment config
                deployment = AgentDeploymentConfig(
                    enabled=deployment_config.get("enabled", False),
                    mode=deployment_config.get("mode", "scheduled"),
                    schedule=deployment_config.get("schedule", "0 * * * *"),
                    restart_on_crash=deployment_config.get(
                        "restart_on_crash", True
                    ),
                    max_retries=deployment_config.get("max_retries", 3),
                    goal=deployment_config.get(
                        "goal", "Execute scheduled task"
                    ),
                )

                agent_config = AgentConfig(
                    name=config_file.stem,
                    config_file=config_file,
                    deployment=deployment,
                )

                self.agents[config_file.stem] = agent_config

                logger.info(
                    f"Loaded agent {config_file.stem} ({deployment.mode} mode)"
                )

            except Exception as e:
                logger.error(f"Error loading {config_file}: {e}")

        logger.info(f"Loaded {len(self.agents)} enabled agents")

    def _run_agent(self, agent_name: str) -> None:
        """Run an agent once.

        Args:
            agent_name: Name of the agent to run
        """
        if agent_name not in self.agents:
            logger.error(f"Agent {agent_name} not found")
            return

        agent = self.agents[agent_name]

        logger.info(f"Running agent {agent_name}")

        try:
            # Get environment variables
            env = os.environ.copy()

            # Spawn agent
            proc_info = self.process_manager.spawn_agent(
                agent_name=agent_name,
                config_file=str(agent.config_file),
                goal=agent.deployment.goal,
                env=env,
            )

            # Update next run time (will be set by scheduler)
            # For now, just mark as running

        except Exception as e:
            logger.error(f"Error running agent {agent_name}: {e}")

    def _setup_scheduled_agent(self, agent_name: str) -> None:
        """Setup schedule for an agent.

        Args:
            agent_name: Name of the agent
        """
        agent = self.agents[agent_name]

        if agent.deployment.mode != "scheduled":
            return

        # Parse cron schedule and convert to schedule library format
        # For now, support simple schedules
        # TODO: Full cron syntax parsing

        cron = agent.deployment.schedule

        # Simple parsing for common patterns
        if cron == "0 * * * *":  # Every hour
            schedule.every().hour.do(self._run_agent, agent_name)
        elif cron.startswith("*/"):
            # Every N minutes
            minutes = int(cron.split()[0][2:])
            schedule.every(minutes).minutes.do(self._run_agent, agent_name)
        elif cron.startswith("0 0 * * *"):  # Daily at midnight
            schedule.every().day.at("00:00").do(self._run_agent, agent_name)
        else:
            # Default to hourly if can't parse
            logger.warning(
                f"Couldn't parse schedule '{cron}' for {agent_name}, defaulting to hourly"
            )
            schedule.every().hour.do(self._run_agent, agent_name)

        logger.info(f"Scheduled agent {agent_name} with cron: {cron}")

    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Scheduler loop started")

        while self.running:
            try:
                # Run pending scheduled tasks
                schedule.run_pending()

                # Check for crashed agents that need restarting
                self._check_and_restart_crashed_agents()

                # Sleep briefly
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5)

        logger.info("Scheduler loop stopped")

    def _check_and_restart_crashed_agents(self) -> None:
        """Check for crashed agents and restart if configured."""
        for agent_name, agent in self.agents.items():
            if not agent.deployment.restart_on_crash:
                continue

            proc_info = self.process_manager.get_process_info(agent_name)

            if proc_info is None:
                continue

            # Check if crashed and under retry limit
            if (
                proc_info.status == "crashed"
                and proc_info.crash_count < agent.deployment.max_retries
            ):
                logger.warning(
                    f"Agent {agent_name} crashed (attempt {proc_info.crash_count}/{agent.deployment.max_retries}), restarting..."
                )

                # Wait a bit before restarting
                time.sleep(5)

                # Restart
                self._run_agent(agent_name)

            elif proc_info.crash_count >= agent.deployment.max_retries:
                logger.error(
                    f"Agent {agent_name} exceeded max retries, giving up"
                )
                # Stop trying to restart
                agent.deployment.restart_on_crash = False

    def start(self, foreground: bool = False) -> None:
        """Start the agent manager daemon.

        Args:
            foreground: Run in foreground instead of daemonizing
        """
        # Check if already running
        if self.pid_file.exists():
            try:
                with open(self.pid_file, "r") as f:
                    old_pid = int(f.read().strip())

                # Check if process is still running
                try:
                    os.kill(old_pid, 0)
                    logger.error(
                        f"Agent manager already running with PID {old_pid}"
                    )
                    return
                except OSError:
                    # Process not running, remove stale PID file
                    self.pid_file.unlink()

            except Exception as e:
                logger.warning(f"Error checking existing PID file: {e}")

        # Write PID file
        with open(self.pid_file, "w") as f:
            f.write(str(os.getpid()))

        logger.info(f"Starting agent manager (PID {os.getpid()})")

        # Load agent configurations
        self.load_agent_configs()

        if not self.agents:
            logger.warning("No enabled agents found, exiting")
            return

        # Setup schedules
        for agent_name in self.agents:
            self._setup_scheduled_agent(agent_name)

        # Start scheduler loop
        self.running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True
        )
        self.scheduler_thread.start()

        logger.info(f"Agent manager started with {len(self.agents)} agents")

        # Save initial state
        self.save_state()

        if foreground:
            # Run in foreground
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                self.stop()
        else:
            # Daemon mode - just return
            logger.info("Running in daemon mode")

    def stop(self) -> None:
        """Stop the agent manager and all agents."""
        logger.info("Stopping agent manager")

        # Stop scheduler
        self.running = False

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        # Stop all agents
        self.process_manager.stop_all()

        # Remove PID file
        if self.pid_file.exists():
            self.pid_file.unlink()

        logger.info("Agent manager stopped")

    def get_status(self) -> dict:
        """Get status of all agents.

        Returns:
            Dict with manager and agent status
        """
        uptime = None
        if self.pid_file.exists():
            try:
                import datetime

                uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(
                    self.pid_file.stat().st_mtime
                )
            except Exception:
                pass

        processes = self.process_manager.get_all_processes()

        agents_status = []
        for agent_name, agent in self.agents.items():
            proc_info = processes.get(agent_name)

            status = {
                "name": agent_name,
                "mode": agent.deployment.mode,
                "schedule": agent.deployment.schedule,
                "status": proc_info.status if proc_info else "not_started",
                "last_run": (
                    proc_info.last_run.isoformat()
                    if proc_info and proc_info.last_run
                    else None
                ),
                "next_run": (
                    proc_info.next_run.isoformat()
                    if proc_info and proc_info.next_run
                    else None
                ),
                "crash_count": proc_info.crash_count if proc_info else 0,
            }

            agents_status.append(status)

        return {
            "running": self.running,
            "pid": os.getpid() if self.running else None,
            "uptime": str(uptime) if uptime else None,
            "agents": agents_status,
        }

    def save_state(self) -> None:
        """Save manager state to disk."""
        try:
            state = self.get_status()

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def get_agent_logs(
        self, agent_name: str, lines: int = 50
    ) -> Optional[list[str]]:
        """Get logs for a specific agent.

        Args:
            agent_name: Name of the agent
            lines: Number of lines to retrieve

        Returns:
            List of log lines or None
        """
        return self.process_manager.tail_logs(agent_name, lines=lines)


def main():
    """Main entry point for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    manager = AgentManager()
    manager.start(foreground=True)


if __name__ == "__main__":
    main()
