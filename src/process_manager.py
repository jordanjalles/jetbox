"""Process manager for spawning and monitoring agent subprocesses."""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """Information about a running agent process."""

    agent_name: str
    process: Optional[subprocess.Popen]
    pid: Optional[int]
    started_at: datetime
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    status: str  # running, scheduled, stopped, crashed
    exit_code: Optional[int]
    crash_count: int
    log_path: Path


class ProcessManager:
    """Manages agent subprocesses and their lifecycle."""

    def __init__(self, log_dir: Path):
        """Initialize process manager.

        Args:
            log_dir: Directory to store agent logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, ProcessInfo] = {}
        self._lock = threading.Lock()

    def spawn_agent(
        self,
        agent_name: str,
        config_file: str,
        goal: str = "Execute scheduled task",
        env: Optional[dict] = None,
    ) -> ProcessInfo:
        """Spawn an agent subprocess.

        Args:
            agent_name: Name of the agent
            config_file: Path to agent config YAML
            goal: Goal/task for the agent
            env: Environment variables to pass

        Returns:
            ProcessInfo for the spawned process
        """
        with self._lock:
            # Create log file
            log_path = self.log_dir / f"{agent_name}.log"

            # Build command
            cmd = ["python", "agent.py", "--config", config_file, goal]

            logger.info(f"Spawning agent {agent_name}: {' '.join(cmd)}")

            try:
                # Open log file for appending
                log_file = open(log_path, "a", encoding="utf-8")

                # Spawn process
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=Path.cwd(),
                )

                # Create process info
                proc_info = ProcessInfo(
                    agent_name=agent_name,
                    process=process,
                    pid=process.pid,
                    started_at=datetime.now(),
                    last_run=datetime.now(),
                    next_run=None,
                    status="running",
                    exit_code=None,
                    crash_count=0,
                    log_path=log_path,
                )

                self.processes[agent_name] = proc_info

                logger.info(
                    f"Agent {agent_name} spawned with PID {process.pid}"
                )

                return proc_info

            except Exception as e:
                logger.error(f"Failed to spawn agent {agent_name}: {e}")
                raise

    def check_process_status(self, agent_name: str) -> Optional[str]:
        """Check if a process is still running.

        Args:
            agent_name: Name of the agent

        Returns:
            Status string or None if not found
        """
        with self._lock:
            if agent_name not in self.processes:
                return None

            proc_info = self.processes[agent_name]

            if proc_info.process is None:
                return proc_info.status

            # Check if process is still running
            poll_result = proc_info.process.poll()

            if poll_result is None:
                # Still running
                proc_info.status = "running"
                return "running"
            else:
                # Process has terminated
                proc_info.exit_code = poll_result
                proc_info.process = None
                proc_info.pid = None

                if poll_result == 0:
                    proc_info.status = "completed"
                    logger.info(
                        f"Agent {agent_name} completed successfully"
                    )
                else:
                    proc_info.status = "crashed"
                    proc_info.crash_count += 1
                    logger.error(
                        f"Agent {agent_name} crashed with exit code {poll_result}"
                    )

                return proc_info.status

    def stop_process(self, agent_name: str, timeout: int = 10) -> bool:
        """Stop an agent process gracefully.

        Args:
            agent_name: Name of the agent
            timeout: Seconds to wait before force killing

        Returns:
            True if stopped successfully
        """
        with self._lock:
            if agent_name not in self.processes:
                logger.warning(f"Agent {agent_name} not found")
                return False

            proc_info = self.processes[agent_name]

            if proc_info.process is None:
                logger.info(f"Agent {agent_name} already stopped")
                proc_info.status = "stopped"
                return True

            logger.info(f"Stopping agent {agent_name} (PID {proc_info.pid})")

            try:
                # Try graceful termination
                proc_info.process.terminate()

                # Wait for termination
                try:
                    proc_info.process.wait(timeout=timeout)
                    logger.info(f"Agent {agent_name} terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if still running
                    logger.warning(
                        f"Agent {agent_name} didn't stop gracefully, force killing"
                    )
                    proc_info.process.kill()
                    proc_info.process.wait()

                proc_info.status = "stopped"
                proc_info.process = None
                proc_info.pid = None
                return True

            except Exception as e:
                logger.error(f"Error stopping agent {agent_name}: {e}")
                return False

    def get_process_info(self, agent_name: str) -> Optional[ProcessInfo]:
        """Get information about a process.

        Args:
            agent_name: Name of the agent

        Returns:
            ProcessInfo or None if not found
        """
        with self._lock:
            # Update status before returning
            self.check_process_status(agent_name)
            return self.processes.get(agent_name)

    def get_all_processes(self) -> dict[str, ProcessInfo]:
        """Get information about all processes.

        Returns:
            Dict of agent_name -> ProcessInfo
        """
        with self._lock:
            # Update all statuses
            for agent_name in list(self.processes.keys()):
                self.check_process_status(agent_name)

            return dict(self.processes)

    def stop_all(self, timeout: int = 10) -> None:
        """Stop all running processes.

        Args:
            timeout: Seconds to wait for each process
        """
        logger.info("Stopping all agent processes")

        with self._lock:
            agent_names = list(self.processes.keys())

        for agent_name in agent_names:
            self.stop_process(agent_name, timeout=timeout)

    def tail_logs(
        self, agent_name: str, lines: int = 50
    ) -> Optional[list[str]]:
        """Get the last N lines from an agent's log file.

        Args:
            agent_name: Name of the agent
            lines: Number of lines to retrieve

        Returns:
            List of log lines or None if not found
        """
        with self._lock:
            if agent_name not in self.processes:
                return None

            log_path = self.processes[agent_name].log_path

        if not log_path.exists():
            return []

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
        except Exception as e:
            logger.error(f"Error reading log file for {agent_name}: {e}")
            return None
