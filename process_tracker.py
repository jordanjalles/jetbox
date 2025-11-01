"""
Process Tracker - Track spawned test processes for safe termination.

This module provides a registry of running test processes that can be safely
killed without accidentally killing the Claude process.

Usage:
    from process_tracker import ProcessTracker

    # Register a process at start
    pid = os.getpid()
    ProcessTracker.register_process(pid, "evaluation", "3-level test suite")

    try:
        # Run tests
        run_tests()
    finally:
        # Unregister on completion
        ProcessTracker.unregister_process(pid)

    # Kill all tracked processes (from another process)
    ProcessTracker.kill_all_tracked_processes()
"""
import os
import signal
import json
from pathlib import Path
from datetime import datetime


class ProcessTracker:
    """Track spawned test processes to enable safe termination."""

    TRACKER_FILE = Path(".agent_context/running_processes.json")

    @classmethod
    def register_process(cls, pid: int, process_type: str, description: str):
        """
        Register a process as running.

        Args:
            pid: Process ID to register
            process_type: Type of process (e.g., "evaluation", "test_suite")
            description: Human-readable description
        """
        cls.TRACKER_FILE.parent.mkdir(exist_ok=True, parents=True)

        processes = cls._load_processes()
        processes[str(pid)] = {
            "pid": pid,
            "type": process_type,
            "description": description,
            "started": datetime.now().isoformat(),
            "ppid": os.getppid()  # Parent process ID
        }

        cls._save_processes(processes)
        print(f"[process_tracker] Registered PID {pid}: {description}")

    @classmethod
    def unregister_process(cls, pid: int):
        """
        Unregister a process (it completed).

        Args:
            pid: Process ID to unregister
        """
        processes = cls._load_processes()
        if str(pid) in processes:
            del processes[str(pid)]
            cls._save_processes(processes)
            print(f"[process_tracker] Unregistered PID {pid}")

    @classmethod
    def kill_all_tracked_processes(cls, exclude_ppid: int = None):
        """
        Kill all tracked processes, optionally excluding children of a specific parent.

        This is the safe way to stop tests without killing Claude.

        Args:
            exclude_ppid: Don't kill processes with this parent PID (e.g., Claude's PID)

        Returns:
            list: List of PIDs that were killed
        """
        processes = cls._load_processes()
        killed = []

        for pid_str, info in list(processes.items()):
            pid = int(pid_str)
            ppid = info.get("ppid")

            # Skip if this is a child of the excluded parent (Claude)
            if exclude_ppid and ppid == exclude_ppid:
                print(f"[process_tracker] Skipping PID {pid} (child of protected process {ppid})")
                continue

            # Check if process still exists
            try:
                os.kill(pid, 0)  # Signal 0 just checks existence
                print(f"[process_tracker] Killing PID {pid}: {info['description']}")
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except ProcessLookupError:
                print(f"[process_tracker] PID {pid} already dead, removing from tracker")
            except PermissionError:
                print(f"[process_tracker] No permission to kill PID {pid}")

            # Remove from tracker
            del processes[pid_str]

        cls._save_processes(processes)
        return killed

    @classmethod
    def list_running_processes(cls):
        """
        List all tracked running processes.

        Returns:
            list: List of process info dicts
        """
        processes = cls._load_processes()

        if not processes:
            print("[process_tracker] No tracked processes running")
            return []

        print(f"[process_tracker] {len(processes)} tracked process(es):")
        for pid_str, info in processes.items():
            print(f"  PID {pid_str}: {info['description']} (started {info['started']})")

        return list(processes.values())

    @classmethod
    def is_process_running(cls, pid: int) -> bool:
        """
        Check if a tracked process is still running.

        Args:
            pid: Process ID to check

        Returns:
            bool: True if process exists and is running
        """
        try:
            os.kill(pid, 0)  # Signal 0 just checks existence
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we can't signal it
            return True

    @classmethod
    def cleanup_dead_processes(cls):
        """Remove dead processes from tracker."""
        processes = cls._load_processes()
        cleaned = 0

        for pid_str in list(processes.keys()):
            pid = int(pid_str)
            if not cls.is_process_running(pid):
                print(f"[process_tracker] Removing dead process PID {pid}")
                del processes[pid_str]
                cleaned += 1

        if cleaned > 0:
            cls._save_processes(processes)
            print(f"[process_tracker] Cleaned up {cleaned} dead process(es)")

        return cleaned

    @classmethod
    def _load_processes(cls) -> dict:
        """Load process registry from file."""
        if not cls.TRACKER_FILE.exists():
            return {}

        try:
            with open(cls.TRACKER_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[process_tracker] ⚠️  Corrupt tracker file, resetting")
            return {}
        except Exception as e:
            print(f"[process_tracker] ⚠️  Error loading tracker: {e}")
            return {}

    @classmethod
    def _save_processes(cls, processes: dict):
        """Save process registry to file."""
        cls.TRACKER_FILE.parent.mkdir(exist_ok=True, parents=True)
        with open(cls.TRACKER_FILE, 'w') as f:
            json.dump(processes, f, indent=2)


if __name__ == "__main__":
    # When run directly, list tracked processes
    print("="*70)
    print("TRACKED PROCESSES")
    print("="*70)
    ProcessTracker.list_running_processes()
    print("\nCleaning up dead processes...")
    cleaned = ProcessTracker.cleanup_dead_processes()
    if cleaned == 0:
        print("✓ All tracked processes are alive")
