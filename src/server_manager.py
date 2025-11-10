"""
Server manager for orchestrator.

Manages long-running server processes that persist across
multiple TaskExecutor invocations.
"""
from __future__ import annotations
import subprocess
import os
import time
import json
import threading
from pathlib import Path
from typing import Any


class ServerManager:
    """
    Manages background server processes for the orchestrator.

    Servers persist across TaskExecutor invocations and are only
    stopped when orchestrator exits or user explicitly stops them.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.servers: dict[str, dict[str, Any]] = {}
        self.running = False
        self.monitor_thread = None

        # Communication files for TaskExecutor IPC
        self.context_dir = workspace / ".agent_context"
        self.context_dir.mkdir(parents=True, exist_ok=True)

        self.request_file = self.context_dir / "server_requests.jsonl"
        self.response_file = self.context_dir / "server_responses.jsonl"

        # Track processed requests to avoid re-processing
        self.processed_count = 0

    def start_monitoring(self):
        """Start background thread to monitor server requests from TaskExecutor."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("[ServerManager] Monitoring started")

    def stop_monitoring(self):
        """Stop monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("[ServerManager] Monitoring stopped")

    def _monitor_loop(self):
        """Background thread that processes server requests from TaskExecutor."""
        while self.running:
            try:
                if not self.request_file.exists():
                    time.sleep(0.1)
                    continue

                # Read all requests
                with open(self.request_file, 'r') as f:
                    lines = f.readlines()

                # Process new requests only
                for line in lines[self.processed_count:]:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        request = json.loads(line)
                        response = self._handle_request(request)

                        # Write response
                        with open(self.response_file, 'a') as f:
                            f.write(json.dumps(response) + '\n')

                        self.processed_count += 1

                    except json.JSONDecodeError as e:
                        print(f"[ServerManager] Invalid JSON in request: {e}")
                        self.processed_count += 1

                time.sleep(0.1)

            except Exception as e:
                print(f"[ServerManager] Error in monitor loop: {e}")
                time.sleep(0.5)

    def _handle_request(self, request: dict) -> dict:
        """Handle a server request from TaskExecutor."""
        action = request.get("action")

        try:
            if action == "start":
                return self.start_server(
                    server_id=request["server_id"],
                    cmd=request["cmd"],
                    cwd=request["cwd"],
                    log_file=request["log_file"],
                )
            elif action == "stop":
                return self.stop_server(request["server_id"])
            elif action == "check":
                return self.check_server(
                    request["server_id"],
                    tail_lines=request.get("tail_lines", 20)
                )
            elif action == "list":
                return self.list_servers()
            else:
                return {"error": f"Unknown action: {action}"}

        except KeyError as e:
            return {"error": f"Missing required field: {e}"}
        except Exception as e:
            return {"error": f"Server operation failed: {e}"}

    def start_server(
        self,
        server_id: str,
        cmd: list[str],
        cwd: str,
        log_file: str,
    ) -> dict:
        """Start a server process in background."""
        # Check if server already running
        if server_id in self.servers:
            server = self.servers[server_id]
            if server["process"].poll() is None:
                return {
                    "error": f"Server '{server_id}' is already running (PID {server['process'].pid})",
                    "server_id": server_id,
                    "pid": server["process"].pid,
                }
            else:
                # Dead process, clean it up
                print(f"[ServerManager] Cleaning up dead server '{server_id}'")
                del self.servers[server_id]

        try:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Start server process
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                    text=True,
                )

            # Give it a moment to start
            time.sleep(0.5)

            # Check if it crashed immediately
            if process.poll() is not None:
                with open(log_file, 'r') as f:
                    error_output = f.read()[:500]
                return {
                    "error": f"Server failed to start (exit code {process.returncode})",
                    "output": error_output,
                }

            # Track the server
            self.servers[server_id] = {
                "process": process,
                "cmd": cmd,
                "cwd": cwd,
                "log_file": log_file,
                "pid": process.pid,
                "started_at": time.time(),
            }

            print(f"[ServerManager] Started server '{server_id}' (PID {process.pid})")

            return {
                "success": True,
                "server_id": server_id,
                "pid": process.pid,
                "log_file": log_file,
                "message": f"Server '{server_id}' started successfully. Logs: {log_file}",
            }

        except Exception as e:
            return {"error": f"Failed to start server: {e}"}

    def stop_server(self, server_id: str, timeout: int = 5) -> dict:
        """Stop a running server."""
        if server_id not in self.servers:
            return {"error": f"No server '{server_id}' is running"}

        server = self.servers[server_id]
        process = server["process"]

        try:
            print(f"[ServerManager] Stopping server '{server_id}' (PID {process.pid})")

            # Graceful shutdown (SIGTERM)
            process.terminate()

            try:
                process.wait(timeout=timeout)
                exit_code = process.returncode
                message = f"Server stopped gracefully (exit code: {exit_code})"
            except subprocess.TimeoutExpired:
                # Force kill (SIGKILL)
                print(f"[ServerManager] Server '{server_id}' didn't stop gracefully, killing...")
                process.kill()
                process.wait()
                message = "Server forcefully killed after timeout"

            # Remove from tracking
            del self.servers[server_id]

            return {
                "success": True,
                "server_id": server_id,
                "message": message,
            }

        except Exception as e:
            return {"error": f"Failed to stop server: {e}"}

    def check_server(self, server_id: str, tail_lines: int = 20) -> dict:
        """Check status of a server and get recent logs."""
        if server_id not in self.servers:
            return {"error": f"No server '{server_id}' is running"}

        server = self.servers[server_id]
        process = server["process"]
        poll_result = process.poll()

        # Read recent log output
        output = ""
        log_file = server["log_file"]
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    output = ''.join(lines[-tail_lines:]) if lines else ""
            except Exception as e:
                output = f"Error reading log: {e}"

        return {
            "success": True,
            "server_id": server_id,
            "pid": server["pid"],
            "status": "running" if poll_result is None else f"stopped (exit code: {poll_result})",
            "log_file": log_file,
            "recent_output": output,
            "uptime_seconds": int(time.time() - server["started_at"]),
        }

    def list_servers(self) -> dict:
        """List all tracked servers."""
        servers = []

        for server_id, server in list(self.servers.items()):
            process = server["process"]
            poll_result = process.poll()

            servers.append({
                "server_id": server_id,
                "pid": server["pid"],
                "status": "running" if poll_result is None else f"stopped ({poll_result})",
                "cmd": ' '.join(server["cmd"]),
                "log_file": server["log_file"],
                "uptime_seconds": int(time.time() - server["started_at"]),
            })

            # Clean up dead processes
            if poll_result is not None:
                print(f"[ServerManager] Cleaning up dead server '{server_id}'")
                del self.servers[server_id]

        return {
            "success": True,
            "servers": servers,
            "count": len(servers),
        }

    def stop_all_servers(self, timeout: int = 5):
        """
        Stop all running servers.

        Called when orchestrator exits.
        """
        if not self.servers:
            return

        print(f"[ServerManager] Stopping {len(self.servers)} server(s)...")

        for server_id in list(self.servers.keys()):
            result = self.stop_server(server_id, timeout=timeout)
            if "error" in result:
                print(f"[ServerManager] Warning: {result['error']}")

        print("[ServerManager] All servers stopped")

    def cleanup_old_requests(self):
        """
        Clean up request/response files when TaskExecutor starts.

        This prevents old requests from being re-processed.
        """
        if self.request_file.exists():
            self.request_file.unlink()
        if self.response_file.exists():
            self.response_file.unlink()

        self.processed_count = 0
